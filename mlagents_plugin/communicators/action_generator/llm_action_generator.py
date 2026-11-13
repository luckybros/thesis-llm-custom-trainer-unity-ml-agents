from mlagents_plugin.communicators.action_generator.llm_settings import LLMSettings
from mlagents_plugin.communicators.action_generator.state_abstration_module import StateAstractionModule
from mlagents_plugin.communicators.action_generator.action_optimization_module import ActionOptimizationModule
from mlagents_plugin.communicators.action_generator.prompt_builder import PromptBuilder
#from mlagents_plugin.communicators.action_generator.langchain_model import LangchainModel
#from mlagents_plugin.communicators.action_generator.cerberus_model import CerberusModel
#from mlagents_plugin.communicators.action_generator.vllm_model import VLLMModel
from mlagents_plugin.communicators.action_generator.ollama_model import OllamaModel
from mlagents_plugin.communicators.action_generator.action_parser import ActionParser
from mlagents_plugin.communicators.action_generator.distribution_generator import DistributionGenerator
from mlagents_plugin.caches.hash_cache import LLMHashCache
#from mlagents_plugin.caches.embedding_cache import LLMEmbeddingCache
import random

class LLMActionGenerator:
    def __init__(self, setting_path: str, other_settings: dict, visual: bool = False):
        self.config = LLMSettings(settings_path=setting_path, other_settings=other_settings)

        self.visual = visual

        self.abstraction_module = StateAstractionModule(self.config)
        self.cache = self._build_cache()
        self.action_optimization_module = ActionOptimizationModule(self.config)
        self.prompt_builder = PromptBuilder(self.config) # comune a tutti a prescindere dal modello
        self.model = OllamaModel(self.config)
        self.action_parser = ActionParser(self.config)
        self.dist_generator = DistributionGenerator(self.config)

        # numero massimo di re-interrogazioni dell'LLM quando l'output non e' parsabile.
        # superato questo limite si usa l'azione di default per non bloccare il flusso.
        self.max_llm_retries = self.config.max_llm_retries

    # ------------------------------------------------------------------ #
    # DRL + LLM: l'LLM restituisce una distribuzione (one-hot) per ramo
    # ------------------------------------------------------------------ #
    def get_llm_policy(self, raw_state):

        agent_id = raw_state['agent_id']
        abstract_state = self.abstraction_module.discretize(raw_state)
        cached_action = self.cache.query(abstract_state, agent_id)

        if cached_action is not None:
            action_dict_from_cached = self.prompt_builder.get_action_from_policy_list(cached_action)
            self.update_prompt_history(agent_id, abstract_state, action_dict_from_cached)
            return cached_action

        prompt = self.prompt_builder.build_prompt(agent_id, abstract_state)
        action_dict = self._query_llm_batch([prompt])[0]
        distributions = self.dist_generator.generate_distributions(action_dict)
        distributions = self._check_distributions(distributions)

        self.update_prompt_history(agent_id, abstract_state, action_dict)

        self.cache.update(abstract_state, distributions, agent_id)
        return distributions

    def get_llm_policy_batch(self, raw_states):
        """
        raw_states: lista di raw_state (uno per agente), stesso step.
        Ritorna: dict {agent_id: distributions}, stesso formato per-agente
                 restituito da get_llm_policy, ma processato in batch.
        """
        agent_ids, abstract_states, results, miss_indices, miss_prompts = \
            self._split_hits_and_misses(raw_states, self.prompt_builder.get_action_from_policy_list)

        action_dicts = self._query_llm_batch(miss_prompts)
        for idx, action_dict in zip(miss_indices, action_dicts):
            agent_id = agent_ids[idx]
            abstract_state = abstract_states[idx]
            distributions = self.dist_generator.generate_distributions(action_dict)
            distributions = self._check_distributions(distributions)
            self.update_prompt_history(agent_id, abstract_state, action_dict)
            self.cache.update(abstract_state, distributions, agent_id)
            results[agent_id] = distributions

        return results

    # ------------------------------------------------------------------ #
    # Solo LLM: l'LLM sceglie direttamente l'azione (lista di indici per ramo)
    # ------------------------------------------------------------------ #
    def get_llm_response_batch(self, raw_states):
        """
        raw_states: lista di raw_state (uno per agente), stesso step.
        Ritorna: dict {agent_id: [idx_ramo_0, idx_ramo_1, ...]}, stesso formato
                 per-agente restituito da get_llm_response, ma processato in batch.
        """
        agent_ids, abstract_states, results, miss_indices, miss_prompts = \
            self._split_hits_and_misses(raw_states, self.prompt_builder.get_action_from_response_list)

        action_dicts = self._query_llm_batch(miss_prompts)
        for idx, action_dict in zip(miss_indices, action_dicts):
            agent_id = agent_ids[idx]
            abstract_state = abstract_states[idx]
            action_chosen = self.dist_generator.get_actions(action_dict)
            self.update_prompt_history(agent_id, abstract_state, action_dict)
            self.cache.update(abstract_state, action_chosen, agent_id)
            results[agent_id] = action_chosen

        return results

    def get_llm_response(self, raw_state):
        agent_id = raw_state['agent_id']
        return self.get_llm_response_batch([raw_state])[agent_id]

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _split_hits_and_misses(self, raw_states, cached_to_action_dict):
        """
        Discretizza gli stati, interroga la cache e separa gli agenti con cache-hit
        (gia' risolti in `results`) da quelli per cui serve chiamare l'LLM.
        """
        agent_ids = [rs['agent_id'] for rs in raw_states]
        abstract_states = []
        for rs in raw_states:
            abstract_state = self.abstraction_module.discretize(rs)
            abstract_state.pop('type', None)
            abstract_states.append(abstract_state)

        results = {}
        miss_indices = []
        miss_prompts = []

        for i, (agent_id, abstract_state) in enumerate(zip(agent_ids, abstract_states)):
            cached_action = self.cache.query(abstract_state, agent_id)

            if cached_action is not None:
                action_dict_from_cached = cached_to_action_dict(cached_action)
                self.update_prompt_history(agent_id, abstract_state, action_dict_from_cached)
                results[agent_id] = cached_action
            else:
                miss_indices.append(i)
                miss_prompts.append(self.prompt_builder.build_prompt(agent_id, abstract_state))

        return agent_ids, abstract_states, results, miss_indices, miss_prompts

    def _query_llm_batch(self, prompts):
        """
        Una sola chiamata batch all'LLM per tutti i prompt; i prompt con output non
        parsabile vengono ri-sottomessi fino a max_llm_retries volte, poi si usa
        default_agent_action. Ritorna una lista di action_dict allineata a `prompts`.
        """
        action_dicts = [None] * len(prompts)
        pending = list(range(len(prompts)))
        attempt = 0

        while pending and attempt <= self.max_llm_retries:
            model_outputs = self.model.call_llm_batch([prompts[i] for i in pending])
            still_pending = []
            for i, model_output in zip(pending, model_outputs):
                action_dict = self.action_parser.parse_actions(model_output)
                if action_dict:
                    action_dicts[i] = action_dict
                else:
                    still_pending.append(i)
            if still_pending:
                print(f"[LLMActionGenerator] {len(still_pending)} output non validi "
                      f"(tentativo {attempt + 1}/{self.max_llm_retries + 1})")
            pending = still_pending
            attempt += 1

        for i in pending:
            action_dicts[i] = self._default_action_dict()

        return action_dicts

    def _default_action_dict(self):
        """default_agent_action dallo YAML; se assente/non valida, prima opzione di ogni ramo."""
        action_dict = self.action_parser.parse_actions(self.config.default_agent_action or "")
        if action_dict:
            return action_dict
        return {0: {
            "continuous": {},
            "discrete": {name: a["options"][0] for name, a in self.config.actions["discrete"].items()},
        }}

    def _check_distributions(self, distributions):
        for agent_id, dists in distributions['discrete'].items():
            for idx, dist in enumerate(dists):
                if sum(dist) != 1:
                    num_options = len(dist)
                    random_idx = random.randint(0, num_options - 1)
                    dist[random_idx] = 1.0
        return distributions

    def clear_cache(self):
        self.cache.clear_cache()

    def _build_cache(self):
        if self.visual:
            # import lazy: torch/transformers/faiss servono solo per la versione visuale
            from mlagents_plugin.caches.visual_embedding_cache import VisualLLMEmbeddingCache
            return VisualLLMEmbeddingCache(self.config.history_length)
        return LLMHashCache(self.config.history_length)

    def update_prompt_history(self, agent_id, abstract_state, action_dict):
        # la history testuale si basa su obs_template: con osservazioni solo visuali non c'e' testo da salvare
        if self.visual and not self.config.use_vectorial_obs:
            return
        self.prompt_builder.update_history(agent_id, abstract_state, action_dict)
