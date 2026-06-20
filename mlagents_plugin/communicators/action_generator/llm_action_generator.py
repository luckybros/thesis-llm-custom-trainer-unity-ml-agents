from mlagents_plugin.communicators.action_generator.llm_settings import LLMSettings
from mlagents_plugin.communicators.action_generator.state_abstration_module import StateAstractionModule
from mlagents_plugin.communicators.action_generator.action_optimization_module import ActionOptimizationModule
from mlagents_plugin.communicators.action_generator.prompt_builder import PromptBuilder
from mlagents_plugin.communicators.action_generator.langchain_model import LangchainModel
#from mlagents_plugin.communicators.action_generator.vllm_model import VLLMModel
from mlagents_plugin.communicators.action_generator.action_parser import ActionParser
from mlagents_plugin.communicators.action_generator.distribution_generator import DistributionGenerator
from mlagents_plugin.caches.hash_cache import LLMHashCache
#from mlagents_plugin.caches.embedding_cache import LLMEmbeddingCache
import random

class LLMActionGenerator:
    def __init__(self, setting_path: str, other_settings: dict):
        self.config = LLMSettings(settings_path=setting_path, other_settings=other_settings)

        self.abstraction_module = StateAstractionModule(self.config)
        self.cache = LLMHashCache()
        self.action_optimization_module = ActionOptimizationModule(self.config)
        self.prompt_builder = PromptBuilder(self.config) # comune a tutti a prescindere dal modello
        self.model = LangchainModel(self.config)
        self.action_parser = ActionParser(self.config)
        self.dist_generator = DistributionGenerator(self.config)

    def get_llm_policy(self, raw_state):

        agent_id = raw_state['agent_id']
        abstract_state = self.abstraction_module.discretize(raw_state)
        cached_action = self.cache.query(abstract_state, agent_id)

        if cached_action is not None:
            action_dict_from_cached = self.prompt_builder.get_action_from_policy_list(cached_action)
            self.prompt_builder.update_history(agent_id, abstract_state, action_dict_from_cached)
            # qui dovrebbe restitui
            return cached_action
        
        #print(f"abstract state: {abstract_state}")
        prompt = self.prompt_builder.build_prompt(agent_id, abstract_state)
        print(f"Prompt: {prompt}")
        model_output = self.model.call_llm(prompt)
        action_dict = self.action_parser.parse_actions(model_output)
        action_chosen = self.dist_generator.get_actions(action_dict)
        distributions = self.dist_generator.generate_distributions(action_dict)
        distributions = self._check_distributions(distributions)

        self.prompt_builder.update_history(agent_id, abstract_state, action_dict)

        self.cache.update(abstract_state, distributions, agent_id)
        return distributions
    
    def get_llm_response(self, raw_state):
        print(f"raw_state: {raw_state}")
        agent_id = raw_state['agent_id']

        abstract_state = self.abstraction_module.discretize(raw_state)
        abstract_state.pop('type')
        cached_action = self.cache.query(abstract_state, agent_id)

        if cached_action is not None:
            obs_text = self.prompt_builder._build_obs_text(abstract_state)
            action_dict_from_cached = self.prompt_builder.get_action_from_list(cached_action)
            self.prompt_builder.update_history(agent_id, obs_text, action_dict_from_cached)
            return cached_action
        
        prompt = self.prompt_builder.build_prompt(agent_id, abstract_state)
        print(prompt)
        model_output = self.model.call_llm(prompt)
        print(f'model output: {model_output}')
        action_dict = self.action_parser.parse_actions(model_output)
        action_chosen = self.dist_generator.get_actions(action_dict)

        self.prompt_builder.update_history(agent_id, abstract_state, action_dict)

        self.cache.update(abstract_state, action_chosen, agent_id)
        return action_chosen
    
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
