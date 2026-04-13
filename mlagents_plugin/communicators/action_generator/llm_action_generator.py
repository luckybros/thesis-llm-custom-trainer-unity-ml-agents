from mlagents_plugin.communicators.action_generator.llm_settings import LLMSettings
from mlagents_plugin.communicators.action_generator.state_abstration_module import StateAstractionModule
from mlagents_plugin.communicators.action_generator.action_optimization_module import ActionOptimizationModule
from mlagents_plugin.communicators.action_generator.prompt_builder import PromptBuilder
from mlagents_plugin.communicators.action_generator.langchain_model import LangchainModel
from mlagents_plugin.communicators.action_generator.action_parser import ActionParser
from mlagents_plugin.communicators.action_generator.distribution_generator import DistributionGenerator
import random

class LLMActionGenerator:
    def __init__(self, setting_path: str, other_settings: dict):
        self.config = LLMSettings(settings_path=setting_path, other_settings=other_settings)

        self.abstraction_module = StateAstractionModule(self.config)
        self.action_optimization_module = ActionOptimizationModule(self.config)
        self.prompt_builder = PromptBuilder(self.config) # comune a tutti a prescindere dal modello
        self.model = LangchainModel(self.config)
        self.action_parser = ActionParser(self.config)
        self.dist_generator = DistributionGenerator(self.config)

    def get_llm_policy(self, raw_state):

        abstract_state = self.abstraction_module.discretize(raw_state)
        prompt = self.prompt_builder.build_prompt(abstract_state)
        #print(f"[PROMPT]: {prompt}")
        model_output = self.model.call_llm(prompt)
        print(f"[MODEL OUTPUT]: {model_output}")
        action_dict = self.action_parser.parse_actions(model_output)
        print(f"[PARSED ACTIONS]: {action_dict}")
        distributions = self.dist_generator.generate_distributions(action_dict)
        
        distributions = self._check_distributions(distributions)
        print(f"[DISTRIBUTIONS]: {distributions}")
        return distributions
    
    def get_llm_response(self, raw_state):
        abstract_state = self.abstraction_module.discretize(raw_state)
        prompt = self.prompt_builder.build_prompt(abstract_state)
        model_output = self.model.call_llm(prompt)
        action_dict = self.action_parser.parse_actions(model_output)
        return action_dict
    
    def _check_distributions(self, distributions):
        for agent_id, dists in distributions['discrete'].items():
            for idx, dist in enumerate(dists):
                if sum(dist) != 1:
                    num_options = len(dist)
                    random_idx = random.randint(0, num_options - 1)
                    dist[random_idx] = 1.0
        return distributions
