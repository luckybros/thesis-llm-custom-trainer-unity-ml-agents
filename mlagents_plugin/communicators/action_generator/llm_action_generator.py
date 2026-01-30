from mlagents_plugin.communicators.action_generator.llm_settings import LLMSettings
from mlagents_plugin.communicators.action_generator.state_abstration_module import StateAstractionModule
from mlagents_plugin.communicators.action_generator.action_optimization_module import ActionOptimizationModule
from mlagents_plugin.communicators.action_generator.prompt_builder import PromptBuilder
from mlagents_plugin.communicators.action_generator.langchain_model import LangchainModel
from mlagents_plugin.communicators.action_generator.action_parser import ActionParser
from mlagents_plugin.communicators.action_generator.distribution_generator import DistributionGenerator

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
        print(f"abstract_state: {abstract_state}")
        prompt = self.prompt_builder.build_prompt(abstract_state)
        model_output = self.model.call_llm(prompt)
        action_dict = self.action_parser.parse_actions(model_output)
        distributions = self.dist_generator.generate_distributions(action_dict)
        
        return distributions
