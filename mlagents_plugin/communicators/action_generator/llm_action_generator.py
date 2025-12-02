from abc import ABC, abstractmethod

class LLMActionGenerator(ABC):

    def __init__(self, discrete_branches: tuple[int], num_continuous_action: int, num_agents: int):
        self.discrete_branches = discrete_branches
        self.num_continuous_action = num_continuous_action
        self.num_agents = num_agents
        
    @abstractmethod
    def encode_state(self, state):
        """
        Function to convert numerical observations to text for LLMs to understand
        """
        pass

    @abstractmethod
    def get_llm_policy(self, text_state, cache):
        """
        Function to call LLMs, returns a probability distribution on the action space
        """ 
        pass