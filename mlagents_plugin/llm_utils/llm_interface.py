from abc import ABC, abstractmethod

class LLMCommunicator(ABC):

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