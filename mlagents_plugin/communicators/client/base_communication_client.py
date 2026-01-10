from abc import ABC, abstractmethod

class BaseCommunicationClient(ABC):
    def __init__(self, discrete_branches, num_continuous_action, num_agents, is_visual):
        self.discrete_branches = discrete_branches
        self.num_continuous_action = num_continuous_action
        self.num_agents = num_agents
        self.is_visual = is_visual

    @abstractmethod
    def recieve_action_from_llm(self, obs):
        pass
