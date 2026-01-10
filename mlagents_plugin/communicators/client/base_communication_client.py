from abc import ABC, abstractmethod

class BaseCommunicationClient(ABC):
    def __init__(
            self, 
            discrete_branches, 
            num_continuous_action, 
            num_agents, 
            use_vectorial_obs,
            use_visual_obs
    ):
        self.discrete_branches = discrete_branches
        self.num_continuous_action = num_continuous_action
        self.num_agents = num_agents
        self.use_vectorial_obs = use_vectorial_obs
        self.use_visual_obs = use_visual_obs

    @abstractmethod
    def recieve_action_from_llm(self, obs):
        pass
