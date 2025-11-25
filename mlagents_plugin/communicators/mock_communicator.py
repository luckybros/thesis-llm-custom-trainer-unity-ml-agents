from typing import Dict, List
import numpy as np
from llm_communicator_interface import LLMCommunicator

class MockCommunicator(LLMCommunicator):

    def __init__(self, discrete_branches: tuple[int], num_agents: int):
        """
        We tell the communicator how many actions he can take
        """
        self.discrete_branches = discrete_branches
        self.num_agents = num_agents
        self.reevaluation_interval = 100
        self._call_count = 0
        # Uniform distribution for every branch
        self._cached_distributions = [[np.ones(size) / size for size in self.discrete_branches] for i in range (self.num_agents)]

    def _generate_random_distributions(self) -> Dict[int, List[float]]:
        """
        Helper method to generate a random distribution
        """
        distributions = {}
        for i in range(self.num_agents):
            logits_per_agent = [np.random.rand(size) for size in self.discrete_branches]
            agent_distributions = []
            for logits in logits_per_agent:
                distribution_sum = np.sum(logits)
                if distribution_sum == 0.0:
                    normalized_dist = np.ones(len(logits)) / len(logits)
                else:
                    normalized_dist = logits / distribution_sum
                normalized_dist = np.log(normalized_dist)
                agent_distributions.append(normalized_dist)
            distributions["agent_0-"+str(i)] = agent_distributions
        return distributions
    
    def encode_state(self, state):
        return state
        
    def get_llm_policy(self, text_state):
        """
        In the mock communicator we return a random distribution over the action space, and we choose one every 100 action
        """
        self._call_count += 1

        # Ogni 100 passi usa le distribuzioni dell'utente
        if self._call_count % self.reevaluation_interval == 0:
            """
            new_distributions = []
            for j in range(self.num_agents):
                agent_distributions = []
                for i, branch_size in enumerate(self.discrete_branches):
                    chosen_action = int(input(f"Agent {j}: Choose the {i} action in between 0 and {branch_size-1}"))
                    distr = np.zeros(branch_size)
                    distr[chosen_action] = 1.0
                    agent_distributions.append(distr)
                new_distributions.append(agent_distributions)
            """
            self._cached_distributions = self._generate_random_distributions()
            return self._cached_distributions
        else:
            return self._generate_random_distributions()
