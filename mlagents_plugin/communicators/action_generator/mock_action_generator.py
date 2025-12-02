from typing import Dict, List
import numpy as np
from mlagents_plugin.communicators.action_generator.llm_action_generator import LLMActionGenerator

class MockActionGenerator(LLMActionGenerator):

    def __init__(self, discrete_branches: tuple[int], num_continuous_action: int, num_agents: int):
        """
        We tell the communicator how many actions he can take
        """
        self.discrete_branches = discrete_branches
        self.num_continuous_action = num_continuous_action
        self.num_agents = num_agents
        self.reevaluation_interval = 100
        self._call_count = 0
        # Uniform distribution for every branch
        # self._cached_distributions = [[np.ones(size) / size for size in self.discrete_branches] for i in range (self.num_agents)]

    def _generate_random_distributions(self) -> Dict[str, List[np.ndarray]]:
        """
        Helper method to generate a random log-prob distribution for discrete actions, per agent.
        """
        distributions = {}
        for i in range(self.num_agents):
            logits_per_agent = [np.random.rand(size) for size in self.discrete_branches]
            agent_distributions = []
            # Normalize
            for logits in logits_per_agent:
                distribution_sum = np.sum(logits)
                if distribution_sum == 0.0:
                    normalized_dist = np.ones(len(logits)) / len(logits)
                else:
                    normalized_dist = logits / distribution_sum
                normalized_dist = np.log(normalized_dist)   # log-probs
                agent_distributions.append(normalized_dist)
            distributions[f"agent_0-{i}"] = agent_distributions

        distributions = {
            k: [arr.tolist() for arr in v]
            for k, v in distributions.items()
        }
        return distributions
    
    def _generate_random_continuous_params(self) -> Dict[str, List[np.ndarray]]:
        """
        Generates mean and std vector (size=num_continuous_action) for each agent
        """
        params = {}
        for i in range(self.num_agents):
            agent_distribution = []
            means = np.random.uniform(-1, 1, self.num_continuous_action)
            stds = np.random.uniform(1e-2, 1.0, self.num_continuous_action)  # use positive std!
            for mean, std in zip(means, stds):
                agent_distribution.append([mean, std])
            params[f"agent_0-{i}"] = agent_distribution

        return params
    
    def encode_state(self, state):
        return state
        
    def get_llm_policy(self, text_state):
        """
        In the mock communicator we return a random distribution over the action space, and we choose one every 100 action
        """
        payload = {}
        if len(self.discrete_branches) > 0:
            payload["discrete"] = self._generate_random_distributions()
        if self.num_continuous_action > 0:
            payload["continuous"] = self._generate_random_continuous_params()
        return payload
        """
        # Ogni 100 passi usa le distribuzioni dell'utente
        if self._call_count % self.reevaluation_interval == 0:

            new_distributions = []
            for j in range(self.num_agents):
                agent_distributions = []
                for i, branch_size in enumerate(self.discrete_branches):
                    chosen_action = int(input(f"Agent {j}: Choose the {i} action in between 0 and {branch_size-1}"))
                    distr = np.zeros(branch_size)
                    distr[chosen_action] = 1.0
                    agent_distributions.append(distr)
                new_distributions.append(agent_distributions)
            return {"discrete", self._generate_random_distributions()}
        else:
            return self._generate_random_distributions()
        """
