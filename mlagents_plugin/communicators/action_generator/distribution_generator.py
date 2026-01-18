import numpy as np
class DistributionGenerator:

    def __init__(self, settings):
        self.discrete_branches = settings.discrete_branches
        self.num_continuous_action = settings.num_continuous_actions
        self.actions = settings.actions

    def generate_distributions(self, actions):
        """
        input:
        {
            0: {  # Agent ID
                "continuous": {
                    "Z Rotation (Steering)": "Rotate left",
                    "X Rotation (Tilt)": "Rotate backward"
                },
                "discrete": {
                    "Move": "Right"
                }
            },
        }
        """
        result = {}
        if len(self.discrete_branches) > 0:
            result["discrete"] = self._generate_discrete_distributions(actions)
        if self.num_continuous_action > 0:
            result["continuous"] = self._generate_continuous_distribuitons(actions)
        return result
    
    def _generate_discrete_distributions(self, result):
        distributions = {}
        
        for idx, agent_values in enumerate(result.values()):
            dists_for_agent = [np.zeros(size) for size in self.discrete_branches]
            discrete_value = agent_values['discrete']

            for i, (k, v) in enumerate(discrete_value.items()):
                # k->Z Rotation (Steering), v->Rotate left
                indx = self.get_index_of_action(k, v, False)
                dists_for_agent[i][indx] = 1.0
            distributions[f"agent_0-{idx}"] = dists_for_agent

        distributions = {
            k: [arr.tolist() for arr in v]
            for k, v in distributions.items()
        }

        return distributions
    
    def _generate_continuous_distribuitons(self, result):
        distributions = {}

        for idx, agent_values in enumerate(result.values()):
            dists_for_agent = [np.zeroes(2) for _ in range(self.num_continuous_action)]
            continuous_value = agent_values['continuous']

            for i, (k, v) in enumerate(continuous_value.items()):
                indx = self.get_index_of_action(k, v, True)
                value = self.get_continuous_value_by_index(k, indx)
                dists_for_agent[i][0] = value
            distributions[f"agent_0-{idx}"] = dists_for_agent

        distributions = {
            k: [arr.tolist() for arr in v]
            for k, v in distributions.items()
        }

        return distributions

    def get_index_of_action(self, action: str, choice: str, continuous: bool) -> int:
        return self.actions['continuous' if continuous else 'discrete'][action]['options'].index(choice)
    
    def get_continuous_value_by_index(self, action, idx) -> float:
        return self.actions['continuous'][action]['values'][idx]
