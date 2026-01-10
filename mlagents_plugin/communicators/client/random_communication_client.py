
from mlagents_plugin.communicators.client.base_communication_client import BaseCommunicationClient
from mlagents_plugin.communicators.action_generator.mock_action_generator import MockActionGenerator

class RandomCommunicationClient(BaseCommunicationClient):

    def __init__(self, discrete_branches: tuple[int], num_continuous_action: int, num_agents: int, is_visual : bool):
        super().__init__(discrete_branches=discrete_branches, num_continuous_action=num_continuous_action, num_agents=num_agents, is_visual=is_visual)
        self.communicator = MockActionGenerator(discrete_branches=discrete_branches, num_continuous_action=num_continuous_action, num_agents=num_agents)

    def recieve_action_from_llm(self, states):

        distributions = self.communicator.get_llm_policy(states)
        #distributions = {
        #    k: [arr.tolist() for arr in v]
        #    for k, v in distributions.items()
        #}
        #response_payload = {"discrete": distributions}
        return distributions
