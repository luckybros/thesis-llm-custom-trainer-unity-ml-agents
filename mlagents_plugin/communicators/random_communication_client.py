
from mlagents_plugin.communicators.base_communication_client import BaseCommunicationClient
from mlagents_plugin.communicators.mock_communicator import MockCommunicator

class RandomCommunicationClient(BaseCommunicationClient):

    def __init__(self, discrete_branches: tuple[int], num_agents: int):

        super().__init__(discrete_branches=discrete_branches, num_agents=num_agents)
        self.communicator = MockCommunicator(discrete_branches=discrete_branches, num_agents=num_agents)

    def recieve_action_from_llm(self, states):

        distributions = self.communicator.get_llm_policy(states)
        distributions = {
            k: [arr.tolist() for arr in v]
            for k, v in distributions.items()
        }
        response_payload = {"discrete": distributions}
        return response_payload
