import zmq
from mlagents_plugin.communicators.base_communication_client import BaseCommunicationClient

class ZMQCommunicatorClient(BaseCommunicationClient):

    def __init__(self, discrete_branches: tuple[int], num_continuous_action: int, num_agents: int):
        super().__init__(discrete_branches=discrete_branches, num_continuous_action=num_continuous_action, num_agents=num_agents)
        self.HOST = "127.0.0.1"
        self.PORT = 65432
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.HOST}:{self.PORT}")
        payload = {
            "type": "init",
            "discrete_branches": discrete_branches,
            "num_continuous_actions": num_continuous_action,
            "num_agents": num_agents
        }
        self.socket.send_json(payload)
        self.socket.recv_json()

    def recieve_action_from_llm(self, obs):
        payload = {
            "states": [state.tolist()] for state in obs
        }
        self.socket.send_json(payload)
        data = self.socket.recv_json()
        return data
