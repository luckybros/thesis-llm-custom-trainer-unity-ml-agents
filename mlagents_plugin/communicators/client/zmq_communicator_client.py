import zmq
from mlagents_plugin.communicators.client.base_communication_client import BaseCommunicationClient
from mlagents_plugin.utils.image_processer import ImageProcesser

class ZMQCommunicatorClient(BaseCommunicationClient):

    def __init__(self, discrete_branches: tuple[int], num_continuous_action: int, num_agents: int, observation_types: dict):
        super().__init__(
            discrete_branches=discrete_branches, 
            num_continuous_action=num_continuous_action, 
            num_agents=num_agents, 
            observation_types=observation_types
        )

        self.image_processer = ImageProcesser()
        self.observation_types = observation_types

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

    def receive_distribution_from_llm(self, obs, selected_index):

        payload = self._get_data_from_obs(obs, selected_index)
        payload["agent_id"] = next(iter(payload['VECTORIAL']))
        payload["type"] = "drl_llm"
        self.socket.send_json(payload)
        data = self.socket.recv_json()
        return data
    
    def receive_action_from_llm(self, obs, selected_index):
        
        payload = self._get_data_from_obs(obs, selected_index)
        payload["type"] = "llm_only"
        payload["agent_id"] = next(iter(payload['VECTORIAL']))

        self.socket.send_json(payload)
        data = self.socket.recv_json()
        return data

    def _get_data_from_obs(self, obs, selected_index):
        
        payload = {}
        for observation_type in self.observation_types:
            data_batch = obs[observation_type['index']]

            agent_data = data_batch[selected_index : selected_index + 1]

            key = observation_type.get('name', observation_type['type'])

            agent_key = f"agent-{selected_index}"

            if observation_type['type'] == 'VISUAL':
                agent_data = self.image_processer.process_batch_images(agent_data)
                agent_data = {agent_key: agent_data[0]}
            elif observation_type['type'] == 'GRID':
                agent_data = self.image_processer.process_grid_images(obs_list=agent_data, settings=observation_type)
                agent_data = {agent_key: agent_data[0]}
            elif observation_type['type'] == 'RAYCAST':
                if (observation_type['name'] == 'RAYCAST_FRONT'):
                    agent_data = agent_data[:, -21*5:]
                    #agent_data = agent_data[:, -15*5:]
                else:
                    agent_data = agent_data[:, -9*5:]
                agent_data = {agent_key: agent_data[0].tolist()}

            else:
                agent_data = {agent_key: agent_data[0].tolist()}

            payload[key] = agent_data

        return payload