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

    def recieve_action_from_llm(self, obs):
        # if self.is_visual:
        payload = {}

        #print(f"Obs in client: {obs}")

        for observation_type in self.observation_types:
            print(f"OBSERVATION TYPE: {observation_type['type']}")
            data = obs[observation_type['index']]
            if observation_type['type'] == 'VISUAL':
                data = self.image_processer.process_batch_images(data)
                data = {f"agent_{i}": img_str for i, img_str in enumerate(data)}
            elif observation_type['type'] == 'GRID':
                data = self.image_processer.process_grid_images(obs_list=data, settings=observation_type)
                data = {f"agent_{i}": img_str for i, img_str in enumerate(data)}
            else: 
                data = {f"agent_{i}": state.tolist() for i, state in enumerate(data)}
            payload[observation_type['type']] = data
        """
        if self.use_visual_obs:
            visual_obs = obs[0]
            vectorial_obs = obs[1]

            imgs = self.image_processer.process_batch_images(visual_obs)

            payload["vectorial"] = {f"agent_{i}": state.tolist() for i, state in enumerate(vectorial_obs)}
            payload["images"] = {f"agent_{i}": img_str for i, img_str in enumerate(imgs)}

        else:  
            if (len(obs) > 1):
                obs = obs[1]    # you have obs with different type of observation, for example visual. for now we obtain only the vector
            else:
                obs = obs[0]
            payload["vectorial"] = {f"agent_{i}": state.tolist() for i, state in enumerate(obs)}
        """
        
        #print(f"Obs in client: {payload}")
        self.socket.send_json(payload)
        data = self.socket.recv_json()
        return data
 