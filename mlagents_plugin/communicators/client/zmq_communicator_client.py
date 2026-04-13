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

    def recieve_action_from_llm(self, obs, selected_index):
        # if self.is_visual:
        payload = {}

        #print(f"Obs in client: {obs}")

        """
        for observation_type in self.observation_types:
            print(f"OBSERVATION TYPE: {observation_type['name']}")
            data = obs[observation_type['index']]
            key = observation_type.get('name', observation_type['type'])
            if observation_type['type'] == 'VISUAL':
                data = self.image_processer.process_batch_images(data)
                data = {f"agent_{i}": img_str for i, img_str in enumerate(data)}
            elif observation_type['type'] == 'GRID':
                data = self.image_processer.process_grid_images(obs_list=data, settings=observation_type)
                data = {f"agent_{i}": img_str for i, img_str in enumerate(data)}
            elif observation_type['type'] == 'RAYCAST':
                # qui il check e lo slice andrebbe fatto se i raycast sono stuckati
                if (observation_type['name'] == 'RAYCAST_FRONT'):
                    data = data[:, -21*5:]
                else:
                    data = data[:, -9*5:]
                data = {f"agent_{i}": state.tolist() for i, state in enumerate(data)}
            else: 
                data = {f"agent_{i}": state.tolist() for i, state in enumerate(data)}
            
            payload[key] = data
        
        """

        
        for observation_type in self.observation_types:
            #print(f"OBSERVATION TYPE: {observation_type['name']}")

            data_batch = obs[observation_type['index']]
            #print(f"Data batch shape: {data_batch}")

            agent_data = data_batch[selected_index : selected_index + 1]
            #print(f"agent_data: {agent_data}")

            key = observation_type.get('name', observation_type['type'])

            agent_key = f"agent_{selected_index}"

            if observation_type['type'] == 'VISUAL':
                agent_data = self.image_processer.process_batch_images(agent_data)
                agent_data = {agent_key: agent_data[0]}
            elif observation_type['type'] == 'GRID':
                agent_data = self.image_processer.process_grid_images(obs_list=agent_data, settings=observation_type)
                agent_data = {agent_key: agent_data[0]}
            elif observation_type['type'] == 'RAYCAST':
                if (observation_type['name'] == 'RAYCAST_FRONT'):
                    agent_data = agent_data[:, -21*5:]
                else:
                    agent_data = agent_data[:, -9*5:]
                agent_data = {agent_key: agent_data[0].tolist()}

            else:
                agent_data = {agent_key: agent_data[0].tolist()}

            payload[key] = agent_data
        
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
 