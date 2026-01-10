import zmq
from mlagents_plugin.communicators.client.base_communication_client import BaseCommunicationClient
from mlagents_plugin.utils.image_processer import ImageProcesser

class ZMQCommunicatorClient(BaseCommunicationClient):

    def __init__(self, discrete_branches: tuple[int], num_continuous_action: int, num_agents: int, use_vectorial_obs: bool, use_visual_obs: bool):
        super().__init__(
            discrete_branches=discrete_branches, 
            num_continuous_action=num_continuous_action, 
            num_agents=num_agents, 
            use_vectorial_obs=use_vectorial_obs, 
            use_visual_obs=use_visual_obs
        )

        if self.use_visual_obs:
            self.image_processer = ImageProcesser()

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

        if self.use_visual_obs:
            print(f"NUMERO DI IMMAGINI: {len(obs[0])}")
            visual_obs = obs[0]
            vectorial_obs = obs[1]

            imgs = self.image_processer.process_batch_images(visual_obs)

            payload["states"] = {f"agent_{i}": state.tolist() for i, state in enumerate(vectorial_obs)}
            payload["images"] = {f"agent_{i}": img_str for i, img_str in enumerate(imgs)}

        else:  
            if (len(obs) > 1):
                obs = obs[1]    # you have obs with different type of observation, for example visual. for now we obtain only the vector
            payload["states"] = {f"agent_{i}": state.tolist() for i, state in enumerate(obs)}
        
        self.socket.send_json(payload)
        data = self.socket.recv_json()
        return data
 