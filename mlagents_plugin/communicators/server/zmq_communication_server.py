import zmq 
import json
import argparse
import yaml
from mlagents_plugin.communicators.action_generator.llm_action_generator import LLMActionGenerator


"""
ACTION_GENERATOR_REGISTRY = {
    "mock": MockActionGenerator,
    "lang": LangchainActionGenerator
}
"""

class ZMQCommunicatorServer:

    #def __init__(self, action_generator_type: str, config_path: str):
    def __init__(self, config_path: str):
        self.HOST = "127.0.0.1"
        self.PORT = 65432
        self.action_generator = None
        #self.act_gen_cls = ACTION_GENERATOR_REGISTRY[action_generator_type]
        self.config_path = config_path
        #self.is_visual = is_visual

    def handle_client_logic(self, data):
        # print(type(data)) è un dict
        if self.action_generator is None:
            payload = {"response": "OK"}
            if data.get("type") == "init":
                discrete_branches = tuple(data["discrete_branches"])
                num_agents = data["num_agents"]
                num_continuous_actions = data["num_continuous_actions"]
                """
                self.action_generator = self.act_gen_cls(
                    discrete_branches=discrete_branches, 
                    num_continuous_action=num_continuous_actions, 
                    num_agents=num_agents, 
                    settings_path=self.config_path
            )
                """
                other_settings = {
                    'discrete_branches': discrete_branches,
                    'num_agents': num_agents,
                    'num_continuous_actions': num_continuous_actions
                }

                self.action_generator = LLMActionGenerator(
                    setting_path=self.config_path,
                    other_settings=other_settings
                )

            return json.dumps(payload)

        # print(f"state: {data}")
        llm_policy = self.action_generator.get_llm_policy(data)
        return llm_policy


def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--actgen", type=str, choices=ACTION_GENERATOR_REGISTRY.keys(), default="mock", help="Type of action generator")
    parser.add_argument("--config", type=str, default="", help="Config file path")
    args = parser.parse_args()

    #server = ZMQCommunicatorServer(action_generator_type=args.actgen, config_path=args.config)
    server = ZMQCommunicatorServer(config_path=args.config)
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://{server.HOST}:{server.PORT}")
    print(f"[ZMQ Server] Listening on port {server.PORT}")

    try:
        while True:
            data = socket.recv_json()
            response = server.handle_client_logic(data)
            socket.send_json(response)
    except KeyboardInterrupt:
        print("\n[ZMQ Server] Stopped by user (Ctrl+C)")
    finally:
        socket.close()
        context.term()
        print("[ZMQ Server] Shutdown.")

if __name__== "__main__":
    main()