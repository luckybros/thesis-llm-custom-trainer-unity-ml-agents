import zmq
import json
import argparse
import yaml
from typing import Optional
from mlagents_plugin.communicators.action_generator.llm_action_generator import LLMActionGenerator
from mlagents_plugin.utils.zmq_port_utils import resolve_zmq_port

class ZMQCommunicatorServer:

    #def __init__(self, action_generator_type: str, config_path: str):
    def __init__(
        self,
        config_path: str,
        visual: bool = False,
        worker_id: int = 0,
        base_port: Optional[int] = None,
        port: Optional[int] = None,
    ):
        self.HOST = "127.0.0.1"
        self.PORT = port if port is not None else resolve_zmq_port(worker_id=worker_id, base_port=base_port)
        self.action_generator = None
        self.config_path = config_path
        self.visual = visual

    def handle_client_logic(self, data):
        # print(type(data)) è un dict
        if self.action_generator is None:
            payload = {"response": "OK"}
            if data.get("type") == "init":
                discrete_branches = tuple(data["discrete_branches"])
                num_agents = data["num_agents"]
                num_continuous_actions = data["num_continuous_actions"]
                other_settings = {
                    'discrete_branches': discrete_branches,
                    'num_agents': num_agents,
                    'num_continuous_actions': num_continuous_actions
                }

                self.action_generator = LLMActionGenerator(
                    setting_path=self.config_path,
                    other_settings=other_settings,
                    visual=self.visual
                )

            return json.dumps(payload)

        if data.get("type") == "init":
            self.action_generator.clear_cache()
            return json.dumps({"response": "OK"})
        
        elif data.get("type") == "llm_only":
            llm_action = self.action_generator.get_llm_response(data)
            return llm_action

        elif data.get("type") == "llm_only_batch":
            raw_states = data["items"]
            llm_action_batch = self.action_generator.get_llm_response_batch(raw_states)
            return llm_action_batch

        elif data.get("type") == "drl_llm":
            llm_policy = self.action_generator.get_llm_policy(data)
            return llm_policy
        
        elif data.get("type") == "drl_llm_batch":
            raw_states = data["items"]
            llm_policy_batch = self.action_generator.get_llm_policy_batch(raw_states)
            return llm_policy_batch
        
        return None
        


def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--actgen", type=str, choices=ACTION_GENERATOR_REGISTRY.keys(), default="mock", help="Type of action generator")
    parser.add_argument("--config", type=str, default="", help="Config file path")
    parser.add_argument("--visual", action="store_true")
    parser.add_argument(
        "--port", type=int, default=None,
        help="Exact ZMQ port to bind to (overrides --base-port/--worker-id and MLAGENTS_ZMQ_BASE_PORT/MLAGENTS_WORKER_ID)",
    )
    parser.add_argument(
        "--base-port", type=int, default=None,
        help="Base ZMQ port; the actual port is base-port + worker-id (default: MLAGENTS_ZMQ_BASE_PORT env var, or an offset derived from ML-Agents' own --base-port, or 65432)",
    )
    parser.add_argument(
        "--worker-id", type=int, default=None,
        help="Worker id offset added to --base-port (default: MLAGENTS_WORKER_ID env var, or 0)",
    )
    args = parser.parse_args()

    #server = ZMQCommunicatorServer(action_generator_type=args.actgen, config_path=args.config)
    server = ZMQCommunicatorServer(
        config_path=args.config,
        visual=args.visual,
        worker_id=args.worker_id,
        base_port=args.base_port,
        port=args.port,
    )
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