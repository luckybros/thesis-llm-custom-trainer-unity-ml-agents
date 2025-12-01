import zmq 
import json
from mlagents_plugin.communicators.mock_communicator import MockCommunicator

class ZMQCommunicatorServer:

    def __init__(self):
        self.HOST = "127.0.0.1"
        self.PORT = 65432
        self.communicator : MockCommunicator = None

    def handle_client_logic(self, data):
        print(f'Recieved: {data}')
        if self.communicator is None:
            payload = {"response": "OK"}
            if data.get("type") == "init":
                discrete_branches = tuple(data["discrete_branches"])
                num_agents = data["num_agents"]
                num_continuous_actions = data["num_continuous_actions"]
                self.communicator = MockCommunicator(discrete_branches=discrete_branches, num_continuous_action=num_continuous_actions, num_agents=num_agents)
            return json.dumps(payload)

        states = data["states"]
        text_states = [self.communicator.encode_state(state) for state in states]
        llm_policy = self.communicator.get_llm_policy(text_states)
        return llm_policy


def main():
    server = ZMQCommunicatorServer()
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