import socket
import json

from mlagents_plugin.communicators.action_generator.mock_action_generator import MockActionGenerator

class MockCommunicationServer:

    def __init__(self):
        self.HOST = "127.0.0.1"
        self.PORT = 65432
        self.communicator : MockActionGenerator = None

    def handle_client_logic(self, data):

        request_json = json.loads(data.decode('utf-8'))

        if self.communicator is None:
            payload = {"response": "OK"}
            if request_json.get("type") == "init":
                discrete_branches = tuple(request_json["discrete_branches"])
                num_agents = request_json["num_agents"]
                self.communicator = MockActionGenerator(discrete_branches=discrete_branches, num_agents=num_agents)
            return json.dumps(payload).encode('utf-8')

        states = request_json["states"]
        text_states = [self.communicator.encode_state(state) for state in states]
        distributions = self.communicator.get_llm_policy(text_states)
        # distributions = [[action_list.tolist() for action_list in agent_list] for agent_list in distributions]
        distributions = {
            k: [arr.tolist() for arr in v]
            for k, v in distributions.items()
        }
        response_payload = {"discrete": distributions}
        return json.dumps(response_payload).encode('utf-8')

def main():

    communication_server = MockCommunicationServer()

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.bind((communication_server.HOST, communication_server.PORT))
            server.listen()

            while True:
                client_conn, client_addr = server.accept()

                with client_conn:
                    data = client_conn.recv(131072)
                    if not data:
                        continue

                    response = communication_server.handle_client_logic(data)
                    client_conn.sendall(response)
    except KeyboardInterrupt:
        print("\nServer interrotto dall'utente (Ctrl+C).")
    #except Exception as e:
    #    e.print_exc()
    finally:
        print("Chiusura del server.")

if __name__ == "__main__":
    main()