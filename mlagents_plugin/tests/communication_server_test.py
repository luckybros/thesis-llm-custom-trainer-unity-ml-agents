import json
from mlagents_plugin.communicators.server.mock_communication_server import MockCommunicationServer

m = MockCommunicationServer()
mock_payload = {
    "type": "init",
    "discrete_branches": (3, 2),  
    "num_agents": 2
}
response = m.handle_client_logic(json.dumps(mock_payload).encode('utf-8'))
print("Init:", response)

mock_state_payload = {
    "states": [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ]
}

response = m.handle_client_logic(json.dumps(mock_state_payload).encode('utf-8'))
print(response)