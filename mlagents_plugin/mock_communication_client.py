#!/usr/bin/env python3

import socket
import json

class MockCommunicationClient:

    def __init__(self, discrete_branches: tuple[int], num_agents: int):
        self.HOST = "127.0.0.1"
        self.PORT = 65432
        self.discrete_branches = discrete_branches
        self.num_agents = num_agents

        payload = {
            "type": "init",
            "discrete_branches": discrete_branches,
            "num_agents": num_agents
        }
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.connect((self.HOST, self.PORT))
            client.sendall(json.dumps(payload).encode('utf-8'))
            data = client.recv(1024)

    def recieve_action_from_llm(self, states):
        payload = {
            "states": [state.tolist() for state in states]
        }

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.connect((self.HOST, self.PORT))
            client.sendall(json.dumps(payload).encode('utf-8'))
            data = client.recv(4096)
            data = json.loads(data.decode('utf-8'))
            probabilities = data["probabilities"]
            user = data["user"]
            return (probabilities, user)


