from mlagents_plugin.communicators.mock_communicator import MockCommunicator

communicator = MockCommunicator((2, 2), 2, 2)
print(communicator.get_llm_policy('a'))