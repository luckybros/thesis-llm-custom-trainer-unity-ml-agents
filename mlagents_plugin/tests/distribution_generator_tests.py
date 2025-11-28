from mlagents_plugin.communicators.mock_communicator import MockCommunicator

communicator = MockCommunicator((), 2, 12)
print(communicator.get_llm_policy('a'))