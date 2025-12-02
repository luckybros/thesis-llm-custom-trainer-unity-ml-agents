from mlagents_plugin.communicators.action_generator.mock_action_generator import MockCommunicator

communicator = MockCommunicator((), 2, 12)
print(communicator.get_llm_policy('a'))