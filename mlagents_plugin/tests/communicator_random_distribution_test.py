from mlagents_plugin.communicators.mock_communicator import MockCommunicator

m = MockCommunicator((3, 2), 2) # example with two different options for each branch of the action and two agents

d = m._generate_random_distributions()


print(d)