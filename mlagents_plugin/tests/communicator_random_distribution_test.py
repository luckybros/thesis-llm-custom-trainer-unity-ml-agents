from mlagents_plugin.communicators.action_generator.mock_action_generator import MockCommunicator

m = MockCommunicator((3, 2), 2) # example with two different options for each branch of the action and two agents

d = m._generate_random_distributions()


print(d)