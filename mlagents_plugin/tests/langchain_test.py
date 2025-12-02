from mlagents_plugin.communicators.action_generator.langchain_action_generator import LangchainActionGenerator

l = LangchainActionGenerator((3), 0, 1)

states = {
    'agent_0': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}

print(l.get_llm_policy(states))