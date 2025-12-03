from mlagents_plugin.communicators.action_generator.langchain_action_generator import LangchainActionGenerator

path_multi_agent = '/Users/luketto/Desktop/Tesi/ML_Agents_tesi/simple_plugin/mlagents_plugin/tests/BasicLLMProva.yaml'
path_one_agent = '/Users/luketto/Desktop/Tesi/ML_Agents_tesi/simple_plugin/mlagents_plugin/config/Basic/BasicLLM.yaml'
l = LangchainActionGenerator((3,), 0, 1, path_one_agent)

states = {
    'agent_0': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #'agent_1': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}

print(l.get_llm_policy(states))