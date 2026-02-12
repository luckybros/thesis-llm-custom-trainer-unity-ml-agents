import numpy as np
import yaml
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_plugin.communicators.action_generator.llm_action_generator import LLMActionGenerator
from mlagents_plugin.utils.image_processer import ImageProcesser
import argparse

"""parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="", help="Config file path")
args = parser.parse_args()"""

with open('/Users/luketto/Desktop/Tesi/ML_Agents_tesi/simple_plugin/mlagents_plugin/config/Tanks/TanksMLAgentsLLMPlugin.yaml', 'r') as f:
    config = yaml.safe_load(f)
observation_types = config['behaviors']['Tanks']['hyperparameters']['observation_types']

image_processer = ImageProcesser()

env = UnityEnvironment(file_name=None, seed=1, side_channels=[])
env.reset()

behavior_name = list(env.behavior_specs.keys())[0]
spec = env.behavior_specs[behavior_name]
decision_steps, terminal_steps = env.get_steps(behavior_name)

other_settings = {
    'discrete_branches': spec.action_spec.discrete_branches,
    'num_agents': len(decision_steps),
    'num_continuous_actions': spec.action_spec.continuous_size
}

action_generator = LLMActionGenerator(setting_path="/Users/luketto/Desktop/Tesi/ML_Agents_tesi/simple_plugin/mlagents_plugin/config/Tanks/TanksLLM.yaml",
                                      other_settings=other_settings)

env.reset()

print("♾️  Start Endless Loop (Premi Ctrl+C per fermare)")

try:
    while True:
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        # if len(decision_steps) == 0 and len(terminal_steps) > 0:
            # env.reset()
            #continue # Salta al prossimo frame subito
        payload = {}

        obs = decision_steps.obs
        for observation_type in observation_types:
            data = obs[observation_type['index']]
            if observation_type['type'] == 'VISUAL':
                data = image_processer.process_batch_images(data)
                data = {f"agent_{i}": img_str for i, img_str in enumerate(data)}
            elif observation_type['type'] == 'GRID':
                data = image_processer.process_grid_images(obs_list=data, settings=observation_type)
                data = {f"agent_{i}": img_str for i, img_str in enumerate(data)}
            else: 
                data = {f"agent_{i}": state.tolist() for i, state in enumerate(data)}
            payload[observation_type['type']] = data

        result = action_generator.get_llm_response(payload)
        print(f"llm_result: {result}")
        n_agents = len(decision_steps)
        action = spec.action_spec.random_action(n_agents)
        env.set_actions(behavior_name, action)

        env.step()

except KeyboardInterrupt:
    print("\n🛑 Stop manuale.")

finally:
    env.close()