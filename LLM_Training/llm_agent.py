import os
import numpy as np
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.stats_side_channel import StatsSideChannel
from mlagents_plugin.oracle_side_channel import OracleSideChannel
from mlagents_plugin.communicators.client.zmq_communicator_client import ZMQCommunicatorClient
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.exception import UnityTimeOutException, UnityCommunicatorStoppedException, UnityEnvironmentException
from torch.utils.tensorboard import SummaryWriter
import time
import yaml
from PIL import ImageGrab

output_frames_dir = os.path.join("results", f"frames")
os.makedirs(output_frames_dir, exist_ok=True)
secondo_monitor_box = (1920, 0, 3840, 1080)

run_id = "Tank_LLM_Baseline"

# --- CONFIGURATION ---
EXPERIMENT_NAME = "Tank_LLM_Baseline"
NUM_SEEDS = 10          
TOTAL_STEP_BUDGET = 500000 
summary_freq = 1000

# Variabili globali per la singola iterazione
global_step_count = 0
run_count = 0
accumulated_stats = {}

# Variabili globali per la singola iterazione
global_step_count = 0
run_count = 0
accumulated_stats = {}

with open('/Users/luketto/Desktop/Tesi/ML_Agents_tesi/simple_plugin/mlagents_plugin/config/Tanks/TanksMLAgentsLLMPlugin.yaml', 'r') as f:
    config = yaml.safe_load(f)

# lista di dict
# [{'name': 'RAYCAST_FRONT', 'type': 'RAYCAST', 'index': 0}, {'name': 'RAYCAST_BACK', 'type': 'RAYCAST', 'index': 1}, {'name': 'VECTORIAL', 'type': 'VECTORIAL', 'index': 2}]
observation_types = config['behaviors']['Tanks']['hyperparameters']['observation_types']

def create_and_run_environment(base_seed, current_writer, run_count):
    global global_step_count, accumulated_stats

    rng = np.random.default_rng(base_seed + run_count)

    start_time = time.time()

    print(f"\n======================================")
    print(f"Avvio Run #{run_count} (Step Globale: {global_step_count})")
    print(f"======================================")

    stats_channel = StatsSideChannel()
    oracle_channel = OracleSideChannel(base_seed, global_step_count)
    engine_channel = EngineConfigurationChannel()
    engine_channel.set_configuration_parameters(width=1920, height=1080)

    side_channels = [stats_channel, oracle_channel, engine_channel]

    print('aaaaaaaaa')
    env = UnityEnvironment(
        file_name='/Users/luketto/Desktop/Tesi/ML_Agents_tesi/simple_plugin/LLM_Training/TanksGameLLM', 
        seed=base_seed, 
        side_channels=side_channels,
        #no_graphics=True,
        timeout_wait=100,
        additional_args=['-nolog'])

    env.reset()
    behavior_names = list(env.behavior_specs.keys())
    spec = env.behavior_specs[behavior_names[0]]

    engine_channel.set_configuration_parameters(time_scale=20)
    engine_channel.set_configuration_parameters(capture_frame_rate=60)

    client = ZMQCommunicatorClient(spec.action_spec.discrete_branches, 
                                    spec.action_spec.continuous_size, 
                                    1,
                                    observation_types
                                    )
    try:
        while global_step_count < TOTAL_STEP_BUDGET:
            agents_stepped_this_frame = 0
            for behavior_name in behavior_names:
                decision_steps, terminal_steps = env.get_steps(behavior_name)
                n_agents = len(decision_steps)

                agents_stepped_this_frame += n_agents
                if n_agents > 0:
                    obs = decision_steps.obs

                    llm_actions = []
                    for i in range (0, n_agents):
                        llm_actions.append(client.receive_action_from_llm(obs, i))

                    action = ActionTuple()
                    action.add_discrete(np.array(llm_actions, dtype=np.int32))
                    env.set_actions(behavior_name, action)

                #screenshot = ImageGrab.grab(bbox=secondo_monitor_box, all_screens=True)
                #frame_path = os.path.join(output_frames_dir, f"frame_{global_step_count}.png")
                #screenshot.save(frame_path)

                env.step()
                global_step_count += agents_stepped_this_frame
                oracle_channel.step_increment(global_step_count)

                stats = stats_channel.get_and_reset_stats()

                for stat_name, stat_info_list in stats.items():
                    values = [info[0] for info in stat_info_list]
                    if values:
                        max_value = values[-1]
                        accumulated_stats[stat_name] = max_value

                if global_step_count % summary_freq == 0:
                    elapsed_time = time.time() - start_time
                    print(f"[INFO] - {global_step_count} steps, {elapsed_time:.1f}s")
                    for stat_name, value in accumulated_stats.items():
                        current_writer.add_scalar(stat_name, value, global_step_count)

    except KeyboardInterrupt:
        print(f"Manual stop {global_step_count}")

    finally:
        env.close()
        time.sleep(2)
        print("Closing")

try:
    for current_seed in range(0, NUM_SEEDS):
        global_step_count = 0
        run_count = 0
        accumulated_stats = {}

        seed_log_dir = os.path.join("results", f"{run_id}_{current_seed}")
        writer = SummaryWriter(log_dir=seed_log_dir)

        if os.path.exists('/Users/luketto/Library/Application Support/DefaultCompany/Tanks/TankStatesLog.txt'):
            os.remove('/Users/luketto/Library/Application Support/DefaultCompany/Tanks/TankStatesLog.txt')
            print("Deleted States Log")
        if os.path.exists('/Users/luketto/Library/Application Support/DefaultCompany/Tanks/GlobalStatesLog.txt'):
            os.remove('/Users/luketto/Library/Application Support/DefaultCompany/Tanks/GlobalStatesLog.txt')
            print("Deleted Global States Log")
    
        while global_step_count < TOTAL_STEP_BUDGET:
            try:
                run_count += 1
                create_and_run_environment(current_seed, writer, run_count)
            except UnityEnvironmentException as e:
                print(f"\n[WARNING] Crash rilevato: {e}")
                print("Riavvio dell'ambiente in corso...")
                time.sleep(1)
                continue
            except KeyboardInterrupt:
                writer.close()
                print(f"\nInterruzione manuale allo step {global_step_count}")

        writer.close()
        
        print(f"Esperimento con Seed {current_seed} completato!")

    print(f"\n=== TUTTI I {NUM_SEEDS} ESPERIMENTI SONO TERMINATI ===")
finally:
    writer.close()
    print("TensorBoard Writer chiuso correttamente.")