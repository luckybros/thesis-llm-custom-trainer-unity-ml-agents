import os
import numpy as np
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.stats_side_channel import StatsSideChannel
from mlagents_plugin.oracle_side_channel import OracleSideChannel
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.exception import UnityTimeOutException, UnityCommunicatorStoppedException, UnityEnvironmentException
from torch.utils.tensorboard import SummaryWriter
import time

run_id = "Tank_Random_Baseline"

# --- CONFIGURATION ---
EXPERIMENT_NAME = "Tank_Random_Baseline"
NUM_SEEDS = 10          
TOTAL_STEP_BUDGET = 1000000 
summary_freq = 1000

# Variabili globali per la singola iterazione
global_step_count = 0
run_count = 0
accumulated_stats = {}

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

    side_channels = [stats_channel, oracle_channel, engine_channel]

    env = UnityEnvironment(
        file_name='mlagents_plugin/config/Tanks/TanksGameRL', 
        seed=base_seed, 
        side_channels=side_channels,
        no_graphics=True,
        timeout_wait=100,
        additional_args=['-nolog'])
    env.reset()

    engine_channel.set_configuration_parameters(time_scale=20)
    engine_channel.set_configuration_parameters(capture_frame_rate=60)

    behavior_names = list(env.behavior_specs.keys())

    try:
        while global_step_count < TOTAL_STEP_BUDGET:

            agents_stepped_this_frame = 0
            for behavior_name in behavior_names:
                decision_steps, terminal_steps = env.get_steps(behavior_name)
                n_agents = len(decision_steps)

                agents_stepped_this_frame += n_agents
                if n_agents > 0:
                    action_spec = env.behavior_specs[behavior_name].action_spec
                    random_actions = np.column_stack([
                        rng.integers(0, branch_size, size=(n_agents)) 
                        for branch_size in action_spec.discrete_branches
                    ])
                    action = ActionTuple()
                    action.add_discrete(np.array(random_actions, dtype=np.int32))
                    env.set_actions(behavior_name, action)

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
        print("Closing...")

try:
    for current_seed in range(1, NUM_SEEDS):
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


    

