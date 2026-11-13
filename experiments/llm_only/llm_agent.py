"""
Esperimento "solo LLM": nessuna rete neurale, nessun training.
Ad ogni decisione le osservazioni di Unity vengono inviate al server LLM
(mlagents_plugin.communicators.server.zmq_communication_server) via ZMQ e
l'azione restituita dall'LLM viene eseguita direttamente nell'ambiente.

Due file di configurazione (vedi configs/):
  - YAML LLM      (llm_settings: prompt, azioni, modello)  -> letto dal server LLM
  - YAML training (behaviors.<nome>.hyperparameters.observation_types,
                   engine_settings)                          -> letto da questo script
Lo YAML training e' lo stesso formato usato da mlagents-learn con il trainer
llm_trainer: qui servono solo observation_types ed engine_settings, gli altri
iperparametri vengono ignorati (ma il file e' riusabile per il training DRL+LLM).

Di norma viene lanciato da orchestrator.py, che avvia prima il server LLM.

Per scoprire le osservazioni esposte dalla build (indici, nomi, shape):
    python experiments/llm_only/llm_agent.py --env <build> --inspect
"""
import argparse
import os
import time

import numpy as np
import yaml
from mlagents_envs.environment import ActionTuple, UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.stats_side_channel import StatsSideChannel
from torch.utils.tensorboard import SummaryWriter

from mlagents_plugin.communicators.client.zmq_communicator_client import ZMQCommunicatorClient

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
DEFAULT_TRAINING_CONFIG = os.path.join(REPO_ROOT, "mlagents_plugin", "config", "FPS", "FPSTrainingLLM.yaml")

# osservazioni che richiedono il rendering di Unity (camera): con --no-graphics arrivano nere
VISUAL_OBS_TYPES = {"VISUAL"}


def load_training_config(path, behavior=None):
    """
    Dallo YAML training ritorna (observation_types, engine_settings).
    behavior: nome del behavior in `behaviors` (default: il primo).
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f) or {}
    behaviors = config.get("behaviors") or {}
    if not behaviors:
        raise ValueError(f"Nessuna sezione 'behaviors' in {path}")
    behavior = behavior or next(iter(behaviors))
    observation_types = behaviors[behavior]["hyperparameters"].get("observation_types")
    if not observation_types:
        raise ValueError(f"Nessun 'observation_types' per il behavior '{behavior}' in {path}")
    return observation_types, config.get("engine_settings", {}) or {}


def guess_obs_type(spec):
    name = spec.name.lower()
    if len(spec.shape) == 3:
        return "GRID" if "grid" in name else "VISUAL"
    if "ray" in name:
        return "RAYCAST"
    return "VECTORIAL"


def inspect_environment(env_path, base_port, seed, no_graphics):
    """Stampa le osservazioni della build nell'ordine in cui arrivano a Python."""
    env = UnityEnvironment(file_name=env_path, seed=seed, worker_id=seed, base_port=base_port,
                           no_graphics=bool(no_graphics), timeout_wait=100)
    try:
        env.reset()
        for behavior_name, spec in env.behavior_specs.items():
            print(f"\nBehavior: {behavior_name}")
            print(f"  Azioni discrete (opzioni per ramo): {spec.action_spec.discrete_branches}")
            print(f"  Azioni continue: {spec.action_spec.continuous_size}")
            print("  Osservazioni (index -> name, shape, tipo suggerito):")
            for i, obs in enumerate(spec.observation_specs):
                print(f"    {i}: {obs.name:35s} shape={obs.shape}  ->  {guess_obs_type(obs)}")
            print("\n  Bozza per lo YAML training, sotto hyperparameters (rinomina i RAYCAST, es. RAYCAST_FRONT):")
            print("  observation_types:")
            for i, obs in enumerate(spec.observation_specs):
                t = guess_obs_type(obs)
                print(f"    - name: {t}\n      type: {t}\n      index: {i}")
    finally:
        env.close()


def create_and_run_environment(env_path, training_config, behavior=None, seed=1, base_port=5400, zmq_port=65200,
                               run_id=None, total_steps=10000, summary_freq=1000,
                               no_graphics=None, timeout_ms=120000,
                               time_scale=None, capture_frame_rate=None, use_oracle=False,
                               clear_files=()):

    observation_types, engine_settings = load_training_config(training_config, behavior)
    is_visual = any(o["type"] in VISUAL_OBS_TYPES for o in observation_types)
    if no_graphics is None:
        no_graphics = not is_visual
    if is_visual and no_graphics:
        print("[WARN] observation_types contiene VISUAL ma --no-graphics e' attivo: "
              "la camera Unity restituira' immagini vuote.")

    run_id = run_id or f"LLM_Only_{seed}"
    writer = SummaryWriter(log_dir=os.path.join("results", run_id))

    # log specifici del gioco da azzerare a inizio run
    for path in clear_files:
        if os.path.exists(path):
            os.remove(path)
            print(f"Deleted {path}")

    accumulated_stats = {}
    step_count = 0
    start_time = time.time()

    print("\n======================================")
    print(f"Avvio Run '{run_id}' (seed={seed}, visual={is_visual}, "
          f"UnityPort={base_port + seed}, ZMQPort={zmq_port + seed})")
    print("======================================")

    stats_channel = StatsSideChannel()
    engine_channel = EngineConfigurationChannel()
    side_channels = [stats_channel, engine_channel]
    oracle_channel = None
    if use_oracle:
        # oracolo di bug-detection dei giochi del laboratorio (FPS/Tanks): serve solo se la build lo implementa
        from mlagents_plugin.oracle_side_channel import OracleSideChannel
        oracle_channel = OracleSideChannel(run_id, step_count)
        side_channels.append(oracle_channel)

    env = UnityEnvironment(
        file_name=env_path,              # None -> attende il Play nell'Editor Unity
        seed=seed,
        worker_id=seed,                  # porta Unity = base_port + seed
        base_port=base_port,
        side_channels=side_channels,
        no_graphics=no_graphics,
        timeout_wait=100)

    try:
        env.reset()
        behavior_names = list(env.behavior_specs.keys())
        spec = env.behavior_specs[behavior_names[0]]
        if spec.action_spec.continuous_size > 0:
            print("[WARN] L'ambiente ha azioni continue: l'esperimento solo-LLM invia solo azioni discrete.")

        engine_channel.set_configuration_parameters(
            time_scale=time_scale or engine_settings.get("time_scale", 20),
            capture_frame_rate=capture_frame_rate or engine_settings.get("capture_frame_rate", 60),
        )

        client = ZMQCommunicatorClient(
            discrete_branches=spec.action_spec.discrete_branches,
            num_continuous_action=spec.action_spec.continuous_size,
            num_agents=1,                # il server riceve un payload per agente
            observation_types=observation_types,
            worker_id=seed,              # porta ZMQ = zmq_port + seed (stessa regola del server)
            base_port=zmq_port,
            timeout_ms=timeout_ms,       # il primo prompt carica il modello in memoria: serve margine
        )
        # azione usata se il server non risponde per un agente (timeout/errore):
        # indice 0 per ogni ramo, cioe' la prima opzione di ogni azione nello YAML LLM
        fallback_action = [0] * len(spec.action_spec.discrete_branches)
        missing_responses = 0

        while step_count < total_steps:
            agents_stepped_this_frame = 0
            for behavior_name in behavior_names:
                decision_steps, _ = env.get_steps(behavior_name)
                n_agents = len(decision_steps)

                agents_stepped_this_frame += n_agents
                if n_agents > 0:
                    obs = decision_steps.obs
                    selected_indices = list(range(n_agents))

                    llm_actions_batch = client.receive_action_from_llm_batch(obs, selected_indices)
                    llm_actions = []
                    for idx in selected_indices:
                        agent_action = llm_actions_batch.get(f"agent-{idx}")
                        if agent_action is None:
                            missing_responses += 1
                            print(f"[WARN] Nessuna risposta LLM per agent-{idx} allo step {step_count} "
                                  f"(totale {missing_responses}), uso l'azione di fallback {fallback_action}")
                            agent_action = fallback_action
                        llm_actions.append(agent_action)

                    action = ActionTuple()
                    action.add_discrete(np.array(llm_actions, dtype=np.int32))
                    env.set_actions(behavior_name, action)

            env.step()
            step_count += agents_stepped_this_frame
            if oracle_channel is not None:
                oracle_channel.step_increment(step_count)

            stats = stats_channel.get_and_reset_stats()
            for stat_name, stat_info_list in stats.items():
                values = [info[0] for info in stat_info_list]
                if values:
                    accumulated_stats[stat_name] = values[-1]

            if agents_stepped_this_frame and step_count % summary_freq == 0:
                elapsed_time = time.time() - start_time
                print(f"[INFO] - {step_count} steps, {elapsed_time:.1f}s")
                for stat_name, value in accumulated_stats.items():
                    writer.add_scalar(stat_name, value, step_count)
                writer.add_scalar("LLM/missing_responses", missing_responses, step_count)

    except KeyboardInterrupt:
        print(f"Manual stop {step_count}")

    finally:
        env.close()
        time.sleep(2)
        writer.close()
        print(f"Run '{run_id}' terminata (step {step_count}).")


def build_parser():
    parser = argparse.ArgumentParser(description="Esperimento solo-LLM su un ambiente Unity ML-Agents")
    parser.add_argument("--env", type=str, default=None,
                        help="Build Unity (.exe / .app / .x86_64). Se omesso si collega all'Editor (premere Play).")
    parser.add_argument("--training-config", type=str, default=DEFAULT_TRAINING_CONFIG,
                        help="YAML training da cui leggere observation_types ed engine_settings")
    parser.add_argument("--behavior", type=str, default=None,
                        help="Nome del behavior nello YAML training (default: il primo)")
    parser.add_argument("--inspect", action="store_true",
                        help="Stampa le osservazioni/azioni della build ed esce (non servono server ne' YAML)")
    parser.add_argument("--seed", type=int, default=1,
                        help="Seed della run; usato anche come worker_id per le porte Unity e ZMQ")
    parser.add_argument("--run-id", type=str, default=None, help="Default: LLM_Only_<seed>")
    parser.add_argument("--base-port", type=int, default=5400, help="Porta base Unity (effettiva: base + seed)")
    parser.add_argument("--zmq-port", type=int, default=65200, help="Porta base ZMQ (effettiva: base + seed)")
    parser.add_argument("--total-steps", type=int, default=10000)
    parser.add_argument("--summary-freq", type=int, default=1000)
    parser.add_argument("--timeout-ms", type=int, default=120000,
                        help="Timeout per singola risposta del server LLM")
    parser.add_argument("--time-scale", type=float, default=None)
    parser.add_argument("--capture-frame-rate", type=int, default=None)
    graphics = parser.add_mutually_exclusive_group()
    graphics.add_argument("--no-graphics", dest="no_graphics", action="store_true", default=None,
                          help="Forza no-graphics (default: attivo solo se non ci sono osservazioni VISUAL)")
    graphics.add_argument("--graphics", dest="no_graphics", action="store_false",
                          help="Forza il rendering (finestra del gioco visibile)")
    parser.add_argument("--oracle", action="store_true",
                        help="Registra l'OracleSideChannel (solo per le build FPS/Tanks del laboratorio)")
    parser.add_argument("--clear-file", action="append", default=[],
                        help="File di log del gioco da cancellare a inizio run (ripetibile)")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.inspect:
        inspect_environment(args.env, args.base_port, args.seed, args.no_graphics)
    else:
        create_and_run_environment(
            env_path=args.env,
            training_config=args.training_config,
            behavior=args.behavior,
            seed=args.seed,
            base_port=args.base_port,
            zmq_port=args.zmq_port,
            run_id=args.run_id,
            total_steps=args.total_steps,
            summary_freq=args.summary_freq,
            no_graphics=args.no_graphics,
            timeout_ms=args.timeout_ms,
            time_scale=args.time_scale,
            capture_frame_rate=args.capture_frame_rate,
            use_oracle=args.oracle,
            clear_files=args.clear_file,
        )
