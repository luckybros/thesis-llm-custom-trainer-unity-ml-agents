"""
Orchestratore dell'esperimento "solo LLM".

Avvia in sequenza:
  1. il server LLM (ZMQ) -> mlagents_plugin.communicators.server.zmq_communication_server
  2. l'agente            -> experiments/llm_only/llm_agent.py (Unity + client ZMQ)
e alla fine (o con Ctrl+C) chiude entrambi.

Due file di configurazione (template in configs/):
  --llm-config       YAML con llm_settings (prompt, azioni, modello)  -> server LLM
  --training-config  YAML training con observation_types / engine_settings -> agente

Server e agente possono girare in due ambienti Python diversi (vedi GUIDA.md):
  --server-python  interprete con langchain-ollama (+ torch/transformers/faiss per il visuale)
  --agent-python   interprete con mlagents / mlagents_envs
Di default entrambi usano l'interprete corrente.

Porte: Unity = --unity-base-port + seed, ZMQ = --zmq-base-port + seed (stessa
regola su client e server), quindi run con seed diversi possono girare in parallelo.

Esempio:
    python experiments/llm_only/orchestrator.py \
        --env Builds/MioGioco.exe \
        --llm-config experiments/llm_only/configs/mio_gioco_llm.yaml \
        --training-config experiments/llm_only/configs/mio_gioco_training.yaml \
        --total-steps 2000
"""
import argparse
import os
import socket
import subprocess
import sys
import time

import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
LLM_AGENT_SCRIPT = os.path.join(CURRENT_DIR, "llm_agent.py")

DEFAULT_LLM_CONFIG = os.path.join(REPO_ROOT, "mlagents_plugin", "config", "FPS", "FPSLLM.yaml")
DEFAULT_TRAINING_CONFIG = os.path.join(REPO_ROOT, "mlagents_plugin", "config", "FPS", "FPSTrainingLLM.yaml")


def uses_visual_obs(training_config):
    with open(training_config, "r") as f:
        config = yaml.safe_load(f) or {}
    for behavior in (config.get("behaviors") or {}).values():
        for obs in behavior["hyperparameters"].get("observation_types", []):
            if obs["type"] == "VISUAL":
                return True
    return False


def wait_for_port(port, server_process, timeout):
    """Attende che il server ZMQ sia in ascolto (invece di uno sleep fisso)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if server_process.poll() is not None:
            return False
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.5)
    return server_process.poll() is None


def main():
    parser = argparse.ArgumentParser(description="Orchestratore esperimento solo-LLM")
    parser.add_argument("--env", type=str, default=None,  # .exe (Windows), .app (macOS), .x86_64 (Linux)
                        help="Eseguibile Unity (se omesso: Editor, premere Play)")
    parser.add_argument("--llm-config", type=str, default=DEFAULT_LLM_CONFIG,
                        help="YAML llm_settings per il server (prompt, azioni, modello)")
    parser.add_argument("--training-config", type=str, default=DEFAULT_TRAINING_CONFIG,
                        help="YAML training da cui l'agente legge observation_types ed engine_settings")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--unity-base-port", type=int, default=5400)
    parser.add_argument("--zmq-base-port", type=int, default=65200)
    parser.add_argument("--total-steps", type=int, default=10000)
    parser.add_argument("--summary-freq", type=int, default=1000)
    parser.add_argument("--visual", action="store_true", default=None,
                        help="Avvia il server in modalita' visuale (default: auto se observation_types contiene VISUAL)")
    parser.add_argument("--server-python", type=str, default=os.environ.get("LLM_SERVER_PYTHON", sys.executable))
    parser.add_argument("--agent-python", type=str, default=os.environ.get("MLAGENTS_PYTHON", sys.executable))
    parser.add_argument("--startup-timeout", type=float, default=60.0,
                        help="Secondi massimi di attesa perche' il server ZMQ sia in ascolto")
    args, agent_extra = parser.parse_known_args()   # argomenti extra -> llm_agent.py (es. --clear-file, --graphics)

    llm_config = os.path.abspath(args.llm_config)
    training_config = os.path.abspath(args.training_config)
    visual = args.visual if args.visual is not None else uses_visual_obs(training_config)
    zmq_port = args.zmq_base_port + args.seed

    server_command = [
        args.server_python, "-u", "-m", "mlagents_plugin.communicators.server.zmq_communication_server",
        "--config", llm_config,
        # stessa regola del client in llm_agent.py: porta = base_port + worker_id (seed)
        "--base-port", str(args.zmq_base_port),
        "--worker-id", str(args.seed),
    ]
    if visual:
        server_command.append("--visual")

    game_command = [
        args.agent_python, "-u", LLM_AGENT_SCRIPT,
        "--training-config", training_config,
        "--seed", str(args.seed),
        "--base-port", str(args.unity_base_port),
        "--zmq-port", str(args.zmq_base_port),
        "--total-steps", str(args.total_steps),
        "--summary-freq", str(args.summary_freq),
    ]
    if args.env:
        game_command += ["--env", args.env]
    if args.run_id:
        game_command += ["--run-id", args.run_id]
    game_command += agent_extra

    # il package mlagents_plugin deve essere importabile da entrambi i processi
    # anche senza `pip install -e .`
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = REPO_ROOT + os.pathsep + child_env.get("PYTHONPATH", "")

    print(f"----------[START ORCHESTRATOR LLM-ONLY seed={args.seed} visual={visual}]----------")

    server_process = None
    game_process = None
    exit_code = 0

    try:
        print(f"\n[1] Avvio Server ZMQ su porta {zmq_port}...")
        print(f"    Comando: {' '.join(server_command)}")
        server_process = subprocess.Popen(server_command, cwd=REPO_ROOT, env=child_env)

        if not wait_for_port(zmq_port, server_process, args.startup_timeout):
            print("ERRORE: il server ZMQ non e' partito (chiuso o non in ascolto).")
            exit_code = 1
        else:
            print(f"\n[2] Server pronto. Avvio LLM Agent (Unity Port: {args.unity_base_port + args.seed})...")
            print(f"    Comando: {' '.join(game_command)}")
            game_process = subprocess.Popen(game_command, env=child_env)
            exit_code = game_process.wait()

    except KeyboardInterrupt:
        print("\n\nInterruzione manuale (Ctrl+C). Spegnimento...")

    finally:
        print("\n[3] Pulizia dei processi...")
        for proc in (game_process, server_process):
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
        print("Orchestratore terminato.")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
