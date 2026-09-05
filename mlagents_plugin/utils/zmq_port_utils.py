"""
Dynamic port resolution for the plugin's ZMQ channel (trainer <-> LLM server).

Running several ML-Agents trainings in parallel on the same machine used to
collide because both ZMQCommunicatorClient and ZMQCommunicatorServer bound to
a single hardcoded port. Ports are now derived at runtime, in order of
priority:

  1. An explicit override (e.g. a `--port`/`--base-port`/`--worker-id` CLI
     argument, or a `zmq_base_port` trainer hyperparameter).
  2. The MLAGENTS_ZMQ_BASE_PORT / MLAGENTS_WORKER_ID environment variables.
  3. An offset derived from ML-Agents' own `--base-port` CLI argument (read
     from this process' argv), so that independent `mlagents-learn` runs -
     which are normally told apart by different --base-port values - also end
     up with disjoint ZMQ port ranges instead of all falling back to the same
     default.
  4. A fixed default, for simple single-run usage.

The final port is always `base_port + worker_id`, mirroring how ML-Agents
itself derives its own per-worker ports.
"""

import os
import sys
from typing import Optional

DEFAULT_ZMQ_BASE_PORT = 65432
ZMQ_BASE_PORT_ENV_VAR = "MLAGENTS_ZMQ_BASE_PORT"
ZMQ_WORKER_ID_ENV_VAR = "MLAGENTS_WORKER_ID"

# Kept well above ML-Agents' own default gRPC port range (5005+) so a base
# port derived from --base-port never collides with ML-Agents' own sockets.
BASE_PORT_DERIVATION_OFFSET = 60000


def _find_cli_base_port(argv: Optional[list] = None) -> Optional[int]:
    """Look for ML-Agents' own --base-port argument in argv, if present."""
    argv = sys.argv if argv is None else argv
    for i, arg in enumerate(argv):
        if arg == "--base-port" and i + 1 < len(argv):
            try:
                return int(argv[i + 1])
            except ValueError:
                return None
        if arg.startswith("--base-port="):
            try:
                return int(arg.split("=", 1)[1])
            except ValueError:
                return None
    return None


def resolve_zmq_base_port(explicit_base_port: Optional[int] = None) -> int:
    if explicit_base_port is not None:
        return explicit_base_port

    env_value = os.environ.get(ZMQ_BASE_PORT_ENV_VAR)
    if env_value is not None:
        return int(env_value)

    cli_base_port = _find_cli_base_port()
    if cli_base_port is not None:
        return cli_base_port + BASE_PORT_DERIVATION_OFFSET

    return DEFAULT_ZMQ_BASE_PORT


def resolve_zmq_worker_id(explicit_worker_id: Optional[int] = None) -> int:
    if explicit_worker_id is not None:
        return explicit_worker_id
    return int(os.environ.get(ZMQ_WORKER_ID_ENV_VAR, 0))


def resolve_zmq_port(
    worker_id: Optional[int] = None, base_port: Optional[int] = None
) -> int:
    return resolve_zmq_base_port(base_port) + resolve_zmq_worker_id(worker_id)
