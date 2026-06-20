import os
import csv
import hashlib
from datetime import datetime
from mlagents_plugin.caches.llm_cache import LLMCache
from typing import Any, Optional, Dict, List


class LLMHashCache(LLMCache):

    LOG_DIR = "cache_logs"
    LOGGING_ENABLED = True

    CSV_FIELDS = [
        "timestamp",
        "event",          
        "hash",
        "flatten_state",
        "history_len",
        "history_buffer", 
        "full_query",     
        "action",         
    ]

    def __init__(self):
        super().__init__()
        self.cache: Dict[str, Any] = {}

        if self.LOGGING_ENABLED:
            os.makedirs(self.LOG_DIR, exist_ok=True)
            # un csv writer/file handle per agente, aperti pigramente
            self._csv_files: Dict[str, Any] = {}
            self._csv_writers: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _csv_path(self, agent_id: str) -> str:
        safe_id = str(agent_id).replace("/", "_").replace("\\", "_")
        return os.path.join(self.LOG_DIR, f"agent_{safe_id}.csv")

    def _get_writer(self, agent_id: str):
        if agent_id not in self._csv_writers:
            path = self._csv_path(agent_id)
        
            f = open(path, "w", newline="", encoding="utf-8")
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
            writer.writeheader()
            self._csv_files[agent_id] = f
            self._csv_writers[agent_id] = writer
        return self._csv_writers[agent_id]

    def _log(self, agent_id: str, event: str, flatten_state: str,
              history: List[str], full_query: str, state_hash: Any,
              action: Any = None):
        if not self.LOGGING_ENABLED:
            return

        writer = self._get_writer(agent_id)
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "event": event,
            "hash": state_hash,
            "flatten_state": flatten_state,
            "history_len": len(history),
            "history_buffer": " || ".join(history),
            "full_query": full_query,
            "action": "" if action is None else repr(action),
        })
        # flush subito cosi' i dati sono leggibili anche se il training
        # crasha o viene interrotto a meta'
        self._csv_files[agent_id].flush()

    def close_logs(self):
        """Chiude esplicitamente tutti i file csv aperti (chiamala a fine training/episodio)."""
        for f in self._csv_files.values():
            f.close()
        self._csv_files.clear()
        self._csv_writers.clear()

    # ------------------------------------------------------------------
    # Cache logic
    # ------------------------------------------------------------------
    def query(self, state: dict, agent_id: str) -> Optional[Dict[str, Dict[str, List[List[int]]]]]:
        self._print_cache_state(state)
        flatten_state = self._flattening(state)
        agent_history = self.agent_history_buffers.get(agent_id, [])
        full_query = f"{flatten_state} [HISTORY] " + " ".join(agent_history)
        state_hash = self._generate_hash(full_query)

        if state_hash in self.cache:
            self.hits += 1
            action_retrieved = self.cache[state_hash]

            self._log(agent_id, "QUERY-HIT", flatten_state, agent_history,
                       full_query, state_hash, action=action_retrieved)
            self._add_to_history(agent_id, flatten_state, action_retrieved)

            return action_retrieved

        self.misses += 1
        self._log(agent_id, "QUERY-MISS", flatten_state, agent_history,
                   full_query, state_hash)
        return None

    def update(self, state, action, agent_id):
        # cache on new state + history
        # then i save the new state in the history buffer
        flat_state = self._flattening(state)
        agent_history = self.agent_history_buffers.get(agent_id, [])
        new_cache = f"{flat_state} [HISTORY] " + " ".join(agent_history)
        state_hash = self._generate_hash(new_cache)

        self._log(agent_id, "UPDATE", flat_state, agent_history,
                   new_cache, state_hash, action=action)

        self.cache[state_hash] = action
        self._add_to_history(agent_id, flat_state, action)

    def _generate_hash(self, state: str) -> str:
        return hashlib.sha256(state.encode("utf-8")).hexdigest()