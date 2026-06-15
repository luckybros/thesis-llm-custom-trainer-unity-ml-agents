import json
from mlagents_plugin.caches.llm_cache import LLMCache
from typing import Any, Optional, Dict, List

class LLMHashCache(LLMCache):

    def __init__(self):
        super().__init__()
        self.cache: Dict[int, Any] = {}

    def query(self, state: dict, agent_id: str) -> Optional[Dict[str, Dict[str, List[List[int]]]]]:
        self._print_cache_state(state)

        flatten_state = self._flattening(state)

        agent_history = self.agent_history_buffers.get(agent_id, []) 

        query = f"{flatten_state} [HISTORY] " + " ".join(agent_history)
        state_hash = self._generate_hash(query)

        if state_hash in self.cache:
            #print(f"Cache hit")
            self.hits += 1
            action_retrived = self.cache[state_hash]
            #self._add_to_history(flatten_state, action_retrived)
            return action_retrived
        
        self.misses += 1

        return None
    
    def update(self, state, action, agent_id):
        # cache on new state + history
        # then i save the new state in the history buffer
        flat_state = self._flattening(state)

        agent_history = self.agent_history_buffers.get(agent_id, [])
        new_cache = f"{flat_state} [HISTORY] " + " ".join(agent_history)

        state_hash = self._generate_hash(new_cache)
        self.cache[state_hash] = action       
        self._add_to_history(agent_id, flat_state, action)

    def _generate_hash(self, state):
        return hash(state)