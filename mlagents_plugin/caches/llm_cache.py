from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List

class LLMCache(ABC):

    def __init__(self):
        #self.cache: List[Any] = []
        self.hits = 0
        self.misses = 0
        self.update_print = 1
        self.agent_history_buffers: Dict[str, List[str]] = {}
        self.history_length = 3

    @abstractmethod
    def query(self, state) -> Optional[Dict[str, Dict[str, List[List[int]]]]]:
        pass

    @abstractmethod
    def update(self):
        pass

    def _flattening(self, state):
        raycast_block = {'RAYCAST_FRONT': 'front', 'RAYCAST_BACK': 'back'}

        result = []

        for block_key, prefix in raycast_block.items():
            if block_key in state:
                inner_dict = state[block_key]
                for agent_name, agent_rays in inner_dict.items():
                    for ray in agent_rays:
                        direction = ray.get('direction', '')
                        
                        for obj in ray.get('objects_detected', ''):
                            obj_name = obj.get('object', '')
                            distance = obj.get('distance', '')
                            result.append(f"{prefix}_{direction}:{obj_name}_{distance}")

        if "VECTORIAL" in state:
            val = state['VECTORIAL']
            for k, v in val.items():
                for key, value in v.items():
                    result.append(f"{key}:{value}")

        result.sort()

        flat_str = " ".join(result)

        if (self.hits + self.misses) % self.update_print == 0:
            a = result
            #print(" ".join(a))
        
        #if action is not None:
        #    flat_str += f" | action:{str(action)}"

        return flat_str
    
    def _add_to_history(self, agent_id: str, current_state, action):
        # state is already flatten
        if agent_id not in self.agent_history_buffers:
            self.agent_history_buffers[agent_id] = []

        buffer = self.agent_history_buffers[agent_id]
        if len(buffer) >= self.history_length:
            buffer.pop(0)

        history_string = current_state + f" | action: {str(action)}"
        buffer.append(history_string)

    def _print_cache_state(self, state):
        if (self.hits + self.misses) % self.update_print == 0:
            print(f"HITS:{self.hits} - MISSES:{self.misses}")
            #print(f"{state}")
    