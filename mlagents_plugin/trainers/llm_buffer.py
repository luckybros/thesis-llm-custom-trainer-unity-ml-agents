from collections import defaultdict
from collections.abc import MutableMapping
import enum
import itertools
from typing import BinaryIO, DefaultDict, Dict, List, Tuple, Union, Optional

import numpy as np

class LLMBufferKey(enum.Enum):
    """
    Key for every single information we want to obtain from the LLM
    """
    LLM_LOG_CONTINUOUS_LOG_PROBS = "llm_continuous_log_probs"
    LLM_LOG_DISCRETE_LOG_PROBS = "llm_discrete_log_probs"


class LLMBuffer:
    def __init__(self):
        self._fields: DefaultDict[LLMBufferKey, List[List[np.ndarray]]] = defaultdict(list)
        
    def __getitem__(self, key: LLMBufferKey) -> List[List[np.ndarray]]:
        return self._fields[key]

    def add_entry(self, key: LLMBufferKey, data: np.ndarray) -> None:
        self[key].append(data)

    def pop_n_entries(self, num_items: int) -> Dict[LLMBufferKey, List[np.ndarray]]:
        """
        Rimuove e restituisce i primi 'num_items' elementi da TUTTI i campi.
        Restituisce un dizionario che mappa ogni chiave alla sua lista di dati,
        pronto per essere processato dal Trainer.
        """
        return_dict: Dict[LLMBufferKey, List[np.ndarray]] = {}
        for key, field in self._fields.items():
            if len(field) < num_items:
                num_to_pop = len(field)
            else:
                num_to_pop = num_items
            
            # Usiamo uno slicing per prendere e rimuovere gli elementi in modo efficiente
            return_dict[key] = field[0:num_to_pop]
            del field[0:num_to_pop]
        
        return return_dict

