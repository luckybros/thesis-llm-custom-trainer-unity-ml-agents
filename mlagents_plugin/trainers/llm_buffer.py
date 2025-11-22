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


class LLMBufferField(list):

    def __init__(self, *args, **kwargs):
        self.padding_value = 0
        super().__init__(*args, **kwargs)

    def __str__(self) -> str:
        return f"LLMAgentBufferField: {super.__str__()}"
    
    def __getitem__(self, index):
        return_data = super().__getitem__(index)
        if isinstance(return_data, list):
            return LLMBufferField(return_data)
        else:
            return return_data
        

class LLMBuffer:
    def __init__(self):
        self._fields: DefaultDict[LLMBufferKey, LLMBufferField] = defaultdict(LLMBufferField)
        
    def __getitem__(self, key: LLMBufferKey) -> LLMBufferField:
        """Permette di accedere a un campo specifico, es. buffer[LLMBufferKey.LOGITS]"""
        return self._fields[key]

    def add_entry(self, key: LLMBufferKey, data: np.ndarray) -> None:
        """Aggiunge una singola entry (un numpy array) a un campo specifico."""
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

    def __len__(self) -> int:
        if not self._fields:
            return 0
        return len(next(iter(self._fields.values())))