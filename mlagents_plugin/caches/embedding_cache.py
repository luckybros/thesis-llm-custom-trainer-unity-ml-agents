import json
import numpy as np
import faiss
from mlagents_plugin.caches.llm_cache import LLMCache
from typing import Any, Optional, Dict, List
from sentence_transformers import SentenceTransformer

class LLMEmbeddingCache(LLMCache):

    def __init__(self):
        super().__init__()
        self.D = 384
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.IndexFlatIP(self.D)
        self.threshold = 0.99

    def update(self, state, action):
        emb = self._get_embedding(state)
        self.index.add(emb)
        self.cache.append(action)

    def query(self, new_state):
        self._print_cache_state(new_state)
        if self.index.ntotal == 0:
            return None
        
        emb = self._get_embedding(new_state)

        score, index = self.index.search(emb, k=1)

        best_score = score[0][0]
        best_index = index[0][0]

        if best_score >= self.threshold:
            self.hits += 1
            return self.cache[best_index]
        else: 
            self.misses += 1
            return None

    def _get_embedding(self, state):
        # per ora stringa, Semantic Flattening, sostuire json.dumps con una funzione
        #state_string = json.dumps(state, sort_keys=True)
        state_string = self._flattening(state)
        emb = self.model.encode(state_string, convert_to_numpy=True)
        matrix = np.array([emb]).astype('float32')
        faiss.normalize_L2(matrix)
        return matrix

