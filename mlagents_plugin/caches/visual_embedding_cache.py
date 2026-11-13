import io
import base64
import numpy as np
import faiss
import torch
import PIL.Image
from typing import Any, Optional, Dict, List
from transformers import AutoProcessor, AutoModel
from mlagents_plugin.caches.llm_cache import LLMCache

class VisualLLMEmbeddingCache(LLMCache):

    def __init__(self, history_length: int = 1):
        super().__init__(history_length)
        self.D = 512
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = AutoModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device).eval()
        self.index = faiss.IndexFlatIP(self.D)
        self.threshold = 0.99
        self.cache: List[Any] = []

    def update(self, state, action, agent_id):
        state = state.get("VISUAL", {})
        if not state:
            return 
        
        emb = self._get_embedding(next(iter(state.values())))
        self.index.add(emb)
        self.cache.append(action)

    def query(self, new_state, agent_id):
        self._print_cache_state(new_state)

        visual_state = new_state.get("VISUAL", {})

        if not visual_state:
            return None
        
        if self.index.ntotal == 0:
            self.misses += 1
            return None
        
        emb = self._get_embedding(next(iter(visual_state.values())))

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
        img_bytes = base64.b64decode(state)
        image = PIL.Image.open(io.BytesIO(img_bytes)).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        emb = features.cpu().numpy().astype("float32")
        faiss.normalize_L2(emb)
        return emb
