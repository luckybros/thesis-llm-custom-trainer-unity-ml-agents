from typing import Dict, List
from cerberus import CerberusClient
import os

class CerberusModel:

    def __init__(self, settings):
        self.model_name = settings.model
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.join(base_dir, "..", "..", "..", "cerberus_project")
        self.model = CerberusClient(project_dir=project_dir)
        self.DEFAULT_AGENT_ACTION = settings.default_agent_action

    def call_llm(self, prompt: Dict[str, str]) -> str:
        return self.call_llm_batch([prompt])[0]
    
    def call_llm_batch(self, prompts: List[Dict[str, str]])-> List[str]:
        """
        prompts: lista di dict {"sys_msg": ..., "hum_mgs": ...}
        return: list of strings of outputs
        """
        if not prompts:
            return []
        
        messages_batch = [
            [
                {"role": "system", "content": p["sys_msg"]},
                {"role": "user", "content": p["hum_msg"]}
            ]
            for p in prompts
        ]

        outputs = self.model.chat_batch(
            "gpt-oss-20b",
            messages=messages_batch
        )

        results = []
        for out in outputs:
            llm_choice = out.content
            if not self._is_output_valid(llm_choice):
                llm_choice = self.DEFAULT_AGENT_ACTION
            results.append(llm_choice)

        return results

    def _is_output_valid(self, output: str) -> bool:
        """
        Controlla se l'output dell'LLM contiene la struttura sintattica minima richiesta.
        """
        if not output:
            return False
            
        # Controlliamo la presenza delle tre macro-azioni necessarie
        required_keywords = ["Move", "Turn", "Shoot"]
        return all(keyword in output for keyword in required_keywords)

