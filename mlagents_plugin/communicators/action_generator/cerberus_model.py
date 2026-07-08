from typing import Dict, List
from cerberus import CerberusClient

class CerberusModel:

    def __init__(self, settings):
        self.model_name = settings.model
        self.model = CerberusClient()

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
                {"role": "system", "content": "Reasoning: low\n\n" + p["sys_msg"]},
                {"role": "user", "content": p["hum_msg"]}
            ]
            for p in prompts
        ]

        outputs = self.model.chat(
            messages=messages_batch
        )

        results = []
        for out in outputs:
            llm_choice = out.outputs[0].text
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

