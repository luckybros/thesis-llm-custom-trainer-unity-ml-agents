from typing import Dict, List
from vllm import LLM, SamplingParams

class VLLMModel:

    def __init__(self, settings):
        self.model_name = settings.model
        self.model = LLM(
            model=self.model_name,
            trust_remote_code=True,
            max_model_len=4096
        )
        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=2048
        )
        self.DEFAULT_AGENT_ACTION = "Agent 0:\n  Move\n    Move forward\n  Turn\n    Stay\n  Shoot\n    Don't shoot"

    """
    def call_llm(self, prompt):
        messages = [
            {"role": "system", "content": prompt["sys_msg"]},
            {"role": "user", "content": prompt["hum_msg"]}
        ]

        outputs = self.model.chat(
            messages=messages,
            sampling_params=self.sampling_params,
            use_tqdm=False
        )

        llm_choice = outputs[0].outputs[0].text

        print(f"llm choice: {llm_choice}")

        if not self._is_output_valid(llm_choice):
            return self.DEFAULT_AGENT_ACTION
        
        return llm_choice
    """
        
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

        outputs = self.model.chat(
            messages=messages_batch,
            sampling_params=self.sampling_params,
            use_tqdm=True
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

