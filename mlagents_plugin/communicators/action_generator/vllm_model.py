from vllm import LLM, SamplingParams

class VLLMModel:

    def __init__(self, settings):
        self.model_name = settings.model
        self.model = LLM(
            model=self.model_name,
            trust_remote_code=True,
            max_model_len=1024
        )

        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=250
        )

        self.DEFAULT_AGENT_ACTION = "Agent 0:\n  Move\n    Move forward\n  Turn\n    Stay\n  Shoot\n    Don't shoot"

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
        
    def _is_output_valid(self, output: str) -> bool:
        """
        Controlla se l'output dell'LLM contiene la struttura sintattica minima richiesta.
        """
        if not output:
            return False
            
        # Controlliamo la presenza delle tre macro-azioni necessarie
        required_keywords = ["Move", "Turn", "Shoot"]
        return all(keyword in output for keyword in required_keywords)

