from typing import Dict, List
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

class OllamaModel:

    def __init__(self, settings):
        self.model_name = settings.model
        self.model = ChatOllama(model=self.model_name)
        self.DEFAULT_AGENT_ACTION = "Agent 0:\n  Move\n    Move forward\n  Turn\n    Stay\n  Shoot\n    Don't shoot"

    def call_llm_batch(self, prompts: List[Dict[str, str]]) -> List[str]:
        """
        prompts: lista di dict {"sys_msg": ..., "hum_mgs": ...}
        return: list of strings of outputs
        """
        if not prompts:
            return []
        
        message_batch = [
            [
                SystemMessage(content=p['sys_msg']),
                HumanMessage(content=p['hum_msg'])
            ]
            for p in prompts
        ]

        outputs = self.model.batch(message_batch)

        results = []
        for out in outputs:
            llm_choice = out.content.strip()
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
