import logging
import re
from typing import Dict, List
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

# I modelli "thinking" (es. qwen3, deepseek-r1) possono restituire il ragionamento
# racchiuso in <think>...</think>: va rimosso prima del parsing, perche' puo'
# contenere a sua volta blocchi "Agent 0:" di prova.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class OllamaModel:
    """
    Backend LLM locale via Ollama (https://ollama.com).

    Funziona con qualunque modello presente in `ollama list`, es.:
      - solo testo:      qwen2.5:7b, qwen3:8b, llama3.2, mistral-small:24b
      - testo + visione: qwen2.5vl:7b, qwen3-vl:8b, llama3.2-vision
    Le immagini arrivano nel prompt come blocchi {"type": "image_url"} (vedi PromptBuilder)
    e ChatOllama le inoltra al modello: serve quindi un modello *vision* se use_visual_obs=True.

    Parametri opzionali da `llm_settings.model_kwargs` (passati a ChatOllama), es.:
      temperature: 0.2
      base_url: http://localhost:11434   # oppure variabile d'ambiente OLLAMA_HOST
      num_ctx: 8192
      reasoning: false                    # disattiva il "thinking" di qwen3
      keep_alive: 30m
    """

    def __init__(self, settings):
        self.model_name = settings.model
        self.model = ChatOllama(model=self.model_name, **settings.model_kwargs)
        self.required_keywords = settings.required_keywords
        self.DEFAULT_AGENT_ACTION = settings.default_agent_action

    def call_llm(self, prompt: Dict[str, str]) -> str:
        return self.call_llm_batch([prompt])[0]

    def call_llm_batch(self, prompts: List[Dict[str, str]]) -> List[str]:
        """
        prompts: lista di dict {"sys_msg": str, "hum_msg": str | list[content block]}
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

        # A remote/unreliable backend (network drop, server-side timeout, malformed
        # response from the underlying model) raises here. Uncaught, this used to
        # propagate out of the ZMQ request handler and kill the whole LLM server
        # process - which then left the training-side ZMQ client (a REQ socket)
        # waiting for a reply that would never come. Falling back to
        # DEFAULT_AGENT_ACTION for the whole batch keeps the server alive and lets
        # the caller's own "invalid output" handling take over, same as a
        # syntactically-invalid LLM reply.
        try:
            outputs = self.model.batch(message_batch)
        except Exception:
            logger.exception(
                "[OllamaModel] LLM call failed (%d prompt(s)) - falling back to "
                "default_agent_action for all of them.", len(message_batch)
            )
            return [self.DEFAULT_AGENT_ACTION] * len(prompts)

        results = []
        for out in outputs:
            llm_choice = _THINK_RE.sub("", out.content or "").strip()
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

        # Controlliamo la presenza di tutte le macro-azioni necessarie
        return all(keyword in output for keyword in self.required_keywords)
