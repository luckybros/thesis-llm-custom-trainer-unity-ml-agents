from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()

class LangchainModel:

    def __init__(self, settings):
        self.model_name = settings.model
        self.model = self._model_constructor(self.model_name)
        self.chain = RunnableLambda(self._format_input) | self.model | StrOutputParser()

        self.fallback_model = ChatOllama(model="qwen2.5:1.5b", temperature=0)
        self.fallback_chain = RunnableLambda(self._format_input) | self.fallback_model | StrOutputParser()

    def call_llm(self, prompt):
        try:
        #print(f'prompt: {prompt}')
            llm_choice = self.chain.invoke(prompt)
        #print(f"llm_choice: {llm_choice}")

            if not self._is_output_valid(llm_choice):
                print(f"Rilevato output incompleto/invalido: '{llm_choice}'. Attivazione fallback...")
                try:
                    llm_choice = self.fallback_chain.invoke(prompt)
                    if not self._is_output_valid(llm_choice):
                        return "Agent 0:\n  Move\n    Move forward\n  Turn\n    Stay\n  Shoot\n    Don't shoot"
                    else:
                        return llm_choice
                except Exception as fallback_e:
                    return "Agent 0:\n  Move\n    Move forward\n  Turn\n    Stay\n  Shoot\n    Don't shoot"
            
            return llm_choice
        
        except Exception as e:
            #print(f"Error calling LLM: {e}")
            error_msg = str(e).lower()

            fallback_triggers = [
                "400", "401", "403", "404", "413", "422", "424", 
                "429", "498", "499", "500", "502", "503", 
                "rate limit", "too many tokens", "quota", "invalid_request_error",
                "connection", "timeout" 
            ]
            if any(trigger in error_msg for trigger in fallback_triggers):
                try:
                    llm_choice = self.fallback_chain.invoke(prompt)
                    if not self._is_output_valid(llm_choice):
                        return "Agent 0:\n  Move\n    Move forward\n  Turn\n    Stay\n  Shoot\n    Don't shoot"
                    else:
                        return llm_choice            
                except Exception as fallback_e:
                    return "Agent 0:\n  Move\n    Move forward\n  Turn\n    Stay\n  Shoot\n    Don't shoot"
                
            return "Agent 0:\n  Move\n    Move forward\n  Turn\n    Stay\n  Shoot\n    Don't shoot"
    
    def _model_constructor(self, model_name):
        if model_name == "gemini-2.5-flash":
            return ChatGoogleGenerativeAI(model=self.model_name)
        elif model_name == "openai/gpt-oss-120b" or model_name == "meta-llama/llama-4-scout-17b-16e-instruct":
            return ChatGroq(temperature=0, model_name=self.model_name) # modellare temperatura
        elif model_name == 'qwen3-vl:2b' or model_name == 'llama3.2':
            return ChatOllama(model=self.model_name, temperature=0)
        
    def _is_output_valid(self, output: str) -> bool:
        """
        Controlla se l'output dell'LLM contiene la struttura sintattica minima richiesta.
        Nel tuo caso, deve contenere i tre blocchi fondamentali: Move, Turn e Shoot.
        """
        if not output:
            return False
            
        # Controlliamo la presenza delle tre macro-azioni necessarie
        required_keywords = ["Move", "Turn", "Shoot"]
        return all(keyword in output for keyword in required_keywords)
    
    def _format_input(self, prompt: dict):
        return [
            SystemMessage(content=prompt['sys_msg']),
            HumanMessage(content=prompt['hum_msg'])
        ]