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

    def call_llm(self, prompt):
        #print(f'prompt: {prompt}')
        llm_choice = self.chain.invoke(prompt)
        print(f"llm_choice: {llm_choice}")
        return llm_choice
    
    def _model_constructor(self, model_name):
        if model_name == "gemini-2.5-flash":
            return ChatGoogleGenerativeAI(model=self.model_name)
        elif model_name == "openai/gpt-oss-120b" or model_name == "meta-llama/llama-4-scout-17b-16e-instruct":
            return ChatGroq(temperature=0, model_name=self.model_name) # modellare temperatura
        elif model_name == 'qwen3-vl:2b' or model_name == 'llama3.2':
            return ChatOllama(model=self.model_name, temperature=0)
        
    def _format_input(self, prompt: dict):
        return [
            SystemMessage(content=prompt['sys_msg']),
            HumanMessage(content=prompt['hum_msg'])
        ]