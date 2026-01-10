import yaml
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

MODEL_CONSTRUCTORS = {
    "gemini-2.5-flash": ChatGoogleGenerativeAI,
    "openai/gpt-oss-120b": ChatGroq,
    "meta-llama/llama-4-scout-17b-16e-instruct": ChatGroq
}

class LangchainActionGeneratorSettings:

    def __init__(self, settings_path: str):
        with open(settings_path, 'r') as f:
            config = yaml.safe_load(f)
        llm_config = config.get('llm_settings')
        self.game_desc = llm_config.get('game_desc', '')
        self.agent_role = llm_config.get('agent_role', '')
        self.history_length = llm_config.get('history_length', 1)
        self.model_name = llm_config.get('model_name', '')
        self.task = llm_config.get('task', '')
        self.actions = llm_config.get('actions', [])
        self.model_constructor = MODEL_CONSTRUCTORS[self.model_name]

        self.use_vectorial_obs = llm_config.get('use_vectorial_obs', False)
        self.use_visual_obs = llm_config.get('use_visual_obs', False)
        self.batch_size = llm_config.get('batch_size', None)

    def model_constructor_generator(self):
        #if self.model_constructor is ChatGoogleGenerativeAI:
        #    return ChatGoogleGenerativeAI(model=self.model_name)
        if self.model_constructor is ChatGroq:
            # aggiungere temperatura
            return ChatGroq(temperature=0, model_name=self.model_name)
        if self.model_constructor is ChatGoogleGenerativeAI:
           return ChatGoogleGenerativeAI(model=self.model_name)

