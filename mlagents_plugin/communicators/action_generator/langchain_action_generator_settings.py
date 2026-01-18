from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from simple_plugin.mlagents_plugin.communicators.action_generator.llm_settings import LLMActionGeneratorSettings

MODEL_CONSTRUCTORS = {
    "gemini-2.5-flash": ChatGoogleGenerativeAI,
    "openai/gpt-oss-120b": ChatGroq,
    "meta-llama/llama-4-scout-17b-16e-instruct": ChatGroq
}

class LangchainActionGeneratorSettings(LLMActionGeneratorSettings):

    def model_constructor_generator(self):
        if self.model_constructor is ChatGroq:
            return ChatGroq(temperature=0, model_name=self.model_name)
        if self.model_constructor is ChatGoogleGenerativeAI:
           return ChatGoogleGenerativeAI(model=self.model_name)

