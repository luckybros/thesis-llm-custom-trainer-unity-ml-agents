# test_fallback.py
from langchain_model import LangchainModel # (o come si chiama il tuo file)
from dotenv import load_dotenv
import os

load_dotenv()

# Creiamo un finto settings object
class DummySettings:
    model = "meta-llama/llama-4-scout-17b-16e-instruct"

# SABOTIAMO LA CHIAVE API PER FORZARE L'ERRORE (Simula il Rate Limit)
os.environ["GROQ_API_KEY"] = "gsk_chiave_falsa_per_far_esplodere_groq_12345"

# --- 3. ESECUZIONE ---
print("Inizializzazione sistema...")
model = LangchainModel(DummySettings())

prompt_di_prova = {
    "sys_msg": "Sei l'intelligenza artificiale di un tank. Rispondi solo con l'azione in formato YAML.",
    "hum_msg": "Il nemico è davanti a te. Cosa fai?\nAgent 0:\n"
}

print("\nAvvio test. Aspettati un errore da Groq e l'intervento di Ollama...")
risultato = model.call_llm(prompt_di_prova)

print("\n--- RISULTATO FINALE RESTITUITO A UNITY ---")
print(risultato)