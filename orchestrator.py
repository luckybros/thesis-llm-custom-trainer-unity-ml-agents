import subprocess
import time
import sys

PYTHON_SERVER = "/Users/luketto/.venv-vllm-metal/bin/python" 

SERVER_COMMAND = [
    PYTHON_SERVER, 
    "-m", "mlagents_plugin.communicators.server.zmq_communication_server",
    "--config", "/Users/luketto/Desktop/Tesi/ML_Agents_tesi/simple_plugin/mlagents_plugin/config/Tanks/TanksLLM.yaml"
]

MLAGENTS_ENV = "/opt/anaconda3/envs/mlagents/bin/python" 
MLAGENTS_LEARN_COMMAND = "/opt/anaconda3/envs/mlagents/bin/mlagents-learn"

GAME_COMMAND = [
    MLAGENTS_ENV,
    MLAGENTS_LEARN_COMMAND,
    "mlagents_plugin/config/Tanks/TanksMLAgentsLLMPlugin.yaml",
    "--run-id=ehisonopiccolo",
    "--force"
]

def main():
    print("Avvio dell'Orchestratore di Tesi...")

    server_process = None
    game_process = None

    try:
        print("\n[1] Avvio del Server ZMQ Communication...")
        print(f"    Comando: {' '.join(SERVER_COMMAND)}")
        
        server_process = subprocess.Popen(SERVER_COMMAND)

        print("    Attendere 15 secondi per l'inizializzazione del server e del modello...")
        time.sleep(15)

        if server_process.poll() is not None:
            print("ERRORE: Il server ZMQ si è chiuso inaspettatamente all'avvio.")
            sys.exit(1)

        print("\n[2] Server pronto! Avvio dell'addestramento ML-Agents...")
        print(f"    Comando: {' '.join(GAME_COMMAND)}")
        
        game_process = subprocess.Popen(GAME_COMMAND)

        game_process.wait()

        print("\n✅ Addestramento/Partita terminata.")

    except KeyboardInterrupt:
        print("\n\n⚠️ Interruzione manuale (Ctrl+C). Hai fermato l'addestramento.")

    except Exception as e:
        print(f"Errore imprevisto nell'orchestratore: {e}")

    finally:
        print("\n[3] Pulizia della memoria e spegnimento in corso...")
        
        if game_process and game_process.poll() is None:
            game_process.terminate()
            
        if server_process and server_process.poll() is None:
            print("    Chiusura del Server ZMQ e liberazione della memoria...")
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()
                
        print("Tutto chiuso in sicurezza. Ciao!")

if __name__ == "__main__":
    main()