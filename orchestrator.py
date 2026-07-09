import subprocess
import time
import sys

PYTHON_SERVER = "/lustre/home/rpietrantuono/.conda/envs/game-test-llm/bin/python" 

SERVER_COMMAND = [
    PYTHON_SERVER, 
    "-m", "mlagents_plugin.communicators.server.zmq_communication_server",
    "--config", "/lustre/home/rpietrantuono/envs/game-test/custom-plugin/mlagents_plugin/config/Tanks/TanksLLM.yaml"
]

MLAGENTS_ENV = "/lustre/home/rpietrantuono/.conda/envs/game-test/bin/python" 

GAME_COMMAND = [
    MLAGENTS_ENV,
    "/lustre/home/rpietrantuono/envs/game-test/custom-plugin/mlagents_train_DRL_LLM.py"
]

def main():
    print("----------[START ORCHESTRATOR]----------")

    server_process = None
    game_process = None

    try:
        print("\n[1] Avvio del Server")
        print(f"    Comando: {' '.join(SERVER_COMMAND)}")
        
        server_process = subprocess.Popen(SERVER_COMMAND)

        time.sleep(15)

        if server_process.poll() is not None:
            print("ERRORE: Il server ZMQ si è chiuso inaspettatamente all'avvio.")
            sys.exit(1)

        print("\n[2] Server pronto, avvio dell'addestramento ML-Agents...")
        print(f"Comando: {' '.join(GAME_COMMAND)}")
        
        game_process = subprocess.Popen(GAME_COMMAND)

        game_process.wait()

    except KeyboardInterrupt:
        print("\n\nInterruzione manuale (Ctrl+C). Hai fermato l'addestramento.")

    except Exception as e:
        print(f"Errore imprevisto nell'orchestratore: {e}")

    finally:
        print("\n[3] Pulizia della memoria e spegnimento in corso...")
        
        if game_process and game_process.poll() is None:
            game_process.terminate()
            
        if server_process and server_process.poll() is None:
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()
                
        print("Orchestratore terminato")

if __name__ == "__main__":
    main()