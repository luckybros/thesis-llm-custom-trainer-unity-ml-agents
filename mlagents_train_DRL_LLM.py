import subprocess
import time
import sys
import argparse
import os

NUM_SEEDS = 10 

def kill_unity_processes(env_name):
    print(f"[PULIZIA] Cerco processi residui di {env_name}...")
    base_name = os.path.basename(env_name)
    os.system(f"pkill -f {base_name}")
    time.sleep(2)

def run_ml_agents(base_seed):
    os.environ["ORACLE_ID"] = str(base_seed)

    results_path = os.path.join("results", f"Tank_DRL_LLM_Baseline_{base_seed}")
    base_cmd = [
        "mlagents-learn", 
        "mlagents_plugin/config/Tanks/TanksMLAgentsLLMPlugin.yaml", 
        f"--run-id=Tank_DRL_LLM_Baseline_{base_seed}", 
        "--env=mlagents_plugin/config/Tanks/TanksGameRL",
        "--no-graphics"
    ]

    if not os.path.exists(results_path):
        print(f"[SUPERVISORE] Nessuna run precedente trovata per '{base_seed}'. Inizio da zero.")
    else:
        print(f"[SUPERVISORE] Run '{base_seed}' già esistente. Aggiungo --resume.")
        base_cmd.append("--resume")
        
    current_run_seed = base_seed
    seed_log_file = f"seed_history_run_{base_seed}.txt"

    with open(seed_log_file, "a") as f:
        f.write(f"\n{'='*30}\nInizio addestramento Base Seed: {base_seed}\n")

    print("\n" + "="*50)
    print(f"[SUPERVISORE] Comando: {' '.join(base_cmd)}")
    print("="*50 + "\n")

    while True:
        try: 
            print(f"\n[SUPERVISORE] Avvio sessione di addestramento: {' '.join(base_cmd)}")
            cmd_to_run = base_cmd + [f"--seed={current_run_seed}"]

            with open(seed_log_file, "a") as f:
                f.write(f"Avvio/Resume con seed: {current_run_seed}\n")
            # Lanciamo il processo
            process = subprocess.Popen(cmd_to_run)
            
            # Attendiamo che il processo finisca
            return_code = process.wait()

            if return_code == 0:
                print("[SUPERVISORE] Addestramento completato con successo.")
                break
            else:
                print(f"[SUPERVISORE] Rilevato crash (Exit Code: {return_code}). Riavvio tra 5 secondi...")
                if "--resume" not in base_cmd:
                    base_cmd.append("--resume")
                # TODO: comando del seed cambiato, aumentato di 100 magari? cosi la prima è 1, 101, 201, e la seconda è 2, 102, 202, altrimenti se aumentassimo di 1 otterrei le run della versione di dopo, e devono comunque essere riproducibili. magari salviamo in un log tutti i seed usati per i valori randomici 
                current_run_seed += 100
                time.sleep(5)
            
        except KeyboardInterrupt:
            print("\n[SUPERVISORE] Interruzione manuale. Uscita...")
            process.terminate()
            sys.exit(0)


if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument("-y", "--yaml", required=True, help="Percorso del file di configurazione")
    #parser.add_argument("-e", "--env", required=True, help="Percorso dell'eseguibile")  # facciamo solo con l'eseguibile x semplicità tanto trainiamo sempre con l'eseguibile
    #parser.add_argument("-n", "--numenvs", type=int, help="Numero degli env in parallelo")
    #parser.add_argument("-g", "--no-graphics", help="Se runnare senza interfaccia grafica")

    #args = parser.parse_args()

    for current_seed in range(1, NUM_SEEDS+1):
        if os.path.exists('/Users/luketto/Library/Application Support/DefaultCompany/Tanks/TankStatesLog.txt'):
            os.remove('/Users/luketto/Library/Application Support/DefaultCompany/Tanks/TankStatesLog.txt')
            print("Deleted States Log")
        if os.path.exists('/Users/luketto/Library/Application Support/DefaultCompany/Tanks/GlobalStatesLog.txt'):
            os.remove('/Users/luketto/Library/Application Support/DefaultCompany/Tanks/GlobalStatesLog.txt')
            print("Deleted Global States Log")
        if os.path.exists('/Users/luketto/Library/Application Support/DefaultCompany/Tanks/ExploredTiles.txt'):
            os.remove(('/Users/luketto/Library/Application Support/DefaultCompany/Tanks/ExploredTiles.txt'))
        run_ml_agents(current_seed)