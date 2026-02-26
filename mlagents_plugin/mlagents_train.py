import subprocess
import time
import sys
import argparse
import os

def kill_unity_processes(env_name):
    print(f"[PULIZIA] Cerco processi residui di {env_name}...")
    base_name = os.path.basename(env_name)
    os.system(f"pkill -f {base_name}")
    time.sleep(2)

def run_ml_agents(args):
    # Il comando base che usi di solito
    # Aggiungi --resume alla lista degli argomenti
    results_path = os.path.join("results", args.id)
    base_cmd = ["mlagents-learn", args.yaml, "--run-id="+ args.id, "--env=" + args.env]

    if args.no_graphics:
        base_cmd.append("--no-graphics")

    if args.numenvs:
        base_cmd.append("--num-envs" + args.numenvs)

    if os.path.exists(results_path):
        print(f"[SUPERVISORE] Cartella {results_path} trovata. Riprendo l'addestramento (--resume).")
        base_cmd.append("--resume")
    else:
        print(f"[SUPERVISORE] Nessuna run precedente trovata per '{args.id}'. Inizio da zero.")

    print("\n" + "="*50)
    print(f"[SUPERVISORE] Comando: {' '.join(base_cmd)}")
    print("="*50 + "\n")

    while True:
        try: 
            print(f"\n[SUPERVISORE] Avvio sessione di addestramento: {' '.join(base_cmd)}")
            
            # Lanciamo il processo
            process = subprocess.Popen(base_cmd)
            
            # Attendiamo che il processo finisca
            return_code = process.wait()
            
            if return_code == 0:
                print("[SUPERVISORE] Addestramento completato con successo.")
                break
            else:
                print(f"[SUPERVISORE] Rilevato crash (Exit Code: {return_code}). Riavvio tra 5 secondi...")
                time.sleep(5)
                
            if "--resume" not in base_cmd:
                base_cmd.append("--resume")

        except KeyboardInterrupt:
            print("\n[SUPERVISORE] Interruzione manuale. Uscita...")
            process.terminate()
            sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", required=True, help="Percorso del file di configurazione")
    parser.add_argument("-e", "--env", required=True, help="Percorso dell'eseguibile")  # facciamo solo con l'eseguibile x semplicità tanto trainiamo sempre con l'eseguibile
    parser.add_argument("-i", "--id", required=True, help="Nome della run")
    parser.add_argument("-n", "--numenvs", type=int, help="Numero degli env in parallelo")
    parser.add_argument("-g", "--no-graphics", help="Se runnare senza interfaccia grafica")

    args = parser.parse_args()
    run_ml_agents(args)