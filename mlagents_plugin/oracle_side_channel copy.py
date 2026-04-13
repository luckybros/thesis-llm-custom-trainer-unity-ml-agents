from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.side_channel import (
    SideChannel, 
    IncomingMessage, 
    OutgoingMessage
)
from mlagents_envs.exception import (
    UnityCommunicationException,
    UnityTimeOutException,
    UnityEnvironmentException,
    UnityCommunicatorStoppedException,
)
import numpy as np
import uuid
import threading
import signal
import time
import os
import csv
import atexit
import psutil
import json
from datetime import datetime

class OracleSideChannel(SideChannel):

    def __init__(self, id, log_folder) -> None:
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7"))

        self.worker = threading.Thread(target=self._on_update, daemon=True)
        self.last_heartbeat_time = time.time()
        self.timeout_seconds = 100
        self.history = []
        self.history_length = 100
        self.csv_path = f"bug_log_drl_{id}.csv"

        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "step", "oracle", "bug_type", "message"])
        self.bug_detected = "None"
        self.step = self._load_base_step(log_folder)
        # atexit.register(self._on_python_exit)

    def on_message_received(self, msg: IncomingMessage) -> None:
        """
        Note: we must implement this method of the SideChannel interface to recieve messages from Unity
        """
        message_content = msg.read_string()
        # --- BUG REPORT--- (format: [GAME_BUG]{ORACLE_NAME}|{BUG_TYPE}|{MESSAGE})
        if message_content.startswith('[GAME_BUG]'):
            parts = message_content.split('|')
            
            oracle_name = parts[1]
            bug_type = parts[2]
            bug_message = parts[3]
            #self.step = int(parts[4])

            # ORACLE_NAME: DAMAGE ORACLE
            print(f"[{oracle_name}] Bug detected")
            print(f"[{oracle_name}] Type: {bug_type}")
            print(f"[{oracle_name}] {bug_message}")

            self._log_to_csv(oracle_name, bug_type, bug_message)

            self.bug_detected = bug_type

            if oracle_name == "CRASH ORACLE":
                print(f"[ORACLE] Errore critico da {oracle_name}. Chiusura di Unity in corso...")
                if bug_message.startswith('DivideByZeroException') or bug_message.startswith('Exception: Projectile Explosion Crash!') or bug_message.startswith('IndexOutOfRangeException'):
                    self.reset_environment()
                else:
                    self._kill_unity_process()
            elif oracle_name == "HANG ORACLE":
                print(f"[ORACLE] Errore critico da {oracle_name}. Chiusura di Unity in corso...")
                self.reset_environment()
            elif oracle_name == "STUCK ORACLE":
                print(f"[STUUUUCK]")
                self.reset_environment()
            else :
                print(f"[ORACLE] Bug non critico da {oracle_name}. Invio comando di RESET a Unity...")
                #self.reset_environment()

        
        # --- START: Unity is ready ---
        elif message_content.startswith("[START]"):
            #self.reset_environment()
            parts = message_content.split('|')
            self.step = int(parts[1])
            self.worker.start()

        # --- ALIVE: heartbeat from HangOracle ---
        elif message_content.startswith('[ALIVE]'):
            self.last_heartbeat_time = time.time()
            message_split = message_content.split('|')
            self.step = int(message_split[1])

    def _log_to_csv(self, oracle: str, bug_type: str, message: str):
        """Append una riga al CSV"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Pulisci il messaggio: togli newline per non rompere il CSV
        clean_msg = message.replace("\n", " | ").replace("\r", "")
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([timestamp, self.step, oracle, bug_type, clean_msg])
        print(f"[ORACLE] Bug salvato in {self.csv_path}: [{oracle}][{bug_type}]")

    def reset_environment(self) -> None:
        self.send_string("RESET")

    def update_history(self, num: int) -> None:
        self.history.append(num)

    def send_string(self, data: str) -> None:
        # Add the string to an OutgoingMessage
        msg = OutgoingMessage()
        msg.write_string(data)
        # Method to queue the data we want to send
        super().queue_message_to_send(msg)

    def _on_update(self) -> None:
        """Thread that monitors hearbeat for hang detection"""
        self.last_heartbeat_time = time.time()
        print(f"[HANG ORACLE] Monitoring started...")
        while True:
            # Hang oracle
            if time.time() - self.last_heartbeat_time > self.timeout_seconds:
                #print(f"[HANG ORACLE] ERROR! Unity does not respond")
                #print(f"[HANG ORACLE] Detected possible hang")
                #print(f"[HANG ORACLE] History: {self.history} ")
                #self._log_to_csv("HANG ORACLE", "hang", "Not recieved ping for timeout seconds")
                self.save_hang_bug()
                #self.bug_detected = True

                # qui decidiamo cosa fare
                # scrivere su un file di log
                # inviare una mail
                # uccidere il processo Python (perché ormai è bloccato)
                # riprendere il processo
                self._kill_unity_process()
                break

            time.sleep(0.01)

    def save_hang_bug(self) -> None:
        print(f"[HANG ORACLE] ERROR! Unity does not respond")
        print(f"[HANG ORACLE] Detected possible hang")
        print(f"[HANG ORACLE] History: {self.history} ")
        self._log_to_csv("HANG ORACLE", "hang", "Not recieved ping for timeout seconds")

        self.bug_detected = "hang"

    """
    def _on_python_exit(self):
        # if mlagents closes but it's not hanging, so an unexpected crash without error
        print(f"self.crash_detected: {self.crash_detected}")
        if not self.crash_detected:
            print(f"[CRASH ORACLE]: ERROR! There was un unexpected error")
            print(f"[CRASH ORACLE]: Detected possible crash")
            print(f"[CRASH ORACLE]: No error message detected")
            self._log_to_csv("crash", "Unexpected crash: no error message from Unity")
    """
    def _load_base_step(self, log_folder: str) -> int:
        """Legge lo step globale dall'ultimo salvataggio di ML-Agents (se esiste)"""
        status_path = os.path.join(log_folder, "training_status.json")
        if os.path.exists(status_path):
            
            try:
                with open(status_path, "r") as f:
                    data = json.load(f)
                    # Cerca lo step per il tuo comportamento (presumibilmente "Tanks")
                    for behavior, info in data.items():
                        if "step" in info:
                            print(f"[ORACLE] Resume rilevato. Partiamo dallo step globale: {info['step']}")
                            return int(info["step"])
            except Exception as e:
                print(f"[ORACLE] Errore lettura training_status.json: {e}")
        return 0

    def _kill_unity_process(self):
        """Find the Unity child process and kill it """
        try:
            # Otteniamo il PID del processo padre (mlagents-learn)
            # e cerchiamo i processi figli (l'eseguibile di Unity)
            current_process = psutil.Process(os.getpid())
            children = current_process.children(recursive=True)
                
            if not children:
                print("[ORACLE] No Unity child processes found.")
                return

            for child in children:
                # Verifichiamo che sia l'eseguibile del gioco (puoi filtrare per nome se vuoi)
                print(f"[ORACLE] Terminating child process: {child.name()} (PID: {child.pid})")
                child.kill() # Prova una chiusura gentile
                # child.kill() # Usa questo se terminate() non bastasse
                    
            print("[ORACLE] Unity killed. ML-Agents should now save and exit.")
                
        except Exception as e:
            print(f"[ORACLE] Error while killing Unity: {e}")

    def step_increment(self, steps):
        self.step = steps