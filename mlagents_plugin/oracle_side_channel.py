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
from datetime import datetime

class OracleSideChannel(SideChannel):

    def __init__(self) -> None:
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7"))

        self.worker = threading.Thread(target=self._on_update, daemon=True)
        self.last_heartbeat_time = time.time()
        self.timeout_seconds = 100
        self.history = []
        self.history_length = 100
        self.crash_detected = False
        self.error_message = ""
        self.csv_path = "bug_log.csv"
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "type", "message"])
        atexit.register(self._on_python_exit)

    def on_message_received(self, msg: IncomingMessage) -> None:
        """
        Note: we must implement this method of the SideChannel interface to recieve messages from Unity
        """
        message_content = msg.read_string()
        #print(message_content)
        #print(f"self.crash_detected: {self.crash_detected}")
        if message_content.startswith('ERROR'):
            self.crash_detected = True
            self.error_message = message_content
            print(f"[CRASH ORACLE]: ERROR! There was un unexpected error")
            print(f"[CRASH ORACLE]: Detected possible crash")
            print(f"[CRASH ORACLE]: {self.error_message}")
            self._log_to_csv("crash", self.error_message)
        if message_content == "START":
            self.worker.start()
        elif message_content.startswith('ALIVE'):
            self.last_heartbeat_time = time.time()
            message_split = message_content.split(' ')
            self.update_history(message_split[1])

    def _log_to_csv(self, bug_type: str, message: str):
        """Append una riga al CSV"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Pulisci il messaggio: togli newline per non rompere il CSV
        clean_msg = message.replace("\n", " | ").replace("\r", "")
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, bug_type, clean_msg])
        print(f"[ORACLE] Bug salvato in {self.csv_path}: [{bug_type}]")

    def update_history(self, num: int) -> None:
        self.history.append(num)

    def send_string(self, data: str) -> None:
        # Add the string to an OutgoingMessage
        msg = OutgoingMessage()
        msg.write_string(data)
        # Method to queue the data we want to send
        super().queue_message_to_send(msg)

    def _on_update(self) -> None:
        self.last_heartbeat_time = time.time()
        print(f"[CRASH ORACLE] Starting...")
        while True:
            # crash oracle
                #os._exit(1)

            # Hang oracle
            if time.time() - self.last_heartbeat_time > self.timeout_seconds:
                print(f"[HANG ORACLE] ERROR! Unity does not respond")
                print(f"[HANG ORACLE] Detected possible hang")
                print(f"[HANG ORACLE] History: {self.history} ")
                self._log_to_csv("hang", "Not recieved ping for 10 seconds")

                # qui decidiamo cosa fare
                # scrivere su un file di log
                # inviare una mail
                # uccidere il processo Python (perché ormai è bloccato)
                # riprendere il processo
                self.crash_detected = True
                self._kill_unity_process()
                break

            time.sleep(0.01)

    def _on_python_exit(self):
        # if mlagents closes but it's not hanging, so an unexpected crash without error
        print(f"self.crash_detected: {self.crash_detected}")
        if not self.crash_detected:
            print(f"[CRASH ORACLE]: ERROR! There was un unexpected error")
            print(f"[CRASH ORACLE]: Detected possible crash")
            print(f"[CRASH ORACLE]: No error message detected")
            self._log_to_csv("crash", "Unexpected crash: no error message from Unity")

    def _kill_unity_process(self):
        """Trova il processo di Unity collegato e lo abbatte"""
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
                child.terminate() # Prova una chiusura gentile
                # child.kill() # Usa questo se terminate() non bastasse
                    
            print("[ORACLE] Unity killed. ML-Agents should now save and exit.")
                
        except Exception as e:
            print(f"[ORACLE] Error while killing Unity: {e}")