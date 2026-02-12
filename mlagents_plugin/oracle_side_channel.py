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
import time
import os

class OracleSideChannel(SideChannel):

    def __init__(self) -> None:
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7"))

        self.worker = threading.Thread(target=self._on_update, daemon=True)
        self.last_heartbeat_time = time.time()
        self.timeout_seconds = 10
        self.history = []
        self.history_length = 100

    def on_message_received(self, msg: IncomingMessage) -> None:
        """
        Note: we must implement this method of the SideChannel interface to recieve messages from Unity
        """
        message_content = msg.read_string()
        #print(message_content)
        if message_content == "START":
            self.worker.start()
        elif message_content.startswith('ALIVE'):
            self.last_heartbeat_time = time.time()
            message_split = message_content.split(' ')
            self.update_history(message_split[1])

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
            if time.time() - self.last_heartbeat_time > self.timeout_seconds:
                print(f"[CRASH ORACLE] ERROR! Unity does not respond")
                print(f"[CRASH ORACLE] Detected possible crash")
                print(f"[CRASH ORACLE] History: {self.history} ")

                # Qui decidi tu cosa fare. Esempi:
                # - Scrivere su un file di log
                # - Inviare una mail
                # - Uccidere il processo Python (perché ormai è bloccato)
                os._exit(1) # Decommenta per terminare brutalmente tutto
            time.sleep(0.01)






