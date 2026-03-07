from mlagents_envs.environment import UnityEnvironment
from mlagents_plugin.oracle_side_channel import OracleSideChannel
import time

env = UnityEnvironment(
    file_name='/Users/luketto/Desktop/Tesi/ML_Agents_tesi/simple_plugin/mlagents_plugin/config/Tanks/TanksGameBug', 
    seed=1, 
    side_channels=[OracleSideChannel()],
    additional_args=[
        "-screen-fullscreen", "0",      
        "-screen-width", "800",         
        "-screen-height", "600"]
    )

env.reset()

while True:
    env.step()
    time.sleep(0.01)