from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

# 1. Creiamo solo il canale per la time_scale
engine_channel = EngineConfigurationChannel()

# Percorso dell'eseguibile
file_path = '/Users/luketto/Desktop/Tesi/ML_Agents_tesi/simple_plugin/mlagents_plugin/config/Tanks/TanksGameAA'

print("Avvio di Unity in corso...")

# 2. Inizializziamo l'ambiente
# ATTENZIONE: no_graphics=False è obbligatorio per vedere il gioco e usare la tastiera
env = UnityEnvironment(
    file_name=file_path,
    side_channels=[engine_channel],
    no_graphics=False, 
    timeout_wait=100
)

env.reset()

# 3. Imposta la time_scale desiderata per i tuoi test (es. 20, 50, 100)
test_time_scale = 20.0
engine_channel.set_configuration_parameters(time_scale=test_time_scale)
print(f"\nAmbiente avviato con time_scale a {test_time_scale}x!")
print("Usa la tastiera per giocare. Premi Ctrl+C qui nel terminale per chiudere.")

try:
    # Loop infinito: tiene in vita l'ambiente e gli fa processare i frame
    while True:
        env.step()

except KeyboardInterrupt:
    print("\nInterruzione manuale (Ctrl+C).")

finally:
    env.close()
    print("Ambiente chiuso correttamente.")