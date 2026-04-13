import sys
import numpy as np
import pygame
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_plugin.oracle_side_channel import OracleSideChannel
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.stats_side_channel import StatsSideChannel
import os

# --- 1. CONFIGURAZIONE ---
env_path = '/Users/luketto/Desktop/Tesi/ML_Agents_tesi/simple_plugin/mlagents_plugin/config/Tanks/TanksGameRL'

# I due behavior separati forniti da Unity
behavior_team_1 = 'Tanks?team=1'
behavior_team_0 = 'Tanks?team=0'

# --- 2. SETUP UNITY ---
print("Avvio di Unity in corso...")
engine_channel = EngineConfigurationChannel()
oracle_channel = OracleSideChannel(0, 0)
stats_channel = StatsSideChannel()

log_assoluto = os.path.abspath("debug_unity.log")

# time_scale=1.0 è fondamentale per giocare a velocità "umana"
engine_channel.set_configuration_parameters(width=1280, height=720, time_scale=1.0) 

env = UnityEnvironment(file_name=env_path, side_channels=[oracle_channel, engine_channel, stats_channel], 
                       additional_args=['-windowed', '-logFile', log_assoluto])
env.reset()

# --- 3. SETUP PYGAME ---
"""
pygame.init()
screen = pygame.display.set_mode((400, 150))
pygame.display.set_caption("Tanks - Multiplayer Manuale")
font = pygame.font.SysFont(None, 24)

# Testo a schermo per i controlli
screen.blit(font.render('TEAM 1: W/A/S/D (Muovi) - Z (Spara)', True, (255, 100, 100)), (20, 20))
screen.blit(font.render('TEAM 0: Frecce (Muovi) - Spazio (Spara)', True, (100, 100, 255)), (20, 60))
screen.blit(font.render('>> Clicca qui per giocare <<', True, (255, 255, 255)), (20, 100))
pygame.display.flip()

clock = pygame.time.Clock()
"""

print("\n=== CONTROLLO MANUALE 1v1 ATTIVATO ===")
print("Assicurati di aver cliccato sulla finestrella nera di Pygame.")
print("Premi la [X] della finestrella per uscire.\n")

# --- 4. LOOP PRINCIPALE ---
try:
    running = True
    while running:
        #clock.tick(60) # Limita a 60 FPS
        
        #for event in pygame.event.get():
        #    if event.type == pygame.QUIT:
       #         running = False
        
        if not running:
            break

        # Legge lo stato della tastiera
        #pygame.event.pump()
        #keys = pygame.key.get_pressed()

        # ==========================================
        # CONTROLLI TEAM 1 (WASD + Z)
        # ==========================================
        decision_steps_1, terminal_steps_1 = env.get_steps(behavior_team_1)
        if len(decision_steps_1) > 0:
            action_spec_1 = env.behavior_specs[behavior_team_1].action_spec
            actions_1 = np.zeros((len(decision_steps_1), len(action_spec_1.discrete_branches)), dtype=np.int32)
            
            for i in range(len(decision_steps_1)):
                # Move
                if keys[pygame.K_w]: actions_1[i, 0] = 1
                elif keys[pygame.K_s]: actions_1[i, 0] = 2
                # Turn
                if keys[pygame.K_d]: actions_1[i, 1] = 1
                elif keys[pygame.K_a]: actions_1[i, 1] = 2
                # Shoot
                if keys[pygame.K_z]: actions_1[i, 2] = 1

            action_tuple_1 = ActionTuple()
            action_tuple_1.add_discrete(actions_1)
            env.set_actions(behavior_team_1, action_tuple_1)

        # ==========================================
        # CONTROLLI TEAM 0 (Frecce + Spazio)
        # ==========================================
        decision_steps_0, terminal_steps_0 = env.get_steps(behavior_team_0)
        if len(decision_steps_0) > 0:
            action_spec_0 = env.behavior_specs[behavior_team_0].action_spec
            actions_0 = np.zeros((len(decision_steps_0), len(action_spec_0.discrete_branches)), dtype=np.int32)
            
            for i in range(len(decision_steps_0)):
                # Move
                if keys[pygame.K_UP]: actions_0[i, 0] = 1
                elif keys[pygame.K_DOWN]: actions_0[i, 0] = 2
                # Turn (1=Sinistra, 2=Destra come da tuo script C#)
                if keys[pygame.K_LEFT]: actions_0[i, 1] = 1
                elif keys[pygame.K_RIGHT]: actions_0[i, 1] = 2
                # Shoot
                if keys[pygame.K_SPACE]: actions_0[i, 2] = 1

            action_tuple_0 = ActionTuple()
            action_tuple_0.add_discrete(actions_0)
            env.set_actions(behavior_team_0, action_tuple_0)
        
        # Avanza di un frame per ENTRAMBI i team contemporaneamente
        env.step()

except KeyboardInterrupt:
    print("\nInterruzione manuale...")

except Exception as e:
    print(f"\n[ERRORE CRITICO INTERCETTATO]: {e}")
    import traceback
    traceback.print_exc()

finally:
    # --- 5. PULIZIA ---
    print("Chiusura in corso...")
    pygame.quit()
    env.close()