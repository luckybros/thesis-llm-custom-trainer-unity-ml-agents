import pandas as pd
from tbparse import SummaryReader
import os

log_dir = "./results_drl" 

print("Lettura dei log in corso...")

# CORREZIONE QUI: Usiamo un set {'dir_name'} invece di un dizionario
reader = SummaryReader(log_dir, extra_columns={'dir_name'})
df = reader.scalars

if not df.empty:
    cartella_output = "risultati_drl_csv"
    os.makedirs(cartella_output, exist_ok=True)
    
    print("Inizio l'esportazione...")
    
    # Ora 'dir_name' esiste ed è pronto per essere raggruppato
    for (run_name, metrica), df_gruppo in df.groupby(['dir_name', 'tag']):
        
        # Pulizia dei nomi per evitare problemi con i percorsi dei file
        run_sicura = run_name.replace('/', '_').replace('\\', '_')
        metrica_sicura = metrica.replace('/', '_').replace('\\', '_')
        
        # Crea il nome del file (es: cartella/Tank_DRL_LLM_Baseline_2___Coverage_UniqueStates.csv)
        nome_file = f"{cartella_output}/{run_sicura}___{metrica_sicura}.csv"
        
        # Manteniamo solo Step e Value
        df_da_salvare = df_gruppo[['wall time', 'step', 'value']]
        
        # Salva il file
        df_da_salvare.to_csv(nome_file, index=False)
        print(f"Salvato: {nome_file}")
        
    print(f"\nFinito! Trovi tutti i tuoi file CSV divisi per Run e Metrica nella cartella '{cartella_output}'.")
else:
    print("Nessun dato scalare trovato in questa cartella.")