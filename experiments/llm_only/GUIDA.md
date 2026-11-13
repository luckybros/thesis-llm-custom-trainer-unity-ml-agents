# Guida rapida: esperimento "solo LLM" in locale

In questo esperimento **un LLM locale gioca da solo** al tuo gioco Unity: niente reinforcement learning, niente `mlagents-learn`, niente training. A ogni decisione dell'agente:

```
Unity (build)  --osservazioni-->  llm_agent.py  --ZMQ-->  server LLM  --prompt-->  Ollama (Qwen, ...)
               <----azione-------                <--------azione-------  <--risposta---
```

1. Il gioco manda le osservazioni (raycast, valori numerici, immagine della camera).
2. Il server le traduce in testo (o immagine), costruisce il prompt e interroga il modello in Ollama.
3. La risposta viene convertita negli indici delle azioni discrete e rimandata a Unity.

Funziona su Windows, macOS e Linux. Serve solo il tuo PC: per modelli da 3–8B bastano 8–16 GB di RAM (meglio con GPU, ma va anche su CPU / Apple Silicon, solo più lento).

File coinvolti:

| File | A cosa serve |
|---|---|
| `experiments/llm_only/orchestrator.py` | avvia il server LLM e poi l'agente; chiude tutto alla fine |
| `experiments/llm_only/llm_agent.py` | collega la build Unity al server; `--inspect` mostra le osservazioni della build |
| `experiments/llm_only/configs/template_llm.yaml` | template **YAML LLM** (prompt, azioni, modello) — letto dal server |
| `experiments/llm_only/configs/template_training.yaml` | template **YAML training** (quali osservazioni inviare) — letto dall'agente |
| `mlagents_plugin/config/FPS/FPSLLM.yaml` + `FPSTrainingLLM.yaml` | esempio completo della coppia di YAML (gioco FPS) |

---

## 1. Installazione (una volta sola)

### 1.1 Ollama + modello

Scarica e installa Ollama da <https://ollama.com/download>, poi:

```bash
ollama pull qwen2.5:7b       # osservazioni testuali
ollama pull qwen2.5vl:7b     # solo se userai osservazioni visuali (camera)
ollama list                  # controlla che ci siano
```

Su Windows/macOS Ollama parte da solo come applicazione; su Linux, se non è attivo: `ollama serve`.
PC poco potente? Usa `qwen2.5:3b` / `qwen2.5vl:3b`.

### 1.2 Due ambienti Python (conda)

Il lato Unity (ML-Agents) e il lato LLM (LangChain) hanno dipendenze diverse: si tengono separati.

```bash
git clone -b only-llm https://github.com/luckybros/thesis-llm-custom-trainer-unity-ml-agents.git
cd thesis-llm-custom-trainer-unity-ml-agents
```

**Ambiente agente** (ML-Agents). La versione di `mlagents` deve essere quella compatibile con il package ML-Agents installato nel progetto Unity (vedi la tabella delle release di ML-Agents):

```bash
conda create -n llm-agent python=3.10 -y
conda activate llm-agent
pip install mlagents==1.1.0        # adatta alla versione del package Unity
pip install -e .
```

**Ambiente server LLM**:

```bash
conda create -n llm-server python=3.10 -y
conda activate llm-server
pip install -r requirements_llm.txt
pip install torch transformers faiss-cpu   # solo per osservazioni visuali
```

Annota il percorso dell'interprete del server, servirà dopo:

```bash
conda activate llm-server
python -c "import sys; print(sys.executable)"
# es. Windows: C:\Users\<te>\miniconda3\envs\llm-server\python.exe
# es. macOS/Linux: /Users/<te>/miniconda3/envs/llm-server/bin/python
```

---

## 2. Preparare il gioco in Unity e fare la build

Nel prefab/oggetto dell'agente:

- **Behavior Parameters** → *Behavior Type* = `Default`, nessun *Model* assegnato (così le decisioni arrivano da Python).
- **Decision Requester** presente (il *Decision Period* stabilisce ogni quanti step fisici l'LLM decide).
- Azioni **discrete** (le continue non sono supportate in questa modalità).
- Annota il **Behavior Name**, l'ordine dei branch in `OnActionReceived` e delle opzioni di ogni branch: serviranno negli YAML.

Poi **File → Build Settings** (Unity 6: *Build Profiles*) → piattaforma del tuo PC → *Build* in una cartella, es. `Builds/`. Otterrai:

| Sistema | Eseguibile da passare a `--env` |
|---|---|
| Windows | `Builds/MioGioco.exe` |
| macOS | `Builds/MioGioco.app` |
| Linux | `Builds/MioGioco.x86_64` |

> In alternativa puoi non fare la build: ometti `--env` e premi **Play** nell'Editor quando lo script dice di farlo.
>
> macOS: se la build non si apre perché "danneggiata/non verificata": `xattr -dr com.apple.quarantine Builds/MioGioco.app`.

---

## 3. Scoprire le osservazioni della build

Dall'ambiente agente:

```bash
conda activate llm-agent
python experiments/llm_only/llm_agent.py --env Builds/MioGioco.exe --inspect
```

Esempio di output:

```
Behavior: Player?team=0
  Azioni discrete (opzioni per ramo): (3, 2)
  Azioni continue: 0
  Osservazioni (index -> name, shape, tipo suggerito):
    0: RayPerceptionSensor                 shape=(20,)  ->  RAYCAST
    1: VectorSensor_size2                  shape=(2,)   ->  VECTORIAL
```

Ti dice gli **indici** delle osservazioni (`observation_types` nello YAML training), quante **opzioni ha ogni branch** di azione (`actions` nello YAML LLM) e stampa una bozza di `observation_types` da incollare.

Per un raycast la shape è `num_rays × (num_detectable_tags + 2)` (× gli eventuali *Stacked Raycasts*): nell'esempio 5 raggi × (2 tag + 2) = 20.

---

## 4. Scrivere i due YAML dell'esperimento

Ogni esperimento usa **due file**, entrambi commentati nei template:

| File | Letto da | Contiene |
|---|---|---|
| YAML LLM | server LLM | `llm_settings`: modello, prompt, azioni, come tradurre le osservazioni in testo |
| YAML training | agente (`llm_agent.py`) | `behaviors.<nome>.hyperparameters.observation_types` (quali osservazioni inviare) ed `engine_settings` |

Lo YAML training ha lo stesso formato usato da `mlagents-learn` con il trainer `llm_trainer`: nell'esperimento solo-LLM servono solo `observation_types` ed `engine_settings`, il resto viene ignorato. Così lo stesso file è già pronto se in seguito si vuole fare il training DRL+LLM.

```bash
cp experiments/llm_only/configs/template_llm.yaml      experiments/llm_only/configs/mio_gioco_llm.yaml
cp experiments/llm_only/configs/template_training.yaml experiments/llm_only/configs/mio_gioco_training.yaml
```

**YAML LLM** (`llm_settings`):
   - `model_name`: nome esatto da `ollama list`.
   - `game_desc`, `agent_role`, `task`: il prompt. In `agent_role` spiega **il formato di risposta** (blocco `Agent 0:` con nome azione e opzione scelta, come nel template).
   - `actions.discrete`: un elemento per branch, **nello stesso ordine di `OnActionReceived`**; le `options` nello stesso ordine degli indici del branch.
   - `default_agent_action`: azione di riserva se il modello risponde male.
   - `obs_template` + `features`: come trasformare i numeri in testo (`BUCKET`, `BOOL`, `ONE_HOT` per i valori di `CollectObservations`; tag, raggi e distanze per i raycast).
   - `use_vectorial_obs` / `use_raycast_obs` / `use_visual_obs`: cosa mettere nel prompt.

**YAML training**:
   - sotto `behaviors:` metti il **Behavior Name** dell'agente (quello di *Behavior Parameters*);
   - `observation_types`: `index` preso dal punto 3, `name` uguale a quello usato nello YAML LLM (es. `RAYCAST_FRONT` in `features.raycast` e in `obs_template`);
   - `engine_settings` (opzionale): `time_scale`, `capture_frame_rate`.

### Versione visuale (immagine della camera)

- Aggiungi una **Camera Sensor** all'agente in Unity e rifai la build.
- YAML training: in `observation_types` aggiungi l'osservazione camera con `name: VISUAL`, `type: VISUAL` e il suo indice.
- YAML LLM: `use_visual_obs: True` e un modello vision, es. `model_name: qwen2.5vl:7b`.
- Puoi tenere anche le osservazioni testuali (immagine + testo) o metterle a `False` (solo immagine).
- Non serve altro: l'orchestratore avvia il server in modalità visuale e apre il gioco con la grafica attiva. Le immagini inviate vengono salvate in `logs_immagini/` (utile per controllare cosa vede il modello; cancellala tra una run e l'altra).

---

## 5. Lanciare l'esperimento

Dall'ambiente agente, nella cartella del repository:

```bash
conda activate llm-agent
python experiments/llm_only/orchestrator.py \
    --env Builds/MioGioco.exe \
    --llm-config experiments/llm_only/configs/mio_gioco_llm.yaml \
    --training-config experiments/llm_only/configs/mio_gioco_training.yaml \
    --server-python <percorso_python_di_llm-server> \
    --total-steps 2000 \
    --run-id prova_qwen
```

(Su Windows/PowerShell scrivi il comando su una riga sola o usa `` ` `` al posto di `\`.)

Cosa succede:
1. `[1] Avvio Server ZMQ su porta 65201...` → il server carica lo YAML LLM.
2. `[2] Server pronto. Avvio LLM Agent...` → si apre il gioco e inizia la run.
3. La prima decisione è lenta (Ollama carica il modello in memoria), poi va a regime.
4. Ogni `--summary-freq` step: `[INFO] - 1000 steps, ...s`.
5. A fine run o con **Ctrl+C** vengono chiusi gioco e server.

Opzioni utili:

| Opzione | Effetto |
|---|---|
| `--total-steps N` | numero di decisioni totali (default 10000) |
| `--seed N` | seed della run; cambia anche le porte, quindi run con seed diversi possono girare in parallelo |
| `--graphics` | mostra la finestra del gioco anche senza osservazioni visuali (per guardare l'LLM giocare) |
| `--time-scale 1` | velocità reale, utile insieme a `--graphics` |
| `--timeout-ms 300000` | più tempo per risposta (modelli grandi / PC lenti) |

---

## 6. Risultati

- `results/<run_id>/`: statistiche TensorBoard inviate dal gioco (`Academy.StatsRecorder`) e `LLM/missing_responses` (risposte perse per timeout).
  ```bash
  tensorboard --logdir results
  ```
- Log del server nel terminale: prompt inviati, risposte non valide, statistiche cache (`HITS/MISSES`: stati già visti non richiamano l'LLM).

---

## 7. Problemi frequenti

| Sintomo | Causa / soluzione |
|---|---|
| `il server ZMQ non e' partito` | errore nello YAML LLM o nell'ambiente server: leggi il traceback sopra; controlla `--server-python` |
| `Connection refused` / `ConnectError` nel server | Ollama non è attivo: aprilo o lancia `ollama serve` |
| `model "..." not found` | `model_name` diverso da `ollama list` → `ollama pull <modello>` |
| Tante `[WARN] Nessuna risposta LLM` | modello troppo lento per il timeout: `--timeout-ms` più alto o modello più piccolo |
| L'agente fa sempre la stessa azione | il modello non rispetta il formato → si usa `default_agent_action`. Controlla nel log del server le risposte; rendi più chiaro il formato in `agent_role`, abbassa `temperature` |
| `KeyError` / indici fuori range nel server | `observation_types` (YAML training), `features.*.indices` o `num_rays`/`num_detectable_tags` non corrispondono alla build: riesegui `--inspect` |
| Azioni "sbagliate" (gira a destra invece che a sinistra) | ordine di `actions` o delle `options` diverso da `OnActionReceived` |
| Immagini nere in `logs_immagini/` | gioco avviato con `--no-graphics`: toglilo |
| `Couldn't connect to the Unity environment` | build sbagliata per il sistema, oppure porta occupata da una run precedente: cambia `--seed` o chiudi i processi rimasti |

Riferimento completo sul framework (formato YAML, cache, porte, modalità DRL+LLM): [`README.md`](../../README.md) nella radice.
