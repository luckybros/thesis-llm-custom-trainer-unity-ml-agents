## 1) Obiettivo del framework

Questo framework collega **Unity ML-Agents** a un **LLM esterno** tramite trainer custom (`llm_trainer`), così l’LLM può contribuire alle decisioni durante l’addestramento.

### Flusso ad alto livello
1. Unity genera osservazioni e riceve azioni.
2. Il trainer custom (Python) processa osservazioni/reward.
3. Ogni `llm_refresh_interval` step, il trainer interroga il server LLM.
4. Il contributo LLM viene pesato da `alpha` (`alpha_initial` → `alpha_final`).

---

## 2) Prerequisiti

- Unity (versione compatibile col progetto)
- ML-Agents (lato Unity + lato Python)
- Ollama (sviluppo locale)
- Conda (consigliato Miniconda)

---

## 3) Setup ambienti Conda (2 ambienti separati)

## 3.1 Ambiente 1: ML-Agents + trainer custom/plugin

Deve essere lo stesso ambiente conda in cui si è installato ML-Agents.

```bash
cd thesis-llm-custom-trainer-unity-ml-agents
pip install -e .
```
Questo installa il trainer custom dalla repository corrente.

## 3.2 Ambiente 2: server LLM (LangChain, ecc.)

```bash
conda create -n llmtrainer-server -y
conda activate llmtrainer-server
pip install -r requirements-server.txt
```

---

## 4) Configurazione lato ML-Agents (trainer YAML)

**IMPORTANTE**: settare nel trainer YAML di ML-Agents nella sezione *hyperparameters*.
Nel file trainer (es. `config/tanks_llm.yaml`) imposta:

- `trainer_type: llm_trainer` (**obbligatorio**)
- `alpha_initial`, `alpha_final` (peso del contributo LLM)
- `alpha_rate_schedule`:
  - `linear`: alpha evolve da initial a final
  - `constant`: alpha resta sul valore iniziale
- `llm_refresh_interval`: ogni quanti step interrogare LLM
- `communicator: zmq` (attualmente implementato)
- `zmq_base_port` (opzionale): base port del canale ZMQ trainer↔server LLM. Se omesso viene risolta automaticamente a runtime — vedi [§6.1) Porte ZMQ dinamiche e run parallele](#61-porte-zmq-dinamiche-e-run-parallele).

### 4.1 Osservazioni inviate all’LLM (`observation_types`)

Tipi supportati:
- `VECTORIAL`
- `RAYCAST`
- `GRID`
- `VISUAL`

Aggiungere l'indice in cui il blocco di dati viene inviato. 

⚠️ **Regola fondamentale**: l’ordine deve essere identico a Unity.  
Il client legge i dati “a blocchi” in base agli indici: se ordine/indici non combaciano, il parsing sarà errato.

### 4.2 Esempio struttura trainer config

```yaml
behaviors:
  Tanks:
    trainer_type: llm_trainer
    hyperparameters:
      batch_size: 512
      buffer_size: 5120
      learning_rate: 0.0003

      # Peso del contributo LLM nel tempo
      alpha_initial: 0.05
      alpha_final: 0.4
      alpha_rate_schedule: linear   # linear | constant

      beta: 0.02
      epsilon: 0.2
      lambd: 0.95
      num_epoch: 3
      learning_rate_schedule: constant

      # Comunicazione con server LLM
      communicator: zmq
      llm_refresh_interval: 10

      # IMPORTANTISSIMO: ordine uguale a Unity
      observation_types:
        - name: RAYCAST_FRONT
          type: RAYCAST
          index: 0
        - name: RAYCAST_BACK
          type: RAYCAST
          index: 1
        - name: VECTORIAL
          type: VECTORIAL
          index: 2

    network_settings:
      normalize: false
      hidden_units: 512
      num_layers: 2

    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
```

---

## 5) Configurazione lato LLM (`llm_settings`)

Questo file descrive **come costruire il prompt**, **quali azioni può scegliere l’agente** e **come decodificare le osservazioni**.

Campi principali:
- `game_desc`: descrizione del gioco
- `agent_role`: istruzioni/ruolo dell’agente
- `model_name`: modello LLM da usare
- `actions`: definizione azioni discrete
- `obs_template`: template testo osservazioni
- `features`: mapping osservazioni (fondamentale)
- `default_agent_action`: format di default nel caso in cui l'inferenza non si risolva correttamente

> Nota: `history_length` e `task` possono essere presenti ma al momento sono meno rilevanti operativamente.

---

## 5.1 Azioni (`actions`) — ordine obbligatorio

⚠️ L’ordine delle azioni deve essere uguale a `OnActionReceived` in Unity C#.  
⚠️ Anche l’ordine delle `options` deve essere coerente con il branch discreto Unity.

### Esempio
```yaml
actions:
  discrete:
    - name: Move
      options: [Stay, Move forward, Move backward]
      description: >
        Move the character forward or backward.
    - name: Turn
      options: [Stay, Turn right, Turn left]
      description: >
        Turn the direction which the character is facing.
    - name: Shoot
      options: [Don't shoot, Shoot]
      description: >
        Shoot a projectile in the direction which the character is facing.
```

---

## 5.2 Template osservazioni (`obs_template`)

Esempio:
```yaml
obs_template: |
  The player is in the {pos_x} {pos_z} section of the map.
  It {can_shoot}, and its health is {health} while its enemy health is {enemy_health}.
  It has {power_up} equipped.
  Looking forward, it sees {RAYCAST_FRONT}.
  Looking backward, it sees {RAYCAST_BACK}.
```

---

## 5.3 Sezione `features` (la più importante)

Qui converti valori numerici raw in etichette testuali leggibili dall’LLM.

### Tipi supportati (`vectorial`)
- **BUCKET**: se il valore cade in `[min,max]`, assegna `label`
- **BOOL**: mappa 0/1 in etichetta
- **ONE_HOT**: mappa indice attivo in categoria

⚠️ Gli `indices` devono combaciare con Unity.

### Esempio `features.vectorial`
```yaml
features:
  vectorial:
    pos_x:
      type: "BUCKET"
      indices: [0]
      mapping:
        - label: "FAR_LEFT"
          min: -1.1
          max: -0.6
        - label: "CENTER_X"
          min: -0.19
          max: 0.2
        - label: "FAR_RIGHT"
          min: 0.61
          max: 1.1

    can_shoot:
      type: "BOOL"
      indices: [2]
      mapping:
        0: "CANT_SHOOT"
        1: "CAN_SHOOT"

    power_up:
      type: "ONE_HOT"
      indices: [5, 6, 7, 8, 9, 10, 11]
      mapping:
        0: "None"
        1: "Speed Up"
        2: "Damage Reduction"
        3: "Fast Shooting"
        4: "Healing"
        5: "Invincibility"
        6: "Damage Multiplier"
```

### Esempio `features.raycast`
```yaml
features:
  raycast:
    RAYCAST_FRONT:
      detectable_tags:
        0: "TANK"
        1: "POWER UP"
        2: "WALL"
      num_detectable_tags: 3
      direction: [FAR_LEFT, LEFT, SLIGHT_LEFT, FORWARD, SLIGHT_RIGHT, RIGHT, FAR_RIGHT]
      distance:
        - label: IMMEDIATE
          min: 0.0
          max: 0.2
        - label: CLOSE
          min: 0.2
          max: 0.5
        - label: MEDIUM
          min: 0.5
          max: 0.8
        - label: FAR
          min: 0.8
          max: 1.1
```

## 5.4 Sezione `default_agent_action`
Settare un'azione di default per l'agente in caso in cui l'inferenza non vada a buon fine
```yaml
default_agent_action: |
    Agent 0:
      Move
        Move forward
      Turn
        Stay
      Shoot
        Don't shoot
```

---

## 6) Avvio training 

Apri due terminali.

### Terminale A — server LLM

```bash
conda activate llmtrainer-server
# avvia ollama se necessario
# avvia server llm del progetto (comando reale repo)
python -m mlagents_plugin.communicators.server.zmq_communication_server --config <file_di_configurazione_llm> --visual
```
Aggiungere visual solo per la versione con osservazioni visive (*you dont say?*)

### Terminale B — ML-Agents training

```bash
conda activate llmtrainer-mlagents
mlagents-learn <file_di_configurazione> --run-id=<run_id> --env=<path_eseguibile>
```

Se non si aggiunge il path eseguibile, semplicemente startare su Unity premendo +*Play**.

---

## 6.1) Porte ZMQ dinamiche e run parallele

Il canale ZMQ tra il trainer (`ZMQCommunicatorClient`) e il server LLM (`ZMQCommunicatorServer`) **non usa più una porta fissa**. La porta effettiva è sempre:

```
porta = base_port + worker_id
```

risolta a runtime da [`mlagents_plugin/utils/zmq_port_utils.py`](mlagents_plugin/utils/zmq_port_utils.py), in questo ordine di priorità:

1. valore esplicito (hyperparametro `zmq_base_port` nello YAML del trainer / flag `--port`, `--base-port`, `--worker-id` del server)
2. offset derivato dal `--base-port` passato a `mlagents-learn` stesso (+60000)
3. default fisso `65432`

Il `worker_id` reale di ML-Agents (quello assegnato ad ogni ambiente quando si usa `--num-envs`) viene propagato automaticamente dal trainer al client — non serve configurarlo a mano lato client.

⚠️ **Limite importante del punto 2 (derivazione automatica)**: funziona **solo lato trainer**, perché il processo `mlagents-learn` legge il proprio `--base-port` dai suoi argomenti e può quindi calcolarsi da solo un `base_port` ZMQ diverso per ogni run. Il **server LLM è un processo completamente separato**, lanciato a mano in un altro terminale: non ha alcun modo di vedere quale `--base-port` è stato passato al `mlagents-learn` corrispondente, quindi **non può replicare automaticamente la stessa derivazione**. Se non gli dici nulla, il server resta sul default `65432`, che può non coincidere con quanto calcolato dal trainer — il risultato non è nemmeno un conflitto di porta, ma un semplice mismatch: il client bussa su una porta, il server ne ascolta un'altra, e non si parlano.

Per questo, se lanci più processi `mlagents-learn` indipendenti (un processo per terminale, **senza** `--num-envs` — lo scenario che si usa in pratica in questo progetto), il `worker_id` resta sempre `0` su tutti — non ti aiuta a distinguerli — e l'unico modo affidabile per evitare collisioni/mismatch è fissare esplicitamente lo **stesso** `base_port` ZMQ tra trainer e suo server (nel file di configurazione e nel comando del server), diverso da quello usato dalle altre run. Vedi sotto.

### Caso 1 — una sola run, un solo processo
Non serve fare nulla: i comandi della sezione 6) funzionano già con i default (porta `65432` su entrambi i lati).

### Caso 2 — più run di `mlagents-learn` in parallelo sulla stessa macchina (un processo per terminale, senza `--num-envs`)

Questo è lo scenario di utilizzo effettivo: ogni run è una coppia di processi indipendenti — un `mlagents-learn` in un terminale e il suo server LLM in un altro — e vuoi poterne lanciare più coppie insieme senza che si pestino i piedi.

**Perché non basta il default.** Con `worker_id` sempre a `0`, la formula `porta = base_port + worker_id` diventa `porta = base_port`. Se non tocchi nulla, ogni run calcola `65432` e tutte collidono. L'unica leva che hai è differenziare manualmente `base_port` run per run.

**Perché non basta nemmeno l'auto-derivazione dal `--base-port` di ML-Agents (punto 2 sopra).** Quella derivazione funziona solo dentro al processo `mlagents-learn`, che può leggere il proprio `--base-port` dai suoi argomenti. Il server è un processo a parte, lanciato in un altro terminale: non ha modo di sapere quale `--base-port` hai usato per il `mlagents-learn` corrispondente, quindi non può replicare lo stesso calcolo. Se ti affidi solo a questo, il trainer calcola una porta e il server ne ascolta un'altra: non è un conflitto, è un mismatch silenzioso (il client non riceve mai risposta).

**La soluzione: fissa lo stesso `base_port` in due posti — il file di configurazione del trainer e il comando del server.**

- **lato trainer**: non esiste un flag CLI di `mlagents-learn` per questo — è un hyperparametro del trainer plugin, quindi va scritto nel file YAML di quella run:
  ```yaml
  behaviors:
    Tanks:
      trainer_type: llm_trainer
      hyperparameters:
        communicator: zmq
        zmq_base_port: 65001   # <-- un valore diverso per ogni run
        ...
  ```

- **lato server**: come flag sulla riga di comando quando lo avvii — non serve nemmeno passare per `base_port`, puoi dare la porta esatta con `--port` (equivalente a `--base-port`, dato che qui `worker_id` è sempre `0`):
  ```bash
  python -m mlagents_plugin.communicators.server.zmq_communication_server --config <file_llm> --port 65001
  ```

Serve quindi **un file YAML dedicato per ogni run** (o una copia modificata ogni volta), perché il valore è scritto dentro al file di configurazione:

```bash
# Run A — file configRunA.yaml con zmq_base_port: 65001
python -m mlagents_plugin.communicators.server.zmq_communication_server --config <file_llm> --port 65001
mlagents-learn configRunA.yaml --run-id=runA --env=<eseguibile_A> --base-port 5005
```
```bash
# Run B — file configRunB.yaml con zmq_base_port: 66000
python -m mlagents_plugin.communicators.server.zmq_communication_server --config <file_llm> --port 66000
mlagents-learn configRunB.yaml --run-id=runB --env=<eseguibile_B> --base-port 6005
```

Da tenere a mente:

- Il valore in `zmq_base_port` (YAML) e quello in `--port`/`--base-port` (server) devono essere **identici** dentro la stessa run; **diversi** tra run diverse.
- `--base-port` di `mlagents-learn` (canale Unity↔Python, meccanismo standard ML-Agents) è un numero indipendente da `zmq_base_port` (canale Python↔server LLM): vanno comunque differenziati entrambi run per run, ma non serve che coincidano tra loro.
- `worker_id` non va toccato in questo scenario: resta sempre `0` di default, sia lato trainer (automatico) sia lato server (non serve passare `--worker-id`).

---

## 7) Checklist anti-errori

- `trainer_type` è `llm_trainer`
- `communicator` è `zmq`
- ordine `observation_types` uguale a Unity
- indici `features.vectorial.*.indices` corretti
- ordine `actions` uguale a `OnActionReceived`
- ordine `options` coerente con Unity
- `detectable_tags` raycast nello stesso ordine di Unity
- modello Ollama disponibile localmente (`model_name` corretto)
- se si eseguono più run in parallelo: stesso valore in `zmq_base_port` (YAML) e `--port`/`--base-port` (server) per la stessa run, diverso da quello delle altre run — vedi [§6.1](#61-porte-zmq-dinamiche-e-run-parallele)

---