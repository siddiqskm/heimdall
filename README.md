# heimdall

**Conversation gatekeeper with dwell-based routing.** Classifies user turns and maps them to system actions (allow progress, suppress, reset context, or no response).

- **Python 3.12+**
- **Poetry** for install and dev

---

## What it does

Heimdall labels each user message as one of four **labels**:

| Label         | Meaning                          |
|---------------|----------------------------------|
| `SILENT`      | Filler, acknowledgements, no intent |
| `REQUEST`     | User wants to continue or make progress |
| `TOPIC_RESET` | User wants to change topic       |
| `HOSTILE`     | Hostile or abusive               |

Those labels are turned into **system actions**:

| Label         | System action   |
|---------------|-----------------|
| SILENT        | `NO_RESPONSE`   |
| REQUEST       | `ALLOW_PROGRESS`|
| TOPIC_RESET   | `RESET_CONTEXT` |
| HOSTILE       | `SUPPRESS`      |

A stateful **dwell** layer keeps intent stable (e.g. “cool” in the middle of a request stays REQUEST), handles hostility (cooldown, recovery), and decays intent on silent streaks. Optional **learning** updates per-chat bias and prototypes from outcomes.

---

## Pipeline

```
text → Embedder → Classifier (LR + prototypes + scores) → LabelDwell (FSM) → decide (confidence gate) → route → SystemAction
```

- **Embedder**: sentence-transformers `all-MiniLM-L6-v2`, normalised embeddings.
- **Classifier**: LR model + per-chat bias (with decay), session/user/offline prototype stores, score engine (hostile / reset / utility) that can override the LR label.
- **LabelDwell**: FSM (IDLE, INTENT, HOSTILE, POST_RESET) for continuity and hostility handling.
- **decide**: Final confidence gate (e.g. low-confidence SILENT → REQUEST).
- **route**: Label → system action.

---

## Install

```bash
git clone <repo>
cd heimdall
poetry install
```

Requires **Python 3.12**. The package ships with a pre-trained LR model and offline prototypes in `heimdall/models/` and `heimdall/assets/`.

---

## Quick start

### As a library

State and learning are **per chat**: one `Classifier` and one `LabelDwell` per chat. Omit `chat_id` for a new chat; pass `chat_id` to resume an existing one.

```python
from pathlib import Path
import heimdall
from heimdall.core.config import HeimdallConfig
from heimdall.core.classifier import Classifier
from heimdall.core.decision import decide
from heimdall.core.dwell import LabelDwell
from heimdall.core.embedder import Embedder
from heimdall.core.router import route

# Optional: enable log output (default is no output)
heimdall.configure_logging()

config = HeimdallConfig(state_dir=Path("~/.heimdall").expanduser())
embedder = Embedder()
clf = Classifier(config=config)                    # new chat (generates chat_id)
dwell = LabelDwell(config=config, chat_id=clf.chat_id)

text = "lets build an auth system"
vec = embedder.encode(text)
pred = clf.predict(vec, text=text)
dwell_label = dwell.apply(pred.label, pred.activation)
final_label = decide(dwell_label, pred.confidence, confidence_threshold=config.confidence_threshold)
action = route(final_label)
# action is one of: NO_RESPONSE, ALLOW_PROGRESS, RESET_CONTEXT, SUPPRESS

# To resume a chat later: clf = Classifier(config=config, chat_id=existing_chat_id)
# To delete chat state when chat is closed: config.delete_chat_state(chat_id)
```

### Playground script

Interactive loop using the same pipeline plus optional learning:

```bash
poetry run python playground.py
```

Logging is enabled when the script is run; it uses `heimdall.configure_logging()` so you see classification and actions.

### Integration example

A minimal example that wires heimdall as a **gate in front of an assistant** (stub LLM/API): branch on system action and only call your backend when the action is `ALLOW_PROGRESS`. See [examples/README.md](examples/README.md) and run:

```bash
poetry run python examples/gatekeeper_bot.py
```

---

## Configuration

Only `state_dir` is required; everything else has defaults. Per-chat state is stored under `state_dir/chats/{chat_id}/`.

**Default:** `default_config()` uses `~/.heimdall`. You can pass any directory to override (e.g. `.playground_state` for the playground, or a path under your app).

```python
from pathlib import Path
from heimdall.core.config import HeimdallConfig, default_config

# Default: ~/.heimdall
config = default_config()

# Or explicit (e.g. playground uses .playground_state)
config = HeimdallConfig(state_dir=Path(".playground_state"))
```

Under `state_dir` the package keeps **one directory per chat**: `state_dir/chats/{chat_id}/` with:

- `delta.json` – bias vector for the LR classifier
- `prototypes.json` – user prototypes for this chat
- `dwell.json` – FSM state (when using LabelDwell with config)

Call `config.delete_chat_state(chat_id)` when a chat is closed so state is not kept forever (caller or a job is responsible).

Key options (see `HeimdallConfig` in `heimdall/core/config.py`):

- **Decision**: `confidence_threshold` (default `0.38`) – below this, SILENT is upgraded to REQUEST.
- **Dwell**: `hostile_cooldown`, `intent_decay_silent_streak`, `hostile_recovery_threshold`.
- **Score thresholds**: `hostile_threshold`, `reset_threshold`, `utility_silent_threshold`.
- **Prototypes**: `session_proto_threshold`, `user_proto_threshold`, `offline_proto_threshold`, and add thresholds for session/user.

---

## Logging

Heimdall follows the usual Python library pattern:

- **No handlers by default** – the package attaches a `NullHandler` to the `"heimdall"` logger so nothing is emitted until the app configures logging.
- **Your app** (or a script like the playground) should enable output by calling `heimdall.configure_logging()` once.

Because the package often runs in the background (e.g. behind a chat API), **writing to a log file** is supported and recommended for production:

```python
import heimdall

# Console only (default)
heimdall.configure_logging()

# Default log file: ~/.heimdall/heimdall.log (creates dir if needed)
heimdall.configure_logging(log_file=True)

# Log file in a different directory (e.g. custom state_dir)
heimdall.configure_logging(log_file=True, log_dir="/var/log/heimdall")

# Explicit log file path; creates parent dirs
heimdall.configure_logging(log_file="/var/log/heimdall/app.log", use_console=False)

# Both console and file
heimdall.configure_logging(log_file=True)
```

Options:

- **`log_file`** – If `True`, a file handler is added at `(log_dir or ~/.heimdall)/heimdall.log`. If a path (str or Path), that path is used. If `None` or `False`, no file.
- **`log_dir`** – When `log_file=True`, directory for the log file (default `~/.heimdall`). Ignored when `log_file` is an explicit path.
- **`use_console`** – If `True` (default) and no custom `handler` is passed, a `StreamHandler` (stderr) is added. Set to `False` when you only want file output.
- **`handler`** – Optional single handler to use instead of the default stream (and/or file when `log_file` is set).
- **`also_configure`** – Optional list of logger names (e.g. `["tests"]`) to attach the same handlers to, so those loggers emit to the same place.

---

## Tests

```bash
# All tests
make test
# or
poetry run pytest -q

# Stop at first failure
make test-strict
poetry run pytest --maxfail=1

# Lint + test
make check        # lint + test
make check-strict # lint + test-strict

# Show heimdall logs while testing (e.g. dwell debug, classifier drift)
poetry run pytest --log-heimdall=DEBUG
```

Lint/format:

```bash
make lint      # ruff check
make lint-fix  # ruff check --fix
make format    # ruff format
```

---

## Training and data

Bootstrap training produces the LR model and offline prototypes used by the classifier.

### Running the trainer

From the repo root:

```bash
poetry run python training/train_bootstrap.py --models-dir ./heimdall/models
```

**Arguments:**

| Argument        | Default            | Description |
|----------------|--------------------|-------------|
| `--models-dir` | `heimdall/models`  | Directory where the trained LR model is written. |

**What it does:**

1. Loads the same embedder as runtime (`sentence-transformers/all-MiniLM-L6-v2`).
2. Encodes all `(text, label)` pairs from the built-in **DATA** list in `training/train_bootstrap.py`.
3. Trains a sklearn **LogisticRegression** classifier on the embeddings.
4. Writes **`lr.joblib`** to `{models-dir}/lr.joblib` (e.g. `./heimdall/models/lr.joblib`).
5. Builds **offline prototypes** from the same DATA and writes **`prototypes_offline.json`** to the package `heimdall/assets/` directory (used at runtime for prototype-based scoring).

After training, the package will use the new `lr.joblib` when `models_dir` points at that directory (or when using the default in-package `heimdall/models/`). Restart any running process (e.g. playground) so it loads the updated model.

### Training data

- **Labels** come from the built-in **DATA** list in `training/train_bootstrap.py`: curated `(text, Label)` pairs for SILENT, REQUEST, TOPIC_RESET, and HOSTILE.
- To add or fix behaviour: extend or edit the **DATA** list in that script (or load your own list of `(text, Label)` tuples), then re-run the command above.
- More detail: [docs/DATA_AND_TRAINING.md](docs/DATA_AND_TRAINING.md).

---

## Benchmarks

Scripts under `benchmarks/` evaluate on stateless gold, stateful gold, and adversarial datasets. Data lives in `benchmarks/data/`. From the repo root:

```bash
./benchmarks/build_all.sh    # build datasets
./benchmarks/evaluate_all.sh # run evaluations
```

---

## Project layout

```
heimdall/
├── heimdall/
│   ├── __init__.py          # NullHandler + configure_logging()
│   ├── core/
│   │   ├── classifier.py    # LR + prototypes + score engine
│   │   ├── config.py        # HeimdallConfig
│   │   ├── decision.py     # confidence gate
│   │   ├── dwell.py        # LabelDwell FSM
│   │   ├── embedder.py     # sentence-transformers
│   │   ├── router.py       # label → system action
│   │   ├── score_engine.py # hostile / reset / utility scores
│   │   ├── types.py        # Label, SystemAction, etc.
│   │   └── ...
│   ├── adapt/               # outcome inference, learning gate, config (DECAY, MAX_BIAS)
│   ├── assets/              # prototypes_offline.json
│   └── models/              # lr.joblib
├── examples/                # gatekeeper_bot.py integration example
├── docs/                    # DATA_AND_TRAINING.md
├── tests/
├── training/                # train_bootstrap.py
├── benchmarks/
├── playground.py            # interactive demo
├── LICENSE                  # MIT
├── pyproject.toml
└── Makefile
```

---

## Author

Siddiq Hussain
