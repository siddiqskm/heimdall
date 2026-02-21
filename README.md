# heimdall

<div align="center">
  <img src="assets/heimdall-logo.png" alt="Heimdall" width="360" />
</div>

**Stateful conversation gatekeeper for LLM systems.**

Heimdall classifies user intent and decides whether to allow, suppress, or reset context — preventing wasted tokens, handling hostility, and stabilizing intent across conversations.

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

## Integration

Use heimdall as a dependency in your own project (Python 3.12+).

### From PyPI (when published)

```bash
pip install heimdall
# or with Poetry
poetry add heimdall
```

Then import from the top level and follow the [Quick start](#quick-start) pattern: create a `HeimdallConfig`, `Embedder`, `Classifier`, and `LabelDwell` (per chat), run `embed → predict → dwell.apply → decide → route` per message.

### From Git

To depend on the repo directly (e.g. a specific branch or before a PyPI release), add heimdall as a Git dependency.

**Poetry** (`pyproject.toml`):

```toml
[tool.poetry.dependencies]
heimdall = { git = "https://github.com/siddiqskm/heimdall.git" }
# or SSH, e.g.:
# heimdall = { git = "git@github.com:siddiqskm/heimdall.git" }
```

Then run `poetry install`. Use the same imports and API as with the PyPI package.

**pip** (from a clone or editable install):

```bash
pip install -e /path/to/heimdall
# or from repo URL (no editable):
pip install git+https://github.com/siddiqskm/heimdall.git
```

---

## Quick start

### As a library

Import only from the top level — do not use `heimdall.core.*` or other internal modules.

State and learning are **per chat**: one `Classifier` and one `LabelDwell` per chat. Omit `chat_id` for a new chat; pass `chat_id` to resume an existing one.

```python
from pathlib import Path
from heimdall import (
    Classifier,
    Embedder,
    HeimdallConfig,
    LabelDwell,
    decide,
    route,
    configure_logging,
)

# Optional: enable log output (default is no output)
configure_logging()

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
poetry run python examples/playground.py
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
from heimdall import HeimdallConfig, default_config

# Default: ~/.heimdall
config = default_config()

# Or explicit (e.g. playground uses .playground_state)
config = HeimdallConfig(state_dir=Path(".playground_state"))
```

Under `state_dir` the package keeps **one directory per chat**: `state_dir/chats/{chat_id}/` with:

- `delta.json` – bias vector for the LR classifier
- `prototypes.json` – user prototypes for this chat
- `dwell.json` – FSM state (when using LabelDwell with config)

These files are written automatically after each `Classifier.predict()` and `LabelDwell.apply()`, so the chat directory is populated as soon as you run the pipeline (no need to call `persist()` yourself in host applications).

Call `config.delete_chat_state(chat_id)` when a chat is closed so state is not kept forever (caller or a job is responsible).

Key options (see `HeimdallConfig` in `heimdall/core/config.py`):

- **Decision**: `confidence_threshold` (default `0.38`) – below this, SILENT is upgraded to REQUEST.
- **Dwell**: `hostile_cooldown`, `intent_decay_silent_streak`, `hostile_recovery_threshold`.
- **Score thresholds**: `hostile_threshold`, `reset_threshold`, `utility_silent_threshold`.
- **Prototypes**: `session_proto_threshold`, `user_proto_threshold`, `offline_proto_threshold`, and add thresholds for session/user.

---

## Logging

Heimdall uses the `"heimdall"` logger and does not add real handlers by default (only a `NullHandler` to avoid "no handler" warnings). Choose one of two patterns depending on how you run heimdall.

### Host apps (recommended when heimdall is embedded)

When heimdall runs inside another application that configures logging (e.g. a web server or API), **do not call `configure_logging()`**. That function attaches heimdall-owned handlers and sets `propagate=False`, so heimdall logs never reach your app’s root logger or file handler and a separate heimdall log file will not appear where you expect.

**Use `set_log_level()` only.** Then heimdall logs propagate to the root logger and go wherever your app routes them (e.g. a single app log file).

```python
import logging
from heimdall import set_log_level

# In your app startup:
set_log_level(logging.INFO)   # or logging.DEBUG
```

Your app configures handlers (file, stream, etc.) on the root logger or on `"heimdall"` as usual; heimdall only needs its level set so that records are emitted and propagated.

### Standalone scripts

When heimdall is the main process (e.g. playground, a small script), you can call `configure_logging()` so heimdall adds its own console and/or file handlers:

```python
from heimdall import configure_logging

# Console only (default)
configure_logging()

# Default log file: ~/.heimdall/heimdall.log
configure_logging(log_file=True)

# Custom path
configure_logging(log_file=True, log_dir="/var/log/heimdall")
```

Options: **`log_file`** – `True` for default path, or a path (str/Path). **`log_dir`** – when `log_file=True`, directory (default `~/.heimdall`). **`use_console`** – add stderr handler (default `True`). **`handler`** – single custom handler. **`also_configure`** – list of logger names to attach the same handlers to.

See [docs/LOGGING.md](docs/LOGGING.md) for more detail on host-app vs standalone integration.

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

After training, the package will use the new `lr.joblib` when `models_dir` points at that directory (or when using the default in-package `heimdall/models/`). Restart any running process (e.g. `examples/playground.py`) so it loads the updated model.

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
├── examples/                # gatekeeper_bot.py, playground.py
├── docs/                    # DATA_AND_TRAINING.md
├── tests/
├── training/                # train_bootstrap.py
├── benchmarks/
├── LICENSE                  # MIT
├── pyproject.toml
└── Makefile
```

---

## Publishing (GitHub & PyPI)

1. **Heimdall repo:** [github.com/siddiqskm/heimdall](https://github.com/siddiqskm/heimdall) (or create and push this project).
2. **Build and publish** (optional, for PyPI):
   ```bash
   poetry build
   # then: poetry publish (or upload to PyPI with twine)
   ```

---

## Author

Siddiq Hussain
