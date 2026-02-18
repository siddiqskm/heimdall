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

A stateful **dwell** layer keeps intent stable (e.g. “cool” in the middle of a request stays REQUEST), handles hostility (cooldown, recovery), and decays intent on silent streaks. Optional **learning** updates per-user bias and prototypes from outcomes.

---

## Pipeline

```
text → Embedder → Classifier (LR + prototypes + scores) → LabelDwell (FSM) → decide (confidence gate) → route → SystemAction
```

- **Embedder**: sentence-transformers `all-MiniLM-L6-v2`, normalised embeddings.
- **Classifier**: LR model + per-user bias (with decay), session/user/offline prototype stores, score engine (hostile / reset / utility) that can override the LR label.
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
clf = Classifier(config=config)
dwell = LabelDwell(config=config)

user_id = "your_user_id"
text = "lets build an auth system"
vec = embedder.encode(text)
predicted, confidence, activation = clf.predict(vec, user_id)
dwell_label = dwell.apply(user_id, predicted, activation)
final_label = decide(dwell_label, confidence, confidence_threshold=config.confidence_threshold)
action = route(final_label)
# action is one of: NO_RESPONSE, ALLOW_PROGRESS, RESET_CONTEXT, SUPPRESS
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

Only `state_dir` is required; everything else has defaults.

```python
from pathlib import Path
from heimdall.core.config import HeimdallConfig, default_config

# Default: ~/.heimdall
config = default_config()

# Or explicit
config = HeimdallConfig(state_dir=Path("/path/to/state"))
```

Under `state_dir` the package uses:

- `user_delta.json` – per-user bias for the LR classifier
- `prototypes_user.json` – user-level prototypes

Key options (see `HeimdallConfig` in `heimdall/core/config.py`):

- **Decision**: `confidence_threshold` (default `0.38`) – below this, SILENT is upgraded to REQUEST.
- **Dwell**: `hostile_cooldown`, `intent_decay_silent_streak`, `hostile_recovery_threshold`.
- **Score thresholds**: `hostile_threshold`, `reset_threshold`, `utility_silent_threshold`.
- **Prototypes**: `session_proto_threshold`, `user_proto_threshold`, `offline_proto_threshold`, and add thresholds for session/user.

---

## Logging

Heimdall follows the usual Python library pattern:

- **No handlers by default** – the package attaches a `NullHandler` to the `"heimdall"` logger so nothing is emitted until the app configures logging.
- **Your app** (or a script like the playground) should enable output by calling:

  ```python
  import heimdall
  heimdall.configure_logging()  # stderr, INFO
  # Or with level / format / file:
  heimdall.configure_logging(level=logging.DEBUG)
  heimdall.configure_logging(handler=logging.FileHandler("heimdall.log"))
  ```

- **No log file is created by default**; logs go to stderr unless you pass a `FileHandler`.

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

Bootstrap training produces the LR model and offline prototypes used by the classifier:

```bash
poetry run python training/train_bootstrap.py --models-dir ./models --state-dir ./state
```

- **Labels** come from the built-in `DATA` list in `training/train_bootstrap.py` (curated `(text, label)` pairs for SILENT, REQUEST, TOPIC_RESET, HOSTILE).
- Uses the same embedder as runtime; trains sklearn `LogisticRegression` and writes `lr.joblib` and state under the given dirs.
- To add your own data: extend the `DATA` list in that script (or load your own list of `(text, Label)` tuples) and re-run. See [docs/DATA_AND_TRAINING.md](docs/DATA_AND_TRAINING.md) for more detail.

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
│   ├── adapt/               # outcome inference, learner, learning gate
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
