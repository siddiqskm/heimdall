# Data and training

## Where labels come from

The default classifier is trained on **bootstrap data** defined in `training/train_bootstrap.py`: a curated list of `(text, label)` pairs for the four labels (SILENT, REQUEST, TOPIC_RESET, HOSTILE). Examples include:

- **SILENT**: fillers and acknowledgements (`"ok"`, `"cool"`, `"hey"`, `"how are you"`).
- **REQUEST**: help-seeking, topic continuation, elaboration (`"need help"`, `"tell me more"`, `"that's interesting"`).
- **TOPIC_RESET**: explicit topic change (`"change topic"`, `"lets talk about something else"`).
- **HOSTILE**: abusive or hostile phrases (in the script’s `DATA` list).

The embedder is **sentence-transformers** `all-MiniLM-L6-v2`. The same embedder is used at training time and at runtime.

## Adding your own training data

1. **Extend the bootstrap list**  
   Edit `training/train_bootstrap.py` and add more `(text, label)` tuples to the `DATA` list. Keep the same four labels. Then re-run:

   ```bash
   poetry run python training/train_bootstrap.py --models-dir ./heimdall/models
   ```

2. **Use your own dataset**  
   Replace or wrap the `DATA` in `train_bootstrap.py` with your own list of `(text, Label)` pairs (e.g. from a CSV or JSON). The training script expects:
   - Normalised text (lowercased, trimmed).
   - Labels from `heimdall.core.types`: `SILENT`, `REQUEST`, `TOPIC_RESET`, `HOSTILE`.

3. **Offline prototypes**  
   The script also writes `prototypes_offline.json` from the same data. You can edit that file or regenerate it to add/change high-confidence prototype phrases per label.

## Benchmarks and gold data

The `benchmarks/` directory uses separate **gold** and **adversarial** datasets (see `benchmarks/data/`) for evaluation. Those are built by the `build_*.sh` scripts and are independent of the bootstrap training data. You can define your own gold sets in the same format and run the evaluation scripts against them.
