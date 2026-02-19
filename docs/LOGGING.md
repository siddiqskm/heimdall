# Logging integration

Heimdall uses the standard library `logging` module under the logger name `"heimdall"`. How you enable output depends on whether heimdall runs as a **standalone** process or inside a **host application**.

## Host applications (Cog, web apps, etc.)

When heimdall is used as a library inside an app that already configures logging, **do not call `configure_logging()`**. That function:

- Attaches its own `StreamHandler` (and optionally a file handler) to the `"heimdall"` logger
- Sets `propagate = False` on the heimdall logger

So heimdall logs never reach the root logger and will not appear in your app’s log file or wherever your app routes logs. You can end up with no heimdall output, or a separate heimdall log file in an unexpected place.

**Recommended:** set only the heimdall logger level and let propagation do the rest. Your app’s existing configuration (e.g. a file handler on the root logger) will then receive heimdall records.

```python
import logging
from heimdall import set_log_level

# Once at startup (e.g. in Cog predictor setup(), or app init):
set_log_level(logging.INFO)   # or logging.DEBUG
```

No handlers are added and `propagate` stays `True` (the default). Logs from `heimdall` and its child loggers (e.g. `heimdall.core.classifier`) propagate to the root logger and go to whatever handlers your app attached there (file, stream, etc.).

## Standalone scripts

When heimdall is the main script (playground, one-off script), it is fine to call `configure_logging()` so heimdall adds its own handlers and controls its own output:

```python
from heimdall import configure_logging

configure_logging()                          # console (stderr)
configure_logging(log_file=True)             # also ~/.heimdall/heimdall.log
configure_logging(log_file=True, log_dir="/var/log/heimdall")
```

See the README “Logging” section for full option list.
