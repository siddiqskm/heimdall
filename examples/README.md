# Examples

## gatekeeper_bot.py

Shows how to use heimdall as a **gate in front of your assistant** (LLM, API, or rule-based backend).

- For each user message, the script runs the full pipeline (embed → classify → dwell → decide → route).
- It branches on the **system action** and only calls the assistant when the action is `ALLOW_PROGRESS`.
- `NO_RESPONSE` → no reply (e.g. filler).
- `SUPPRESS` → generic de-escalation message.
- `RESET_CONTEXT` → clear conversation context and prompt for a new topic.
- `ALLOW_PROGRESS` → call your backend and return its reply.

Replace `call_assistant()` with your own LLM/API call and wire `gate_turn()` into your HTTP handler or message loop.

**Run from repo root:**

```bash
poetry run python examples/gatekeeper_bot.py
```
