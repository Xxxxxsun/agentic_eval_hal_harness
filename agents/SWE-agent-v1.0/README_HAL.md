We made a few changes to the SWE-agent-v1.0 to make it work with HAL.

1. Added `main.py` as the main entry to the agent.
2. Some model mapping is not supported by litellm. Including `openrouter/anthropic/claude-opus-4.1`. When running these models, we need to set `LITELLM_LOCAL_MODEL_COST_MAP=True`, and add the following code to `model_prices_and_context_window_backup.json` install locally in litellm:
```json
"openrouter/anthropic/claude-opus-4": {
        "input_cost_per_token": 0.000015,
        "output_cost_per_token": 0.000075
    },
    "openrouter/anthropic/claude-opus-4-20250514": {
        "input_cost_per_token": 0.000015,
        "output_cost_per_token": 0.000075
    },
    "openrouter/anthropic/claude-opus-4.1": {
        "input_cost_per_token": 0.000015,
        "output_cost_per_token": 0.000075
    },
    "openrouter/anthropic/claude-opus-4.1-20250805": {
        "input_cost_per_token": 0.000015,
        "output_cost_per_token": 0.000075
    },
    "openrouter/anthropic/claude-sonnet-4": {
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015
    }
```
