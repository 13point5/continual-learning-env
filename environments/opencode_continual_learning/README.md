# opencode-continual-learning

### Overview

- **Environment ID**: `opencode-continual-learning`
- **Short description**: Continual learning environment that resumes OpenCode agent sessions from prior rollouts and evaluates whether the agent successfully completes the task.
- **Tags**: continual-learning, opencode, agent, multi-turn

### Task

- **Type**: multi-turn, agent-based
- **Output format expectations**: Plain text (agent completion log)
- **Rubric overview**: Deterministic reward: 1.0 if the agent completes successfully; 0.0 otherwise.

### Quickstart

Run an evaluation with default settings:

```bash
prime eval run opencode-continual-learning
```

Notes:

- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments

| Arg       | Type | Default                             | Description                                                      |
| --------- | ---- | ----------------------------------- | ---------------------------------------------------------------- |
| `dataset` | str  | `"13point5/opencode-rollouts-test"` | Hugging Face dataset repo ID containing rollouts (`train.jsonl`) |

### Metrics

| Metric            | Meaning                                                              |
| ----------------- | -------------------------------------------------------------------- |
| `reward`          | Deterministic reward: 1.0 if agent succeeded, else 0.0               |
| `agent_succeeded` | 1.0 if agent completed (exit code 0, no timeout, no error), else 0.0 |

### Dataset Format

The environment expects a Hugging Face dataset with `train.jsonl` containing rows with:

```json
{
  "session_id": "...",
  "agent": "...",
  "exported_at": "...",
  "metadata": { "remote_url": "..." },
  "session": {
    "messages": [
      {
        "info": { "role": "user" },
        "parts": [{ "type": "text", "text": "..." }]
      },
      {
        "info": { "role": "assistant" },
        "parts": [
          { "type": "reasoning" },
          { "type": "text" },
          { "type": "tool" }
        ]
      }
    ]
  }
}
```

Each row is transformed to extract the prompt (up to and including the last user message) and session state for resumption.
