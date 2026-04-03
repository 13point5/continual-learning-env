# OpenCode Continual Learning Notes

## Goal

Resume an exported OpenCode session inside Verifiers, but treat the last user message as the live turn for the current rollout.

## How This Differs From Base `OpenCodeEnv`

Base `OpenCodeEnv` in `verifiers/envs/experimental/opencode_env.py` does three simple things:

1. Uploads one prompt string.
2. Writes an OpenCode config that derives provider/model from `OPENAI_MODEL`.
3. Runs plain `opencode run`.

Our continual-learning env changes that flow:

1. Convert the exported OpenCode session into Verifiers `prompt` messages so the saved rollout contains prior context.
2. Slice the imported session to the prefix before the last user turn.
3. Use the last user turn as the live prompt for the current rollout.
4. Upload `session.json` and resume with `opencode import` + `opencode run --session ...`.

## Why The OpenCode Config Had To Change

The base config builder assumes `OPENAI_MODEL` looks like `provider/model` and splits it like this:

- provider id: `${OPENAI_MODEL%%/*}`
- model id: `${OPENAI_MODEL##*/}`

That works for models like `openai/gpt-4o-mini`, but it breaks for Prime Inference style ids like `minimax/minimax-m2.5`.

For `minimax/minimax-m2.5`, the base behavior effectively turns the upstream request into:

- provider: `minimax`
- model: `minimax-m2.5`

That caused two problems during resume:

1. Imported sessions could keep using the old session model like `big-pickle`.
2. Prime Inference rejected the leaf model name without the full id.

The fix was:

1. Use a fixed OpenCode provider alias: `eval`.
2. Preserve the full upstream model id as the model key.
3. Pass `--model eval/$OPENAI_MODEL` explicitly on every `opencode run`.
4. Pass `--dir /app` explicitly on every `opencode run`.

So OpenCode now sees:

- provider: `eval`
- model id: `minimax/minimax-m2.5`

instead of trying to reinterpret the upstream model shape itself.

## Why The Session Transform Matters

The exported dataset rows are completed sessions. There is no separate "next prompt" field.

To make that work with `OpenCodeEnv` semantics:

1. Find the last user message in the exported session.
2. Import only the prefix before that user message.
3. Use that last user message as `resume_prompt`.
4. Build Verifiers `prompt` from the prefix plus that last user turn so the saved rollout has the full context visible.

This matches how base `OpenCodeEnv` expects to operate: a single live prompt is uploaded, and any history must come from the imported session.

## What The A/B Eval Proved

We ran two important Minimax evals after the model/config fix.

### Without path rewriting

Run: `44cc2039`

Observed behavior:

- OpenCode resumed with the correct eval model.
- The imported session still referenced the original host path:
  - `/Users/13point5/projects/rollouts/...`
- The agent hit:
  - `permission requested: external_directory`
  - auto-reject
  - failed turn

Conclusion:

- Resume/model forcing worked.
- Imported session paths still leaked host-specific absolute paths.

### With path rewriting restored

Run: `9864d997`

Observed behavior:

- No references to `/Users/13point5/projects/rollouts/...`
- The agent now looked for files under `/app/src/rollouts/...`
- The run made real progress:
  - `num_turns = 10`
  - `total_tool_calls = 9`

Conclusion:

- Path rewriting is needed.
- It successfully moved the agent off the old host path.

## What The Current Blocker Is

The current blocker is no longer model selection or imported host paths.

The current failure is that `/app` inside the sandbox is empty, so the agent looks in the right logical place but the repo is not actually present there.

Symptoms from run `9864d997`:

- `File not found: /app/src/rollouts/storage/db.py`
- `No such file or directory: '/app/src/rollouts'`
- `ls -la /app/` shows an empty directory

## Current Working Mental Model

The continual-learning env now correctly does the following:

1. Load raw exported JSONL rows.
2. Convert each row into Verifiers prompt/history.
3. Import only the prefix session.
4. Resume on the last user turn.
5. Force OpenCode onto the eval model instead of the imported model.
6. Rewrite old absolute session paths to the sandbox workdir.

So the next fix should focus on sandbox workspace population, not session/model wiring.
