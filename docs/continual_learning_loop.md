## First time:

1. `rollouts learn`
2. This will run `rollouts hf push --agent opencode --name my-opencode-sessions` and get the new batch id
3. Write a new `config.toml` file for RL run with the new batch id
4. Start an RL run with the new batch id: `prime rl run config-batch-<id>.toml`
5. Start cron job
6. Every X minutes, check status of the RL run: `prime rl get <run-id>`

If the run is in progress, do nothing.

Else:

1. Get deployment id for the run using a command/function in the rollouts CLI
2. Deploy the new model using `prime deployments create <model-id> --yes`
3. Update opencode config to use this new model

so now im using rft-1 in opencode

to train rft-2 i need to start a new run from the rft-1 checkpoint with new data.
If I collected more data during the training of rft-1, I can start a new run with the new data.
However if I did not collect more data, I need to wait for the next batch of data and then do `rollouts learn` again which should know the last checkpoint and start
this whole process again.
