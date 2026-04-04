import asyncio
import importlib
import json
import random
import tempfile
from pathlib import Path

from datasets import Dataset
from huggingface_hub import hf_hub_download
from prime_tunnel import TunnelConnectionError, TunnelTimeoutError
import verifiers as vf


def _ensure_sandbox_monitor_rubric() -> None:
    """Patch older/broken verifiers installs that omit SandboxMonitorRubric."""
    sandbox_mixin = importlib.import_module("verifiers.envs.experimental.sandbox_mixin")
    if hasattr(sandbox_mixin, "SandboxMonitorRubric"):
        return

    class SandboxMonitorRubric(vf.Rubric):
        async def sandbox_oom(self, state: vf.State) -> float:
            return float(bool(state.get("sandbox_oom")))

        async def sandbox_timeout(self, state: vf.State) -> float:
            return float(bool(state.get("sandbox_timeout")))

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.add_metric(self.sandbox_oom)
            self.add_metric(self.sandbox_timeout)

    sandbox_mixin.SandboxMonitorRubric = SandboxMonitorRubric


_ensure_sandbox_monitor_rubric()

from verifiers.envs.experimental.opencode_env import OpenCodeEnv

from utils import transform_row


DEFAULT_INSTALL_COMMAND = "curl -fsSL https://opencode.ai/install | bash -s -- --version v1.3.13"
OPENCODE_PROVIDER_ID = "eval"


DEFAULT_RUN_COMMAND_TEMPLATE = """\
set -e

apt-get update && apt-get install -y curl git

for install_attempt in 1 2 3; do
    if {install_command}; then
        break
    fi
    if [ "$install_attempt" -eq 3 ]; then
        echo "OpenCode installation failed after 3 attempts" >&2
        exit 1
    fi
    echo "OpenCode install attempt $install_attempt/3 failed, retrying in 5s..." >&2
    sleep 5
done

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$HOME/.opencode/bin:$PATH"

mkdir -p ~/.config/opencode

SCHEMA_DOLLAR='$'

cat > ~/.config/opencode/opencode.json << EOFCONFIG
{config_json}
EOFCONFIG

if [ -n "$ROLLOUTS_REMOTE_URL" ] && [ -n "$ROLLOUTS_MESSAGE_ID" ]; then
    uv tool install agent-rollouts
    rm -rf {workspace_path}
    rollouts restore --repo "$ROLLOUTS_REMOTE_URL" --session "$OPENCODE_SESSION_ID" --message "$ROLLOUTS_MESSAGE_ID" --dest {workspace_path}
else
    mkdir -p {workspace_path}
fi

cd {workspace_path}
if [ -s {session_path} ] && [ -n "$OPENCODE_SESSION_ID" ]; then
    opencode import {session_path}
    cat {prompt_path} | opencode run --session "$OPENCODE_SESSION_ID" --model {opencode_model} --dir {workspace_path} 2>&1 | tee {logs_path}
else
    cat {prompt_path} | opencode run --model {opencode_model} --dir {workspace_path} 2>&1 | tee {logs_path}
fi
"""


class ContinualLearningDummyRewardRubric(vf.Rubric):
    """Assign a simple dummy reward based on whether the agent run succeeded."""

    def __init__(self):
        super().__init__()
        self.add_reward_func(self.dummy_success_reward)
        self.add_metric(self.agent_succeeded)

    @staticmethod
    def _agent_succeeded(state: vf.State) -> bool:
        return (
            state.get("stop_condition") == "agent_completed"
            and not state.get("agent_timed_out", False)
            and state.get("agent_exit_code") == 0
            and state.get("error") is None
        )

    async def dummy_success_reward(self, state: vf.State) -> float:
        if not self._agent_succeeded(state):
            return 0.0
        return random.uniform(0.1, 1.0)

    async def agent_succeeded(self, state: vf.State) -> float:
        return float(self._agent_succeeded(state))


class ContinualLearningEnv(OpenCodeEnv):
    DEFAULT_INSTALL_COMMAND = DEFAULT_INSTALL_COMMAND
    DEFAULT_RUN_COMMAND_TEMPLATE = DEFAULT_RUN_COMMAND_TEMPLATE

    @property
    def remote_session_path(self) -> str:
        return f"{self.asset_dir}/session.json"

    @property
    def remote_workspace_path(self) -> str:
        return f"{self.agent_workdir}/repo"

    def __init__(
        self,
        *args,
        install_command: str | None = DEFAULT_INSTALL_COMMAND,
        run_command_template: str | None = DEFAULT_RUN_COMMAND_TEMPLATE,
        tunnel_startup_max_retries: int = 2,
        tunnel_retry_delay_seconds: float = 2.0,
        tunnel_retry_backoff: float = 2.0,
        **kwargs,
    ):
        self.tunnel_startup_max_retries = max(0, tunnel_startup_max_retries)
        self.tunnel_retry_delay_seconds = max(0.0, tunnel_retry_delay_seconds)
        self.tunnel_retry_backoff = max(1.0, tunnel_retry_backoff)
        super().__init__(
            *args,
            install_command=install_command or self.DEFAULT_INSTALL_COMMAND,
            run_command_template=run_command_template or self.DEFAULT_RUN_COMMAND_TEMPLATE,
            **kwargs,
        )
        self.add_rubric(ContinualLearningDummyRewardRubric())

    async def _retry_tunnel_startup(self, start_tunnel) -> str:
        attempts = self.tunnel_startup_max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                return await start_tunnel()
            except (TunnelConnectionError, TunnelTimeoutError) as exc:
                if attempt >= attempts:
                    raise

                delay_seconds = self.tunnel_retry_delay_seconds * (
                    self.tunnel_retry_backoff ** (attempt - 1)
                )
                self.logger.warning(
                    "Tunnel startup failed on attempt %s/%s with %s: %s. Retrying in %.1fs.",
                    attempt,
                    attempts,
                    type(exc).__name__,
                    exc,
                    delay_seconds,
                )
                await asyncio.sleep(delay_seconds)

        raise RuntimeError("unreachable")

    async def get_tunnel_url(self) -> str:
        return await self._retry_tunnel_startup(super().get_tunnel_url)

    def build_run_command(
        self,
        run_command_template: str,
        agent_workdir: str,
        disabled_tools: list[str] | None = None,
        system_prompt: str | None = None,
        install_command: str = DEFAULT_INSTALL_COMMAND,
        disable_compaction: bool = True,
        enable_interleaved: bool = True,
    ) -> str:
        config_json = self.build_opencode_config(
            disabled_tools,
            self.remote_system_prompt_path if system_prompt else None,
            disable_compaction=disable_compaction,
            enable_interleaved=enable_interleaved,
        )

        return run_command_template.format(
            config_json=config_json,
            agent_workdir=agent_workdir,
            prompt_path=self.remote_prompt_path,
            logs_path=self.remote_logs_path,
            install_command=install_command,
            session_path=self.remote_session_path,
            opencode_model=f"{OPENCODE_PROVIDER_ID}/$OPENAI_MODEL",
            workspace_path=self.remote_workspace_path,
        )

    def build_opencode_config(
        self,
        disabled_tools: list[str] | None = None,
        system_prompt_path: str | None = None,
        disable_compaction: bool = True,
        enable_interleaved: bool = True,
    ) -> str:
        config = {
            "${SCHEMA_DOLLAR}schema": "https://opencode.ai/config.json",
            "provider": {
                OPENCODE_PROVIDER_ID: {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": OPENCODE_PROVIDER_ID,
                    "options": {
                        "baseURL": "$OPENAI_BASE_URL",
                        "apiKey": "intercepted",
                        "timeout": self.provider_timeout_ms,
                    },
                    "models": {
                        "$OPENAI_MODEL": {
                            "name": "$OPENAI_MODEL",
                            "modalities": {
                                "input": ["text", "image"],
                                "output": ["text"],
                            },
                            "interleaved": {"field": "reasoning_content"}
                            if enable_interleaved
                            else False,
                        }
                    },
                }
            },
            "model": f"{OPENCODE_PROVIDER_ID}/$OPENAI_MODEL",
        }

        if disable_compaction:
            config["compaction"] = {"auto": False, "prune": False}

        if system_prompt_path or disabled_tools:
            build_config = {}
            if system_prompt_path:
                build_config["prompt"] = "{file:" + system_prompt_path + "}"
            if disabled_tools:
                build_config["tools"] = {tool: False for tool in disabled_tools}
            config["agent"] = {"build": build_config}

        return json.dumps(config, indent=2)

    async def build_env_vars(self, state: vf.State) -> dict[str, str]:
        env_vars = await super().build_env_vars(state)
        env_vars["OPENCODE_SESSION_ID"] = state["info"]["session_id"]
        env_vars["ROLLOUTS_REMOTE_URL"] = state["info"]["remote_url"]
        env_vars["ROLLOUTS_MESSAGE_ID"] = state["info"]["restore_message_id"]
        return env_vars

    async def post_sandbox_setup(self, state: vf.State) -> None:
        await super().post_sandbox_setup(state)

        if not state["info"]["session_json"]:
            return

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            f.write(
                state["info"]["session_json"].replace(
                    "__AGENT_WORKDIR__", self.remote_workspace_path
                )
            )
            local_session_path = f.name

        try:
            await self.sandbox_client.upload_file(
                state["sandbox_id"],
                self.remote_session_path,
                local_session_path,
            )
        finally:
            Path(local_session_path).unlink(missing_ok=True)

    def build_prompt(self, state: vf.State) -> str:
        return state["info"]["resume_prompt"]


def load_environment(
    dataset: str | None = "13point5/test-2-rollouts-learn", **kwargs
) -> vf.Environment:
    """
    Loads a custom environment.
    """

    data_path = hf_hub_download(
        repo_id=dataset,
        repo_type="dataset",
        filename="train.jsonl",
    )
    with open(data_path) as f:
        dataset = Dataset.from_list([transform_row(json.loads(line)) for line in f])

    return ContinualLearningEnv(dataset=dataset, **kwargs)
