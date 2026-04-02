import json
import tempfile
from pathlib import Path

from datasets import load_dataset
import verifiers as vf
from verifiers.envs.experimental.opencode_env import OpenCodeEnv

DEFAULT_INSTALL_COMMAND = "curl -fsSL https://opencode.ai/install | bash -s -- --version v1.3.13"


DEFAULT_RUN_COMMAND_TEMPLATE = """\
set -e

apt-get update && apt-get install -y curl

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
export PATH="$HOME/.opencode/bin:$PATH"

mkdir -p ~/.config/opencode

SCHEMA_DOLLAR='$'

cat > ~/.config/opencode/opencode.json << EOFCONFIG
{config_json}
EOFCONFIG

opencode import {session_path}

cd {agent_workdir}
cat {prompt_path} | opencode run --session "$OPENCODE_SESSION_ID" 2>&1 | tee {logs_path}
"""


def _session_to_messages(session: dict) -> list[dict]:
    messages = []

    for session_message in session["messages"]:
        role = session_message["info"]["role"]
        parts = session_message.get("parts", [])

        if role == "user":
            content = "\n\n".join(part["text"] for part in parts if part.get("type") == "text")
            if content:
                messages.append({"role": "user", "content": content})
            continue

        reasoning = "\n\n".join(part["text"] for part in parts if part.get("type") == "reasoning")
        content = "\n\n".join(part["text"] for part in parts if part.get("type") == "text")
        tool_parts = [part for part in parts if part.get("type") == "tool"]

        if content or reasoning or tool_parts:
            assistant_message = {"role": "assistant", "content": content}
            if reasoning:
                assistant_message["reasoning_content"] = reasoning
            if tool_parts:
                assistant_message["tool_calls"] = [
                    {
                        "id": part["callID"],
                        "name": part["tool"],
                        "arguments": json.dumps(
                            part["state"].get("input", {}),
                            separators=(",", ":"),
                        ),
                    }
                    for part in tool_parts
                ]
            messages.append(assistant_message)

        for part in tool_parts:
            output = part["state"].get("output", "")
            if not isinstance(output, str):
                output = json.dumps(output)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": part["callID"],
                    "content": output,
                }
            )

    return messages


def _last_user_prompt(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message["role"] == "user":
            return message["content"]

    return ""


def _transform_row(row: dict) -> dict:
    prompt = _session_to_messages(row["session"])

    return {
        "prompt": prompt,
        "info": {
            "session_id": row["session_id"],
            "agent": row["agent"],
            "exported_at": row["exported_at"],
            "metadata": row["metadata"],
            "session_json": row["session"],
            "resume_prompt": _last_user_prompt(prompt),
        },
    }


class ContinualLearningEnv(OpenCodeEnv):
    DEFAULT_INSTALL_COMMAND = DEFAULT_INSTALL_COMMAND
    DEFAULT_RUN_COMMAND_TEMPLATE = DEFAULT_RUN_COMMAND_TEMPLATE

    @property
    def remote_session_path(self) -> str:
        return f"{self.asset_dir}/session.json"

    def __init__(
        self,
        *args,
        install_command: str | None = DEFAULT_INSTALL_COMMAND,
        run_command_template: str | None = DEFAULT_RUN_COMMAND_TEMPLATE,
        **kwargs,
    ):
        super().__init__(
            *args,
            install_command=install_command or self.DEFAULT_INSTALL_COMMAND,
            run_command_template=run_command_template or self.DEFAULT_RUN_COMMAND_TEMPLATE,
            **kwargs,
        )

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
        )

    async def build_env_vars(self, state: vf.State) -> dict[str, str]:
        env_vars = await super().build_env_vars(state)
        env_vars["OPENCODE_SESSION_ID"] = state["info"]["session_id"]
        return env_vars

    async def post_sandbox_setup(self, state: vf.State) -> None:
        await super().post_sandbox_setup(state)

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            json.dump(state["info"]["session_json"], f)
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


def load_environment(**kwargs) -> vf.Environment:
    """
    Loads a custom environment.
    """

    dataset = load_dataset("13point5/opencode-rollouts-test", split="train")
    dataset = dataset.map(_transform_row)

    return ContinualLearningEnv(dataset=dataset, **kwargs)
