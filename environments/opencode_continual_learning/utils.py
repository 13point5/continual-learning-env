import json


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


def _serialize_session(session: dict) -> str:
    session_json = json.dumps(session, separators=(",", ":"))
    roots = {
        path
        for message in session["messages"]
        for path in (message.get("info", {}).get("path") or {}).values()
        if isinstance(path, str) and path.startswith("/")
    }

    if isinstance(session.get("directory"), str) and session["directory"].startswith("/"):
        roots.add(session["directory"])

    for root in sorted(roots, key=len, reverse=True):
        session_json = session_json.replace(root, "__AGENT_WORKDIR__")

    return session_json


def transform_row(row: dict) -> dict:
    session = row["session"]
    session_messages = session["messages"]
    last_user_idx = max(
        idx for idx, message in enumerate(session_messages) if message["info"]["role"] == "user"
    )
    prefix_session = {**session, "messages": session_messages[:last_user_idx]}
    prompt = _session_to_messages({**session, "messages": session_messages[: last_user_idx + 1]})

    return {
        "prompt": prompt,
        "info": {
            "session_id": row["session_id"],
            "agent": row["agent"],
            "exported_at": row["exported_at"],
            "metadata": row["metadata"],
            "session_json": _serialize_session(prefix_session)
            if prefix_session["messages"]
            else "",
            "resume_prompt": prompt[-1]["content"],
        },
    }
