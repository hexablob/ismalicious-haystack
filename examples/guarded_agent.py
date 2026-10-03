import os

os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "false"
os.environ["HAYSTACK_TELEMETRY_ENABLED"] = "false"

import httpx
from haystack import component
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import Tool

from ismalicious_haystack import GateRefusal, IsMaliciousGateHook
from ismalicious_haystack.gate import REFUSAL


def fetch_page(url: str) -> str:
    if url != "https://example.com/":
        raise GateRefusal()
    with httpx.stream("GET", url, follow_redirects=False, timeout=15, trust_env=False) as response:
        response.raise_for_status()
        parts = []
        size = 0
        for part in response.iter_bytes():
            size += len(part)
            if size > 512 * 1024:
                raise GateRefusal()
            parts.append(part)
        return b"".join(parts).decode("utf-8", errors="strict")


@component
class DemoGenerator:
    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage], tools: list[Tool] | None = None) -> dict[str, object]:
        if messages[-1].tool_call_results:
            return {"replies": [ChatMessage.from_assistant("The allowed page was received.")]}
        return {
            "replies": [
                ChatMessage.from_assistant(
                    tool_calls=[ToolCall(tool_name="fetch_page", arguments={"url": "https://example.com/"}, id="demo")]
                )
            ]
        }


def main() -> None:
    fetch = Tool(
        name="fetch_page",
        description="Read the example.com demonstration page",
        parameters={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
        function=fetch_page,
    )
    agent = Agent(
        chat_generator=DemoGenerator(),
        tools=[fetch],
        hooks={
            "before_tool": [IsMaliciousGateHook(phase="before_tool", tool_name="fetch_page")],
            "after_tool": [IsMaliciousGateHook(phase="after_tool", tool_name="fetch_page")],
        },
        streaming_callback=None,
        tool_streaming_callback_passthrough=False,
        raise_on_tool_invocation_failure=True,
    )
    try:
        result = agent.run(messages=[ChatMessage.from_user("Read the example page.")])
        print(result["last_message"].text)
    except GateRefusal:
        print(REFUSAL)


if __name__ == "__main__":
    main()
