import json
import os

os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "false"
os.environ["HAYSTACK_TELEMETRY_ENABLED"] = "false"

import httpx
import pytest
from haystack import component, tracing
from haystack.components.agents import Agent
from haystack.components.agents.state import State
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import Tool

from ismalicious_haystack import GateClient, GateRefusal, IsMaliciousGateHook
from ismalicious_haystack.gate import MAX_BODY_BYTES, REFUSAL, scan_body

URL = "https://example.com/path?q=a,b&x=1#part"
CONTENT = "Complete untrusted content: å🚀, not truncated."


def response(request: httpx.Request, verdict: str = "allow", truncated: bool = False) -> httpx.Response:
    if request.url.path == "/gate/url":
        data = {
            "url": request.url.params["u"],
            "entity": "example.com",
            "verdict": verdict,
            "sources": 0,
            "latency_ms": 1,
        }
    else:
        data = {
            "verdict": verdict,
            "injection": {"score": 0.0, "families": [], "spans": []},
            "links": [{"url": URL, "entity": "example.com", "verdict": "unknown", "sources": 0}],
            "links_truncated": truncated,
            "mode": "fast",
            "latency_ms": 1,
        }
    return httpx.Response(200, json=data)


@component
class RecordingGenerator:
    def __init__(self, tool_name: str = "fetch_page") -> None:
        self.inputs = []
        self.tool_name = tool_name

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage], tools: list[Tool] | None = None) -> dict[str, object]:
        self.inputs.append(messages)
        if messages[-1].tool_call_results:
            return {"replies": [ChatMessage.from_assistant("Done")]}
        return {
            "replies": [
                ChatMessage.from_assistant(
                    tool_calls=[ToolCall(tool_name=self.tool_name, arguments={"url": URL}, id="call-1")]
                )
            ]
        }

    @component.output_types(replies=list[ChatMessage])
    async def run_async(self, messages: list[ChatMessage], tools: list[Tool] | None = None) -> dict[str, object]:
        return self.run(messages, tools)


class Recorder(IsMaliciousGateHook):
    def _clear_results(self, state: State) -> None:
        super()._clear_results(state)
        self.state_after_refusal = state.get("messages")


def build(handler, content: str = CONTENT, tool_name: str = "fetch_page", outputs_to_state=None):
    calls = []

    def fetch_page(url: str) -> str:
        calls.append(url)
        return content

    client = GateClient("test-key", "test-secret", transport=httpx.MockTransport(handler))
    before = Recorder(phase="before_tool", tool_name="fetch_page", client=client)
    after = Recorder(phase="after_tool", tool_name="fetch_page", client=client)
    generator = RecordingGenerator(tool_name)
    tool = Tool(
        name=tool_name,
        description="Fetch a page",
        parameters={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
        function=fetch_page,
        outputs_to_state=outputs_to_state,
    )
    agent = Agent(
        chat_generator=generator,
        tools=[tool],
        hooks={"before_tool": [before], "after_tool": [after]},
        streaming_callback=None,
        tool_streaming_callback_passthrough=False,
        raise_on_tool_invocation_failure=True,
        state_schema={"raw": {"type": str}} if outputs_to_state else None,
    )
    return agent, generator, calls, after


@pytest.mark.parametrize("async_run", [False, True])
async def test_native_agent_allowed_payload_is_unchanged(async_run):
    requests = []

    def handler(request):
        requests.append(request)
        return response(request)

    agent, generator, calls, _ = build(handler)
    result = (
        await agent.run_async(messages=[ChatMessage.from_user("Fetch")])
        if async_run
        else agent.run(messages=[ChatMessage.from_user("Fetch")])
    )
    assert calls == [URL]
    assert requests[0].url.params["u"] == URL
    assert json.loads(requests[1].content) == {"content": CONTENT, "source_url": URL, "mode": "fast"}
    assert generator.inputs[1][-1].tool_call_results[0].result == CONTENT
    assert result["messages"][-2].tool_call_results[0].result == CONTENT
    assert result["last_message"].text == "Done"


@pytest.mark.parametrize("async_run", [False, True])
@pytest.mark.parametrize("phase", ["url", "scan"])
@pytest.mark.parametrize("verdict", ["block", "warn", "unknown"])
async def test_native_agent_refusal_stops_fetch_or_next_model(async_run, phase, verdict):
    def handler(request):
        return response(request, verdict if request.url.path.endswith(phase) else "allow")

    agent, generator, calls, after = build(handler)
    with pytest.raises(GateRefusal, match=REFUSAL) as exc:
        if async_run:
            await agent.run_async(messages=[ChatMessage.from_user("Fetch")])
        else:
            agent.run(messages=[ChatMessage.from_user("Fetch")])
    assert len(generator.inputs) == 1
    assert calls == ([] if phase == "url" else [URL])
    assert CONTENT not in str(exc.value) and URL not in str(exc.value)
    if phase == "scan":
        assert all(not message.tool_call_results for message in after.state_after_refusal)


@pytest.mark.parametrize("failure", ["429", "302", "500", "invalid-json", "missing-field", "timeout", "truncated"])
@pytest.mark.parametrize("async_run", [False, True])
async def test_native_agent_service_failures_are_closed(failure, async_run):
    def handler(request):
        if request.url.path == "/gate/url":
            return response(request)
        if failure.isdigit():
            return httpx.Response(int(failure), headers={"Location": "https://attacker.example"})
        if failure == "invalid-json":
            return httpx.Response(200, content=b"not-json")
        if failure == "missing-field":
            return httpx.Response(200, json={"verdict": "allow"})
        if failure == "timeout":
            raise httpx.ReadTimeout("do-not-echo", request=request)
        return response(request, truncated=True)

    agent, generator, calls, after = build(handler)
    with pytest.raises(GateRefusal):
        if async_run:
            await agent.run_async(messages=[ChatMessage.from_user("Fetch")])
        else:
            agent.run(messages=[ChatMessage.from_user("Fetch")])
    assert len(generator.inputs) == 1
    assert calls == [URL]
    assert all(not message.tool_call_results for message in after.state_after_refusal)


def test_serialized_utf8_body_limit_without_truncation():
    with pytest.raises(GateRefusal):
        scan_body("🚀" * (MAX_BODY_BYTES // 4), URL)
    body = scan_body(CONTENT, URL)
    assert json.loads(body)["content"] == CONTENT


def test_selected_tool_with_outputs_to_state_refused_before_side_effect():
    agent, generator, calls, _ = build(response, outputs_to_state={"raw": {}})
    with pytest.raises(GateRefusal):
        agent.run(messages=[ChatMessage.from_user("Fetch")])
    assert calls == [] and len(generator.inputs) == 1


def test_content_tracing_refused():
    agent, _, calls, _ = build(response)
    previous = tracing.tracer.is_content_tracing_enabled
    tracing.tracer.is_content_tracing_enabled = True
    try:
        with pytest.raises(GateRefusal):
            agent.run(messages=[ChatMessage.from_user("Fetch")])
        assert calls == []
    finally:
        tracing.tracer.is_content_tracing_enabled = previous


def test_other_tools_are_not_scanned():
    def handler(request):
        pytest.fail("Unselected tool must not invoke the scanner")

    agent, _, calls, _ = build(handler, tool_name="local_lookup")
    agent.run(messages=[ChatMessage.from_user("Fetch")])
    assert calls == [URL]


def test_hook_roundtrip_uses_only_secret_environment_names():
    hook = IsMaliciousGateHook(phase="before_tool", tool_name="fetch_page")
    data = hook.to_dict()
    assert "ISMALICIOUS_API_KEY" in json.dumps(data)
    restored = IsMaliciousGateHook.from_dict(data)
    assert restored.tool_name == "fetch_page" and restored.phase == "before_tool"


def test_missing_credentials_refused_without_exposing_environment(monkeypatch):
    monkeypatch.delenv("ISMALICIOUS_API_KEY", raising=False)
    monkeypatch.delenv("ISMALICIOUS_API_SECRET", raising=False)
    hook = IsMaliciousGateHook(phase="before_tool", tool_name="fetch_page")
    with pytest.raises(GateRefusal, match=REFUSAL):
        hook._gate()


@pytest.mark.parametrize(
    "bad", [None, 123, "https://user:pass@example.com/", "file:///tmp/a", "https://example.com/\n"]
)
def test_invalid_url_never_calls_service_or_fetch(bad):
    hook = IsMaliciousGateHook(
        phase="before_tool",
        tool_name="fetch_page",
        client=GateClient("a", "b", transport=httpx.MockTransport(lambda r: pytest.fail("Unexpected request"))),
    )
    state = State(
        schema={},
        data={
            "messages": [
                ChatMessage.from_assistant(tool_calls=[ToolCall(tool_name="fetch_page", arguments={"url": bad})])
            ]
        },
    )
    with pytest.raises(GateRefusal):
        hook.run(state)


def test_non_text_result_is_removed_before_refusal():
    from haystack.dataclasses.chat_message import TextContent

    call = ToolCall(tool_name="fetch_page", arguments={"url": URL}, id="x")
    state = State(
        schema={}, data={"messages": [ChatMessage.from_tool(tool_result=[TextContent(text=CONTENT)], origin=call)]}
    )
    hook = IsMaliciousGateHook(phase="after_tool", tool_name="fetch_page")
    with pytest.raises(GateRefusal):
        hook.run(state)
    assert state.get("messages") == []


@pytest.mark.parametrize("invalid", ["backward-span", "latency-overflow"])
def test_malformed_response_semantics_do_not_reach_next_model(invalid):
    def handler(request):
        if request.url.path == "/gate/url":
            return response(request)
        body = response(request).json()
        if invalid == "backward-span":
            body["injection"]["spans"] = [{"start": 9, "end": 2, "family": "instruction"}]
        else:
            body["latency_ms"] = 2**63
        return httpx.Response(200, json=body)

    agent, generator, calls, after = build(handler)
    with pytest.raises(GateRefusal):
        agent.run(messages=[ChatMessage.from_user("Fetch")])
    assert len(generator.inputs) == 1 and calls == [URL]
    assert all(not m.tool_call_results for m in after.state_after_refusal)
