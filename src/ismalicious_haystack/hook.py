from __future__ import annotations

from typing import Literal

from haystack import default_from_dict, default_to_dict, tracing
from haystack.components.agents.state import State
from haystack.components.agents.state.state_utils import replace_values
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.utils import Secret

from .gate import GateClient, GateRefusal, original_url


class IsMaliciousGateHook:
    def __init__(
        self,
        *,
        phase: Literal["before_tool", "after_tool"],
        tool_name: str,
        url_argument: str = "url",
        api_key: Secret | None = None,
        api_secret: Secret | None = None,
        client: GateClient | None = None,
    ) -> None:
        if phase not in {"before_tool", "after_tool"} or not tool_name or not url_argument:
            raise ValueError("Choose a supported phase and one selected tool.")
        self.phase = phase
        self.allowed_hook_points = (phase,)
        self.tool_name = tool_name
        self.url_argument = url_argument
        self.api_key = api_key or Secret.from_env_var("ISMALICIOUS_API_KEY")
        self.api_secret = api_secret or Secret.from_env_var("ISMALICIOUS_API_SECRET")
        self._client = client

    def _gate(self) -> GateClient:
        if self._client is not None:
            return self._client
        try:
            return GateClient(self.api_key.resolve_value() or "", self.api_secret.resolve_value() or "")
        except (ValueError, RuntimeError):
            raise GateRefusal() from None

    def _checks(self, state: State) -> list[tuple[str, str | None]]:
        if tracing.tracer.is_content_tracing_enabled:
            raise GateRefusal()
        messages: list[ChatMessage] = state.get("messages", [])
        if self.phase == "before_tool":
            tools = state.get("tools", [])
            if any(tool.name == self.tool_name and tool.outputs_to_state for tool in tools):
                raise GateRefusal()
            if not messages:
                return []
            return [
                (original_url(call.arguments.get(self.url_argument)), None)
                for call in messages[-1].tool_calls
                if call.tool_name == self.tool_name
            ]
        results = []
        for message in reversed(messages):
            if message.role != ChatRole.TOOL:
                break
            for result in message.tool_call_results:
                if result.origin.tool_name != self.tool_name:
                    continue
                if result.error or not isinstance(result.result, str):
                    raise GateRefusal()
                results.append((original_url(result.origin.arguments.get(self.url_argument)), result.result))
        return results

    def _clear_results(self, state: State) -> None:
        messages: list[ChatMessage] = state.get("messages", [])
        first = len(messages)
        while first and messages[first - 1].role == ChatRole.TOOL:
            first -= 1
        state.set("messages", messages[:first], handler_override=replace_values)

    def run(self, state: State) -> None:
        try:
            checks = self._checks(state)
            if checks:
                client = self._gate()
                for url, content in checks:
                    client.check(url, content)
        except GateRefusal:
            self._clear_results(state)
            raise GateRefusal() from None

    async def run_async(self, state: State) -> None:
        try:
            checks = self._checks(state)
            if checks:
                client = self._gate()
                for url, content in checks:
                    await client.check_async(url, content)
        except GateRefusal:
            self._clear_results(state)
            raise GateRefusal() from None

    def to_dict(self) -> dict[str, object]:
        if self._client is not None:
            raise ValueError("Injected test transports cannot be serialized.")
        return default_to_dict(
            self,
            phase=self.phase,
            tool_name=self.tool_name,
            url_argument=self.url_argument,
            api_key=self.api_key.to_dict(),
            api_secret=self.api_secret.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> IsMaliciousGateHook:
        parameters = dict(data["init_parameters"])
        parameters["api_key"] = Secret.from_dict(parameters["api_key"])
        parameters["api_secret"] = Secret.from_dict(parameters["api_secret"])
        return default_from_dict(cls, {**data, "init_parameters": parameters})
