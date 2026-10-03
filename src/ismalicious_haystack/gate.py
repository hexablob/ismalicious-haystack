from __future__ import annotations

import base64
import json
from typing import Literal
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

MAX_BODY_BYTES = 1024 * 1024
REFUSAL = "The selected tool was stopped by the content gate."


class GateRefusal(RuntimeError):
    def __init__(self) -> None:
        super().__init__(REFUSAL)


class _Response(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)
    verdict: Literal["allow", "warn", "block"]
    latency_ms: int = Field(ge=0)


class _Link(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)
    url: str
    entity: str
    verdict: Literal["clean", "suspicious", "malicious", "unknown"]
    sources: int = Field(ge=0)


class _Span(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    family: str


class _Injection(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)
    score: float = Field(ge=0, le=1)
    families: list[str]
    spans: list[_Span]


class _Scan(_Response):
    injection: _Injection
    links: list[_Link]
    links_truncated: bool
    mode: Literal["fast", "thorough"]
    source: _Link | None = None
    sanitized_content: str | None = None


class _Url(_Response):
    url: str
    entity: str
    sources: int = Field(ge=0)


def scan_body(content: str, source_url: str) -> bytes:
    if not isinstance(content, str) or not content:
        raise GateRefusal()
    try:
        body = json.dumps(
            {"content": content, "source_url": source_url, "mode": "fast"},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (UnicodeError, ValueError):
        raise GateRefusal() from None
    if len(body) > MAX_BODY_BYTES:
        raise GateRefusal()
    return body


def original_url(value: object) -> str:
    if not isinstance(value, str) or not value or any(ord(char) < 33 for char in value):
        raise GateRefusal()
    try:
        parsed = urlsplit(value)
        if parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username or parsed.password:
            raise GateRefusal()
    except ValueError:
        raise GateRefusal() from None
    return value


class GateClient:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key or not api_secret:
            raise GateRefusal()
        self._headers = {
            "X-API-KEY": base64.b64encode(f"{api_key}:{api_secret}".encode()).decode(),
            "Content-Type": "application/json",
        }
        self._transport = transport

    @staticmethod
    def _inspect(response: httpx.Response, scan: bool, url: str) -> dict[str, object]:
        try:
            response.raise_for_status()
            parsed = (_Scan if scan else _Url).model_validate_json(response.content)
        except (httpx.HTTPError, ValidationError, ValueError):
            raise GateRefusal() from None
        if parsed.verdict != "allow":
            raise GateRefusal()
        if isinstance(parsed, _Scan) and parsed.links_truncated:
            raise GateRefusal()
        if isinstance(parsed, _Url) and parsed.url != url:
            raise GateRefusal()
        return parsed.model_dump()

    def check(self, url: str, content: str | None = None) -> dict[str, object]:
        checked_url = original_url(url)
        try:
            with httpx.Client(
                base_url="https://api.ismalicious.com",
                headers=self._headers,
                transport=self._transport,
                follow_redirects=False,
                timeout=15,
                verify=True,
                trust_env=False,
            ) as client:
                response = (
                    client.get("/gate/url", params={"u": checked_url})
                    if content is None
                    else client.post("/gate/scan", content=scan_body(content, checked_url))
                )
                return self._inspect(response, content is not None, checked_url)
        except httpx.HTTPError:
            raise GateRefusal() from None

    async def check_async(self, url: str, content: str | None = None) -> dict[str, object]:
        checked_url = original_url(url)
        try:
            async with httpx.AsyncClient(
                base_url="https://api.ismalicious.com",
                headers=self._headers,
                transport=self._transport,
                follow_redirects=False,
                timeout=15,
                verify=True,
                trust_env=False,
            ) as client:
                response = (
                    await client.get("/gate/url", params={"u": checked_url})
                    if content is None
                    else await client.post("/gate/scan", content=scan_body(content, checked_url))
                )
                return self._inspect(response, content is not None, checked_url)
        except httpx.HTTPError:
            raise GateRefusal() from None
