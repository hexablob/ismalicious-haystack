# IsMalicious hooks for Haystack Agents

Inspect the URL before a selected fetch tool runs, then inspect its complete textual result before the next model call. This package uses Haystack's native `before_tool` and `after_tool` hooks and IsMalicious's hosted Gate API. It requires Haystack 3.3 or later

```bash
python -m pip install "ismalicious-haystack @ git+https://github.com/hexablob/ismalicious-haystack.git@v0.1.1"
```

Set `ISMALICIOUS_API_KEY` and `ISMALICIOUS_API_SECRET` from [your account](https://ismalicious.com/app/account) in your secret manager. The integration serializes their environment variable names, not their values. Do not put raw secrets in a workflow or log

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from ismalicious_haystack import GateRefusal, IsMaliciousGateHook

agent = Agent(
    chat_generator=OpenAIChatGenerator(),
    tools=[fetch_page],  # your Tool with a string result and a url argument
    hooks={
        "before_tool": [IsMaliciousGateHook(phase="before_tool", tool_name="fetch_page")],
        "after_tool": [IsMaliciousGateHook(phase="after_tool", tool_name="fetch_page")],
    },
    streaming_callback=None,
    tool_streaming_callback_passthrough=False,
    raise_on_tool_invocation_failure=True,
)
try:
    result = agent.run(messages=messages)
except GateRefusal:
    result = {"refusal": "The selected tool was stopped by the content gate."}
```

The complete runnable example is [`examples/guarded_agent.py`](examples/guarded_agent.py). The same hooks support `await agent.run_async(...)`. The example uses a deterministic native Haystack generator to demonstrate the tool path without an LLM provider charge. Its fetch is restricted to `https://example.com/`; adapt that allowlist to your application and implement SSRF protection for arbitrary destinations

## Policy and boundaries

Top-level `warn` and `block`, scan failures, timeout, quota errors, redirects, invalid schemas, incomplete link inspection and oversized serialized UTF-8 request bodies all refuse. No retries or truncation are used. A valid `allow` retains the original content. It means the service did not block under its current rules, not that content or an unknown URL is proven benign. Inner link records may contain `unknown`; the API contract is documented at [Gate API](https://ismalicious.com/api-docs)

The URL reputation request does not fetch the destination. Only the named tool is protected. This package accepts string tool results; binary, multipart and streaming results are outside the first release. The scan quota is distinct from the indicator lookup quota

Disable `HAYSTACK_CONTENT_TRACING_ENABLED` and all model/tool streaming callbacks. The hook rejects enabled Haystack content tracing and selected tools with `outputs_to_state`. Native tool-result streaming can happen before `after_tool`, so a callback configured on the Agent, its generator, or inside the tool would bypass this boundary. Do not install such callbacks. Do not log raw tool results. Register the gate first at both hook points and do not add later hooks that rewrite or expose uninspected tool content

On a refusal, the hook removes the current tool-result messages from the live State and raises a fixed exception without the URL, result or credentials. The caller must return a fixed refusal and must not recover that State or retry the Agent without the gates. Results necessarily exist in memory before scanning; tool side effects cannot be undone. This is not protection for network calls inside arbitrary tools, model-native search or tools not selected by name

## Validation

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -e ".[test]"
.venv/bin/python -m pytest
```

Tests execute the real Haystack Agent synchronously and asynchronously with a deterministic generator and a local HTTP transport fixture. They establish URL preservation, refusal before fetch, refusal before the next generator/client result, unchanged allowed content, error behavior and secret-free serialization. Synthetic service replies do not measure the detector's accuracy. No claim of production API validation is made by these tests

This integration was prepared with AI assistance. IsMalicious maintains its code, tests and examples
