# miniloop

A tiny **provider-agnostic tool-calling agent loop** in pure Python stdlib —
no dependencies. Bring any LLM; miniloop handles the think → act → respond loop,
tool execution, transcript management, and result collection.

Extracted from [CareLoop](https://github.com/lunadadxx-del/careloop) (built for the
Build, Ship, Shape: Amazon Developer Hackathon).

## Install

```bash
pip install miniloop   # (or: pip install -e .)
```

## Usage

```python
from miniloop import AgentLoop, LLMProvider, ProviderResponse, ToolCall

class MyProvider(LLMProvider):
    def complete(self, messages, tools, system):
        # call your LLM here; return text and/or tool calls
        ...

def get_weather(city: str) -> dict:
    return {"ok": True,
            "message": f"Sunny in {city}.",
            "card": {"type": "weather", "city": city}}

tools = {
    "get_weather": get_weather,
}
schemas = [{
    "name": "get_weather",
    "description": "Get the weather for a city.",
    "parameters": {"type": "object",
                   "properties": {"city": {"type": "string"}},
                   "required": ["city"]},
}]

agent = AgentLoop(provider=MyProvider(), tools=tools,
                  tool_schemas=schemas,
                  system_prompt="You are a helpful assistant.")
result = agent.chat([], "what's the weather in Bengaluru?")
print(result["reply"])       # the final answer
print(result["cards"])       # structured cards collected from tools
print(result["tool_calls"])  # audit trail of what ran
```

### Message format

The transcript passed to your provider uses plain dicts:

- `{"role": "user", "content": "..."}`
- `{"role": "assistant", "content": "..."|None, "tool_calls": [{"id", "name", "arguments"}]}`
- `{"role": "tool", "tool_call_id": ..., "name": ..., "content": "..."}`

Map these to your provider's native format (OpenAI, Bedrock Converse `toolUse`/
`toolResult`, etc.) inside `complete()`.

### Tool results

Return a dict from each tool:

```python
{"ok": True, "message": "human-readable summary",
 "card": {...}}   # optional structured payload for your UI
```

Tool exceptions never kill the chat — they're converted into `{"ok": False, ...}`
results the agent can react to.

## Design notes

- **Zero dependencies.** Copy-pasteable into any project.
- **Bounded history.** Only the last 20 transcript messages are kept.
- **Honest by default.** `result["tool_calls"]` records exactly which tools ran
  with which arguments — useful for logs, evals, and demos.

## License

MIT — see [LICENSE](LICENSE).
