"""miniloop: a tiny provider-agnostic tool-calling agent loop.

This module is intentionally dependency-free (stdlib only) so it can be
extracted as its own open-source package. An LLM provider implements one
method — ``complete()`` — and the loop handles tool execution, transcript
management, and card collection.

Tool schemas use a simple OpenAI-style format::

    {"name": ..., "description": ..., "parameters": {"type": "object",
     "properties": {...}, "required": [...]}}

Each tool implementation is a callable returning a dict like::

    {"ok": True, "message": "human-readable summary",
     "card": {"type": "medication", ...} or None}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict = field(default_factory=dict)


@dataclass
class ProviderResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMProvider:
    """Implement ``complete()`` to plug in any LLM."""

    def complete(self, messages: list[dict], tools: list[dict], system: str) -> ProviderResponse:
        raise NotImplementedError


def _result_text(result: Any) -> str:
    if isinstance(result, dict):
        return str(result.get("message", result))
    return str(result)


class AgentLoop:
    """Runs the think -> act (tools) -> respond loop."""

    def __init__(
        self,
        provider: LLMProvider,
        tools: dict[str, Callable[..., dict]],
        tool_schemas: list[dict],
        system_prompt: str,
        max_iterations: int = 6,
    ):
        self.provider = provider
        self.tools = tools
        self.tool_schemas = tool_schemas
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations

    def chat(self, history: list[dict], user_message: str) -> dict:
        messages = list(history) + [{"role": "user", "content": user_message}]
        cards: list[dict] = []
        calls_made: list[dict] = []

        for _ in range(self.max_iterations):
            resp = self.provider.complete(messages, self.tool_schemas, self.system_prompt)
            assistant_msg: dict = {"role": "assistant", "content": resp.content}
            if resp.tool_calls:
                assistant_msg["tool_calls"] = [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    for tc in resp.tool_calls
                ]
            messages.append(assistant_msg)

            if not resp.tool_calls:
                break

            for tc in resp.tool_calls:
                fn = self.tools.get(tc.name)
                if fn is None:
                    result = {"ok": False, "message": f"I don't know how to '{tc.name}'."}
                else:
                    try:
                        result = fn(**tc.arguments)
                        if not isinstance(result, dict):
                            result = {"ok": True, "message": str(result)}
                    except TypeError as e:
                        result = {"ok": False,
                                  "message": f"Wrong arguments for '{tc.name}': {e}."}
                    except Exception as e:  # tool bugs must not kill the chat
                        result = {"ok": False, "message": f"'{tc.name}' failed: {e}."}
                calls_made.append({"tool": tc.name, "arguments": tc.arguments,
                                   "ok": bool(result.get("ok"))})
                if isinstance(result, dict) and result.get("card"):
                    cards.append(result["card"])
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "name": tc.name, "content": _result_text(result)})

        reply = ""
        for m in reversed(messages):
            if m["role"] == "assistant" and m.get("content"):
                reply = m["content"]
                break
        if not reply and calls_made:
            # Provider spoke only through tools — summarize what happened.
            reply = " ".join(
                c["result"] if isinstance(c.get("result"), str) else "" for c in calls_made
            ).strip()
        if not reply:
            # Fall back to the tool result messages themselves.
            summaries = []
            for m in messages:
                if m["role"] == "tool" and m.get("content"):
                    summaries.append(m["content"])
            reply = " ".join(summaries).strip() or "Done."

        new_history = messages[-20:]  # keep the transcript bounded
        return {"reply": reply, "cards": cards, "tool_calls": calls_made,
                "history": new_history}
