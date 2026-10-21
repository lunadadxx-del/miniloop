"""Tests for miniloop."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from miniloop import AgentLoop, LLMProvider, ProviderResponse, ToolCall


class EchoProvider(LLMProvider):
    """Always calls the 'shout' tool once, then answers."""
    def __init__(self):
        self.calls = 0

    def complete(self, messages, tools, system):
        self.calls += 1
        if any(m.get("role") == "tool" for m in messages):
            return ProviderResponse(content="Done shouting.")
        return ProviderResponse(
            content=None,
            tool_calls=[ToolCall(id="1", name="shout", arguments={"text": "hello"})])


def shout(text: str = "") -> dict:
    return {"ok": True, "message": f"SHOUTED: {text}",
            "card": {"type": "shout", "text": text}}


def _agent(provider=None):
    return AgentLoop(
        provider=provider or EchoProvider(),
        tools={"shout": shout},
        tool_schemas=[{"name": "shout", "description": "Shout",
                       "parameters": {"type": "object",
                                     "properties": {"text": {"type": "string"}}}}],
        system_prompt="sys")


def test_tool_called_and_card_collected():
    res = _agent().chat([], "shout please")
    assert res["reply"] == "Done shouting."
    assert res["cards"] == [{"type": "shout", "text": "hello"}]
    assert res["tool_calls"][0]["tool"] == "shout"
    assert res["tool_calls"][0]["ok"] is True


def test_unknown_tool_is_graceful():
    class Rogue(LLMProvider):
        def complete(self, messages, tools, system):
            if any(m.get("role") == "tool" for m in messages):
                return ProviderResponse(content="recovered")
            return ProviderResponse(content=None, tool_calls=[
                ToolCall(id="x", name="nope", arguments={})])

    res = _agent(Rogue()).chat([], "hi")
    assert res["reply"] == "recovered"
    assert res["tool_calls"][0]["ok"] is False


def test_tool_exception_does_not_kill_chat():
    def boom() -> dict:
        raise RuntimeError("kaput")

    class Once(LLMProvider):
        def complete(self, messages, tools, system):
            if any(m.get("role") == "tool" for m in messages):
                return ProviderResponse(content="survived")
            return ProviderResponse(content=None,
                                    tool_calls=[ToolCall(id="x", name="boom", arguments={})])

    agent = AgentLoop(provider=Once(), tools={"boom": boom}, tool_schemas=[],
                      system_prompt="sys")
    res = agent.chat([], "go")
    assert res["reply"] == "survived"


def test_history_bounded():
    agent = _agent()
    history = [{"role": "user", "content": f"m{i}"} for i in range(50)]
    res = agent.chat(history, "one more")
    assert len(res["history"]) <= 20
