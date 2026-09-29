"""Chat with tool calling. Ollama and Groq both accept OpenAI-style tool specs."""
import json
from dataclasses import dataclass, field

import requests

from agent.config import GROQ_API_KEY, GROQ_MODEL, LLM_PROVIDER, OLLAMA_MODEL, OLLAMA_URL


class LLMError(RuntimeError):
    pass


@dataclass
class ToolCall:
    name: str
    args: dict
    id: str | None = None      # Groq/OpenAI need it to match results; Ollama does not


@dataclass
class Reply:
    text: str
    calls: list[ToolCall] = field(default_factory=list)
    raw: dict = field(default_factory=dict)   # assistant message to append to the history


def _args(a):
    if isinstance(a, dict):
        return a
    try:
        return json.loads(a or "{}")
    except json.JSONDecodeError:
        return {}


def chat(messages, tools, provider=LLM_PROVIDER) -> Reply:
    if provider == "ollama":
        try:
            r = requests.post(f"{OLLAMA_URL}/api/chat", timeout=600, json={
                "model": OLLAMA_MODEL, "messages": messages, "tools": tools, "stream": False,
                "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 600}})  # cap runaway generations
            r.raise_for_status()
        except requests.RequestException as e:
            raise LLMError(f"Ollama not reachable at {OLLAMA_URL} ({e}).") from e
        msg = r.json()["message"]
        calls = [ToolCall(c["function"]["name"], _args(c["function"].get("arguments")))
                 for c in msg.get("tool_calls") or []]
        return Reply(msg.get("content") or "", calls, msg)
    if provider == "groq":
        if not GROQ_API_KEY:
            raise LLMError("GROQ_API_KEY is not set. Add it to .env (see .env.example).")
        r = requests.post("https://api.groq.com/openai/v1/chat/completions", timeout=120,
                          headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                          json={"model": GROQ_MODEL, "messages": messages, "tools": tools, "temperature": 0})
        if r.status_code != 200:
            raise LLMError(f"Groq error {r.status_code}: {r.text[:200]}")
        msg = r.json()["choices"][0]["message"]
        calls = [ToolCall(c["function"]["name"], _args(c["function"]["arguments"]), c["id"])
                 for c in msg.get("tool_calls") or []]
        raw = {k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")}
        return Reply(msg.get("content") or "", calls, raw)
    raise LLMError(f"Unknown LLM_PROVIDER '{provider}'.")


def tool_message(call: ToolCall, content: str) -> dict:
    m = {"role": "tool", "content": content}
    if call.id:
        m["tool_call_id"] = call.id
    else:
        m["tool_name"] = call.name
    return m


def model_name(provider=LLM_PROVIDER):
    return OLLAMA_MODEL if provider == "ollama" else GROQ_MODEL
