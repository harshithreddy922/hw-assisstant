"""Local model via Ollama's native chat API: nothing leaves the laptop.

Needs a tool-calling capable model, e.g. `ollama pull qwen2.5:7b` (default) or `llama3.1:8b`.
Small local models follow instructions less reliably than Claude, so the Assistant's argument
validation and grounding check matter even more here.
"""
from __future__ import annotations

import json
import os
import uuid

import httpx

from .base import SYSTEM_PROMPT, Step, ToolCall

DEFAULT_HOST = "http://127.0.0.1:11434"


def ollama_available(host: str | None = None) -> bool:
    try:
        r = httpx.get((host or os.environ.get("OLLAMA_HOST", DEFAULT_HOST)).rstrip("/") + "/api/tags",
                      timeout=0.5, trust_env=False)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


class OllamaBackend:
    name = "ollama"
    is_local = True
    is_llm = True

    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or os.environ.get("HWQA_OLLAMA_MODEL", "qwen2.5:7b")
        self.host = (host or os.environ.get("OLLAMA_HOST", DEFAULT_HOST)).rstrip("/")
        self.client = httpx.Client(timeout=180, trust_env=False)  # local only: never via a proxy

    @staticmethod
    def _tool_specs(tools) -> list[dict]:
        return [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                  "parameters": t.input_schema}} for t in tools]

    @staticmethod
    def _to_wire(messages: list[dict]) -> list[dict]:
        wire = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in messages:
            if m["role"] == "assistant":
                msg = {"role": "assistant", "content": m.get("content") or ""}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [{"function": {"name": c["name"], "arguments": c["args"]}}
                                         for c in m["tool_calls"]]
                wire.append(msg)
            elif m["role"] == "tool":
                wire.append({"role": "tool", "content": m["content"], "tool_name": m["name"]})
            else:
                wire.append({"role": "user", "content": m["content"]})
        return wire

    def step(self, messages: list[dict], tools: list) -> Step:
        try:
            r = self.client.post(f"{self.host}/api/chat", json={
                "model": self.model, "messages": self._to_wire(messages), "tools": self._tool_specs(tools),
                "stream": False, "options": {"temperature": 0}})
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            return Step(error=f"Ollama error {e.response.status_code}: {e.response.text[:200]}")
        except httpx.HTTPError as e:
            return Step(error=f"Could not reach Ollama at {self.host}: {e}")
        msg = r.json().get("message", {})
        calls = []
        for c in msg.get("tool_calls") or []:
            fn = c.get("function", {})
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"__invalid_json__": args}
            calls.append(ToolCall(id=c.get("id") or uuid.uuid4().hex[:12], name=fn.get("name", ""), args=args))
        return Step(text=(msg.get("content") or "").strip() or None, tool_calls=calls, raw=msg)
