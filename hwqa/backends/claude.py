"""Claude backend (Anthropic SDK), with strict tool schemas so tool arguments always match the schema."""
from __future__ import annotations

import logging
import os

from .base import SYSTEM_PROMPT, Step, ToolCall

log = logging.getLogger("hwqa.claude")


class ClaudeBackend:
    name = "claude"
    is_local = False
    is_llm = True

    def __init__(self, model: str | None = None, effort: str | None = None):
        import anthropic

        self._a = anthropic
        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("HWQA_CLAUDE_MODEL", "claude-opus-5-5")
        self.effort = effort or os.environ.get("HWQA_EFFORT", "medium")

    @staticmethod
    def _tool_specs(tools) -> list[dict]:
        return [{"name": t.name, "description": t.description, "input_schema": t.input_schema, "strict": True}
                for t in tools]

    @staticmethod
    def _to_wire(messages: list[dict]) -> list[dict]:
        wire: list[dict] = []
        for m in messages:
            if m["role"] == "user":
                wire.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                # Send back the exact content blocks Claude produced (incl. thinking blocks).
                wire.append({"role": "assistant", "content": m.get("raw") or m.get("content") or ""})
            elif m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                if m.get("is_error"):
                    block["is_error"] = True
                # all results for one assistant turn go back in a single user message
                if wire and wire[-1]["role"] == "user" and isinstance(wire[-1]["content"], list) \
                        and wire[-1]["content"] and wire[-1]["content"][0].get("type") == "tool_result":
                    wire[-1]["content"].append(block)
                else:
                    wire.append({"role": "user", "content": [block]})
        return wire

    def step(self, messages: list[dict], tools: list) -> Step:
        a = self._a
        try:
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                tools=self._tool_specs(tools),
                messages=self._to_wire(messages),
                output_config={"effort": self.effort},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except a.AuthenticationError:
            return Step(error="Claude rejected the API key (check ANTHROPIC_API_KEY).")
        except a.RateLimitError:
            return Step(error="Claude rate limit reached; try again shortly.")
        except a.APIStatusError as e:
            return Step(error=f"Claude API error {e.status_code}: {e.message}")
        except a.APIConnectionError:
            return Step(error="Could not reach the Claude API (offline?). Try --backend offline or ollama.")

        if resp.stop_reason == "refusal":
            return Step(error="The model declined to answer this request.")
        content = [b.model_dump(exclude_none=True) for b in resp.content]
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in resp.content if b.type == "tool_use"]
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        if resp.stop_reason == "max_tokens" and not calls:
            return Step(text=text or None, raw=content, error="answer was cut off (max_tokens)")
        return Step(text=text or None, tool_calls=calls, raw=content)
