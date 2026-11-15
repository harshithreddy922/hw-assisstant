"""The orchestration loop shared by every backend.

    question -> backend.step() -> tool calls? -> validate -> (cache | run tool) -> redact if remote
             -> append results -> backend.step() ... -> final answer -> grounding check
             -> (LLM only) one self-correction round if numbers can't be traced to tool output
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from .backends.base import Backend, ToolCall
from .executor import Executor
from .guards import redact, ungrounded_claims
from .tools import HardwareTools

MAX_STEPS = 8
MAX_TOOL_CALLS_PER_TURN = 12


@dataclass
class Answer:
    text: str
    tool_calls: list[dict] = field(default_factory=list)   # name, args, cached, ok, sources
    ungrounded: list[str] = field(default_factory=list)
    audit: list = field(default_factory=list)              # commands/files touched for this answer
    error: str | None = None


def validate_args(schema: dict, args) -> str | None:
    """Strict validation of model-produced arguments (never trust them, even with strict schemas)."""
    if not isinstance(args, dict):
        return "arguments must be a JSON object"
    props = schema.get("properties", {})
    for k, v in args.items():
        if k not in props:
            return f"unknown argument '{k}'"
        spec = props[k]
        t = spec.get("type")
        if t == "boolean" and not isinstance(v, bool):
            return f"'{k}' must be a boolean"
        if t == "string":
            if not isinstance(v, str):
                return f"'{k}' must be a string"
            if "enum" in spec and v not in spec["enum"]:
                return f"'{k}' must be one of {spec['enum']}"
    for k in schema.get("required", []):
        if k not in args:
            return f"missing argument '{k}'"
    return None


class Assistant:
    def __init__(self, backend: Backend, executor: Executor | None = None):
        self.backend = backend
        self.x = executor or Executor()
        self.hw = HardwareTools(self.x)
        self.tools = self.hw.available()
        self.history: list[dict] = []
        self.cache: dict[tuple, tuple[float, dict]] = {}
        self.session_results: list[dict] = []   # everything the model has seen (for grounding)

    def reset(self) -> None:
        self.history.clear()
        self.cache.clear()
        self.session_results.clear()
        if hasattr(self.backend, "last_intents"):
            self.backend.last_intents = []

    # ------------------------------------------------------------------ tools
    def _run_tool(self, call: ToolCall) -> tuple[dict, dict]:
        meta = {"name": call.name, "args": call.args, "cached": False, "ok": False, "sources": []}
        tool = self.tools.get(call.name)
        if tool is None:
            return {"error": f"unknown or unavailable tool '{call.name}'"}, meta
        err = validate_args(tool.input_schema, call.args)
        if err:
            return {"error": f"invalid arguments: {err}"}, meta
        key = (call.name, json.dumps(call.args, sort_keys=True))
        hit = self.cache.get(key)
        if hit and time.time() - hit[0] < tool.ttl_s:
            result, meta["cached"] = hit[1], True
        else:
            result = self.hw.call(call.name, call.args)
            self.cache[key] = (time.time(), result)
        meta["ok"], meta["sources"] = result["ok"], result["sources"]
        return result, meta

    # ------------------------------------------------------------------ main loop
    def ask(self, question: str) -> Answer:
        audit_start = len(self.x.audit)
        self.history.append({"role": "user", "content": question})
        calls_meta: list[dict] = []
        tool_specs = list(self.tools.values())
        final, error = None, None

        for _ in range(MAX_STEPS):
            step = self.backend.step(self.history, tool_specs)
            if step.error and not step.text and not step.tool_calls:
                error = step.error
                break
            self.history.append({"role": "assistant", "content": step.text or "",
                                 "tool_calls": [{"id": c.id, "name": c.name, "args": c.args} for c in step.tool_calls],
                                 "raw": step.raw})
            if not step.tool_calls:
                final = step.text
                error = step.error
                break
            for call in step.tool_calls[:MAX_TOOL_CALLS_PER_TURN]:
                result, meta = self._run_tool(call)
                calls_meta.append(meta)
                if "error" not in result:
                    self.session_results.append(result)
                payload = json.dumps(result, ensure_ascii=False)
                if not self.backend.is_local:
                    payload = redact(payload)
                self.history.append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                                     "content": payload, "is_error": "error" in result})
            for call in step.tool_calls[MAX_TOOL_CALLS_PER_TURN:]:
                self.history.append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                                     "content": json.dumps({"error": "too many tool calls in one turn"}),
                                     "is_error": True})
        else:
            error = "stopped after too many steps without an answer"

        if final is None:
            final = "Sorry, I couldn't produce an answer." + (f" ({error})" if error else "")
            # keep the transcript well-formed for the next question
            if self.history and self.history[-1]["role"] != "assistant":
                self.history.append({"role": "assistant", "content": final, "tool_calls": [], "raw": None})
            return Answer(final, calls_meta, [], self.x.audit[audit_start:], error)

        bad = ungrounded_claims(final, self.session_results, question)
        if bad and self.backend.is_llm:
            self.history.append({"role": "user", "content":
                                 "[automatic accuracy check] These figures in your answer do not appear in any tool "
                                 f"output: {', '.join(bad)}. Rewrite the answer using only values present in the tool "
                                 "results, or say you don't know. Reply with the corrected answer only."})
            step = self.backend.step(self.history, tool_specs)
            if step.text and not step.tool_calls:
                self.history.append({"role": "assistant", "content": step.text, "tool_calls": [], "raw": step.raw})
                final = step.text
                bad = ungrounded_claims(final, self.session_results, question)
            else:  # correction failed: keep the original answer (flagged) and a well-formed transcript
                self.history.append({"role": "assistant", "content": final, "tool_calls": [], "raw": None})
        return Answer(final, calls_meta, bad, self.x.audit[audit_start:], error)
