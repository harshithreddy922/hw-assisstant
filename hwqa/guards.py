"""Accuracy and privacy guards that sit between the tools and the language model.

grounding: every number in the final answer must be traceable to a tool output (or the user's
           question). Unverifiable figures are flagged; LLM backends get one chance to self-correct.
privacy:   before tool output goes to a *remote* model, strip anything identifying. (Collection is
           already whitelisted in parsers.py; this is a second, pattern-based safety net.)
"""
from __future__ import annotations

import json
import re

# ----------------------------------------------------------------------------- grounding

_NUM = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)(?![\w])")
_UNIT_NUM = re.compile(r"(\d+(?:\.\d+)?)\s*(?:GB|GiB|TB|MB|GHz|MHz|MT/s|%|W|Wh|°C|cores?|threads?|slots?)", re.I)


def _numbers(text: str) -> set[float]:
    out = set()
    for m in _NUM.finditer(text):
        try:
            out.add(float(m.group(1).replace(",", "")))
        except ValueError:
            pass
    return out


def evidence_numbers(tool_results: list[dict], question: str = "") -> set[float]:
    blob = json.dumps(tool_results, ensure_ascii=False) + " " + question
    nums = _numbers(blob)
    # values expressed in other common units (MHz <-> GHz, MB <-> GB) also count as grounded
    derived = set()
    for n in nums:
        if n >= 100:
            derived |= {round(n / 1000, 1), round(n / 1000, 2), round(n / 1024, 1)}
    return nums | derived


def ungrounded_claims(answer: str, tool_results: list[dict], question: str = "") -> list[str]:
    """Numbers attached to hardware units in the answer that appear in no tool output."""
    ev = evidence_numbers(tool_results, question)
    bad = []
    for m in _UNIT_NUM.finditer(answer):
        v = float(m.group(1))
        if not any(abs(v - e) <= max(0.051, 0.005 * abs(e)) for e in ev):
            bad.append(m.group(0))
    return sorted(set(bad))


# ----------------------------------------------------------------------------- privacy

_REDACTIONS = [
    (re.compile(r"\b[0-9a-f]{2}(?::[0-9a-f]{2}){5}\b", re.I), "<mac>"),
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I), "<uuid>"),
    (re.compile(r"(/(?:home|media|run/media|Users)/)[^/\s\"]+"), r"\1<user>"),
    (re.compile(r"(?i)(serial(?:[ _]number)?[\"']?\s*[:=]\s*[\"']?)[^\"',\s}]+"), r"\1<redacted>"),
]


def redact(text: str) -> str:
    for pat, rep in _REDACTIONS:
        text = pat.sub(rep, text)
    return text
