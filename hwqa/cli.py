"""Command-line interface.

    hwqa                       interactive chat
    hwqa ask "Is my SSD NVMe?" one-shot question
    hwqa doctor                what can be inspected on this machine, and what is missing
    hwqa tool get_cpu_info     run one tool and print its normalised JSON
    sudo hwqa collect-privileged   one-time root snapshot of RAM slot data (serials removed)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .assistant import Answer, Assistant
from .executor import Executor
from .tools import SNAPSHOT_PATH, HardwareTools

DIM, BOLD, YEL, RST = ("\033[2m", "\033[1m", "\033[33m", "\033[0m") if sys.stdout.isatty() else ("", "", "", "")


def build_backend(choice: str):
    from .backends.offline import OfflineBackend
    if choice == "offline":
        return OfflineBackend()
    if choice == "ollama":
        from .backends.ollama import OllamaBackend
        return OllamaBackend()
    if choice == "claude":
        from .backends.claude import ClaudeBackend
        return ClaudeBackend()
    # auto: prefer a local model (privacy), then Claude if the user configured a key, then offline
    from .backends.ollama import ollama_available
    if ollama_available():
        from .backends.ollama import OllamaBackend
        return OllamaBackend()
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        from .backends.claude import ClaudeBackend
        return ClaudeBackend()
    return OfflineBackend()


def privacy_line(backend) -> str:
    if backend.is_local:
        return f"backend: {backend.name}. Everything stays on this machine."
    return (f"backend: {backend.name} (remote). Normalised tool output is sent to the API after redaction; "
            "serial numbers, MAC addresses, UUIDs and user names are never sent. Use --backend ollama or "
            "offline to keep everything local.")


def print_answer(a: Answer, verbose: bool = False) -> None:
    print(f"\n{a.text}\n")
    srcs = []
    for c in a.tool_calls:
        for s in c["sources"]:
            if s not in srcs:
                srcs.append(s)
    if srcs:
        print(f"{DIM}sources: {', '.join(srcs)}{RST}")
    if a.ungrounded:
        print(f"{YEL}⚠ could not verify against system data: {', '.join(a.ungrounded)}{RST}")
    if verbose:
        print_trace(a)


def print_trace(a: Answer) -> None:
    for c in a.tool_calls:
        print(f"{DIM}  tool {c['name']}({json.dumps(c['args'])}){' [cached]' if c['cached'] else ''}{RST}")
    for e in a.audit:
        mark = "✓" if e.ok else "✗"
        print(f"{DIM}    {mark} {e.kind:7} {e.target} {e.detail}{RST}")


HELP = """commands:
  /trace     show tools, commands and files used for the last answer
  /tools     list available tools
  /raw NAME  print a tool's normalised JSON (e.g. /raw get_memory_hardware)
  /privacy   where your data goes
  /reset     forget the conversation
  /quit      exit"""


def chat(assistant: Assistant, verbose: bool) -> None:
    print(f"{BOLD}Hardware assistant{RST}. Ask about your CPU, RAM, storage, GPU, battery… (/help for commands)")
    print(f"{DIM}{privacy_line(assistant.backend)}{RST}")
    if assistant.x.is_root:
        print(f"{YEL}Note: running as root. This works, but isn't needed: use `sudo hwqa collect-privileged` once instead.{RST}")
    last: Answer | None = None
    while True:
        try:
            q = input(f"\n{BOLD}you>{RST} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            continue
        if q in ("/quit", "/exit"):
            return
        if q == "/help":
            print(HELP)
        elif q == "/trace":
            print_trace(last) if last else print("nothing yet")
        elif q == "/tools":
            for t in assistant.tools.values():
                print(f"  {t.name}: {t.description}")
        elif q.startswith("/raw"):
            name = q[4:].strip()
            if name in assistant.tools:
                print(json.dumps(assistant.hw.call(name, {}), indent=2))
            else:
                print("usage: /raw <tool name>  (see /tools)")
        elif q == "/privacy":
            print(privacy_line(assistant.backend))
        elif q == "/reset":
            assistant.reset()
            print("conversation cleared")
        else:
            last = assistant.ask(q)
            print_answer(last, verbose)


def doctor(allow_sudo: bool) -> int:
    x = Executor(allow_sudo=allow_sudo)
    hw = HardwareTools(x)
    print(f"running as root: {x.is_root}   sudo -n allowed: {allow_sudo}   snapshot: "
          f"{'present' if os.path.exists(SNAPSHOT_PATH) else 'none'} ({SNAPSHOT_PATH})\n")
    for row in hw.capability_report():
        status = "OK " if row["available"] else "-- "
        print(f"  {status} {row['tool']:<24} found: {', '.join(row['sources_found']) or '-':<34} "
              f"missing: {', '.join(row['sources_missing']) or '-'}")
    print("\nbinaries:")
    for b in ("lscpu", "lsblk", "lspci", "lsusb", "dmidecode", "nvidia-smi"):
        print(f"  {b:<11} {x.which(b) or 'not installed'}")
    return 0


def collect_privileged() -> int:
    """Run as root ONCE. Saves dmidecode memory tables without serial numbers / asset tags, readable
    by the unprivileged assistant. The assistant itself never needs root."""
    if os.geteuid() != 0:
        print("This must run as root:  sudo hwqa collect-privileged", file=sys.stderr)
        return 1
    x = Executor()
    res = x.run("dmidecode", "-t", "memory")
    if not res.ok:
        print(f"dmidecode failed: {res.error} {res.stderr.strip()}", file=sys.stderr)
        return 1
    clean = "\n".join(line for line in res.stdout.splitlines()
                      if not line.strip().startswith(("Serial Number:", "Asset Tag:", "UUID:")))
    os.makedirs(os.path.dirname(SNAPSHOT_PATH), mode=0o755, exist_ok=True)
    tmp = SNAPSHOT_PATH + ".tmp"
    with open(tmp, "w") as f:
        f.write(clean)
    os.chmod(tmp, 0o644)
    os.replace(tmp, SNAPSHOT_PATH)
    print(f"Saved memory hardware snapshot (serial numbers removed) to {SNAPSHOT_PATH}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hwqa", description="Ask questions about this Linux machine's hardware.")
    ap.add_argument("--backend", choices=["auto", "claude", "ollama", "offline"],
                    default=os.environ.get("HWQA_BACKEND", "auto"))
    ap.add_argument("--allow-sudo", action="store_true",
                    help="allow `sudo -n` for the single allow-listed root command (needs a NOPASSWD rule)")
    ap.add_argument("-v", "--verbose", action="store_true", help="show tools/commands used for each answer")
    sub = ap.add_subparsers(dest="cmd")
    p_ask = sub.add_parser("ask", help="ask one question and exit")
    p_ask.add_argument("question", nargs="+")
    sub.add_parser("chat", help="interactive chat (default)")
    sub.add_parser("doctor", help="show what can be inspected on this machine")
    p_tool = sub.add_parser("tool", help="run one tool and print JSON")
    p_tool.add_argument("name")
    p_tool.add_argument("--args", default="{}", help='JSON arguments, e.g. \'{"category": "network"}\'')
    sub.add_parser("collect-privileged", help="(root) save RAM slot data for unprivileged use")
    a = ap.parse_args(argv)

    if a.cmd == "doctor":
        return doctor(a.allow_sudo)
    if a.cmd == "collect-privileged":
        return collect_privileged()
    x = Executor(allow_sudo=a.allow_sudo)
    if a.cmd == "tool":
        hw = HardwareTools(x)
        if a.name not in hw.tools:
            print(f"unknown tool. available: {', '.join(hw.tools)}", file=sys.stderr)
            return 2
        print(json.dumps(hw.call(a.name, json.loads(a.args)), indent=2))
        return 0
    assistant = Assistant(build_backend(a.backend), x)
    if a.cmd == "ask":
        print_answer(assistant.ask(" ".join(a.question)), a.verbose)
        return 0
    chat(assistant, a.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
