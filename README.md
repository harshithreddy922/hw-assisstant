# hwqa: a hardware Q&A assistant for your Linux laptop

Ask plain-language questions about **this** machine's hardware. The assistant gathers facts with
safe, read-only system tools, then answers, and says "I don't know" (with the fix) when it can't
see something.

```
you> Can I add more RAM, and how much?

You have 16 GB in 1 of 2 slots (DDR4 3200 MT/s). The board supports up to 64 GB, so you can add a
stick of up to 32 GB in the free slot, for 48 GB in total. Use a matching DDR4 3200 MT/s SODIMM/DIMM
for dual-channel performance. (The maximum comes from firmware and is occasionally understated; the
service manual is authoritative.)
sources: dmidecode -t memory (as root), /proc/meminfo

you> and what about the GPU?

You have 2 GPUs: Intel Alder Lake-P GT2 [Iris Xe Graphics] (integrated), using the i915 driver;
NVIDIA GeForce RTX 3050 Laptop GPU (discrete) with 4 GB VRAM, using the nvidia driver.
sources: lspci -mm -nn, nvidia-smi
```

---

## Quick start

Requires Python 3.10+ on Linux.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

| What | Command |
|---|---|
| Interactive chat | `python -m hwqa` |
| One question | `python -m hwqa ask "Is my SSD NVMe?"` |
| Show tools / commands used per answer | `python -m hwqa -v ask "How much RAM do I have?"` (or `/trace` in chat) |
| What can be inspected on this machine | `python -m hwqa doctor` |
| Raw normalised JSON from one tool | `python -m hwqa tool get_memory_hardware` |
| One-time root snapshot for RAM-slot data | `sudo python -m hwqa collect-privileged` |
| Tests (51) | `pytest` |

(`pip install -e .` also installs an `hwqa` command.)

**Choosing the brain** (`--backend`, default `auto`):

| Backend | Data leaves the machine? | Notes |
|---|---|---|
| `ollama` | **No** | Local model via Ollama, e.g. `ollama pull qwen2.5:7b` (set `HWQA_OLLAMA_MODEL`) |
| `claude` | Redacted tool output only | Best answer quality; needs `ANTHROPIC_API_KEY` (`HWQA_CLAUDE_MODEL`, default `claude-opus-5-5`) |
| `offline` | **No** | No model at all: rule-based intent routing + templated answers built only from tool data |

`auto` picks a running local Ollama first (privacy), then Claude if a key is configured, else offline.

In chat: `/trace` (commands and files behind the last answer), `/tools`, `/raw <tool>`, `/privacy`,
`/reset`, `/quit`.

---

## Architecture

```
 user question
      │
      ▼
┌──────────────┐  next step?   ┌─────────────────────────────┐
│  Assistant   │──────────────▶│ Backend: Claude │ Ollama │ │
│  (tool loop) │◀──────────────│          offline rules      │
└──────┬───────┘ tool calls /  └─────────────────────────────┘
       │         final answer
       │ 1. validate tool name + args against strict JSON schema
       │ 2. session cache (static hw: long TTL, live readings: seconds)
       ▼
┌──────────────────────────────┐   12 typed, read-only tools
│ HardwareTools (tools.py)     │   get_cpu_info, get_memory_hardware, get_storage_devices, …
│ best source → fallbacks      │   each returns {data, sources, unavailable[{what, reason, how_to_fix}]}
└──────┬───────────────────────┘
       ▼
┌──────────────────────────────┐   pure functions, unit-tested on real output
│ Parsers (parsers.py)         │   raw text → normalised JSON; derived facts computed in code
└──────┬───────────────────────┘   (e.g. RAM upgrade analysis); serials/MACs never extracted
       ▼
┌──────────────────────────────┐   the ONLY code that touches the OS
│ Executor (executor.py)       │   allow-listed binaries + exact argv, no shell, trusted dirs,
│                              │   read-only allow-listed paths, timeouts, audit log
└──────────────────────────────┘
       │
 final answer ──▶ grounding check: every number must appear in tool output
                  (LLMs get one automatic self-correction round) ──▶ user, with sources
```

| File | Role |
|---|---|
| `hwqa/executor.py` | Security boundary: command/path allow-lists, no shell, privilege handling, audit |
| `hwqa/tools.py` | The 12 tools, their schemas, fallback chains, explicit "unavailable" reporting |
| `hwqa/parsers.py` | Parsers for lscpu, /proc/cpuinfo, /proc/meminfo, dmidecode, lsblk, lspci, pci.ids, nvidia-smi, lsusb, os-release, power_supply |
| `hwqa/assistant.py` | Tool loop: validation, caching, redaction, grounding, follow-up context |
| `hwqa/guards.py` | Grounding checker + privacy redaction |
| `hwqa/backends/` | `claude.py` (Anthropic SDK), `ollama.py` (local), `offline.py` (no model) |
| `hwqa/cli.py` | Chat REPL, `ask`, `doctor`, `tool`, `collect-privileged` |

---

## 1. Tool-calling design: how the LLM gathers facts instead of guessing

- **Typed tools, not a shell.** The model only sees 12 tools such as `get_memory_hardware()` or
  `get_pci_devices(category: enum)`. Each has a strict JSON schema (`additionalProperties: false`,
  enums for every string argument). There is no "run command" tool, so there is nothing to inject into.
- **Arguments are re-validated** by the Assistant before execution, even with strict schemas (local
  models don't honour schemas reliably). Unknown tools and invalid arguments come back to the model as
  error results, and nothing runs. The test `test_model_cannot_call_unknown_tools_or_bad_args` scripts a
  misbehaving model asking for `run_shell("rm -rf /")` and `category="$(reboot)"`.
- **Tools return normalised, structured data plus provenance.** For example, `get_storage_devices` says
  `"kind": "NVMe SSD", "is_nvme": true, "size": "512 GB"`, not raw `lsblk` output. Each result lists its
  `sources` (shown to the user) and an explicit `unavailable` list with the reason and how to fix it.
- **Derived facts are computed in code.** The RAM-upgrade answer needs arithmetic: per-slot maximum,
  headroom, largest stick you can add, and soldered-memory detection. `parsers.upgrade_analysis` does it
  deterministically, so the model only has to phrase it.
- **Follow-ups.** The conversation transcript and a session cache persist. "and what about the GPU?" is a
  new topic. "is it integrated?" or "Can I add more?" inherit the previous topic, and repeated facts come
  from cache (static hardware cached for the session; battery, temperature and disk usage for 5–15 s).
- **One loop, three brains.** Validation, caching, audit, redaction and grounding live in the
  `Assistant`. A backend only decides "call these tools" or "here's the answer". Swapping Claude for a
  local model changes no safety property.

## 2. Safety: read-only, allow-listed, no arbitrary execution

| Control | Implementation |
|---|---|
| No shell | `subprocess.run(argv, shell=False)`: `;`, `&&`, `\|`, `$()` are just bytes |
| Allow-listed commands | `ALLOWED_COMMANDS` maps each binary to the **exact** argument vectors permitted (`lscpu -J`, `dmidecode -t memory`, …). Anything else raises `NotAllowed` |
| Binary hijacking | Binaries resolve only from `/usr/bin`, `/bin`, `/usr/sbin`, `/sbin`, `/usr/local/bin`, never from `$PATH`, cwd or home (tested with a planted `lscpu`) |
| Read-only file access | Only `/proc/`, `/sys/`, os-release, pci.ids and the snapshot dir. Both the normalised path and its symlink-resolved target must match, so `/sys/../etc/shadow` is refused (tested) |
| Resource limits | 5 s timeout per command, 2 MB output cap, stdin closed, minimal env (`LC_ALL=C` for parseable output) |
| Read-only by design | No tool writes anything. Requests to install, delete or change things are declined |
| Audit | Every command and file read is logged with status and duration (`/trace`, `-v`) |
| Prompt injection via tool data | Tool output is labelled as data in the system prompt. The model cannot act on it beyond calling other read-only tools |

## 3. Permissions: what needs root, and how it's handled

Only one thing commonly needs root: the firmware memory tables (`dmidecode`), which hold RAM slots,
module types and the board's maximum. **The assistant itself never runs as root.** It tries these in order:

1. **Already root?** Run `dmidecode -t memory` directly (it works, but the CLI warns that root isn't needed).
2. **Privileged snapshot (recommended).** `sudo hwqa collect-privileged` runs once, as root, executes
   only `dmidecode -t memory`, **strips serial numbers and asset tags**, and writes
   `/var/lib/hwqa/dmi-memory.txt` (0644). The unprivileged assistant reads it and cites it as the source.
   RAM rarely changes, so this is a good trade-off.
3. **`--allow-sudo` (opt-in).** Uses `sudo -n` (never prompts) for exactly one allow-listed argv. It needs
   an admin-created rule scoped to that command:
   `youruser ALL=(root) NOPASSWD: /usr/sbin/dmidecode -t memory`
4. **Otherwise: a partial answer plus the fix.** "I can't determine the RAM slots … requires root. To fix:
   `sudo hwqa collect-privileged`. What I can see without that: 15.7 GB of usable RAM." It never guesses.

Everything else (CPU, memory usage, drives, GPU via sysfs, battery, temperatures, network, product
model via `/sys/class/dmi/id`) works unprivileged. The design deliberately never collects
root-only identifiers such as serial numbers.

## 4. Accuracy: avoiding hallucinated specs

- **System prompt rules:** answer only from tool output, prefer precomputed fields, report
  `unavailable` items as "I can't determine X because Y; to fix: Z", and never fill gaps with typical values.
- **Grounding check (`guards.py`):** every number-with-unit in the final answer (GB, GHz, MT/s, %, °C,
  cores, slots…) must match a value in the tool outputs seen this session, allowing for unit
  conversions like 4700 MHz → 4.7 GHz. If something doesn't match, an LLM backend gets one automatic
  "these figures don't appear in tool output: … rewrite or say you don't know" round. Anything still
  unverifiable is shown to the user with a ⚠ marker. Tested with a scripted model that claims "32 GB"
  on a 16 GB machine.
- **Honest semantics in the data:** `MemTotal` is labelled "usable (slightly below installed)"; disk sizes
  are decimal like drive labels; DMI maximum capacity carries a caveat that firmware sometimes
  understates it; integrated GPUs are marked as sharing system RAM; inside a VM the tools say the
  disk type and CPU are virtual.
- **The offline engine** writes only templated sentences filled from tool fields, so it cannot hallucinate.
  It doubles as the deterministic oracle for the tests.

## 5. Portability: distros, hardware and missing tools

- **Capability discovery:** at start-up each tool probes for its data sources, and only tools with at
  least one available source are offered to the model. `hwqa doctor` prints the matrix.
- **Fallback chains** in every tool: `lscpu -J` → `/proc/cpuinfo`; `lsblk -J` (with an older-util-linux
  column variant) → `/sys/block`; `lspci` → `/sys/bus/pci/devices` + `pci.ids` names; `lsusb` →
  `/sys/bus/usb/devices`; NVIDIA VRAM via `nvidia-smi`, AMD via `mem_info_vram_total`; batteries with
  energy (Wh) or charge (Ah) counters.
- **Parsers handle real format variation**, e.g. flat and nested `lscpu -J` (util-linux ≥ 2.38 nests
  fields), dmidecode sizes in MB or GB, "No Module Installed", soldered "Row Of Chips" memory, and
  multiple memory arrays.
- **Tested on a machine that lacks most tools:** this was developed in an Ubuntu 24.04 VM with no
  `dmidecode`, `lspci`, `lsusb`, DMI data, battery or GPU. Every tool degrades to a clear explanation
  there (see the session below), and the laptop paths are tested against fixtures.

## 6. Privacy: does data leave the machine?

- **Local by default when possible:** `auto` prefers a local Ollama model; `offline` needs no model. With
  either, nothing leaves the laptop. The Ollama client also ignores proxy environment variables.
- **Minimal collection:** parsers whitelist fields. Serial numbers, asset tags, UUIDs and MAC addresses
  are never read into results, and the root snapshot strips serials before writing.
- **Redaction before any remote call:** for Claude, tool output additionally passes through
  pattern-based redaction (MACs, UUIDs, `/home/<user>` and `/media/<user>` paths, serials) as a second
  safety net. `/privacy` and the start-up banner state exactly where data goes.
- **Trade-off:** a local 7B model is private but follows tool-use instructions less reliably. That's
  why argument validation and grounding checks sit outside the model, so they protect every backend equally.

## Example session on the development VM (offline backend)

```
you> How much RAM do I have?
The OS sees 15.7 GB usable; 460 MB is in use and 15.3 GB is available (2.9% used).
sources: /proc/meminfo

you> Can I add more?
I can't determine the RAM slots, module sizes/types/speeds, soldered or not, maximum supported RAM:
dmidecode is not installed. To fix: sudo apt install dmidecode (or dnf/pacman install dmidecode),
then: sudo hwqa collect-privileged. What I can see without that: 15.7 GB of usable RAM.

you> Is my SSD NVMe?
I can't tell: this is a virtual machine. Drives: vda: 275 GB virtual disk (VM); …
```

## Testing

`pytest` runs 51 tests (one is skipped when run as root):
- **Parsers:** real captured output from the dev VM (lscpu, lsblk, meminfo) plus representative
  laptop output (dmidecode normal and soldered, lspci, nvidia-smi, lsusb, nested lscpu, battery).
- **Security:** refused commands, argument injection, path traversal, `$PATH` hijack, no sudo without
  opt-in, `sudo -n` only for the allow-listed argv, and a misbehaving scripted model.
- **End to end:** full conversations on a fake laptop (`tests/fake_system.py`) that keeps the real
  allow-list checks: the brief's RAM example as root, via snapshot and without root; soldered RAM; NVMe;
  GPU follow-ups with caching; battery, Wi-Fi, temperatures, model; read-only refusals.
- **Guards:** grounding detection, the LLM self-correction round, redaction of remote payloads, and the
  Claude wire format (parallel tool results grouped in one message).
- **Live smoke test** on the real machine.

## Limitations / next steps

- The Claude and Ollama backends are implemented against their documented APIs, but the automated tests
  drive the loop with scripted and offline backends. Run `python -m hwqa --backend claude` (with
  `ANTHROPIC_API_KEY`) or `--backend ollama` to try them live.
- The offline engine understands common question shapes only. The LLM backends handle open-ended phrasing.
- More tools would be easy to add the same way: display/EDID resolution, `smartctl` disk health
  (root, via the snapshot pattern), `upower`, Bluetooth, and per-core frequencies.
- The per-slot maximum is estimated as board maximum ÷ slots. Some boards differ, which the answer
  states as a caveat.
