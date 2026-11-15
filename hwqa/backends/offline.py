"""Offline engine: keyword intent routing + templated answers. No model, no network.

Not as flexible as an LLM, but every sentence it writes is built directly from tool data, so it
cannot hallucinate. It is the fallback when no model is available, and the deterministic oracle the
tests run against. It goes through the same tool loop (validation, caching, audit) as the LLMs.
"""
from __future__ import annotations

import json
import re
import uuid

from .base import Step, ToolCall

INTENTS: list[tuple[str, str, list[tuple[str, dict]]]] = [
    # (intent, regex, tools)
    ("overview", r"\b(specs?|specifications|overview|summary|everything|tell me about (my|this) (machine|computer|laptop|pc|system))\b",
     [("get_system_summary", {}), ("get_cpu_info", {}), ("get_memory_usage", {}), ("get_memory_hardware", {}),
      ("get_storage_devices", {}), ("get_gpu_info", {})]),
    ("ram_upgrade", r"\b(upgrad\w*|add (more )?(ram|memory)|more (ram|memory)|max(imum)? (ram|memory)|slots?|sticks?|"
                    r"dimms?|sodimm|soldered|expand\w*|how much (ram|memory) can)\b",
     [("get_memory_hardware", {}), ("get_memory_usage", {})]),
    ("ram", r"\b(ram|memory)\b", [("get_memory_usage", {}), ("get_memory_hardware", {})]),
    ("disk_space", r"\b(space|free|full|left|usage|used)\b.*\b(disk|drive|storage|ssd|partition|filesystem)s?\b|"
                   r"\b(disk|drive|storage|ssd)s?\b.*\b(space|free|full|left)\b|\bdisk space\b",
     [("get_filesystem_usage", {})]),
    ("storage", r"\b(ssd|nvme|hdd|hard (disk|drive)|disks?|drives?|storage|sata|m\.2)\b",
     [("get_storage_devices", {})]),
    ("cpu", r"\b(cpu|processor|cores?|threads?|ghz|clock|avx\w*|hyper-?threading)\b", [("get_cpu_info", {})]),
    ("gpu", r"\b(gpu|graphics|video card|nvidia|radeon|vram|geforce|display adapter)\b", [("get_gpu_info", {})]),
    ("battery", r"\b(battery|batteries|charg\w*|cycle count|unplugged)\b", [("get_battery_status", {})]),
    ("temps", r"\b(temp\w*|hot|heat\w*|thermal|overheat\w*|cooling)\b", [("get_temperatures", {})]),
    ("network", r"\b(wi-?fi|wireless|ethernet|network|lan|nic|internet adapter|bluetooth)\b",
     [("get_network_interfaces", {}), ("get_pci_devices", {"category": "network"})]),
    ("usb", r"\b(usb|webcam|camera|peripherals?)\b", [("get_usb_devices", {})]),
    ("system", r"\b(model|make|manufacturer|brand|laptop|desktop|operating system|os|distro\w*|linux version|kernel|"
               r"bios|uefi|firmware|vm|virtual( machine)?|what (computer|machine))\b",
     [("get_system_summary", {})]),
]
FOLLOW_UP = re.compile(r"\b(it|its|that|this|they|them|those|what about|and|how (fast|big|much|many)|which one|"
                       r"more|else|also)\b", re.I)
UPGRADE_WORDS = re.compile(r"\b(upgrad\w*|add|more|expand\w*|max\w*|bigger|extra)\b", re.I)
CHANGE_REQUEST = re.compile(r"\b(install|delete|remove|format|overclock|change|set|enable|disable|update|kill|rm)\b", re.I)


def detect_intents(question: str) -> list[str]:
    q = question.lower()
    found = [name for name, rx, _ in INTENTS if re.search(rx, q)]
    if "ram_upgrade" in found and "ram" in found:
        found.remove("ram")
    if "overview" in found:
        found = ["overview"]
    if "disk_space" in found and "storage" in found and not re.search(r"\b(nvme|ssd|hdd|type)\b", q):
        found.remove("storage")
    return found


class OfflineBackend:
    name = "offline"
    is_local = True
    is_llm = False

    def __init__(self):
        self.last_intents: list[str] = []
        self.pending: list[str] = []

    # ------------------------------------------------------------------ step
    def step(self, messages: list[dict], tools: list) -> Step:
        available = {t.name for t in tools}
        last = messages[-1]
        if last["role"] == "user":
            q = last["content"]
            intents = detect_intents(q)
            if not intents and self.last_intents:
                # elliptical follow-ups inherit the previous topic: "Can I add more?" after a RAM question
                if UPGRADE_WORDS.search(q) and set(self.last_intents) & {"ram", "ram_upgrade"}:
                    intents = ["ram_upgrade"]
                elif FOLLOW_UP.search(q) or len(q.split()) <= 5:
                    intents = list(self.last_intents)
            if not intents:
                if CHANGE_REQUEST.search(q):
                    return Step(text="I'm a read-only assistant: I can inspect your hardware but not change, "
                                     "install or configure anything.")
                return Step(text="I can answer questions about this computer's CPU, RAM (including upgrades), "
                                 "storage and disk space, GPU, battery, temperatures, network adapters, USB devices "
                                 "and system/OS details. What would you like to know?")
            self.pending = intents
            self.last_intents = intents
            calls, seen = [], set()
            for name, _, tl in INTENTS:
                if name in intents:
                    for tool, args in tl:
                        key = (tool, json.dumps(args, sort_keys=True))
                        if tool in available and key not in seen:
                            seen.add(key)
                            calls.append(ToolCall(uuid.uuid4().hex[:12], tool, args))
            if calls:
                return Step(tool_calls=calls)
            return Step(text="The tools needed for that question aren't available on this system.")
        # last message is a tool result -> compose the final answer from this turn's results
        results: dict[str, dict] = {}
        question = ""
        for m in reversed(messages):
            if m["role"] == "tool":
                results.setdefault(m["name"], json.loads(m["content"]))
            elif m["role"] == "user":
                question = m["content"]
                break
        return Step(text=compose(self.pending, results, question))


# ====================================================================== composition
def _gaps(r: dict | None) -> list[str]:
    out = []
    for g in (r or {}).get("unavailable", []):
        s = f"I can't determine the {g['what']}: {g['reason']}."
        if g.get("how_to_fix"):
            s += f" To fix: {g['how_to_fix'].rstrip('.')}."
        out.append(s)
    return out


def compose(intents: list[str], R: dict[str, dict], q: str) -> str:
    parts: list[str] = []
    for intent in intents:
        fn = COMPOSERS.get(intent)
        if fn:
            text = fn(R, q.lower())
            if text and text not in parts:
                parts.append(text)
    return "\n\n".join(parts) or "I couldn't find information for that question."


def _d(R, name):
    return (R.get(name) or {}).get("data") or {}


def c_ram(R, q):
    u, h = _d(R, "get_memory_usage"), _d(R, "get_memory_hardware")
    lines = []
    if h.get("slots"):
        spec = " ".join(filter(None, [h.get("memory_type"), h.get("memory_speed")]))
        lines.append(f"You have {h['installed']} of RAM installed{f' ({spec})' if spec else ''}, "
                     f"using {h['used_slots']} of {h['total_slots']} slots.")
    if u.get("total"):
        lines.append(f"The OS sees {u['total']} usable; {u['used']} is in use and {u['available']} is available "
                     f"({u['used_percent']}% used).")
    if not h.get("slots"):
        g = _gaps(R.get("get_memory_hardware"))
        if g and re.search(r"type|speed|ddr|slot|fast|install", q):
            lines += g
    return " ".join(lines)


def c_ram_upgrade(R, q):
    h = _d(R, "get_memory_hardware")
    if not h.get("slots"):
        out = _gaps(R.get("get_memory_hardware"))
        if h.get("usable_ram"):
            out.append(f"What I can see without that: {h['usable_ram']} of usable RAM.")
        return " ".join(out)
    v = h.get("upgrade_verdict")
    spec = " ".join(filter(None, [h.get("memory_type"), h.get("memory_speed")]))
    base = f"You have {h['installed']} in {h['used_slots']} of {h['total_slots']} slots"
    base += f" ({spec})." if spec else "."
    if v == "can_add_module":
        s = (f"{base} The board supports up to {h['max_capacity']}, so you can add a stick of up to "
             f"{h['largest_module_to_add']} in the free slot, for {h['total_after_adding_largest']} in total.")
        if spec:
            s += f" Use a matching {spec} SODIMM/DIMM for dual-channel performance."
    elif v == "replace_modules_to_upgrade":
        s = (f"{base} All slots are full, but the board supports up to {h['max_capacity']}: to upgrade you'd "
             f"replace modules (up to about {h['max_per_slot_estimate']} each).")
    elif v == "not_upgradeable_soldered":
        s = f"{base} The memory is soldered to the motherboard, so it can't be upgraded."
    elif v == "at_max_capacity":
        s = f"{base} That's already the maximum the board supports ({h['max_capacity']})."
    else:
        s = f"{base} The firmware doesn't report a maximum capacity, so I can't tell how far it can be upgraded."
    if v in ("can_add_module", "replace_modules_to_upgrade"):
        s += " (The maximum comes from firmware and is occasionally understated; the service manual is authoritative.)"
    return s


def c_cpu(R, q):
    c = _d(R, "get_cpu_info")
    if not c:
        return " ".join(_gaps(R.get("get_cpu_info")))
    model = c.get("model_name") or "processor of unknown model"
    s = f"Your CPU is {'an' if model[:1].lower() in 'aeiou' else 'a'} {model}"
    cores, threads = c.get("physical_cores"), c.get("logical_cpus")
    if cores and threads:
        s += f" with {cores} physical cores and {threads} threads"
    elif threads:
        s += f" with {threads} logical CPUs"
    if c.get("max_ghz"):
        s += f", up to {c['max_ghz']} GHz"
    elif c.get("base_ghz_from_model_name"):
        s += f", rated {c['base_ghz_from_model_name']} GHz"
    s += "."
    if re.search(r"avx|instruction|flag", q):
        s += " Notable instruction sets: " + (", ".join(c.get("notable_flags") or []) or "none detected") + "."
    if re.search(r"virtuali", q):
        s += f" Hardware virtualization: {c.get('virtualization_support') or 'not reported'}."
    if c.get("vm_note"):
        s += " " + c["vm_note"]
    return s


def c_storage(R, q):
    d = _d(R, "get_storage_devices")
    disks = d.get("disks", [])
    if not disks:
        return "I couldn't find any drives. " + " ".join(_gaps(R.get("get_storage_devices")))
    desc = "; ".join(f"{x['name']}: {x['size']} {x['kind']}" + (f" ({x['model']})" if x.get("model") else "")
                     for x in disks)
    s = ""
    if "nvme" in q:
        nv = [x for x in disks if x["is_nvme"]]
        s = (f"Yes, {nv[0]['name']} is an NVMe SSD. " if len(nv) == 1 else
             f"Yes, {len(nv)} drives are NVMe. " if nv else "No, none of your drives is NVMe. ")
    elif re.search(r"\bssd\b", q):
        ss = [x for x in disks if x["is_ssd"]]
        s = "Yes, you have an SSD. " if ss else "No SSD detected. "
    elif re.search(r"\bhdd\b|hard (disk|drive)", q):
        hd = [x for x in disks if x["kind"].startswith("HDD")]
        s = "Yes, you have a spinning hard drive. " if hd else "No spinning hard drive detected. "
    if d.get("vm_note") and re.search(r"nvme|ssd|hdd|hard", q):
        s = "I can't tell: this is a virtual machine. "
    s += f"Drives: {desc}."
    if d.get("vm_note"):
        s += " " + d["vm_note"]
    return s


def c_disk_space(R, q):
    fss = _d(R, "get_filesystem_usage").get("filesystems", [])
    if not fss:
        return "I couldn't read filesystem usage."
    return "Disk space: " + "; ".join(f"{f['mountpoint']} has {f['free']} free of {f['size']} ({f['used_percent']}% used)"
                                      for f in fss[:6]) + "."


def c_gpu(R, q):
    g = _d(R, "get_gpu_info")
    gpus = g.get("gpus", [])
    if not gpus:
        return " ".join(_gaps(R.get("get_gpu_info"))) or "No GPU found."
    lines = []
    for x in gpus:
        s = f"{x['name']}"
        if x.get("type"):
            s += f" ({x['type']})"
        if x.get("vram"):
            s += f" with {x['vram']} VRAM"
        if x.get("driver"):
            s += f", using the {x['driver']} driver"
        lines.append(s)
    out = ("You have " + str(len(gpus)) + " GPUs: " if len(gpus) > 1 else "Your GPU: ") + "; ".join(lines) + "."
    gaps = _gaps(R.get("get_gpu_info"))
    return out + (" " + " ".join(gaps) if gaps else "")


def c_battery(R, q):
    b = _d(R, "get_battery_status")
    if not b.get("batteries"):
        return " ".join(_gaps(R.get("get_battery_status")))
    out = []
    for x in b["batteries"]:
        s = f"Battery {x['name']}: {x['charge_percent']}% charged, {str(x['status']).lower()}"
        if x.get("health_percent") is not None:
            s += f". Health is {x['health_percent']}% of design capacity ({x['full_capacity']} of {x['design_capacity']})"
        if x.get("cycle_count"):
            s += f", {x['cycle_count']} charge cycles"
        if x.get("estimated_hours_remaining"):
            s += f", about {x['estimated_hours_remaining']} hours remaining at the current draw"
        out.append(s + ".")
    if b.get("ac_connected") is not None:
        out.append("AC adapter is " + ("connected." if b["ac_connected"] else "not connected."))
    return " ".join(out)


def c_temps(R, q):
    t = _d(R, "get_temperatures").get("readings", [])
    if not t:
        return " ".join(_gaps(R.get("get_temperatures")))
    hot = max(t, key=lambda x: x["celsius"])
    return ("Temperatures: " + "; ".join(f"{x['sensor']} {x['celsius']} °C" for x in t[:8])
            + f". Hottest: {hot['sensor']} at {hot['celsius']} °C.")


def c_network(R, q):
    ifs = [i for i in _d(R, "get_network_interfaces").get("interfaces", []) if i["kind"] not in ("loopback", "virtual")]
    pci = _d(R, "get_pci_devices").get("devices", [])
    out = []
    if ifs:
        out.append("Network interfaces: " + "; ".join(
            f"{i['name']} ({i['kind']}, {i['state']}" + (f", {i['link_speed_mbps']} Mb/s" if i.get("link_speed_mbps") else "")
            + (f", driver {i['driver']}" if i.get("driver") else "") + ")" for i in ifs) + ".")
    if pci:
        out.append("Network hardware: " + "; ".join(f"{d['vendor']} {d['device']}" for d in pci) + ".")
    if re.search(r"wi-?fi|wireless", q):
        out.insert(0, "Yes, you have Wi-Fi." if any(i["kind"] == "Wi-Fi" for i in ifs) else "No Wi-Fi adapter detected.")
    return " ".join(out) or "No physical network interfaces found."


def c_usb(R, q):
    d = _d(R, "get_usb_devices").get("devices", [])
    if not d:
        return " ".join(_gaps(R.get("get_usb_devices")))
    return f"{len(d)} USB devices: " + "; ".join(x.get("description") or f"{x['vendor_id']}:{x['product_id']}" for x in d) + "."


def c_system(R, q):
    s = _d(R, "get_system_summary")
    if not s:
        return ""
    who = " ".join(filter(None, [s.get("manufacturer"), s.get("product_name")]))
    out = f"This is a {who}" if who else "This computer is"
    if s.get("form_factor"):
        out += f" ({s['form_factor'].lower()})"
    osname = (s.get("os") or {}).get("name")
    out += f" running {osname or 'Linux'}, kernel {s.get('kernel')} on {s.get('architecture')}."
    if s.get("virtual_machine"):
        out += " It is a virtual machine" + (f" ({s['hypervisor']})" if s.get("hypervisor") else "") + "."
    fw = s.get("firmware") or {}
    if fw.get("version") and re.search(r"bios|uefi|firmware", q):
        out += f" Firmware: {fw.get('vendor')} {fw['version']} ({fw.get('date')}), boot mode {s.get('boot_mode')}."
    if not who and re.search(r"model|make|manufacturer|brand|laptop|computer|machine|\bpc\b", q):
        out += " " + " ".join(_gaps(R.get("get_system_summary")))
    return out


def c_overview(R, q):
    parts = [c_system(R, q), c_cpu(R, q), c_ram(R, ""), c_storage(R, ""), c_gpu(R, "")]
    return "\n".join(p for p in parts if p)


COMPOSERS = {"overview": c_overview, "ram_upgrade": c_ram_upgrade, "ram": c_ram, "disk_space": c_disk_space,
             "storage": c_storage, "cpu": c_cpu, "gpu": c_gpu, "battery": c_battery, "temps": c_temps,
             "network": c_network, "usb": c_usb, "system": c_system}
