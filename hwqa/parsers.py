"""Pure parsing/normalisation functions: raw command output or file text -> structured dicts.

They are side-effect free, so every one is unit-tested against captured real-world output
(tests/fixtures). Fields are *whitelisted*: identifiers such as serial numbers, asset tags, UUIDs
and MAC addresses are never extracted, so they can never reach an LLM (privacy by construction).
"""
from __future__ import annotations

import json
import re

GIB = 1024 ** 3
MIB = 1024 ** 2

# ----------------------------------------------------------------------------- units


def human_bytes(n: int | float | None, binary: bool = True) -> str | None:
    """binary=True  -> GiB-based but labelled the way people (and RAM vendors) say it: '16 GB'.
    binary=False -> decimal, the way disk vendors label capacity: '512 GB', '1 TB'."""
    if n is None:
        return None
    base = 1024 if binary else 1000
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    v = float(n)
    i = 0
    while v >= base and i < len(units) - 1:
        v /= base
        i += 1
    s = f"{v:.1f}".rstrip("0").rstrip(".") if v < 100 else f"{v:.0f}"
    return f"{s} {units[i]}"


_SIZE_RE = re.compile(r"^\s*([\d.]+)\s*(B|bytes|kB|KB|KiB|MB|MiB|GB|GiB|TB|TiB)\s*$", re.I)
_MULT = {"b": 1, "bytes": 1, "kb": 1024, "kib": 1024, "mb": MIB, "mib": MIB, "gb": GIB, "gib": GIB,
         "tb": GIB * 1024, "tib": GIB * 1024}


def parse_size(text: str | None) -> int | None:
    """dmidecode sizes ('16 GB', '8192 MB') are binary units. Returns bytes or None."""
    if not text:
        return None
    m = _SIZE_RE.match(text)
    if not m:
        return None
    return int(float(m.group(1)) * _MULT[m.group(2).lower()])


def _clean(v: str | None) -> str | None:
    if v is None:
        return None
    v = v.strip()
    if v.lower() in ("", "unknown", "not specified", "not provided", "none", "to be filled by o.e.m.",
                     "default string", "null", "n/a", "[not supported]"):
        return None
    return v


# ----------------------------------------------------------------------------- CPU


def parse_lscpu_json(text: str) -> dict:
    data = json.loads(text)
    flat: dict[str, str] = {}

    def walk(items):
        for it in items:
            flat[it["field"].rstrip(":").strip()] = it.get("data")
            walk(it.get("children", []))

    walk(data.get("lscpu", []))
    return normalize_cpu(flat)


def parse_cpuinfo(text: str) -> dict:
    """Fallback when lscpu is missing: /proc/cpuinfo."""
    procs = [b for b in text.strip().split("\n\n") if b.strip()]
    first: dict[str, str] = {}
    cores = set()
    sockets = set()
    for b in procs:
        kv = {}
        for line in b.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                kv[k.strip()] = v.strip()
        if not first:
            first = kv
        sockets.add(kv.get("physical id", "0"))
        cores.add((kv.get("physical id", "0"), kv.get("core id", kv.get("processor"))))
    flat = {
        "Architecture": None,
        "Model name": first.get("model name") or first.get("Model") or first.get("cpu model"),
        "Vendor ID": first.get("vendor_id"),
        "CPU(s)": str(len(procs)),
        "Socket(s)": str(len(sockets)),
        "Flags": first.get("flags"),
        "CPU max MHz": None,
    }
    out = normalize_cpu(flat)
    out["physical_cores"] = len(cores) if "core id" in first else None
    out["source_note"] = "parsed from /proc/cpuinfo (lscpu unavailable)"
    return out


INTERESTING_FLAGS = ["avx", "avx2", "avx512f", "aes", "sha_ni", "sse4_2", "vmx", "svm", "hypervisor"]


def _int(v) -> int | None:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def _float(v) -> float | None:
    try:
        return float(str(v).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None


def normalize_cpu(f: dict) -> dict:
    sockets = _int(f.get("Socket(s)")) or None
    cps = _int(f.get("Core(s) per socket"))
    tpc = _int(f.get("Thread(s) per core"))
    logical = _int(f.get("CPU(s)"))
    flags = set((f.get("Flags") or "").split())
    caches = {}
    for k in ("L1d cache", "L1i cache", "L2 cache", "L3 cache", "L1d", "L1i", "L2", "L3"):
        if f.get(k):
            caches[k.replace(" cache", "")] = f[k]
    out = {
        "model_name": _clean(f.get("Model name")),
        "vendor": _clean(f.get("Vendor ID")),
        "architecture": _clean(f.get("Architecture")),
        "sockets": sockets,
        "cores_per_socket": cps,
        "physical_cores": (cps * sockets) if (cps and sockets) else None,
        "threads_per_core": tpc,
        "logical_cpus": logical,
        "max_mhz": _float(f.get("CPU max MHz")),
        "min_mhz": _float(f.get("CPU min MHz")),
        "caches": caches,
        "virtualization_support": _clean(f.get("Virtualization")),
        "hypervisor_vendor": _clean(f.get("Hypervisor vendor")),
        "running_in_vm": bool(f.get("Hypervisor vendor")) or "hypervisor" in flags,
        "notable_flags": [x for x in INTERESTING_FLAGS if x in flags],
    }
    if out["max_mhz"]:
        out["max_ghz"] = round(out["max_mhz"] / 1000, 2)
    else:
        m = re.search(r"@\s*([\d.]+)\s*GHz", out["model_name"] or "")
        if m:
            out["base_ghz_from_model_name"] = float(m.group(1))
    return out


# ----------------------------------------------------------------------------- memory


def parse_meminfo(text: str) -> dict:
    kv = {}
    for line in text.splitlines():
        m = re.match(r"(\w+):\s+(\d+)\s*kB", line)
        if m:
            kv[m.group(1)] = int(m.group(2)) * 1024
    total = kv.get("MemTotal")
    avail = kv.get("MemAvailable")
    used = total - avail if total is not None and avail is not None else None
    return {
        "total_bytes": total, "total": human_bytes(total),
        "available_bytes": avail, "available": human_bytes(avail),
        "used_bytes": used, "used": human_bytes(used),
        "used_percent": round(100 * used / total, 1) if used is not None and total else None,
        "swap_total": human_bytes(kv.get("SwapTotal")), "swap_free": human_bytes(kv.get("SwapFree")),
        "note": "MemTotal is usable RAM after firmware/GPU reservations, so it is a bit below the installed size.",
    }


def _dmi_sections(text: str) -> list[tuple[int, str, dict]]:
    out = []
    for block in re.split(r"\n(?=Handle 0x)", text):
        m = re.match(r"Handle (0x[0-9A-Fa-f]+), DMI type (\d+)", block.strip())
        if not m:
            continue
        kv = {}
        for line in block.splitlines()[2:]:
            if ":" in line and line.startswith("\t") and not line.startswith("\t\t"):
                k, v = line.strip().split(":", 1)
                kv[k.strip()] = v.strip()
        out.append((int(m.group(2)), m.group(1), kv))
    return out


def parse_dmidecode_memory(text: str) -> dict:
    sections = _dmi_sections(text)
    arrays = {h: kv for t, h, kv in sections if t == 16 and kv.get("Use", "System Memory") == "System Memory"}
    slots = []
    for t, h, kv in sections:
        if t != 17 or (arrays and kv.get("Array Handle") not in arrays):
            continue
        size = parse_size(kv.get("Size"))
        ff = _clean(kv.get("Form Factor"))
        locator = _clean(kv.get("Locator"))
        slots.append({
            "locator": locator,
            "bank": _clean(kv.get("Bank Locator")),
            "installed": bool(size),
            "size_bytes": size,
            "size": human_bytes(size),
            "type": _clean(kv.get("Type")) if size else None,
            "speed": _clean(kv.get("Speed")) if size else None,
            "configured_speed": _clean(kv.get("Configured Memory Speed")) if size else None,
            "form_factor": ff,
            "manufacturer": _clean(kv.get("Manufacturer")) if size else None,
            "part_number": _clean(kv.get("Part Number")) if size else None,
            "soldered": bool(ff and ff.lower() == "row of chips")
                        or bool(locator and re.search(r"onboard|soldered", locator, re.I)),
            # Serial Number / Asset Tag deliberately not extracted.
        })
    max_cap = sum(parse_size(a.get("Maximum Capacity")) or 0 for a in arrays.values()) or None
    declared_slots = sum(_int(a.get("Number Of Devices")) or 0 for a in arrays.values()) or None
    return {
        "max_capacity_bytes": max_cap,
        "max_capacity": human_bytes(max_cap),
        "declared_slots": declared_slots,
        "error_correction": next((_clean(a.get("Error Correction Type")) for a in arrays.values()), None),
        "slots": slots,
        **upgrade_analysis(slots, max_cap),
    }


def upgrade_analysis(slots: list[dict], max_cap: int | None) -> dict:
    """All arithmetic happens here in code, so the LLM never has to compute (or invent) numbers."""
    total = len(slots)
    used = [s for s in slots if s["installed"]]
    installed = sum(s["size_bytes"] for s in used)
    free = total - len(used)
    soldered = [s for s in used if s["soldered"]]
    a: dict = {
        "total_slots": total,
        "used_slots": len(used),
        "free_slots": free,
        "installed_bytes": installed,
        "installed": human_bytes(installed),
        "all_memory_soldered": bool(used) and len(soldered) == len(used),
        "some_memory_soldered": bool(soldered),
        "memory_type": next((s["type"] for s in used if s["type"]), None),
        "memory_speed": next((s["speed"] for s in used if s["speed"]), None),
    }
    if max_cap and total:
        per_slot = max_cap // total
        headroom = max(0, max_cap - installed)
        a["max_per_slot_estimate_bytes"] = per_slot
        a["max_per_slot_estimate"] = human_bytes(per_slot)
        a["headroom_bytes"] = headroom
        a["headroom"] = human_bytes(headroom)
        if a["all_memory_soldered"]:
            verdict = "not_upgradeable_soldered"
            add = 0
        elif free > 0:
            add = min(per_slot, headroom) * 1  # largest single module for one free slot
            verdict = "can_add_module" if add > 0 else "at_max_capacity"
        else:
            add = 0
            verdict = "replace_modules_to_upgrade" if headroom > 0 else "at_max_capacity"
        a["upgrade_verdict"] = verdict
        a["largest_module_to_add_bytes"] = add
        a["largest_module_to_add"] = human_bytes(add) if add else None
        a["total_after_adding_largest"] = human_bytes(installed + add) if add else None
        a["max_total_if_all_slots_filled"] = human_bytes(min(max_cap, per_slot * total))
        a["caveats"] = [
            "Per-slot maximum is estimated as board maximum / number of slots.",
            "'Maximum Capacity' comes from firmware (DMI) and is sometimes understated or outdated; "
            "the laptop's service manual is authoritative.",
            "For dual-channel performance, matching module sizes and speeds is recommended.",
        ]
    else:
        a["upgrade_verdict"] = "unknown_max_capacity"
    return a


# ----------------------------------------------------------------------------- storage


def _kind(dev: dict) -> str:
    name = dev.get("name") or ""
    tran = (dev.get("tran") or "").lower()
    if name.startswith(("zram", "ram")):
        return "zram (compressed RAM swap)"
    if name.startswith("loop"):
        return "loop (file-backed)"
    if tran == "nvme" or name.startswith("nvme"):
        return "NVMe SSD"
    if tran == "usb" or dev.get("rm") or dev.get("hotplug"):
        return "USB/removable drive"
    if tran in ("virtio",) or name.startswith(("vd", "xvd")):
        return "virtual disk (VM)"
    if name.startswith("mmcblk"):
        return "eMMC/SD card"
    rota = dev.get("rota")
    if rota in (True, "1", 1):
        return "HDD (spinning)"
    if rota in (False, "0", 0):
        return "SSD (SATA)" if tran == "sata" else "SSD"
    return "unknown"


VIRTUAL_PREFIXES = ("zram", "loop", "ram")


def parse_lsblk_json(text: str, include_virtual: bool = False) -> dict:
    data = json.loads(text)
    disks = []
    for dev in data.get("blockdevices", []):
        if dev.get("type") not in ("disk", "rom"):
            continue
        name = dev.get("name", "")
        if not include_virtual and (name.startswith(VIRTUAL_PREFIXES) or (dev.get("size") in (0, "0"))):
            continue
        size = _int(dev.get("size"))
        parts = []
        for ch in dev.get("children", []) or []:
            mps = [m for m in (ch.get("mountpoints") or [ch.get("mountpoint")]) if m]
            parts.append({"name": ch.get("name"), "size": human_bytes(_int(ch.get("size")), binary=False),
                          "fstype": ch.get("fstype"), "mountpoints": mps})
        mps = [m for m in (dev.get("mountpoints") or [dev.get("mountpoint")]) if m]
        disks.append({
            "name": name,
            "kind": _kind(dev),
            "is_nvme": _kind(dev) == "NVMe SSD",
            "is_ssd": _kind(dev) in ("NVMe SSD", "SSD", "SSD (SATA)"),
            "interface": dev.get("tran"),
            "size_bytes": size,
            "size": human_bytes(size, binary=False),
            "size_binary": human_bytes(size),
            "model": _clean(dev.get("model")),
            "vendor": _clean(dev.get("vendor")),
            "removable": bool(dev.get("rm")) or bool(dev.get("hotplug")),
            "mountpoints": mps,
            "partitions": parts,
        })
    return {"disks": disks, "count": len(disks),
            "note": "Disk sizes use decimal units like drive labels (1 GB = 10^9 bytes)."}


# ----------------------------------------------------------------------------- PCI / GPU / USB

_LSPCI_LINE = re.compile(r'^(\S+)\s+"([^"]*)"\s+"([^"]*)"\s+"([^"]*)"(.*)$')
_ID = re.compile(r"^(.*?)\s*\[([0-9a-fA-F]{4})\]$")


def _split_id(s: str) -> tuple[str, str | None]:
    m = _ID.match(s.strip())
    return (m.group(1), m.group(2).lower()) if m else (s.strip(), None)


def parse_lspci_mm(text: str) -> list[dict]:
    devs = []
    for line in text.splitlines():
        m = _LSPCI_LINE.match(line.strip())
        if not m:
            continue
        cls, cls_id = _split_id(m.group(2))
        vendor, vendor_id = _split_id(m.group(3))
        device, device_id = _split_id(m.group(4))
        devs.append({"slot": m.group(1), "class": cls, "class_id": cls_id, "vendor": vendor,
                     "vendor_id": vendor_id, "device": device, "device_id": device_id})
    return devs


PCI_CATEGORIES = {
    "gpu": ("0300", "0302", "0380"),
    "network": ("0200", "0280"),
    "audio": ("0401", "0403"),
    "usb_controller": ("0c03",),
    "storage_controller": ("0100", "0101", "0104", "0106", "0107", "0108"),
}


def pci_category(class_id: str | None) -> str:
    for cat, ids in PCI_CATEGORIES.items():
        if class_id and class_id[:4] in ids:
            return cat
    return "other"


def parse_pci_ids(text: str) -> dict[str, tuple[str, dict[str, str]]]:
    """pci.ids database -> {vendor_id: (vendor_name, {device_id: device_name})}."""
    db: dict[str, tuple[str, dict[str, str]]] = {}
    cur = None
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("C "):
            break  # class section
        if not line.startswith("\t"):
            vid, _, name = line.partition("  ")
            cur = vid.strip().lower()
            db[cur] = (name.strip(), {})
        elif not line.startswith("\t\t") and cur:
            did, _, name = line.strip().partition("  ")
            db[cur][1][did.strip().lower()] = name.strip()
    return db


def parse_nvidia_smi(text: str) -> list[dict]:
    out = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            mib = _int(parts[1])
            out.append({"name": parts[0], "vram_bytes": mib * MIB if mib else None,
                        "vram": human_bytes(mib * MIB) if mib else None, "driver_version": parts[2],
                        "pci_bus_id": parts[3] if len(parts) > 3 else None})
    return out


_LSUSB = re.compile(r"^Bus (\d+) Device (\d+): ID ([0-9a-f]{4}):([0-9a-f]{4})\s*(.*)$")


def parse_lsusb(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        m = _LSUSB.match(line.strip())
        if m:
            desc = m.group(5).strip()
            out.append({"bus": m.group(1), "vendor_id": m.group(3), "product_id": m.group(4),
                        "description": desc or None, "is_root_hub": "root hub" in desc.lower()})
    return out


# ----------------------------------------------------------------------------- misc


def parse_os_release(text: str) -> dict:
    kv = {}
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip().strip('"')
    return {"name": kv.get("PRETTY_NAME") or kv.get("NAME"), "id": kv.get("ID"), "version": kv.get("VERSION_ID")}


def parse_uevent(text: str) -> dict:
    kv = {}
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.replace("POWER_SUPPLY_", "").lower()] = v.strip()
    return kv


def normalize_battery(kv: dict) -> dict:
    full = _int(kv.get("energy_full") or kv.get("charge_full"))
    design = _int(kv.get("energy_full_design") or kv.get("charge_full_design"))
    unit = "Wh" if kv.get("energy_full") else "Ah"
    health = round(100 * full / design, 1) if full and design else None
    power = _int(kv.get("power_now"))
    now = _int(kv.get("energy_now"))
    out = {
        "name": kv.get("name"),
        "status": kv.get("status"),
        "charge_percent": _int(kv.get("capacity")),
        "health_percent": health,
        "full_capacity": f"{full / 1e6:.1f} {unit}" if full else None,
        "design_capacity": f"{design / 1e6:.1f} {unit}" if design else None,
        "cycle_count": _int(kv.get("cycle_count")) or None,
        "technology": _clean(kv.get("technology")),
        "manufacturer": _clean(kv.get("manufacturer")),
        "model": _clean(kv.get("model_name")),
        # serial_number deliberately dropped
    }
    if power and now and kv.get("status") == "Discharging":
        out["estimated_hours_remaining"] = round(now / power, 1)
    return out
