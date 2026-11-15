"""Tool registry: the complete list of things the assistant can look at.

The LLM sees each tool's name, description and a strict JSON schema, and can do nothing else.
Every tool returns the same envelope:

    {"tool": ..., "ok": bool, "data": {...},
     "sources": ["lscpu -J", "/proc/meminfo", ...],          # provenance, shown to the user
     "unavailable": [{"what", "reason", "how_to_fix"}],      # explicit gaps -> "I don't know" answers
     "collected_at": "..."}

Each tool tries its best data source first and degrades gracefully (portability):
    lscpu -J -> /proc/cpuinfo
    lsblk -J -> /sys/block
    lspci    -> /sys/bus/pci/devices (+ pci.ids names when installed)
    lsusb    -> /sys/bus/usb/devices
    dmidecode (root) -> privileged snapshot -> sudo -n (opt-in) -> partial answer from /proc/meminfo
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from . import parsers as P
from .executor import Executor

SNAPSHOT_PATH = os.environ.get("HWQA_SNAPSHOT", "/var/lib/hwqa/dmi-memory.txt")

CHASSIS_TYPES = {
    "1": "Other", "2": "Unknown", "3": "Desktop", "4": "Low Profile Desktop", "6": "Mini Tower",
    "7": "Tower", "8": "Portable", "9": "Laptop", "10": "Notebook", "11": "Hand Held", "13": "All in One",
    "14": "Sub Notebook", "17": "Main Server Chassis", "23": "Rack Mount Chassis", "30": "Tablet",
    "31": "Convertible", "32": "Detachable", "35": "Mini PC",
}


@dataclass
class ToolResult:
    tool: str
    data: dict = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    unavailable: list[dict] = field(default_factory=list)
    ok: bool = True

    def gap(self, what: str, reason: str, how_to_fix: str | None = None) -> None:
        self.unavailable.append({"what": what, "reason": reason, **({"how_to_fix": how_to_fix} if how_to_fix else {})})

    def to_dict(self) -> dict:
        return {"tool": self.tool, "ok": self.ok, "data": self.data, "sources": self.sources,
                "unavailable": self.unavailable,
                "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., ToolResult]
    properties: dict = field(default_factory=dict)
    ttl_s: float = 3600.0            # cache lifetime: static hardware long, live readings short
    probes: tuple[str, ...] = ()     # binaries or paths; tool is offered only if at least one exists

    @property
    def input_schema(self) -> dict:
        return {"type": "object", "properties": self.properties, "required": [],
                "additionalProperties": False}


class HardwareTools:
    def __init__(self, executor: Executor):
        self.x = executor
        self._pci_names: dict | None = None
        self.tools: dict[str, Tool] = {t.name: t for t in self._define()}

    # ======================================================================== registry
    def _define(self) -> list[Tool]:
        return [
            Tool("get_system_summary",
                 "Computer make/model, form factor (laptop/desktop), OS distribution, kernel, CPU architecture, "
                 "firmware (BIOS/UEFI) version, boot mode, and whether this is a virtual machine.",
                 self.system_summary, probes=("/proc/version",)),
            Tool("get_cpu_info",
                 "Processor model, vendor, physical cores, threads, sockets, max clock speed, caches, "
                 "virtualization support and notable instruction-set flags (AVX2, AVX-512, AES...).",
                 self.cpu_info, probes=("lscpu", "/proc/cpuinfo")),
            Tool("get_memory_usage",
                 "Current RAM usage from the OS: total usable, used, available, swap. Live values; no root needed.",
                 self.memory_usage, ttl_s=10, probes=("/proc/meminfo",)),
            Tool("get_memory_hardware",
                 "Physical RAM hardware from firmware: number of slots, which are used/free, size/type/speed of "
                 "each module, whether RAM is soldered, the board's maximum supported capacity, and a precomputed "
                 "upgrade analysis. Use for questions about upgrading or adding RAM.",
                 self.memory_hardware, probes=("/proc/meminfo",)),
            Tool("get_storage_devices",
                 "Physical drives: type (NVMe SSD, SATA SSD, HDD, USB), interface, capacity, model, partitions "
                 "and mount points.",
                 self.storage_devices,
                 properties={"include_virtual": {"type": "boolean",
                                                 "description": "Also list zram/loop devices. Default false."}},
                 probes=("lsblk", "/sys/block")),
            Tool("get_filesystem_usage",
                 "Disk space per mounted filesystem: size, used, free, percent used. Live values.",
                 self.filesystem_usage, ttl_s=15, probes=("/proc/mounts",)),
            Tool("get_gpu_info",
                 "Graphics hardware: GPU models, vendor, integrated vs discrete, kernel driver in use, VRAM when "
                 "the driver exposes it (NVIDIA via nvidia-smi, AMD via sysfs).",
                 self.gpu_info, probes=("lspci", "/sys/bus/pci/devices")),
            Tool("get_pci_devices",
                 "PCI devices filtered by category (network adapters, audio, USB controllers, storage controllers).",
                 self.pci_devices,
                 properties={"category": {"type": "string",
                                          "enum": ["gpu", "network", "audio", "usb_controller",
                                                   "storage_controller", "other", "all"],
                                          "description": "Device category to list."}},
                 probes=("lspci", "/sys/bus/pci/devices")),
            Tool("get_network_interfaces",
                 "Network interfaces: Wi-Fi vs Ethernet vs virtual, link state, link speed, kernel driver. "
                 "(MAC addresses are never collected.)",
                 self.network_interfaces, ttl_s=15, probes=("/sys/class/net",)),
            Tool("get_battery_status",
                 "Battery charge, charging status, health (current full capacity vs design), cycle count, "
                 "and whether AC power is connected.",
                 self.battery_status, ttl_s=10, probes=("/sys/class/power_supply",)),
            Tool("get_temperatures",
                 "Current temperature sensor readings (CPU package, NVMe, ACPI zones...) in °C.",
                 self.temperatures, ttl_s=5, probes=("/sys/class/thermal", "/sys/class/hwmon")),
            Tool("get_usb_devices",
                 "Connected USB devices (webcams, receivers, Bluetooth, storage...).",
                 self.usb_devices, ttl_s=15, probes=("lsusb", "/sys/bus/usb/devices")),
        ]

    def available(self) -> dict[str, Tool]:
        out = {}
        for name, t in self.tools.items():
            if not t.probes or any(self._probe(p) for p in t.probes):
                out[name] = t
        return out

    def capability_report(self) -> list[dict]:
        rows = []
        for name, t in self.tools.items():
            found = [p for p in t.probes if self._probe(p)]
            rows.append({"tool": name, "available": bool(found) or not t.probes, "sources_found": found,
                         "sources_missing": [p for p in t.probes if p not in found]})
        return rows

    def _probe(self, p: str) -> bool:
        return self.x.exists(p) if p.startswith("/") else self.x.which(p) is not None

    def call(self, name: str, args: dict) -> dict:
        tool = self.tools[name]
        try:
            r = tool.func(**args)
        except Exception as e:  # a broken parser must never crash the chat
            r = ToolResult(name, ok=False)
            r.gap(name, f"internal error while collecting data: {type(e).__name__}: {e}")
        return r.to_dict()

    # ======================================================================== helpers
    def _sys(self, path: str) -> str | None:
        v = self.x.read_value(path)
        return P._clean(v) if v is not None else None

    def _pci_db(self) -> dict:
        if self._pci_names is None:
            self._pci_names = {}
            for p in ("/usr/share/misc/pci.ids", "/usr/share/hwdata/pci.ids"):
                text = self.x.read(p, 5_000_000)
                if text:
                    self._pci_names = P.parse_pci_ids(text)
                    break
        return self._pci_names

    # ======================================================================== tools
    def system_summary(self) -> ToolResult:
        r = ToolResult("get_system_summary")
        d = r.data
        dmi = "/sys/class/dmi/id/"
        if self.x.exists(dmi):
            d["manufacturer"] = self._sys(dmi + "sys_vendor")
            d["product_name"] = self._sys(dmi + "product_name")
            d["product_version"] = self._sys(dmi + "product_version")
            d["product_family"] = self._sys(dmi + "product_family")
            d["board"] = " ".join(filter(None, [self._sys(dmi + "board_vendor"), self._sys(dmi + "board_name")])) or None
            ct = self._sys(dmi + "chassis_type")
            d["form_factor"] = CHASSIS_TYPES.get(ct or "", None)
            d["firmware"] = {"vendor": self._sys(dmi + "bios_vendor"), "version": self._sys(dmi + "bios_version"),
                             "date": self._sys(dmi + "bios_date")}
            r.sources.append("/sys/class/dmi/id/*")
        else:
            r.gap("manufacturer, model, form factor, BIOS version",
                  "this system exposes no DMI/SMBIOS data (common in virtual machines and some ARM boards)")
        osr = self.x.read("/etc/os-release") or self.x.read("/usr/lib/os-release")
        if osr:
            d["os"] = P.parse_os_release(osr)
            r.sources.append("/etc/os-release")
        u = os.uname()
        d["kernel"] = self.x.read_value("/proc/sys/kernel/osrelease") or u.release
        d["architecture"] = u.machine
        d["boot_mode"] = "UEFI" if self.x.exists("/sys/firmware/efi") else "Legacy BIOS or not detectable"
        cpuinfo = self.x.read("/proc/cpuinfo") or ""
        hyp = self._sys("/sys/hypervisor/type")
        d["virtual_machine"] = bool(hyp) or " hypervisor" in cpuinfo
        if hyp:
            d["hypervisor"] = hyp
        up = self.x.read_value("/proc/uptime")
        if up:
            d["uptime_hours"] = round(float(up.split()[0]) / 3600, 1)
        r.sources += ["uname", "/proc/cpuinfo", "/proc/uptime"]
        return r

    def cpu_info(self) -> ToolResult:
        r = ToolResult("get_cpu_info")
        res = self.x.run("lscpu", "-J")
        if res.ok:
            r.data = P.parse_lscpu_json(res.stdout)
            r.sources.append("lscpu -J")
        else:
            text = self.x.read("/proc/cpuinfo")
            if not text:
                r.ok = False
                r.gap("CPU details", "neither lscpu nor /proc/cpuinfo is available")
                return r
            r.data = P.parse_cpuinfo(text)
            r.sources.append("/proc/cpuinfo")
            r.gap("clock speeds, caches", "lscpu is not installed", "install util-linux")
        freq = self._sys("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
        if freq:
            r.data["current_mhz_cpu0"] = round(int(freq) / 1000)
            r.sources.append("/sys/devices/system/cpu/cpu0/cpufreq")
        if r.data.get("running_in_vm"):
            r.data["vm_note"] = "Running in a virtual machine: these are the virtual CPUs given to this VM, not the host's."
        return r

    def memory_usage(self) -> ToolResult:
        r = ToolResult("get_memory_usage")
        text = self.x.read("/proc/meminfo")
        if not text:
            r.ok = False
            r.gap("memory usage", "/proc/meminfo is not readable")
            return r
        r.data = P.parse_meminfo(text)
        r.sources.append("/proc/meminfo")
        return r

    def memory_hardware(self) -> ToolResult:
        r = ToolResult("get_memory_hardware")
        text, how = None, None
        if self.x.is_root:
            res = self.x.run("dmidecode", "-t", "memory")
            if res.ok:
                text, how = res.stdout, "dmidecode -t memory (as root)"
        if text is None:
            snap = self.x.read(SNAPSHOT_PATH)
            if snap:
                text, how = snap, f"privileged snapshot {SNAPSHOT_PATH} (collected earlier by `sudo hwqa collect-privileged`)"
        if text is None and self.x.allow_sudo and not self.x.is_root:
            res = self.x.run("dmidecode", "-t", "memory", privileged=True)
            if res.ok:
                text, how = res.stdout, "sudo -n dmidecode -t memory"
        parsed = P.parse_dmidecode_memory(text) if text else None
        if parsed and parsed["slots"]:
            r.data = parsed
            r.sources.append(how)
            return r

        # Degraded: what we can know without firmware tables.
        mi = self.x.read("/proc/meminfo")
        if mi:
            m = P.parse_meminfo(mi)
            r.data = {"usable_ram": m["total"], "usable_ram_bytes": m["total_bytes"],
                      "note": "Usable RAM as seen by the OS (slightly below installed size)."}
            r.sources.append("/proc/meminfo")
        if text is not None and not (parsed and parsed["slots"]):
            reason = "firmware reports no memory devices (typical for virtual machines)"
            fix = None
        elif self.x.which("dmidecode") is None:
            reason = "dmidecode is not installed"
            fix = "sudo apt install dmidecode  (or dnf/pacman install dmidecode), then: sudo hwqa collect-privileged"
        else:
            reason = "reading firmware memory tables (dmidecode) requires root, and hwqa does not run as root"
            fix = ("run once: sudo hwqa collect-privileged   (stores a serial-number-free snapshot), "
                   "or start hwqa with --allow-sudo if a NOPASSWD sudo rule exists for exactly "
                   "'/usr/sbin/dmidecode -t memory'")
        r.gap("RAM slots, module sizes/types/speeds, soldered or not, maximum supported RAM", reason, fix)
        return r

    def storage_devices(self, include_virtual: bool = False) -> ToolResult:
        r = ToolResult("get_storage_devices")
        res = self.x.run("lsblk", "-J", "-b", "-o",
                         "NAME,KNAME,PKNAME,TYPE,SIZE,MODEL,VENDOR,ROTA,TRAN,RM,HOTPLUG,FSTYPE,MOUNTPOINTS")
        if not res.ok and res.error != "missing_binary":  # util-linux < 2.37 has no MOUNTPOINTS column
            res = self.x.run("lsblk", "-J", "-b", "-o",
                             "NAME,KNAME,PKNAME,TYPE,SIZE,MODEL,VENDOR,ROTA,TRAN,RM,HOTPLUG,FSTYPE,MOUNTPOINT")
        if res.ok:
            r.data = P.parse_lsblk_json(res.stdout, include_virtual)
            r.sources.append("lsblk -J")
        else:
            r.data = self._storage_from_sysfs(include_virtual)
            r.sources.append("/sys/block")
            r.gap("partitions and mount points", "lsblk unavailable", "install util-linux")
        if any(d["kind"] == "virtual disk (VM)" for d in r.data.get("disks", [])):
            r.data["vm_note"] = "Virtual disks: the physical drive type (SSD/HDD/NVMe) of the host is not visible from inside a VM."
        return r

    def _storage_from_sysfs(self, include_virtual: bool) -> dict:
        devs = []
        for name in self.x.listdir("/sys/block"):
            base = f"/sys/block/{name}/"
            size = int(self.x.read_value(base + "size") or 0) * 512
            dev = {"name": name, "type": "disk", "size": size,
                   "rota": self.x.read_value(base + "queue/rotational") == "1",
                   "rm": self.x.read_value(base + "removable") == "1",
                   "model": self._sys(base + "device/model"),
                   "tran": "nvme" if name.startswith("nvme") else ("virtio" if name.startswith("vd") else None)}
            devs.append(dev)
        import json
        return P.parse_lsblk_json(json.dumps({"blockdevices": devs}), include_virtual)

    def filesystem_usage(self) -> ToolResult:
        r = ToolResult("get_filesystem_usage")
        real_fs = {"ext2", "ext3", "ext4", "xfs", "btrfs", "vfat", "exfat", "ntfs", "ntfs3", "fuseblk", "f2fs",
                   "zfs", "jfs", "reiserfs", "overlay", "bcachefs"}
        mounts = self.x.read("/proc/mounts") or ""
        seen, out = set(), []
        for line in mounts.splitlines():
            parts = line.split()
            if len(parts) < 3 or parts[2] not in real_fs:
                continue
            dev, mp, fs = parts[0], parts[1].replace("\\040", " "), parts[2]
            if dev in seen and fs != "overlay":
                continue  # bind mounts of the same device
            if mp.startswith(("/snap/", "/var/lib/docker", "/run/")):
                continue
            st = self.x.statvfs(mp)
            if not st or st.f_blocks == 0:
                continue
            seen.add(dev)
            size = st.f_blocks * st.f_frsize
            free = st.f_bavail * st.f_frsize
            used = size - st.f_bfree * st.f_frsize
            out.append({"mountpoint": mp, "filesystem": fs, "device": dev.rsplit("/", 1)[-1],
                        "size": P.human_bytes(size, binary=False), "used": P.human_bytes(used, binary=False),
                        "free": P.human_bytes(free, binary=False), "size_bytes": size, "free_bytes": free,
                        "used_percent": round(100 * used / (used + free), 1) if used + free else None})
        r.data = {"filesystems": out}
        r.sources += ["/proc/mounts", "statvfs()"]
        return r

    def _pci(self) -> tuple[list[dict], str]:
        res = self.x.run("lspci", "-mm", "-nn")
        if res.ok:
            devs = P.parse_lspci_mm(res.stdout)
            src = "lspci -mm -nn"
        else:
            devs, db = [], self._pci_db()
            for slot in self.x.listdir("/sys/bus/pci/devices"):
                base = f"/sys/bus/pci/devices/{slot}/"
                cls = (self.x.read_value(base + "class") or "0x000000")[2:6]
                vid = (self.x.read_value(base + "vendor") or "0x")[2:]
                did = (self.x.read_value(base + "device") or "0x")[2:]
                vname, dnames = db.get(vid, (None, {}))
                devs.append({"slot": slot.split(":", 1)[-1], "class_id": cls, "class": None, "vendor_id": vid,
                             "device_id": did, "vendor": vname or f"vendor id {vid}",
                             "device": dnames.get(did) or f"device id {did}"})
            src = "/sys/bus/pci/devices" + (" + pci.ids" if db else " (no pci.ids database: names unknown)")
        for d in devs:
            d["category"] = P.pci_category(d.get("class_id"))
            full = d["slot"] if d["slot"].count(":") == 2 else f"0000:{d['slot']}"
            d["driver"] = self.x.readlink_name(f"/sys/bus/pci/devices/{full}/driver")
        return devs, src

    def gpu_info(self) -> ToolResult:
        r = ToolResult("get_gpu_info")
        devs, src = self._pci()
        r.sources.append(src)
        gpus = []
        for d in (x for x in devs if x["category"] == "gpu"):
            vid = d.get("vendor_id")
            kind = {"8086": "integrated (Intel)", "10de": "discrete (NVIDIA)"}.get(vid or "", None)
            if vid == "1002":
                kind = "AMD (integrated or discrete)"
            if vid in ("1af4", "1234", "15ad", "80ee", "1b36"):
                kind = "virtual GPU (VM)"
            g = {"name": f"{d['vendor']} {d['device']}".strip(), "vendor": d["vendor"], "type": kind,
                 "pci_slot": d["slot"], "driver": d.get("driver")}
            full = d["slot"] if d["slot"].count(":") == 2 else f"0000:{d['slot']}"
            vram = self.x.read_value(f"/sys/bus/pci/devices/{full}/mem_info_vram_total")
            if vram and vram.isdigit():
                g["vram"] = P.human_bytes(int(vram))
                g["type"] = "discrete (AMD)" if int(vram) > 2 * P.GIB else "integrated (AMD APU)"
                r.sources.append("amdgpu sysfs mem_info_vram_total")
            gpus.append(g)
        if self.x.which("nvidia-smi"):
            res = self.x.run("nvidia-smi", "--query-gpu=name,memory.total,driver_version,pci.bus_id",
                             "--format=csv,noheader,nounits")
            if res.ok:
                r.sources.append("nvidia-smi")
                for n in P.parse_nvidia_smi(res.stdout):
                    match = next((g for g in gpus if g["vendor"].lower().startswith("nvidia")), None)
                    if match:
                        match.update({"name": n["name"], "vram": n["vram"], "nvidia_driver": n["driver_version"]})
        elif any(g["vendor"].lower().startswith("nvidia") for g in gpus):
            r.gap("NVIDIA VRAM and driver version", "nvidia-smi not found (proprietary driver not installed?)")
        r.data = {"gpus": gpus, "count": len(gpus)}
        if not gpus:
            r.gap("GPU", "no display controller found on the PCI bus (common for servers and VMs)")
        for g in gpus:
            if g["type"] in ("integrated (Intel)", "integrated (AMD APU)"):
                g["vram_note"] = "Integrated GPUs share system RAM; there is no dedicated VRAM."
        return r

    def pci_devices(self, category: str = "all") -> ToolResult:
        r = ToolResult("get_pci_devices")
        devs, src = self._pci()
        r.sources.append(src)
        sel = devs if category == "all" else [d for d in devs if d["category"] == category]
        r.data = {"category": category, "devices": [
            {k: d.get(k) for k in ("slot", "category", "vendor", "device", "driver")} for d in sel],
            "count": len(sel)}
        return r

    def network_interfaces(self) -> ToolResult:
        r = ToolResult("get_network_interfaces")
        out = []
        for name in self.x.listdir("/sys/class/net"):
            base = f"/sys/class/net/{name}/"
            is_physical = self.x.exists(base + "device")
            if name == "lo":
                kind = "loopback"
            elif self.x.exists(base + "wireless") or self.x.exists(base + "phy80211"):
                kind = "Wi-Fi"
            elif not is_physical:
                kind = "virtual"
            elif self.x.read_value(base + "type") == "1":
                kind = "Ethernet"
            else:
                kind = "other"
            speed = self.x.read_value(base + "speed")
            out.append({"name": name, "kind": kind, "physical": is_physical,
                        "state": self.x.read_value(base + "operstate"),
                        "link_speed_mbps": int(speed) if speed and speed.lstrip("-").isdigit() and int(speed) > 0 else None,
                        "driver": self.x.readlink_name(base + "device/driver") if is_physical else None})
        r.data = {"interfaces": out}
        r.sources.append("/sys/class/net")
        return r

    def battery_status(self) -> ToolResult:
        r = ToolResult("get_battery_status")
        bats, ac = [], None
        for name in self.x.listdir("/sys/class/power_supply"):
            ue = self.x.read(f"/sys/class/power_supply/{name}/uevent")
            if not ue:
                continue
            kv = P.parse_uevent(ue)
            if kv.get("type") == "Battery" and kv.get("scope", "").lower() != "device":
                bats.append(P.normalize_battery(kv))
            elif kv.get("type") in ("Mains", "USB") and kv.get("online") is not None:
                ac = ac or kv.get("online") == "1"
        r.data = {"batteries": bats, "ac_connected": ac}
        r.sources.append("/sys/class/power_supply/*/uevent")
        if not bats:
            r.gap("battery", "no battery detected: this is a desktop, a VM, or the battery driver isn't loaded")
        return r

    def temperatures(self) -> ToolResult:
        r = ToolResult("get_temperatures")
        readings = []
        for z in self.x.listdir("/sys/class/thermal"):
            if not z.startswith("thermal_zone"):
                continue
            t = self.x.read_value(f"/sys/class/thermal/{z}/temp")
            if t and t.lstrip("-").isdigit():
                readings.append({"sensor": self.x.read_value(f"/sys/class/thermal/{z}/type") or z,
                                 "celsius": round(int(t) / 1000, 1)})
        for h in self.x.listdir("/sys/class/hwmon"):
            base = f"/sys/class/hwmon/{h}/"
            chip = self.x.read_value(base + "name") or h
            for f in self.x.listdir(base):
                if f.startswith("temp") and f.endswith("_input"):
                    t = self.x.read_value(base + f)
                    label = self.x.read_value(base + f.replace("_input", "_label"))
                    if t and t.lstrip("-").isdigit():
                        readings.append({"sensor": f"{chip} {label or f[:-6]}", "celsius": round(int(t) / 1000, 1)})
        r.data = {"readings": readings}
        r.sources += ["/sys/class/thermal", "/sys/class/hwmon"]
        if not readings:
            r.gap("temperatures", "no temperature sensors are exposed (common in VMs; on laptops try loading "
                                  "the coretemp/k10temp module)")
        return r

    def usb_devices(self) -> ToolResult:
        r = ToolResult("get_usb_devices")
        res = self.x.run("lsusb")
        if res.ok:
            devs = [d for d in P.parse_lsusb(res.stdout) if not d["is_root_hub"]]
            r.sources.append("lsusb")
        else:
            devs = []
            for d in self.x.listdir("/sys/bus/usb/devices"):
                base = f"/sys/bus/usb/devices/{d}/"
                vid = self.x.read_value(base + "idVendor")
                if not vid or d.startswith("usb"):
                    continue
                desc = " ".join(filter(None, [self._sys(base + "manufacturer"), self._sys(base + "product")]))
                devs.append({"vendor_id": vid, "product_id": self.x.read_value(base + "idProduct"),
                             "description": desc or None})
            r.sources.append("/sys/bus/usb/devices")
        r.data = {"devices": devs, "count": len(devs)}
        if not devs:
            r.gap("USB devices", "no USB devices found (or no USB bus, e.g. in a VM)")
        return r
