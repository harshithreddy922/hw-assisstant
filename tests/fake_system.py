"""A fake laptop for end-to-end tests: the real Executor security checks still apply (allow-listed
commands and paths), but outputs come from fixtures instead of the OS."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from hwqa.executor import ALLOWED_COMMANDS, PRIVILEGED_ARGV, CommandResult, Executor, NotAllowed
from hwqa.tools import SNAPSHOT_PATH

FX = Path(__file__).parent / "fixtures"


def fx(name: str) -> str:
    return (FX / name).read_text()


MEMINFO_16G = "MemTotal:       16066000 kB\nMemFree:         9000000 kB\nMemAvailable:   11000000 kB\nSwapTotal:       8388604 kB\nSwapFree:        8388604 kB\n"
CPUINFO = "processor\t: 0\nvendor_id\t: GenuineIntel\nmodel name\t: 12th Gen Intel(R) Core(TM) i7-1260P\nflags\t\t: fpu avx2\n"


def laptop_files(dmi_root: bool = True) -> dict[str, str]:
    dmi = "/sys/class/dmi/id/"
    files = {
        "/proc/meminfo": MEMINFO_16G,
        "/proc/cpuinfo": CPUINFO,
        "/proc/uptime": "7200.5 1000.0",
        "/proc/version": "Linux version 6.8.0",
        "/proc/sys/kernel/osrelease": "6.8.9-300.fc40.x86_64",
        "/proc/mounts": "/dev/nvme0n1p2 / ext4 rw 0 0\n/dev/nvme0n1p1 /boot/efi vfat rw 0 0\nproc /proc proc rw 0 0\n",
        "/etc/os-release": 'PRETTY_NAME="Fedora Linux 40 (Workstation Edition)"\nID=fedora\nVERSION_ID=40\n',
        dmi + "sys_vendor": "LENOVO", dmi + "product_name": "21AH00FGUS", dmi + "product_family": "ThinkPad T14 Gen 3",
        dmi + "product_version": "ThinkPad T14 Gen 3", dmi + "chassis_type": "10", dmi + "bios_vendor": "LENOVO",
        dmi + "bios_version": "R23ET65W (1.35 )", dmi + "bios_date": "02/14/2024", dmi + "board_vendor": "LENOVO",
        dmi + "board_name": "21AH00FGUS",
        "/sys/class/power_supply/BAT0/uevent": fx("power_supply/BAT0/uevent"),
        "/sys/class/power_supply/AC/uevent": fx("power_supply/AC/uevent"),
        "/sys/class/net/wlp0s20f3/operstate": "up", "/sys/class/net/wlp0s20f3/type": "1",
        "/sys/class/net/wlp0s20f3/wireless": "", "/sys/class/net/wlp0s20f3/device": "",
        "/sys/class/net/lo/operstate": "unknown", "/sys/class/net/lo/type": "772",
        "/sys/class/thermal/thermal_zone0/temp": "52000", "/sys/class/thermal/thermal_zone0/type": "x86_pkg_temp",
        "/sys/bus/pci/devices": "", "/sys/firmware/efi": "",
    }
    return files


class FakeExecutor(Executor):
    def __init__(self, *, root: bool = False, files: dict | None = None, binaries: dict | None = None,
                 allow_sudo: bool = False, dmidecode_text: str | None = None):
        super().__init__(allow_sudo=allow_sudo)
        self._root = root
        self.files = files if files is not None else laptop_files()
        self.commands = binaries if binaries is not None else {
            "lscpu": {("-J",): fx("lscpu_laptop.json")},
            "lsblk": {("-J", "-b", "-o", "NAME,KNAME,PKNAME,TYPE,SIZE,MODEL,VENDOR,ROTA,TRAN,RM,HOTPLUG,FSTYPE,MOUNTPOINTS"):
                      fx("lsblk_laptop.json")},
            "lspci": {("-mm", "-nn"): fx("lspci_laptop.txt")},
            "lsusb": {(): fx("lsusb_laptop.txt")},
            "nvidia-smi": {("--query-gpu=name,memory.total,driver_version,pci.bus_id",
                            "--format=csv,noheader,nounits"): fx("nvidia_smi.csv")},
            "dmidecode": {("-t", "memory"): dmidecode_text if dmidecode_text is not None else fx("dmidecode_laptop.txt")},
        }
        self.links = {"/sys/bus/pci/devices/0000:01:00.0/driver": "nvidia",
                      "/sys/bus/pci/devices/0000:00:02.0/driver": "i915",
                      "/sys/class/net/wlp0s20f3/device/driver": "iwlwifi"}
        self.ran: list[list[str]] = []

    @property
    def is_root(self) -> bool:
        return self._root

    def which(self, binary):
        if binary == "sudo":
            return "/usr/bin/sudo"
        return f"/usr/bin/{binary}" if binary in self.commands else None

    def run(self, binary, *args, privileged=False):
        if binary not in ALLOWED_COMMANDS or tuple(args) not in ALLOWED_COMMANDS[binary]:
            raise NotAllowed(f"command not allow-listed: {binary} {' '.join(args)}")
        if binary not in self.commands:
            return CommandResult(ok=False, error="missing_binary")
        argv = [binary, *args]
        if binary == "dmidecode" and not self._root:
            if not privileged or (binary, *args) not in PRIVILEGED_ARGV or not self.allow_sudo:
                self.ran.append(argv)
                return CommandResult(ok=False, error="permission_denied", stderr="/dev/mem: Permission denied")
            argv = ["sudo", "-n", *argv]
        self.ran.append(argv)
        out = self.commands[binary].get(tuple(args))
        if out is None:
            return CommandResult(ok=False, error="failed")
        self._log("exec", " ".join(argv), True)
        return CommandResult(ok=True, stdout=out, argv=argv)

    def read(self, path, max_bytes=1_000_000):
        self._check_path(os.path.normpath(path)) if not path.startswith("/var/lib/hwqa/") else None
        if path in self.files:
            self._log("read", path, True)
            return self.files[path]
        return None

    def listdir(self, path):
        prefix = path.rstrip("/") + "/"
        names = {p[len(prefix):].split("/")[0] for p in self.files if p.startswith(prefix)}
        if path.rstrip("/") == "/sys/bus/pci/devices":
            names |= {"0000:00:02.0", "0000:01:00.0"}
        return sorted(names)

    def exists(self, path):
        return path in self.files or any(p.startswith(path.rstrip("/") + "/") for p in self.files) \
            or (path.startswith("/usr/bin/") and path[9:] in self.commands)

    def readlink_name(self, path):
        return self.links.get(path)

    def statvfs(self, mountpoint):
        if mountpoint == "/":
            return SimpleNamespace(f_blocks=122_000_000, f_frsize=4096, f_bavail=60_000_000, f_bfree=62_000_000)
        if mountpoint == "/boot/efi":
            return SimpleNamespace(f_blocks=66_000, f_frsize=4096, f_bavail=50_000, f_bfree=50_000)
        return None


def snapshot_files() -> dict:
    files = laptop_files()
    files[SNAPSHOT_PATH] = fx("dmidecode_laptop.txt")
    return files
