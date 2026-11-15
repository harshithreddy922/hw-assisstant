"""The only code in the project that touches the operating system.

Security model (the LLM never reaches this layer directly - it can only pick a tool by name):
  * No shell. Commands run via execve (subprocess with shell=False), so `;`, `|`, `&&`, `$()` are inert.
  * Allow-listed binaries only, resolved from trusted system directories (never from $PATH, cwd or
    the user's home), so a planted `lscpu` in ~/bin can't hijack a tool.
  * Allow-listed argument vectors only: each tool asks for an exact argv defined in code. Free-form
    arguments cannot reach this layer.
  * Read-only file access, limited to /proc, /sys and a few /etc and /usr/share files, checked after
    resolving symlinks.
  * Minimal environment (LC_ALL=C for parseable output), stdin closed, timeouts, output size cap.
  * Privilege: nothing runs as root unless the process already is. The only escalation is
    `sudo -n` (non-interactive, never prompts) for commands in PRIVILEGED_ARGV, and only when the user
    opted in. Most users instead run the one-time `hwqa collect-privileged` helper.
  * Every command and file read is recorded in an audit log the user can inspect (/trace).
"""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

TRUSTED_DIRS = ("/usr/bin", "/bin", "/usr/sbin", "/sbin", "/usr/local/bin")

# binary -> exact argument vectors that may be used with it (prefix-free, compared as tuples)
ALLOWED_COMMANDS: dict[str, set[tuple[str, ...]]] = {
    "lscpu": {("-J",), ()},
    "lsblk": {("-J", "-b", "-o", "NAME,KNAME,PKNAME,TYPE,SIZE,MODEL,VENDOR,ROTA,TRAN,RM,HOTPLUG,FSTYPE,MOUNTPOINTS"),
              ("-J", "-b", "-o", "NAME,KNAME,PKNAME,TYPE,SIZE,MODEL,VENDOR,ROTA,TRAN,RM,HOTPLUG,FSTYPE,MOUNTPOINT")},
    "lspci": {("-mm", "-nn"), ("-mm", "-nn", "-k")},
    "lsusb": {()},
    "dmidecode": {("-t", "memory"), ("-t", "16,17")},
    "nvidia-smi": {("--query-gpu=name,memory.total,driver_version,pci.bus_id", "--format=csv,noheader,nounits")},
}

# Commands that need root. Only these may ever be prefixed with `sudo -n`.
PRIVILEGED_ARGV: set[tuple[str, ...]] = {("dmidecode", "-t", "memory")}

READ_ALLOWED_PREFIXES = ("/proc/", "/sys/", "/etc/os-release", "/usr/lib/os-release",
                         "/usr/share/misc/pci.ids", "/usr/share/hwdata/pci.ids", "/usr/share/hwdata/usb.ids",
                         "/var/lib/usbutils/usb.ids", "/var/lib/hwqa/")
MAX_OUTPUT = 2_000_000
SAFE_ENV = {"LC_ALL": "C", "LANG": "C", "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}


class NotAllowed(Exception):
    pass


@dataclass
class CommandResult:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None
    error: str | None = None          # "missing_binary" | "permission_denied" | "timeout" | "failed" | ...
    argv: list[str] = field(default_factory=list)


@dataclass
class AuditEntry:
    kind: str           # "exec" | "read"
    target: str
    ok: bool
    detail: str = ""
    ms: float = 0.0


class Executor:
    def __init__(self, allow_sudo: bool = False, timeout: float = 5.0):
        self.allow_sudo = allow_sudo
        self.timeout = timeout
        self.audit: list[AuditEntry] = []
        self._which_cache: dict[str, str | None] = {}

    # ------------------------------------------------------------------ discovery
    def which(self, binary: str) -> str | None:
        """Resolve a binary from trusted directories only (never from $PATH)."""
        if binary not in self._which_cache:
            found = None
            for d in TRUSTED_DIRS:
                p = os.path.join(d, binary)
                if os.path.isfile(p) and os.access(p, os.X_OK):
                    found = p
                    break
            self._which_cache[binary] = found
        return self._which_cache[binary]

    @property
    def is_root(self) -> bool:
        return hasattr(os, "geteuid") and os.geteuid() == 0

    # ------------------------------------------------------------------ commands
    def run(self, binary: str, *args: str, privileged: bool = False) -> CommandResult:
        if binary not in ALLOWED_COMMANDS or tuple(args) not in ALLOWED_COMMANDS[binary]:
            raise NotAllowed(f"command not allow-listed: {binary} {' '.join(args)}")
        path = self.which(binary)
        if path is None:
            self._log("exec", binary, False, "missing binary")
            return CommandResult(ok=False, error="missing_binary", argv=[binary, *args])
        argv = [path, *args]
        if privileged and not self.is_root:
            if (binary, *args) not in PRIVILEGED_ARGV:
                raise NotAllowed(f"privilege escalation not allowed for {binary}")
            if not self.allow_sudo:
                return CommandResult(ok=False, error="needs_root", argv=argv)
            sudo = self.which("sudo")
            if sudo is None:
                return CommandResult(ok=False, error="needs_root", argv=argv)
            argv = [sudo, "-n", *argv]  # -n: never prompt; fails unless a NOPASSWD rule exists
        t0 = time.perf_counter()
        try:
            p = subprocess.run(argv, shell=False, capture_output=True, text=True, timeout=self.timeout,
                               env=SAFE_ENV, stdin=subprocess.DEVNULL, errors="replace")
        except subprocess.TimeoutExpired:
            self._log("exec", " ".join(argv), False, "timeout", t0)
            return CommandResult(ok=False, error="timeout", argv=argv)
        except OSError as e:
            self._log("exec", " ".join(argv), False, str(e), t0)
            return CommandResult(ok=False, error="failed", stderr=str(e), argv=argv)
        out = p.stdout[:MAX_OUTPUT]
        err = p.stderr[:4000]
        if p.returncode != 0:
            low = err.lower()
            kind = "permission_denied" if ("permission denied" in low or "password is required" in low
                                            or "must be root" in low or "/dev/mem" in low) else "failed"
            self._log("exec", " ".join(argv), False, f"exit {p.returncode}: {err.strip()[:120]}", t0)
            return CommandResult(ok=False, stdout=out, stderr=err, returncode=p.returncode, error=kind, argv=argv)
        self._log("exec", " ".join(argv), True, f"{len(out)} bytes", t0)
        return CommandResult(ok=True, stdout=out, stderr=err, returncode=0, argv=argv)

    # ------------------------------------------------------------------ files
    @staticmethod
    def _allowed(p: str) -> bool:
        return any(p == pre or p.startswith(pre) if pre.endswith("/") else p == pre
                   for pre in READ_ALLOWED_PREFIXES)

    def _check_path(self, path: str) -> str:
        """Both the requested path (normalised, so `..` can't escape) and its symlink-resolved target
        must be allow-listed. /sys links stay inside /sys; /etc/os-release links to /usr/lib/os-release."""
        norm = os.path.normpath(path) + ("/" if path.endswith("/") else "")
        real = os.path.realpath(path) + ("/" if path.endswith("/") else "")
        if not (self._allowed(norm) and self._allowed(real)):
            raise NotAllowed(f"read not allow-listed: {path}")
        return real

    def read(self, path: str, max_bytes: int = 1_000_000) -> str | None:
        self._check_path(path)
        try:
            with open(path, "r", errors="replace") as f:
                data = f.read(max_bytes)
            self._log("read", path, True)
            return data
        except (FileNotFoundError, NotADirectoryError):
            return None
        except PermissionError:
            self._log("read", path, False, "permission denied")
            return None
        except OSError:
            return None

    def read_value(self, path: str) -> str | None:
        v = self.read(path, 4096)
        return v.strip() if v is not None else None

    def listdir(self, path: str) -> list[str]:
        self._check_path(path.rstrip("/") + "/")
        try:
            return sorted(os.listdir(path))
        except OSError:
            return []

    def exists(self, path: str) -> bool:
        self._check_path(path)
        return os.path.exists(path)

    def readlink_name(self, path: str) -> str | None:
        """Basename of a symlink target, e.g. a device's driver."""
        self._check_path(path)
        try:
            return Path(os.readlink(path)).name
        except OSError:
            return None

    def statvfs(self, mountpoint: str):
        """Filesystem usage metadata (read-only, no file contents)."""
        try:
            st = os.statvfs(mountpoint)
            self._log("statvfs", mountpoint, True)
            return st
        except OSError as e:
            self._log("statvfs", mountpoint, False, str(e))
            return None

    # ------------------------------------------------------------------ audit
    def _log(self, kind: str, target: str, ok: bool, detail: str = "", t0: float | None = None) -> None:
        ms = (time.perf_counter() - t0) * 1000 if t0 else 0.0
        self.audit.append(AuditEntry(kind, target, ok, detail, round(ms, 1)))
