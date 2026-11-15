"""The executor is the security boundary: prove it refuses everything outside the allow-list."""
import os

import pytest

from hwqa.assistant import Assistant, validate_args
from hwqa.backends.base import Step, ToolCall
from hwqa.executor import Executor, NotAllowed
from tests.fake_system import FakeExecutor


@pytest.mark.parametrize("binary,args", [
    ("rm", ("-rf", "/")),
    ("bash", ("-c", "id")),
    ("lscpu", ("--help",)),
    ("lscpu", ("-J; rm -rf /",)),
    ("lsblk", ("-J", "&&", "reboot")),
    ("dmidecode", ("-t", "memory", "|", "nc", "evil", "1")),
])
def test_commands_outside_allow_list_are_refused(binary, args):
    with pytest.raises(NotAllowed):
        Executor().run(binary, *args)


@pytest.mark.parametrize("path", ["/etc/shadow", "/root/.ssh/id_rsa", "/sys/../etc/shadow",
                                  "/proc/../etc/passwd", "/home/user/.bashrc", "/etc/os-release/../shadow"])
def test_file_reads_outside_allow_list_are_refused(path):
    with pytest.raises(NotAllowed):
        Executor().read(path)


def test_allowed_reads_work():
    assert "MemTotal" in Executor().read("/proc/meminfo")


def test_binaries_resolved_only_from_trusted_dirs(tmp_path, monkeypatch):
    fake = tmp_path / "lscpu"
    fake.write_text("#!/bin/sh\necho pwned\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    path = Executor().which("lscpu")
    assert path is None or not path.startswith(str(tmp_path))


def test_no_shell_metacharacter_interpretation():
    # shell=False + fixed argv: even an allow-listed command can't be chained
    r = Executor().run("lscpu", "-J")
    assert r.ok and r.stdout.lstrip().startswith("{")


def test_privilege_escalation_only_for_allow_listed_command():
    x = Executor(allow_sudo=True)
    if x.is_root:
        pytest.skip("running as root")
    with pytest.raises(NotAllowed):
        x.run("lscpu", "-J", privileged=True)


def test_sudo_never_used_without_opt_in():
    x = FakeExecutor(root=False, allow_sudo=False)
    from hwqa.tools import HardwareTools
    r = HardwareTools(x).call("get_memory_hardware", {})
    assert not any(a[0] == "sudo" for a in x.ran)
    assert r["unavailable"] and "root" in r["unavailable"][0]["reason"]


def test_sudo_n_used_when_opted_in():
    x = FakeExecutor(root=False, allow_sudo=True)
    from hwqa.tools import HardwareTools
    r = HardwareTools(x).call("get_memory_hardware", {})
    assert ["sudo", "-n", "dmidecode", "-t", "memory"] in x.ran
    assert r["data"]["installed"] == "16 GB"


def test_argument_validation():
    schema = {"properties": {"category": {"type": "string", "enum": ["gpu", "all"]},
                             "include_virtual": {"type": "boolean"}}, "required": []}
    assert validate_args(schema, {"category": "gpu"}) is None
    assert "one of" in validate_args(schema, {"category": "; rm -rf /"})
    assert "unknown argument" in validate_args(schema, {"cmd": "ls"})
    assert "boolean" in validate_args(schema, {"include_virtual": "yes"})
    assert validate_args(schema, "lscpu") is not None


class ScriptedBackend:
    """Pretends to be an LLM that tries to misbehave."""
    name, is_local, is_llm = "scripted", True, True

    def __init__(self, steps):
        self.steps = list(steps)
        self.seen = []

    def step(self, messages, tools):
        self.seen.append([dict(m) for m in messages])
        return self.steps.pop(0)


def test_model_cannot_call_unknown_tools_or_bad_args():
    b = ScriptedBackend([
        Step(tool_calls=[ToolCall("1", "run_shell", {"cmd": "rm -rf /"}),
                         ToolCall("2", "get_pci_devices", {"category": "$(reboot)"}),
                         ToolCall("3", "get_cpu_info", {})]),
        Step(text="Your CPU has 12 cores."),
    ])
    x = FakeExecutor()
    a = Assistant(b, x).ask("cpu?")
    tool_msgs = [m for m in b.seen[1] if m["role"] == "tool"]
    assert "unknown or unavailable tool" in tool_msgs[0]["content"]
    assert "invalid arguments" in tool_msgs[1]["content"]
    assert '"ok": true' in tool_msgs[2]["content"]
    assert x.ran == [["lscpu", "-J"]]  # nothing else executed
    assert a.text == "Your CPU has 12 cores."
