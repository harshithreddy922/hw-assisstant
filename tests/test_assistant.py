"""End-to-end conversations on a fake laptop, plus accuracy/privacy guards."""
import json

from hwqa.assistant import Assistant
from hwqa.backends.base import Step, ToolCall
from hwqa.backends.claude import ClaudeBackend
from hwqa.backends.offline import OfflineBackend, detect_intents
from hwqa.executor import Executor
from hwqa.guards import redact, ungrounded_claims
from tests.fake_system import FakeExecutor, fx, snapshot_files
from tests.test_security import ScriptedBackend


def laptop(**kw) -> Assistant:
    return Assistant(OfflineBackend(), FakeExecutor(**kw))


# ------------------------------------------------------------------ the brief's example
def test_brief_example_can_i_add_more_ram_as_root():
    a = laptop(root=True).ask("Can I add more RAM, and how much?")
    t = a.text
    assert "16 GB in 1 of 2 slots" in t and "64 GB" in t and "up to 32 GB" in t and "48 GB" in t
    assert a.ungrounded == []
    assert any("dmidecode" in s for c in a.tool_calls for s in c["sources"])


def test_ram_upgrade_from_privileged_snapshot_without_root():
    a = laptop(root=False, files=snapshot_files()).ask("Can I add more RAM, and how much?")
    assert "up to 32 GB" in a.text
    assert any("snapshot" in s for c in a.tool_calls for s in c["sources"])


def test_ram_upgrade_without_root_says_it_does_not_know():
    a = laptop(root=False).ask("Can I add more RAM?")
    assert "can't determine" in a.text and "requires root" in a.text and "collect-privileged" in a.text
    assert "64 GB" not in a.text and "32 GB" not in a.text   # no invented specs


def test_soldered_ram():
    a = laptop(root=True, dmidecode_text=fx("dmidecode_soldered.txt")).ask("can I upgrade my memory?")
    assert "soldered" in a.text and "can't be upgraded" in a.text


# ------------------------------------------------------------------ other questions
def test_how_much_ram():
    a = laptop(root=True).ask("How much RAM do I have?")
    assert "16 GB of RAM installed (DDR4 3200 MT/s)" in a.text and "1 of 2 slots" in a.text


def test_is_my_ssd_nvme():
    a = laptop().ask("Is my SSD NVMe?")
    assert a.text.startswith("Yes, nvme0n1 is an NVMe SSD") and "512 GB" in a.text


def test_gpu_follow_up_and_cache():
    asst = laptop(root=True)
    asst.ask("How much RAM do I have?")
    a = asst.ask("and what about the GPU?")
    assert "RTX 3050" in a.text and "4 GB VRAM" in a.text and "Iris Xe" in a.text and "nvidia" in a.text
    b = asst.ask("is it integrated?")      # pronoun follow-up -> same topic, served from cache
    assert "integrated (Intel)" in b.text
    assert all(c["cached"] for c in b.tool_calls)


def test_battery_wifi_temps_system():
    asst = laptop()
    assert "85.0% of design capacity" in asst.ask("How healthy is my battery?").text
    assert asst.ask("Do I have wifi?").text.startswith("Yes, you have Wi-Fi.")
    assert "52.0 °C" in asst.ask("how hot is it running?").text
    s = asst.ask("what laptop model is this?").text
    assert "LENOVO 21AH00FGUS (notebook)" in s and "Fedora Linux 40" in s and "6.8.9-300.fc40" in s


def test_cpu_answer():
    t = laptop().ask("How many cores does my CPU have?").text
    assert "12 physical cores and 16 threads" in t and "4.7 GHz" in t


def test_elliptical_follow_ups_inherit_topic():
    asst = laptop(root=True)
    asst.ask("How much RAM do I have?")
    assert "up to 32 GB" in asst.ask("Can I add more?").text
    asst.ask("What CPU do I have?")
    assert "12 physical cores" in asst.ask("is it fast?").text


def test_read_only_refusal_and_unknown_question():
    asst = laptop()
    assert "read-only" in asst.ask("please install a new driver").text
    assert "I can answer questions" in asst.ask("what's the weather?").text


def test_intents():
    assert detect_intents("Can I add more RAM, and how much?") == ["ram_upgrade"]
    assert detect_intents("how much disk space is left?") == ["disk_space"]
    assert "storage" in detect_intents("Is my SSD NVMe?")


# ------------------------------------------------------------------ accuracy guard
def test_grounding_detects_invented_numbers():
    results = [{"data": {"installed": "16 GB", "max_capacity": "64 GB", "largest_module_to_add": "32 GB"}}]
    assert ungrounded_claims("You can add up to 32 GB (64 GB max).", results) == []
    assert ungrounded_claims("Your board supports 128 GB.", results) == ["128 GB"]
    assert ungrounded_claims("It runs at 4.7 GHz", [{"max_mhz": 4700}]) == []


def test_llm_hallucination_triggers_self_correction():
    b = ScriptedBackend([
        Step(tool_calls=[ToolCall("1", "get_memory_usage", {})]),
        Step(text="You have 32 GB of RAM."),                      # invented
        Step(text="The OS reports 15.3 GB usable RAM."),          # corrected
    ])
    a = Assistant(b, FakeExecutor()).ask("How much RAM?")
    assert "automatic accuracy check" in b.seen[2][-1]["content"]
    assert a.text == "The OS reports 15.3 GB usable RAM." and a.ungrounded == []


# ------------------------------------------------------------------ privacy
def test_redaction():
    s = redact('{"mac": "a4:5e:60:c2:11:9f", "uuid": "123e4567-e89b-12d3-a456-426614174000", '
               '"mount": "/media/harshith/USB", "Serial Number": "XYZ123"}')
    assert "a4:5e" not in s and "123e4567" not in s and "harshith" not in s and "XYZ123" not in s


def test_remote_backend_gets_redacted_tool_output():
    class Remote(ScriptedBackend):
        is_local = False
    b = Remote([Step(tool_calls=[ToolCall("1", "get_storage_devices", {})]), Step(text="ok")])
    Assistant(b, FakeExecutor()).ask("drives?")
    payload = [m for m in b.seen[1] if m["role"] == "tool"][0]["content"]
    assert "/media/<user>/USB" in payload and "/media/user/" not in payload


def test_claude_wire_format_groups_parallel_tool_results():
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "raw": [{"type": "tool_use", "id": "a", "name": "x", "input": {}},
                                                          {"type": "tool_use", "id": "b", "name": "y", "input": {}}]},
            {"role": "tool", "tool_call_id": "a", "name": "x", "content": "{}"},
            {"role": "tool", "tool_call_id": "b", "name": "y", "content": "{}", "is_error": True}]
    wire = ClaudeBackend._to_wire(msgs)
    assert len(wire) == 3 and [b["tool_use_id"] for b in wire[2]["content"]] == ["a", "b"]
    assert wire[2]["content"][1]["is_error"] is True


# ------------------------------------------------------------------ this real machine
def test_live_smoke_on_this_machine():
    a = Assistant(OfflineBackend(), Executor()).ask("How much RAM do I have?")
    assert "usable" in a.text and a.error is None and a.audit
    assert json.dumps([c["sources"] for c in a.tool_calls])
