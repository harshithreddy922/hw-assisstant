"""Parsers against captured real output (VM) and representative laptop output."""
from hwqa import parsers as P
from tests.fake_system import fx


def test_human_bytes():
    assert P.human_bytes(16 * P.GIB) == "16 GB"
    assert P.human_bytes(512110190592, binary=False) == "512 GB"
    assert P.human_bytes(1000204886016, binary=False) == "1 TB"
    assert P.parse_size("16 GB") == 16 * P.GIB
    assert P.parse_size("8192 MB") == 8 * P.GIB
    assert P.parse_size("No Module Installed") is None


def test_lscpu_real_vm_output():
    c = P.parse_lscpu_json(fx("lscpu_vm.json"))
    assert c["architecture"] == "x86_64"
    assert c["logical_cpus"] >= 1 and c["physical_cores"] >= 1
    assert c["running_in_vm"] is True


def test_lscpu_nested_laptop_output():
    c = P.parse_lscpu_json(fx("lscpu_laptop.json"))
    assert c["model_name"] == "12th Gen Intel(R) Core(TM) i7-1260P"
    assert (c["physical_cores"], c["logical_cpus"], c["threads_per_core"]) == (12, 16, 2)
    assert c["max_ghz"] == 4.7
    assert c["virtualization_support"] == "VT-x"
    assert {"avx2", "aes", "vmx"} <= set(c["notable_flags"])
    assert c["running_in_vm"] is False


def test_cpuinfo_fallback():
    c = P.parse_cpuinfo("processor : 0\nvendor_id : AuthenticAMD\nmodel name : AMD Ryzen 7 5800U\nphysical id : 0\n"
                        "core id : 0\n\nprocessor : 1\nvendor_id : AuthenticAMD\nmodel name : AMD Ryzen 7 5800U\n"
                        "physical id : 0\ncore id : 0\n")
    assert c["model_name"] == "AMD Ryzen 7 5800U" and c["logical_cpus"] == 2 and c["physical_cores"] == 1


def test_meminfo_real():
    m = P.parse_meminfo(fx("meminfo_vm.txt"))
    assert m["total_bytes"] > 0 and m["available_bytes"] <= m["total_bytes"]


def test_dmidecode_brief_example():
    m = P.parse_dmidecode_memory(fx("dmidecode_laptop.txt"))
    assert (m["total_slots"], m["used_slots"], m["free_slots"]) == (2, 1, 1)
    assert m["installed"] == "16 GB" and m["max_capacity"] == "64 GB"
    assert m["upgrade_verdict"] == "can_add_module"
    assert m["largest_module_to_add"] == "32 GB" and m["total_after_adding_largest"] == "48 GB"
    assert m["memory_type"] == "DDR4" and m["memory_speed"] == "3200 MT/s"
    # privacy: serials are never extracted
    assert "4A1B2C3D" not in str(m)


def test_dmidecode_soldered():
    m = P.parse_dmidecode_memory(fx("dmidecode_soldered.txt"))
    assert m["all_memory_soldered"] and m["upgrade_verdict"] == "not_upgradeable_soldered"
    assert m["installed"] == "16 GB" and m["memory_type"] == "LPDDR5"


def test_upgrade_all_slots_full():
    slots = [{"installed": True, "size_bytes": 8 * P.GIB, "soldered": False, "type": "DDR4", "speed": "2666 MT/s"}] * 2
    a = P.upgrade_analysis(slots, 32 * P.GIB)
    assert a["upgrade_verdict"] == "replace_modules_to_upgrade" and a["max_per_slot_estimate"] == "16 GB"


def test_lsblk_laptop():
    d = P.parse_lsblk_json(fx("lsblk_laptop.json"))
    by = {x["name"]: x for x in d["disks"]}
    assert "zram0" not in by
    assert by["nvme0n1"]["is_nvme"] and by["nvme0n1"]["size"] == "512 GB"
    assert by["sda"]["kind"] == "HDD (spinning)" and by["sda"]["size"] == "1 TB"
    assert by["sdb"]["kind"] == "USB/removable drive"
    assert by["nvme0n1"]["partitions"][1]["mountpoints"] == ["/"]


def test_lsblk_real_vm():
    d = P.parse_lsblk_json(fx("lsblk_vm.json"))
    assert d["disks"] and all(x["kind"] == "virtual disk (VM)" for x in d["disks"])


def test_lspci_and_categories():
    devs = P.parse_lspci_mm(fx("lspci_laptop.txt"))
    gpus = [d for d in devs if P.pci_category(d["class_id"]) == "gpu"]
    assert {d["vendor_id"] for d in gpus} == {"8086", "10de"}
    assert any("RTX 3050" in d["device"] for d in gpus)
    assert any(P.pci_category(d["class_id"]) == "network" and "WiFi" in d["device"] for d in devs)


def test_pci_ids_database():
    db = P.parse_pci_ids("# comment\n8086  Intel Corporation\n\t46a6  Alder Lake-P GT2 [Iris Xe Graphics]\n"
                         "\t\t17aa 3803  subsystem\n10de  NVIDIA Corporation\nC 00  Unclassified device\n")
    assert db["8086"][0] == "Intel Corporation" and db["8086"][1]["46a6"].startswith("Alder Lake")
    assert "10de" in db


def test_nvidia_smi_and_lsusb():
    n = P.parse_nvidia_smi(fx("nvidia_smi.csv"))[0]
    assert n["vram"] == "4 GB" and n["driver_version"] == "550.54.14"
    usb = P.parse_lsusb(fx("lsusb_laptop.txt"))
    assert sum(not u["is_root_hub"] for u in usb) == 3


def test_battery_health():
    b = P.normalize_battery(P.parse_uevent(fx("power_supply/BAT0/uevent")))
    assert b["health_percent"] == 85.0 and b["cycle_count"] == 312 and b["charge_percent"] == 70
    assert b["estimated_hours_remaining"] == 3.7
    assert "serial" not in str(b).lower()
