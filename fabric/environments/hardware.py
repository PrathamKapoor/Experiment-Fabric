"""Hardware capture.

Recorded on every run so that "does this result survive moving from an
A100 to an H100?" is a query rather than a research project.  Capture is
best-effort by design: a missing GPU probe must never fail a run, it
just makes that axis unmeasurable and Fabric says so.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import time
from functools import lru_cache
from typing import Any

from fabric.core.entities import Hardware


def _run(command: list[str], timeout: float = 5.0) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def detect_gpus() -> list[dict[str, Any]]:
    """Enumerate NVIDIA GPUs via ``nvidia-smi``.

    Only NVIDIA is probed directly; other accelerators are reported
    through the framework metadata in the environment record instead of
    being guessed at here.
    """
    output = _run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ])
    if not output:
        return []
    gpus: list[dict[str, Any]] = []
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        entry: dict[str, Any] = {"index": _int(parts[0]), "name": parts[1]}
        entry["memory_mb"] = _int(parts[2])
        if len(parts) > 3:
            entry["driver"] = parts[3]
        if len(parts) > 4:
            entry["compute_capability"] = parts[4]
        gpus.append(entry)
    return gpus


def _int(text: str) -> int:
    try:
        return int(float(text))
    except ValueError:
        return 0


def cpu_model() -> str:
    """A human-readable CPU name across platforms."""
    system = platform.system()
    if system == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif system == "Darwin":
        name = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if name:
            return name
    elif system == "Windows":
        name = os.environ.get("PROCESSOR_IDENTIFIER", "")
        if name:
            return name
    return platform.processor() or platform.machine() or "unknown"


def total_memory_bytes() -> int:
    """Physical RAM, or 0 when it cannot be determined."""
    try:
        if hasattr(os, "sysconf"):
            names = os.sysconf_names  # type: ignore[attr-defined]
            if "SC_PAGE_SIZE" in names and "SC_PHYS_PAGES" in names:
                return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        pass
    if platform.system() == "Windows":
        try:
            import ctypes

            class _MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatus()
            status.dwLength = ctypes.sizeof(_MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
                return int(status.ullTotalPhys)
        except Exception:  # pragma: no cover - platform specific
            return 0
    return 0


def machine_identifier() -> str:
    """A stable, non-identifying machine fingerprint.

    Hashed rather than stored raw: a hostname can carry a person's name
    or an institution's internal topology, and the only thing Fabric
    needs is whether two runs happened on the *same* machine.
    """
    raw = "|".join([platform.node(), platform.machine(), platform.system()])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def distributed_state() -> dict[str, Any]:
    """Distributed-training topology, read from the usual env conventions."""
    state: dict[str, Any] = {}
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "SLURM_JOB_ID",
                "SLURM_NTASKS", "SLURM_NODELIST"):
        value = os.environ.get(key)
        if value:
            state[key.lower()] = value if key != "MASTER_ADDR" else "set"
    return state


@lru_cache(maxsize=1)
def capture_hardware() -> Hardware:
    """Capture the current machine's hardware context."""
    gpus = detect_gpus()
    return Hardware(
        machine_id=machine_identifier(),
        cpu_model=cpu_model(),
        cpu_count=os.cpu_count() or 0,
        ram_bytes=total_memory_bytes(),
        architecture=platform.machine(),
        gpus=gpus,
        accelerator=(gpus[0]["name"] if gpus else "cpu"),
        driver=(gpus[0].get("driver", "") if gpus else ""),
        distributed=distributed_state(),
        captured_at=time.time(),
    )


def describe(hardware: Hardware) -> list[tuple[str, str]]:
    """Flatten hardware into display rows."""
    rows = [
        ("machine", hardware.machine_id),
        ("cpu", f"{hardware.cpu_model} ({hardware.cpu_count} logical cores)"),
        ("architecture", hardware.architecture),
        ("memory", f"{hardware.ram_bytes / 1e9:.1f} GB" if hardware.ram_bytes else "unknown"),
        ("accelerator", hardware.accelerator),
    ]
    for gpu in hardware.gpus:
        rows.append((f"gpu:{gpu.get('index', 0)}",
                     f"{gpu.get('name', 'unknown')} {gpu.get('memory_mb', 0)} MB"))
    if hardware.distributed:
        rows.append(("distributed", ", ".join(f"{k}={v}" for k, v in hardware.distributed.items())))
    return rows
