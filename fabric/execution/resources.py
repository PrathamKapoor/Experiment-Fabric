"""Resource accounting.

Every run records what it cost.  This turns the standard leaderboard
comparison -- "ours: 91.4, theirs: 91.2" -- into the comparison that
actually informs a decision: "ours: 91.4 at 92 GPU-hours, theirs: 91.2 at
14".

Measurement is best-effort and honest about it: ``measured`` says which
fields were really observed rather than estimated, and the reporter never
presents an estimate as a measurement.
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field
from typing import Any

from fabric.core.entities import Resources

try:  # pragma: no cover - POSIX only
    import resource as _posix_resource
except ImportError:  # pragma: no cover - Windows
    _posix_resource = None


def peak_rss_bytes() -> int:
    """Peak resident set size of this process, or 0 when unavailable."""
    if _posix_resource is not None:
        usage = _posix_resource.getrusage(_posix_resource.RUSAGE_SELF)
        # ru_maxrss is kilobytes on Linux and bytes on macOS.
        raw = int(usage.ru_maxrss)
        return raw * 1024 if platform.system() == "Linux" else raw
    if platform.system() == "Windows":
        try:
            import ctypes
            from ctypes import wintypes

            class _Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _Counters()
            counters.cb = ctypes.sizeof(_Counters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()  # type: ignore[attr-defined]
            if ctypes.windll.psapi.GetProcessMemoryInfo(  # type: ignore[attr-defined]
                handle, ctypes.byref(counters), counters.cb
            ):
                return int(counters.PeakWorkingSetSize)
        except Exception:  # pragma: no cover - platform specific
            return 0
    return 0


def gpu_memory_bytes() -> int:
    """Currently used GPU memory across visible devices, or 0."""
    import subprocess

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return 0
    if result.returncode != 0:
        return 0
    total = 0
    for line in result.stdout.splitlines():
        try:
            total += int(float(line.strip())) * 1024 * 1024
        except ValueError:
            continue
    return total


@dataclass
class ResourceMonitor:
    """Context manager that measures one run's resource use."""

    track_gpu: bool = False
    cost_per_gpu_hour: float | None = None
    cost_per_cpu_hour: float | None = None
    energy_watts: float | None = None
    """Declared average power draw, used only if the user supplies it.

    Fabric will not invent an energy figure from thin air; without a
    declared wattage the field stays ``None`` and the report says
    "not measured" instead of printing a fabricated number.
    """

    started_wall: float = field(default=0.0, init=False)
    started_cpu: float = field(default=0.0, init=False)
    baseline_gpu: int = field(default=0, init=False)
    measured: dict[str, bool] = field(default_factory=dict, init=False)
    result: Resources = field(default_factory=Resources, init=False)

    def __enter__(self) -> ResourceMonitor:
        self.started_wall = time.perf_counter()
        self.started_cpu = time.process_time()
        self.baseline_gpu = gpu_memory_bytes() if self.track_gpu else 0
        return self

    def __exit__(self, *exc_info: Any) -> None:
        wall = max(0.0, time.perf_counter() - self.started_wall)
        cpu = max(0.0, time.process_time() - self.started_cpu)
        gpu_memory = 0
        if self.track_gpu:
            gpu_memory = max(0, gpu_memory_bytes() - self.baseline_gpu)
        rss = peak_rss_bytes()

        cost: float | None = None
        if self.cost_per_gpu_hour is not None and self.track_gpu:
            cost = self.cost_per_gpu_hour * wall / 3600.0
        elif self.cost_per_cpu_hour is not None:
            cost = self.cost_per_cpu_hour * wall / 3600.0

        energy: float | None = None
        if self.energy_watts is not None:
            energy = self.energy_watts * wall

        self.measured = {
            "wall_seconds": True,
            "cpu_seconds": True,
            "gpu_seconds": self.track_gpu,
            "peak_rss_bytes": rss > 0,
            "gpu_memory_bytes": self.track_gpu and gpu_memory > 0,
            "energy_joules": energy is not None,
            "cost_usd": cost is not None,
        }
        self.result = Resources(
            wall_seconds=wall,
            cpu_seconds=cpu,
            gpu_seconds=wall if self.track_gpu else 0.0,
            peak_rss_bytes=rss,
            gpu_memory_bytes=gpu_memory,
            energy_joules=energy,
            energy_watts=self.energy_watts,
            cost_usd=cost,
        )


def aggregate(resources: list[Resources]) -> dict[str, Any]:
    """Total resource use across many runs."""
    if not resources:
        return {"runs": 0}
    total_wall = sum(r.wall_seconds for r in resources)
    total_cpu = sum(r.cpu_seconds for r in resources)
    total_gpu = sum(r.gpu_seconds for r in resources)
    costs = [r.cost_usd for r in resources if r.cost_usd is not None]
    energies = [r.energy_joules for r in resources if r.energy_joules is not None]
    watts = sorted({r.energy_watts for r in resources if r.energy_watts is not None})
    return {
        "runs": len(resources),
        "wall_seconds": total_wall,
        "wall_hours": total_wall / 3600.0,
        "cpu_seconds": total_cpu,
        "gpu_seconds": total_gpu,
        "gpu_hours": total_gpu / 3600.0,
        "peak_rss_bytes": max((r.peak_rss_bytes for r in resources), default=0),
        "cost_usd": sum(costs) if costs else None,
        "energy_joules": sum(energies) if energies else None,
        "energy_source": (
            "estimated: declared " + "/".join(f"{w:g}" for w in watts) + " W x wall time"
            if energies else "not measured: no energy_watts was declared"
        ),
        "energy_watts": watts[0] if len(watts) == 1 else (watts or None),
        "mean_wall_seconds": total_wall / len(resources),
    }
