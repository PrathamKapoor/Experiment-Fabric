"""Automatic environment capture.

Captured once per Fabric process and attached to every run.  The point
is not to produce a pretty manifest: it is that when a number fails to
reproduce six months later, the difference between the two environments
is a set operation rather than an interview.

Two deliberate choices:

* Environment variables are captured from a fixed **allowlist**.  Dumping
  ``os.environ`` into a shareable record is how API keys end up in
  version control.
* Package versions come from installed distribution metadata, not from
  importing anything.  Importing every package to read ``__version__``
  would be slow, and would execute third-party code during capture.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from functools import lru_cache
from typing import Any

from fabric.core.entities import Environment
from fabric.util.hashing import file_digest

RELEVANT_ENV_VARS = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "PYTHONHASHSEED",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
    "CUBLAS_WORKSPACE_CONFIG",
    "TF_DETERMINISTIC_OPS",
    "TORCH_USE_DETERMINISTIC_ALGORITHMS",
    "XLA_FLAGS",
    "JAX_PLATFORMS",
    "FABRIC_PROFILE",
)
"""Variables that can change a numerical result, and nothing else.

``CUBLAS_WORKSPACE_CONFIG`` and friends are here because they silently
decide whether GPU matrix multiplication is deterministic at all.
"""

LOCKFILES = (
    "requirements.txt",
    "requirements.lock",
    "uv.lock",
    "poetry.lock",
    "Pipfile.lock",
    "environment.yml",
    "conda-lock.yml",
    "pdm.lock",
    "pyproject.toml",
)


def _run(command: list[str], cwd: str | None = None, timeout: float = 5.0) -> str:
    """Run a command, returning stdout or an empty string on any failure."""
    try:
        result = subprocess.run(
            command, cwd=cwd, capture_output=True, text=True,
            timeout=timeout, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def installed_packages() -> dict[str, str]:
    """Installed distributions and their versions."""
    packages: dict[str, str] = {}
    try:
        from importlib import metadata
    except ImportError:  # pragma: no cover - Python < 3.8
        return packages
    for dist in metadata.distributions():
        try:
            name = dist.metadata["Name"]
            version = dist.version
        except Exception:  # pragma: no cover - malformed distribution metadata
            continue
        if name:
            packages[str(name).lower()] = str(version)
    return dict(sorted(packages.items()))


def git_state(path: str | None = None) -> tuple[str, bool, str]:
    """Return ``(commit, dirty, branch)`` for the repository at ``path``."""
    cwd = path or os.getcwd()
    commit = _run(["git", "rev-parse", "HEAD"], cwd)
    if not commit:
        return "", False, ""
    status = _run(["git", "status", "--porcelain"], cwd)
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd)
    return commit, bool(status.strip()), branch


def detect_container() -> str:
    """Best-effort container identification."""
    for variable in ("FABRIC_CONTAINER", "DOCKER_IMAGE", "SINGULARITY_CONTAINER",
                     "APPTAINER_CONTAINER", "KUBERNETES_SERVICE_HOST"):
        value = os.environ.get(variable)
        if value:
            return f"{variable}={value}"
    if os.path.exists("/.dockerenv"):
        return "docker (unidentified image)"
    try:
        with open("/proc/1/cgroup", "r", encoding="utf-8", errors="ignore") as fh:
            content = fh.read()
        for marker in ("docker", "kubepods", "containerd", "lxc"):
            if marker in content:
                return f"{marker} (from /proc/1/cgroup)"
    except OSError:
        pass
    return ""


def accelerator_runtime() -> dict[str, str]:
    """Accelerator library versions, without importing heavy frameworks.

    Framework versions are read from distribution metadata; only the
    CUDA driver is probed with ``nvidia-smi``, because there is no other
    way to see it.
    """
    runtime: dict[str, str] = {}
    packages = _cached_packages()
    for name in ("torch", "jax", "jaxlib", "tensorflow", "cupy", "triton",
                 "nvidia-cublas-cu12", "nvidia-cudnn-cu12"):
        if name in packages:
            runtime[name] = packages[name]
    smi = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if smi:
        runtime["nvidia_driver"] = smi.splitlines()[0].strip()
    cuda = _run(["nvcc", "--version"])
    if cuda:
        for line in cuda.splitlines():
            if "release" in line:
                runtime["nvcc"] = line.split("release")[-1].strip().strip(",")
                break
    return runtime


@lru_cache(maxsize=1)
def _cached_packages() -> dict[str, str]:
    return installed_packages()


def lockfile_digests(root: str) -> dict[str, str]:
    """Digests of any dependency lockfiles found at ``root``."""
    digests: dict[str, str] = {}
    for name in LOCKFILES:
        path = os.path.join(root, name)
        if os.path.isfile(path):
            digests[name] = file_digest(path)
    return digests


@lru_cache(maxsize=8)
def capture_environment(root: str = "", *, include_packages: bool = True) -> Environment:
    """Capture the current software environment.

    Cached per root: capture costs tens of milliseconds and an experiment
    with four hundred runs should pay it once.
    """
    root = root or os.getcwd()
    commit, dirty, _branch = git_state(root)
    packages = _cached_packages() if include_packages else {}
    return Environment(
        python_version=platform.python_version(),
        platform=platform.platform(),
        os_name=platform.system(),
        os_release=platform.release(),
        packages=packages,
        interpreter=f"{platform.python_implementation()} {sys.version.split()[0]}",
        git_commit=commit,
        git_dirty=dirty,
        container=detect_container(),
        lockfiles=lockfile_digests(root),
        env_vars={k: os.environ[k] for k in RELEVANT_ENV_VARS if k in os.environ},
        accelerator_runtime=accelerator_runtime(),
        captured_at=time.time(),
    )


def describe(environment: Environment) -> list[tuple[str, str]]:
    """Flatten an environment into display rows."""
    rows = [
        ("python", environment.python_version),
        ("interpreter", environment.interpreter),
        ("platform", environment.platform),
        ("git commit", (environment.git_commit[:12] + (" (dirty)" if environment.git_dirty else ""))
         if environment.git_commit else "not a git repository"),
        ("container", environment.container or "none detected"),
        ("packages", str(len(environment.packages))),
    ]
    for key, value in environment.accelerator_runtime.items():
        rows.append((key, value))
    for key, value in environment.env_vars.items():
        rows.append((f"env:{key}", value))
    return rows
