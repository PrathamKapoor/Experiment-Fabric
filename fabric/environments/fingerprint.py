"""Environment and hardware differencing.

When a reproduction deviates, the useful output is not "the environments
differ" but *which four packages* differ.  These functions produce that
list, and rank it so the things most likely to move a number appear
first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fabric.core.entities import Environment, Hardware

NUMERICALLY_RELEVANT = (
    "torch", "tensorflow", "jax", "jaxlib", "numpy", "scipy", "cupy", "triton",
    "nvidia_driver", "nvcc", "scikit-learn", "transformers", "flash-attn",
    "sentencepiece", "tokenizers", "xgboost", "lightgbm",
)
"""Packages whose version changes plausibly change a floating-point result.

Ordering the diff by this list is a heuristic, and labelled as one in the
report: a NumPy bump is a likelier explanation for a shifted mean than a
linter upgrade, but it is not proof.
"""


@dataclass(frozen=True)
class Difference:
    """One field that differs between two captured contexts."""

    field: str
    left: Any
    right: Any
    relevance: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {"field": self.field, "left": self.left, "right": self.right,
                "relevance": self.relevance}

    def __str__(self) -> str:
        return f"{self.field}: {self.left!r} -> {self.right!r}"


def _relevance(name: str) -> str:
    base = name.split(":", 1)[-1]
    if base in NUMERICALLY_RELEVANT:
        return "high"
    if base.startswith(("nvidia", "cuda", "cudnn")):
        return "high"
    return "low"


def environment_diff(left: Environment, right: Environment) -> list[Difference]:
    """Every field that differs between two environments, ranked."""
    differences: list[Difference] = []
    for field in ("python_version", "platform", "os_name", "os_release",
                  "interpreter", "git_commit", "git_dirty", "container"):
        a, b = getattr(left, field), getattr(right, field)
        if a != b:
            relevance = "high" if field in ("python_version", "git_commit") else "medium"
            differences.append(Difference(field, a, b, relevance))

    for name in sorted(set(left.packages) | set(right.packages)):
        a = left.packages.get(name)
        b = right.packages.get(name)
        if a != b:
            differences.append(Difference(f"package:{name}", a, b, _relevance(name)))

    for name in sorted(set(left.accelerator_runtime) | set(right.accelerator_runtime)):
        a = left.accelerator_runtime.get(name)
        b = right.accelerator_runtime.get(name)
        if a != b:
            differences.append(Difference(f"runtime:{name}", a, b, "high"))

    for name in sorted(set(left.env_vars) | set(right.env_vars)):
        a = left.env_vars.get(name)
        b = right.env_vars.get(name)
        if a != b:
            differences.append(Difference(f"env:{name}", a, b, "high"))

    for name in sorted(set(left.lockfiles) | set(right.lockfiles)):
        a = left.lockfiles.get(name)
        b = right.lockfiles.get(name)
        if a != b:
            differences.append(Difference(f"lockfile:{name}", a, b, "medium"))

    order = {"high": 0, "medium": 1, "low": 2, "unknown": 3}
    differences.sort(key=lambda d: (order[d.relevance], d.field))
    return differences


def hardware_diff(left: Hardware, right: Hardware) -> list[Difference]:
    """Every hardware field that differs between two machines."""
    differences: list[Difference] = []
    for field, relevance in (
        ("accelerator", "high"), ("driver", "high"), ("architecture", "high"),
        ("cpu_model", "medium"), ("cpu_count", "low"), ("machine_id", "low"),
    ):
        a, b = getattr(left, field), getattr(right, field)
        if a != b:
            differences.append(Difference(field, a, b, relevance))
    if len(left.gpus) != len(right.gpus):
        differences.append(Difference("gpu_count", len(left.gpus), len(right.gpus), "high"))
    order = {"high": 0, "medium": 1, "low": 2, "unknown": 3}
    differences.sort(key=lambda d: (order[d.relevance], d.field))
    return differences


def summarise_differences(differences: list[Difference], *, limit: int = 8) -> str:
    """One-line summary suitable for a report or a terminal."""
    if not differences:
        return "identical"
    high = [d for d in differences if d.relevance == "high"]
    head = (high or differences)[:limit]
    text = "; ".join(str(d) for d in head)
    remaining = len(differences) - len(head)
    return text + (f" (+{remaining} more)" if remaining > 0 else "")


def likely_causes(differences: list[Difference]) -> list[str]:
    """Plausible explanations for a numerical deviation, most likely first.

    Explicitly *plausible*, not established: these are hypotheses for a
    human to test, and the report labels them that way.
    """
    causes: list[str] = []
    fields = {d.field for d in differences}
    if any(f.startswith("runtime:") or f.startswith("package:torch")
           or f.startswith("package:jax") or f.startswith("package:tensorflow") for f in fields):
        causes.append(
            "a deep-learning framework or its accelerator runtime changed version, which "
            "can alter kernel selection and therefore floating-point summation order"
        )
    if "package:numpy" in fields or "package:scipy" in fields:
        causes.append(
            "NumPy/SciPy changed version; BLAS backends and reduction orders differ between builds"
        )
    if any(f.startswith("env:") for f in fields):
        causes.append(
            "a numerically relevant environment variable changed (thread counts and "
            "determinism flags change reduction order)"
        )
    if "python_version" in fields:
        causes.append("the Python version changed, which can change dict/set iteration and hashing")
    if "git_commit" in fields:
        causes.append("the source tree is at a different commit, so the implementation itself differs")
    if "accelerator" in fields or "driver" in fields or "gpu_count" in fields:
        causes.append(
            "the accelerator changed, so different kernels, tensor-core paths and "
            "reduction trees were used"
        )
    return causes
