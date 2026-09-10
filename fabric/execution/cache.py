"""Execution caching.

If the dataset, the model, the code, the configuration, the seed and the
environment are all identical, the computation has already been done.
Reusing it can save weeks of GPU time on a sweep that changes one arm.

The danger is the opposite failure: a cache that hits when it should
not, silently serving a stale number from before you fixed a bug.  Fabric
guards against that by putting the *source digest* of the entrypoint into
the key, so editing the method invalidates its cached runs, and by
recording in every cached run which original run it came from.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from fabric.core.entities import Environment, Run
from fabric.environments.fingerprint import NUMERICALLY_RELEVANT
from fabric.util.hashing import content_id


@dataclass(frozen=True)
class CachePolicy:
    """What counts as "the same computation"."""

    enabled: bool = True
    strict_environment: bool = False
    """Include the *entire* environment in the key.

    Off by default: a linter upgrade should not invalidate a week of
    results.  On, nothing is reused across any environment change at all,
    which is the right setting when the environment is itself under study.
    """
    include_hardware: bool = False
    """Include the machine identity in the key.

    Off by default because most experiments are hardware-insensitive by
    assumption; turn it on when that assumption is what you are testing.
    """

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "strict_environment": self.strict_environment,
            "include_hardware": self.include_hardware,
        }


def environment_key(environment: Environment, *, strict: bool) -> dict[str, Any]:
    """The part of an environment that participates in the cache key."""
    if strict:
        return {"environment_id": environment.id}
    return {
        "python": environment.python_version,
        "git_commit": environment.git_commit,
        "git_dirty": environment.git_dirty,
        "packages": {
            name: version
            for name, version in environment.packages.items()
            if name in NUMERICALLY_RELEVANT
        },
        "runtime": dict(environment.accelerator_runtime),
        "env_vars": dict(environment.env_vars),
    }


def cache_key(
    run: Run,
    *,
    algorithm_id: str,
    source_digest: str,
    environment: Environment,
    policy: CachePolicy,
    hardware_id: str = "",
    env_overlay: Mapping[str, str] | None = None,
) -> str:
    """Compute the cache key for one run.

    A determinism replicate never matches the cache: its entire purpose is
    to execute the same inputs again, so serving it from cache would make
    every pipeline look perfectly deterministic.
    """
    payload: dict[str, Any] = {
        "algorithm": algorithm_id,
        "source": source_digest,
        "dataset": run.dataset_id,
        "model": run.model_id,
        "config": dict(run.config),
        "seed": run.seed,
        "ablated": sorted(run.ablated),
        "environment": environment_key(environment, strict=policy.strict_environment),
        "env_overlay": dict(env_overlay or {}),
    }
    if policy.include_hardware:
        payload["hardware"] = hardware_id
    return "ck-" + content_id(payload, length=24)


def is_cacheable(run: Run, policy: CachePolicy) -> bool:
    """Whether this run may be served from, or written to, the cache."""
    if not policy.enabled:
        return False
    return not run.replicate
