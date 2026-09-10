"""Entrypoints used by the tests.

Kept deliberately small and deterministic: these exist to exercise
Fabric's execution contract, not to model anything.
"""

from __future__ import annotations

from fabric.execution.adapters import RunContext


def noop(ctx: RunContext) -> dict[str, float]:
    """A run whose result depends only on its seed and configuration."""
    rng = ctx.rng()
    scale = float(ctx.get("scale", 1.0))
    base = 0.80 + 0.05 * scale
    return {
        "accuracy": base + rng.gauss(0.0, 0.01),
        "latency_ms": 100.0 * scale + rng.uniform(-2.0, 2.0),
    }


def ablatable(ctx: RunContext) -> dict[str, float]:
    """A run whose accuracy depends on which components survive.

    ``scale`` shifts the result, so a configuration perturbation has
    something to perturb.
    """
    rng = ctx.rng()
    score = 0.5 - 0.05 * (float(ctx.get("scale", 1.0)) - 1.0)
    if ctx.enabled("alpha"):
        score += 0.2
    if ctx.enabled("beta"):
        score += 0.05
    ctx.write_artifact("trace.txt", f"enabled={sorted(set(('alpha', 'beta')) - set(ctx.ablated))}")
    return {"accuracy": score + rng.gauss(0.0, 0.005)}


def deterministic(ctx: RunContext) -> dict[str, float]:
    """Always the same value for a given seed."""
    return {"accuracy": 0.5 + (ctx.seed % 7) / 100.0}


def always_fails(ctx: RunContext) -> dict[str, float]:
    """Raises a failure the classifier should recognise as an OOM."""
    raise MemoryError("CUDA out of memory: tried to allocate 8.00 GiB")


def fails_above_threshold(ctx: RunContext) -> dict[str, float]:
    """Fails only for large configurations, so failures cluster."""
    if float(ctx.get("scale", 1.0)) >= 4.0:
        raise MemoryError("CUDA out of memory: tried to allocate 8.00 GiB")
    return noop(ctx)


def returns_nothing(ctx: RunContext) -> None:
    """Produces no metrics at all, which must be treated as a failure."""
    return None
