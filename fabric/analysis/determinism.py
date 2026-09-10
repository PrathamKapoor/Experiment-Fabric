"""Determinism testing.

Almost everyone believes that fixing the seed makes a pipeline
deterministic.  Very often it does not: thread scheduling, non-associative
floating-point reduction, hash ordering, and non-deterministic GPU kernels
all survive a fixed seed.

Fabric tests the belief instead of assuming it.  A determinism replicate
is the *same run executed again* -- same seed, same code, same data, same
machine -- and any difference between replicates is a lower bound on the
noise that no amount of seed-fixing will remove.

That bound matters downstream: an effect smaller than a pipeline's own
non-determinism cannot be measured by that pipeline at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from fabric.analysis.results import ConditionResults, ResultSet
from fabric.core.entities import MetricSpec


@dataclass
class MetricDeterminism:
    """Determinism evidence for one metric of one condition."""

    metric: str
    condition: str
    seed: int
    values: tuple[float, ...]
    n_replicates: int

    @property
    def identical(self) -> bool:
        return len(set(self.values)) <= 1

    @property
    def max_deviation(self) -> float:
        if len(self.values) < 2:
            return 0.0
        return max(self.values) - min(self.values)

    @property
    def relative_deviation(self) -> float:
        centre = sum(self.values) / len(self.values) if self.values else 0.0
        if centre == 0:
            return 0.0
        return self.max_deviation / abs(centre)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "condition": self.condition,
            "seed": self.seed,
            "values": list(self.values),
            "n_replicates": self.n_replicates,
            "identical": self.identical,
            "max_deviation": self.max_deviation,
            "relative_deviation": self.relative_deviation,
        }


@dataclass
class DeterminismReport:
    """Whether the pipeline reproduces itself bit for bit."""

    tested: bool = False
    results: list[MetricDeterminism] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def deterministic(self) -> bool:
        return self.tested and all(r.identical for r in self.results)

    @property
    def verdict(self) -> str:
        if not self.tested:
            return "not_tested"
        return "deterministic" if self.deterministic else "non_deterministic"

    @property
    def worst(self) -> MetricDeterminism | None:
        candidates = [r for r in self.results if not r.identical]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.relative_deviation)

    def noise_floor(self, metric: str) -> float:
        """Largest observed deviation between identical runs of one metric.

        This is the smallest effect the pipeline could ever detect, and
        the reproducibility analysis uses it to say when a "difference"
        is smaller than the machine's own jitter.
        """
        deviations = [r.max_deviation for r in self.results if r.metric == metric]
        return max(deviations) if deviations else 0.0

    def describe(self) -> str:
        if not self.tested:
            return (
                "Determinism was not tested. Add "
                "'reproducibility: {determinism_replicates: 3}' to the specification "
                "to have Fabric execute identical runs and check."
            )
        if self.deterministic:
            n = max((r.n_replicates for r in self.results), default=0)
            return (
                f"Deterministic: {n} executions of an identical run produced identical "
                "values on every metric."
            )
        worst = self.worst
        assert worst is not None
        return (
            f"NOT deterministic. Identical runs differ by up to {worst.max_deviation:.3g} "
            f"on {worst.metric} ({worst.relative_deviation * 100:.4f}% of its value). "
            "Any effect smaller than this cannot be measured by this pipeline, "
            "however many seeds you run."
        )

    def to_dict(self) -> dict[str, Any]:
        worst = self.worst
        return {
            "tested": self.tested,
            "deterministic": self.deterministic,
            "verdict": self.verdict,
            "results": [r.to_dict() for r in self.results],
            "worst": worst.to_dict() if worst else None,
            "description": self.describe(),
            "notes": self.notes,
        }


def analyse_determinism(results: ResultSet, *,
                        metrics: Sequence[MetricSpec]) -> DeterminismReport:
    """Compare determinism replicates against their original runs."""
    report = DeterminismReport()
    for condition in results.conditions.values():
        replicates = condition.replicates
        if not replicates:
            continue
        by_seed: dict[int, list[float]] = {}
        for spec in metrics:
            grouped: dict[int, list[float]] = {}
            for run in condition.runs:
                if run.succeeded and spec.name in run.metrics:
                    grouped.setdefault(run.seed, []).append(run.metrics[spec.name])
            for seed, values in grouped.items():
                if len(values) < 2:
                    continue
                report.tested = True
                report.results.append(MetricDeterminism(
                    metric=spec.name,
                    condition=condition.name,
                    seed=seed,
                    values=tuple(values),
                    n_replicates=len(values),
                ))
    if report.tested and not report.deterministic:
        report.notes.append(
            "Non-determinism under a fixed seed usually comes from thread scheduling, "
            "non-associative floating-point reduction, or non-deterministic GPU kernels. "
            "Setting CUBLAS_WORKSPACE_CONFIG and enabling deterministic algorithms in "
            "your framework will often, but not always, remove it."
        )
    return report
