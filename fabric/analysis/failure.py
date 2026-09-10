"""Failure analysis.

Failures carry information that successes do not.  "This method OOMs
above batch size 64" is a property of the method; dropping those runs and
reporting the mean of what survived is a form of selection that quietly
flatters whichever arm crashed most.

This module reports the failure rate, classifies causes, and looks for
configuration values that failures cluster around.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from fabric.analysis.results import ResultSet
from fabric.core.entities import FailureClass, Run
from fabric.execution import failures as failure_module
from fabric.statistics.confidence import proportion_interval
from fabric.statistics.hypothesis import chi_square_independence


@dataclass
class Association:
    """A configuration value that failures cluster around."""

    parameter: str
    value: str
    failure_rate: float
    n_runs: int
    n_failed: int
    baseline_rate: float
    p_value: float = float("nan")

    @property
    def lift(self) -> float:
        return self.failure_rate / self.baseline_rate if self.baseline_rate else float("inf")

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "value": self.value,
            "failure_rate": self.failure_rate,
            "n_runs": self.n_runs,
            "n_failed": self.n_failed,
            "baseline_rate": self.baseline_rate,
            "lift": self.lift,
            "p_value": self.p_value,
        }


@dataclass
class FailureAnalysis:
    """Every failure in an experiment, and what they have in common."""

    total_runs: int = 0
    failed_runs: int = 0
    by_class: dict[str, int] = field(default_factory=dict)
    by_condition: dict[str, dict[str, Any]] = field(default_factory=dict)
    associations: list[Association] = field(default_factory=list)
    examples: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        return self.failed_runs / self.total_runs if self.total_runs else 0.0

    @property
    def primary_class(self) -> str:
        return next(iter(self.by_class), "")

    def to_dict(self) -> dict[str, Any]:
        interval = proportion_interval(self.failed_runs, self.total_runs)
        return {
            "total_runs": self.total_runs,
            "failed_runs": self.failed_runs,
            "failure_rate": self.failure_rate,
            "failure_rate_interval": interval.to_dict(),
            "by_class": self.by_class,
            "by_condition": self.by_condition,
            "associations": [a.to_dict() for a in self.associations],
            "examples": self.examples,
            "primary_class": self.primary_class,
            "notes": self.notes,
        }

    def summary_line(self) -> str:
        if not self.total_runs:
            return "no runs"
        if not self.failed_runs:
            return f"0 failures in {self.total_runs} runs"
        return (
            f"{self.failed_runs}/{self.total_runs} runs failed "
            f"({self.failure_rate * 100:.1f}%), most often: {self.primary_class}"
        )


def analyse_failures(results: ResultSet, *, min_group: int = 2) -> FailureAnalysis:
    """Summarise and attribute the failures in an experiment."""
    analysis = FailureAnalysis()
    all_runs = results.all_runs
    analysis.total_runs = len(all_runs)
    failed = [r for r in all_runs if r.failure is not None]
    analysis.failed_runs = len(failed)
    if not failed:
        return analysis

    analysis.by_class = failure_module.summarise([r.failure for r in failed if r.failure])

    for condition in results.conditions.values():
        if not condition.failed:
            continue
        analysis.by_condition[condition.name] = {
            "n_runs": len(condition.runs),
            "n_failed": len(condition.failed),
            "failure_rate": condition.failure_rate,
            "classes": failure_module.summarise(
                [r.failure for r in condition.failed if r.failure]
            ),
        }

    seen: set[str] = set()
    for run in failed:
        if run.failure is None:
            continue
        key = run.failure.failure_class.value
        if key in seen:
            continue
        seen.add(key)
        analysis.examples.append({
            "run_id": run.id,
            "condition": run.condition_name,
            "seed": run.seed,
            "class": key,
            "message": run.failure.message.strip().splitlines()[0][:300]
            if run.failure.message.strip() else "",
        })

    analysis.associations = _associations(all_runs, analysis.failure_rate, min_group)
    _add_notes(analysis)
    return analysis


def _associations(runs: Sequence[Run], baseline_rate: float,
                  min_group: int) -> list[Association]:
    """Find configuration values whose runs fail unusually often."""
    buckets: dict[tuple[str, str], list[Run]] = {}
    for run in runs:
        for key, value in run.config.items():
            buckets.setdefault((str(key), _format(value)), []).append(run)

    associations: list[Association] = []
    for (parameter, value), group in sorted(buckets.items()):
        if len(group) < min_group:
            continue
        failed = sum(1 for r in group if r.failure is not None)
        if failed == 0:
            continue
        rate = failed / len(group)
        if rate <= baseline_rate:
            continue
        others = [r for r in runs if r not in group]
        other_failed = sum(1 for r in others if r.failure is not None)
        test = chi_square_independence([
            [failed, len(group) - failed],
            [other_failed, len(others) - other_failed],
        ])
        associations.append(Association(
            parameter=parameter,
            value=value,
            failure_rate=rate,
            n_runs=len(group),
            n_failed=failed,
            baseline_rate=baseline_rate,
            p_value=test.p_value,
        ))
    associations.sort(key=lambda a: (-a.failure_rate, a.parameter))
    return associations


def _add_notes(analysis: FailureAnalysis) -> None:
    infrastructure = sum(
        count for name, count in analysis.by_class.items()
        if failure_module.is_infrastructure(FailureClass(name))
    )
    if infrastructure:
        analysis.notes.append(
            f"{infrastructure} failure(s) were infrastructure problems (network or "
            "dependency), which say nothing about the methods under test."
        )
    method_failures = analysis.failed_runs - infrastructure
    if method_failures:
        analysis.notes.append(
            f"{method_failures} failure(s) came from the experiment itself and are "
            "part of the result: a method that cannot complete a configuration has not "
            "scored zero on it, it has failed to produce a score at all."
        )
    for association in analysis.associations[:3]:
        if association.p_value == association.p_value and association.p_value < 0.05:
            analysis.notes.append(
                f"Failures concentrate at {association.parameter}={association.value}: "
                f"{association.n_failed}/{association.n_runs} runs failed there "
                f"({association.lift:.1f}x the overall rate, chi-square p="
                f"{association.p_value:.4f})."
            )


def _format(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)
