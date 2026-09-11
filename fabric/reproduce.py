"""Reproduction: running an experiment again and comparing honestly.

A reproduction attempt is itself an experiment, with the original result
as its reference.  So it gets the same treatment: paired where possible,
tested rather than eyeballed, and reported with an interval.

The verdict has four levels, and the distinction between the middle two
is the one that matters:

``PASS``
    The reproduced result is statistically equivalent to the original,
    within the tolerance the metric declared *in advance*.
``PASS WITH DEVIATION``
    The difference is real but the environment differed in ways that
    plausibly explain it; the deviations are named.
``INCONCLUSIVE``
    Neither equivalence nor a difference could be established. Usually a
    sample-size problem, and reported as one rather than rounded to
    "pass".
``FAIL``
    The numbers differ and the environment does not explain it -- either
    beyond the declared tolerance, or with no tolerance declared to judge
    them against, in which case the report says so rather than implying a
    threshold nobody set.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from fabric.analysis.results import ResultSet
from fabric.core.entities import Environment, Hardware, Run
from fabric.environments.fingerprint import (
    Difference,
    environment_diff,
    hardware_diff,
    likely_causes,
)
from fabric.execution.planner import ExperimentPlan
from fabric.statistics.compare import Comparison, compare

PASS = "PASS"
PASS_WITH_DEVIATION = "PASS WITH DEVIATION"
INCONCLUSIVE = "INCONCLUSIVE"
FAIL = "FAIL"


@dataclass
class MetricReproduction:
    """One metric of one condition, original against reproduced."""

    condition: str
    metric: str
    original_mean: float
    reproduced_mean: float
    comparison: Comparison | None = None
    tolerance: float | None = None
    n_original: int = 0
    n_reproduced: int = 0
    exact: bool = False
    """Every seed reproduced its original value bit for bit.

    This is stronger evidence than any tolerance test, and it needs no
    declared margin to be conclusive: there is no difference to bound.
    Treating it as INCONCLUSIVE because nobody wrote down a tolerance
    would report the best possible outcome as the most uncertain one.
    """

    @property
    def difference(self) -> float:
        return self.reproduced_mean - self.original_mean

    @property
    def relative(self) -> float:
        if self.original_mean == 0:
            return float("nan")
        return self.difference / abs(self.original_mean) * 100.0

    @property
    def verdict(self) -> str:
        if self.exact:
            return PASS
        if self.comparison is None:
            return INCONCLUSIVE
        if self.comparison.equivalent:
            return PASS
        interval = self.comparison.difference_interval
        if self.tolerance is not None and interval.low == interval.low:
            if interval.low > abs(self.tolerance) or interval.high < -abs(self.tolerance):
                return FAIL
            return INCONCLUSIVE
        return INCONCLUSIVE if not self.comparison.significant else FAIL

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "metric": self.metric,
            "original_mean": self.original_mean,
            "reproduced_mean": self.reproduced_mean,
            "difference": self.difference,
            "relative_percent": self.relative,
            "tolerance": self.tolerance,
            "n_original": self.n_original,
            "n_reproduced": self.n_reproduced,
            "exact": self.exact,
            "verdict": self.verdict,
            "comparison": self.comparison.to_dict(include_values=False)
            if self.comparison else None,
        }


@dataclass
class ReproductionReport:
    """The outcome of a reproduction attempt."""

    experiment_id: str
    name: str = ""
    primary_metric: str = ""
    metrics: list[MetricReproduction] = field(default_factory=list)
    environment_differences: list[Difference] = field(default_factory=list)
    hardware_differences: list[Difference] = field(default_factory=list)
    causes: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    notes: list[str] = field(default_factory=list)

    def for_metric(self, metric: str) -> list[MetricReproduction]:
        return [m for m in self.metrics if m.metric == metric]

    @property
    def headline(self) -> MetricReproduction | None:
        """The primary metric on the primary condition."""
        candidates = self.for_metric(self.primary_metric)
        return candidates[0] if candidates else None

    @property
    def environment_changed(self) -> bool:
        return bool(self.environment_differences or self.hardware_differences)

    @property
    def verdict(self) -> str:
        if not self.metrics:
            return INCONCLUSIVE
        verdicts = {m.verdict for m in self.metrics}
        if FAIL in verdicts:
            return PASS_WITH_DEVIATION if self.environment_changed else FAIL
        if INCONCLUSIVE in verdicts:
            return INCONCLUSIVE
        return PASS_WITH_DEVIATION if self.environment_changed else PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "primary_metric": self.primary_metric,
            "verdict": self.verdict,
            "metrics": [m.to_dict() for m in self.metrics],
            "environment_differences": [d.to_dict() for d in self.environment_differences],
            "hardware_differences": [d.to_dict() for d in self.hardware_differences],
            "likely_causes": self.causes,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "notes": self.notes,
        }

    def render(self) -> str:
        """The terminal rendering of ``fabric reproduce``."""
        lines = ["REPRODUCTION COMPLETE", ""]
        headline = self.headline
        if headline:
            exactness = "  (exact: every seed reproduced its value)" if headline.exact else ""
            lines.extend([
                f"  original result       {headline.original_mean:.4f}",
                f"  reproduced result     {headline.reproduced_mean:.4f}",
                f"  difference            {headline.difference:+.4f} "
                f"({headline.relative:+.2f}%){exactness}",
            ])
            if headline.comparison:
                lines.append(
                    f"  95% CI on difference  "
                    f"{headline.comparison.difference_interval.format()}"
                )
                if headline.comparison.equivalence:
                    lines.append(
                        f"  equivalence (TOST)    p="
                        f"{headline.comparison.equivalence.p_value:.4f} against a margin "
                        f"of {headline.tolerance:g}"
                    )
            lines.append("")

        if len(self.metrics) > 1:
            lines.append("  per metric")
            width = max(len(m.metric) for m in self.metrics)
            for item in self.metrics:
                marker = " (exact)" if item.exact else ""
                lines.append(
                    f"    {item.metric:<{width}}  {item.original_mean:>10.4f} -> "
                    f"{item.reproduced_mean:>10.4f}  {item.difference:+.4f}  "
                    f"{item.verdict}{marker}"
                )
            lines.append("")

        lines.append(f"  reproducibility       {self.verdict}")
        lines.append("")
        if self.environment_differences or self.hardware_differences:
            lines.append("  deviations from the original environment")
            for difference in (self.hardware_differences + self.environment_differences)[:8]:
                lines.append(f"    [{difference.relevance}] {difference}")
            lines.append("")
        if self.causes:
            lines.append("  plausible explanations (hypotheses, not findings)")
            for cause in self.causes:
                lines.append(f"    - {cause}")
            lines.append("")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


def compare_runs(
    plan: ExperimentPlan,
    original: Sequence[Run],
    reproduced: Sequence[Run],
    *,
    original_environment: Environment | None = None,
    reproduced_environment: Environment | None = None,
    original_hardware: Hardware | None = None,
    reproduced_hardware: Hardware | None = None,
    alpha: float = 0.05,
    conditions: Sequence[str] = (),
) -> ReproductionReport:
    """Compare a reproduction against the original runs."""
    spec = plan.spec
    metrics = list(spec.metrics)
    primary = spec.primary_metric

    report = ReproductionReport(
        experiment_id=plan.experiment.id,
        name=spec.name,
        primary_metric=primary,
    )

    original_set = ResultSet.build(original, plan.conditions, metrics)
    reproduced_set = ResultSet.build(reproduced, plan.conditions, metrics)

    targets = list(conditions)
    if not targets:
        reference = original_set.primary_treatment(primary)
        targets = [reference.name] if reference else []
    if not targets:
        report.notes.append("The original experiment has no condition to reproduce.")
        return report

    for condition_name in targets:
        if condition_name not in original_set.conditions:
            continue
        for spec_metric in metrics:
            original_values = original_set[condition_name].values(spec_metric.name)
            reproduced_values = (
                reproduced_set[condition_name].values(spec_metric.name)
                if condition_name in reproduced_set.conditions else []
            )
            if not original_values or not reproduced_values:
                continue
            paired = _paired_by_seed(original_set, reproduced_set, condition_name,
                                     spec_metric.name)
            exact = paired and _identical_by_seed(
                original_set, reproduced_set, condition_name, spec_metric.name)
            comparison = compare(
                reproduced_values, original_values,
                metric=spec_metric.name,
                treatment="reproduction",
                baseline="original",
                higher_is_better=spec_metric.direction.sign >= 0,
                paired=paired,
                alpha=alpha,
                tolerance=spec_metric.tolerance,
                group="reproduction",
            )
            report.metrics.append(MetricReproduction(
                condition=condition_name,
                metric=spec_metric.name,
                original_mean=sum(original_values) / len(original_values),
                reproduced_mean=sum(reproduced_values) / len(reproduced_values),
                comparison=comparison,
                tolerance=spec_metric.tolerance,
                n_original=len(original_values),
                n_reproduced=len(reproduced_values),
                exact=exact,
            ))

    report.metrics.sort(key=lambda m: (m.metric != primary, m.metric))

    if original_environment and reproduced_environment:
        report.environment_differences = environment_diff(
            original_environment, reproduced_environment)
    if original_hardware and reproduced_hardware:
        report.hardware_differences = hardware_diff(original_hardware, reproduced_hardware)
    report.causes = likely_causes(
        report.environment_differences + report.hardware_differences)

    missing_tolerance = [m.metric for m in report.metrics
                         if m.tolerance is None and not m.exact]
    if missing_tolerance:
        report.notes.append(
            f"No tolerance was declared for {', '.join(sorted(set(missing_tolerance)))}, "
            "so equivalence cannot be established for them -- only the absence of a "
            "detected difference, which is a weaker statement. Declare a 'tolerance' "
            "on each metric to make reproduction a testable claim."
        )
    if report.verdict == PASS_WITH_DEVIATION:
        report.notes.append(
            "The environment differed from the original. The listed causes are "
            "plausible explanations to test, not established ones."
        )
    if report.verdict == INCONCLUSIVE:
        report.notes.append(
            "Neither equivalence nor a difference could be established. This is "
            "usually a sample-size limit rather than a property of the method: more "
            "seeds would narrow the interval."
        )
    report.finished_at = time.time()
    return report


def _identical_by_seed(original: ResultSet, reproduced: ResultSet,
                       condition: str, metric: str) -> bool:
    """Whether every seed reproduced its value exactly."""
    left = original[condition].series(metric)
    right = reproduced[condition].series(metric)
    if not left or set(left) != set(right):
        return False
    return all(left[seed] == right[seed] for seed in left)


def _paired_by_seed(original: ResultSet, reproduced: ResultSet,
                    condition: str, metric: str) -> bool:
    """Whether the reproduction used exactly the original seed set."""
    if condition not in reproduced.conditions:
        return False
    left = set(original[condition].series(metric))
    right = set(reproduced[condition].series(metric))
    return bool(left) and left == right
