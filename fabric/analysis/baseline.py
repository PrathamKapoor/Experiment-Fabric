"""Baseline comparison.

The step most often skipped.  "Our model achieves 92.3%" is not a result;
"our model achieves 92.3% where the baseline achieves 92.1%, seed
variance is 0.4, and the paired difference is not distinguishable from
zero" is.

This module produces the second sentence, for every baseline and every
metric, with the multiplicity of the comparison family accounted for.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from fabric.analysis.results import ResultSet
from fabric.core.entities import MetricSpec, Role
from fabric.statistics.compare import Comparison, compare
from fabric.statistics.hypothesis import TestResult, one_way_anova
from fabric.statistics.multiple import Correction, adjust


@dataclass
class BaselineAnalysis:
    """Every treatment-versus-baseline comparison in one experiment."""

    comparisons: list[Comparison] = field(default_factory=list)
    corrections: dict[str, Correction] = field(default_factory=dict)
    omnibus: dict[str, TestResult] = field(default_factory=dict)
    treatment: str = ""
    treatment_selection_note: str = ""
    notes: list[str] = field(default_factory=list)

    def for_metric(self, metric: str) -> list[Comparison]:
        return [c for c in self.comparisons if c.metric == metric]

    def best_supported_claim(self, metric: str) -> Comparison | None:
        """The weakest significant comparison on a metric.

        Deliberately the *weakest*: a method is only better than "the
        baselines" if it beats the strongest one, so the honest headline
        is the comparison that came closest to failing.
        """
        metric_comparisons = [c for c in self.for_metric(metric) if c.verdict == "improvement"]
        if not metric_comparisons or len(metric_comparisons) != len(self.for_metric(metric)):
            return None
        return min(metric_comparisons, key=lambda c: c.difference)

    def to_dict(self) -> dict[str, Any]:
        return {
            "treatment": self.treatment,
            "treatment_selection_note": self.treatment_selection_note,
            "comparisons": [c.to_dict() for c in self.comparisons],
            "corrections": {k: v.to_dict() for k, v in self.corrections.items()},
            "omnibus": {k: v.to_dict() for k, v in self.omnibus.items()},
            "notes": self.notes,
        }


def analyse_baselines(
    results: ResultSet,
    *,
    metrics: Sequence[MetricSpec],
    alpha: float = 0.05,
    correction: str = "holm",
    paired_policy: str = "auto",
    bootstrap_resamples: int = 5000,
    bootstrap_seed: int = 12345,
    treatment_name: str = "",
    omnibus: bool = True,
) -> BaselineAnalysis:
    """Compare the treatment against every baseline, on every metric."""
    analysis = BaselineAnalysis()
    baselines = results.baselines

    # Resolve the treatment *before* the no-baseline exit. Without it the
    # headline reports "no treatment produced results" for an experiment
    # that ran perfectly well and simply had nothing to compare against --
    # a false statement in the most prominent line of the report.
    primary = metrics[0].name if metrics else ""
    treatment = (
        results.conditions.get(treatment_name)
        if treatment_name else results.primary_treatment(primary)
    )
    if treatment is not None:
        analysis.treatment = treatment.name

    if not baselines:
        analysis.notes.append(
            "No baselines were declared, so no comparative claim can be made. "
            "The numbers below describe the method; they do not establish that it is better."
        )
        return analysis

    if treatment is None:
        analysis.notes.append("No treatment condition produced results.")
        return analysis

    candidates = results.treatments
    if len(candidates) > 1:
        analysis.treatment_selection_note = (
            f"{len(candidates)} treatment configurations were run; '{treatment.name}' had the "
            f"best mean {primary} and is the arm compared below. Selecting the best "
            "configuration after seeing the results inflates the apparent effect. "
            "The per-configuration table reports all of them."
        )
        analysis.notes.append(analysis.treatment_selection_note)

    for spec in metrics:
        if not treatment.series(spec.name):
            continue
        groups = [treatment.values(spec.name)]
        labels = [treatment.name]
        for baseline in baselines:
            if baseline.series(spec.name):
                groups.append(baseline.values(spec.name))
                labels.append(baseline.name)

        if omnibus and len(groups) >= 3:
            # An omnibus test first: if nothing differs overall, the
            # pairwise tests below are exploratory, not confirmatory.
            analysis.omnibus[spec.name] = one_way_anova(*groups, alpha=alpha)

        for baseline in baselines:
            if not baseline.series(spec.name):
                continue
            left, right, seeds = results.aligned(treatment.name, baseline.name, spec.name)
            if len(seeds) < 2:
                continue
            if paired_policy == "never":
                paired: bool | None = False
            elif paired_policy == "always":
                paired = True
            else:
                paired = results.can_pair(treatment.name, baseline.name, spec.name)
            comparison = compare(
                left, right,
                metric=spec.name,
                treatment=treatment.name,
                baseline=baseline.name,
                higher_is_better=spec.direction.sign >= 0,
                paired=paired,
                alpha=alpha,
                tolerance=spec.tolerance,
                bootstrap_resamples=bootstrap_resamples,
                bootstrap_seed=bootstrap_seed,
                group="baseline",
            )
            analysis.comparisons.append(comparison)

    # Correct within each metric's family: the multiplicity that matters
    # is "how many baselines did we test on this metric", not the total
    # number of numbers in the report.
    for spec in metrics:
        family = analysis.for_metric(spec.name)
        if len(family) < 1:
            continue
        result = adjust([(c.baseline, c.p_value) for c in family],
                        method=correction, alpha=alpha)
        analysis.corrections[spec.name] = result
        for index, comparison in enumerate(analysis.comparisons):
            if comparison.metric != spec.name:
                continue
            entry = result.by_label(comparison.baseline)
            if entry:
                analysis.comparisons[index] = comparison.with_adjustment(
                    entry.adjusted_p, correction
                )

    return analysis


def cost_effectiveness(results: ResultSet, metric: str) -> list[dict[str, Any]]:
    """Accuracy against compute, for every condition that recorded both.

    Turns "is +0.2 points worth 6.5x the compute?" into a column a reader
    can scan, rather than a question nobody asks.
    """
    rows: list[dict[str, Any]] = []
    spec = results.metric_spec(metric)
    reference: float | None = None
    for condition in results.conditions.values():
        if condition.perturbation or not condition.series(metric):
            continue
        summary = condition.summary(metric)
        resources = condition.resources()
        wall = resources.get("wall_seconds", 0.0)
        rows.append({
            "condition": condition.name,
            "role": condition.role.value,
            metric: summary.mean,
            "wall_seconds": wall,
            "wall_hours": resources.get("wall_hours", 0.0),
            "gpu_hours": resources.get("gpu_hours", 0.0),
            "cost_usd": resources.get("cost_usd"),
            "runs": len(condition.successful),
        })
        if condition.role == Role.BASELINE and reference is None:
            reference = summary.mean

    baseline_rows = [r for r in rows if r["role"] == "baseline"]
    if baseline_rows:
        anchor = min(baseline_rows, key=lambda r: r["wall_seconds"] or 0.0)
        for row in rows:
            delta = (row[metric] - anchor[metric]) * (spec.direction.sign or 1)
            extra_time = (row["wall_seconds"] or 0.0) - (anchor["wall_seconds"] or 0.0)
            row["vs_cheapest_baseline"] = anchor["condition"]
            row["metric_delta"] = delta
            row["extra_wall_seconds"] = extra_time
            row["compute_multiple"] = (
                (row["wall_seconds"] / anchor["wall_seconds"])
                if anchor["wall_seconds"] else None
            )
            row["metric_gain_per_extra_hour"] = (
                delta / (extra_time / 3600.0) if extra_time > 0 else None
            )
    rows.sort(key=lambda r: -r[metric] * (spec.direction.sign or 1))
    return rows
