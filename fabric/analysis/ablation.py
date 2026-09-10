"""Ablation analysis.

An ablation asks a causal question -- *what does this component
contribute?* -- so the analysis has to do more than print a table of
accuracies.  For each removed component Fabric measures the drop against
the full system on every metric, tests it, sizes it, and then reports two
things a plain table hides:

* which components are **load-bearing** and which are decoration;
* where the whole is more than the sum of its parts, by comparing a joint
  ablation against the sum of the individual ones (an interaction).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from fabric.analysis.results import ConditionResults, ResultSet
from fabric.core.entities import MetricSpec
from fabric.statistics.compare import Comparison, compare
from fabric.statistics.multiple import Correction, adjust


@dataclass
class ComponentEffect:
    """What removing one component (or set of components) did."""

    ablation: str
    removed: tuple[str, ...]
    comparisons: dict[str, Comparison] = field(default_factory=dict)

    def delta(self, metric: str) -> float:
        """Signed change caused by *removal*, positive meaning it got worse.

        Reported from the ablation's point of view: a large positive
        number means the component was doing real work.
        """
        comparison = self.comparisons.get(metric)
        return -comparison.difference if comparison else float("nan")

    def significant(self, metric: str) -> bool:
        comparison = self.comparisons.get(metric)
        return bool(comparison and comparison.significant)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ablation": self.ablation,
            "removed": list(self.removed),
            "comparisons": {k: v.to_dict(include_values=False) for k, v in self.comparisons.items()},
            "deltas": {k: self.delta(k) for k in self.comparisons},
        }


@dataclass
class Interaction:
    """Whether two components interact, rather than adding up."""

    components: tuple[str, ...]
    joint_effect: float
    additive_prediction: float
    metric: str

    @property
    def excess(self) -> float:
        """Joint effect minus what independence would have predicted."""
        return self.joint_effect - self.additive_prediction

    @property
    def kind(self) -> str:
        if abs(self.excess) < 1e-9:
            return "additive"
        return "synergistic" if self.excess > 0 else "redundant"

    def describe(self) -> str:
        if self.kind == "additive":
            return (
                f"Removing {' and '.join(self.components)} together costs exactly what "
                "removing them separately does; they act independently."
            )
        if self.kind == "synergistic":
            return (
                f"Removing {' and '.join(self.components)} together costs "
                f"{self.excess:+.4f} more {self.metric} than the sum of the individual "
                "removals: the components reinforce each other."
            )
        return (
            f"Removing {' and '.join(self.components)} together costs {abs(self.excess):.4f} "
            f"less {self.metric} than the sum of the individual removals: their roles "
            "overlap, so one partly covers for the other."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "components": list(self.components),
            "metric": self.metric,
            "joint_effect": self.joint_effect,
            "additive_prediction": self.additive_prediction,
            "excess": self.excess,
            "kind": self.kind,
            "description": self.describe(),
        }


@dataclass
class AblationAnalysis:
    """The full ablation study."""

    reference: str = ""
    effects: list[ComponentEffect] = field(default_factory=list)
    interactions: list[Interaction] = field(default_factory=list)
    corrections: dict[str, Correction] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def matrix(self, metrics: Sequence[MetricSpec]) -> list[dict[str, Any]]:
        """The ablation matrix, as rows ready for a table."""
        rows = []
        for effect in self.effects:
            row: dict[str, Any] = {
                "configuration": effect.ablation,
                "removed": list(effect.removed),
            }
            for spec in metrics:
                comparison = effect.comparisons.get(spec.name)
                if comparison:
                    row[spec.name] = comparison.treatment_summary.mean
                    row[f"{spec.name}_delta"] = effect.delta(spec.name)
                    row[f"{spec.name}_p"] = comparison.effective_p
                    row[f"{spec.name}_significant"] = comparison.significant
            rows.append(row)
        return rows

    def ranking(self, metric: str) -> list[tuple[str, float, bool]]:
        """Components ordered by how much removing them hurts."""
        single = [e for e in self.effects if len(e.removed) == 1]
        ranked = [
            (e.removed[0], e.delta(metric), e.significant(metric))
            for e in single if metric in e.comparisons
        ]
        ranked.sort(key=lambda item: -item[1])
        return ranked

    def load_bearing(self, metric: str) -> list[str]:
        return [name for name, delta, significant in self.ranking(metric)
                if significant and delta > 0]

    def decorative(self, metric: str) -> list[str]:
        return [name for name, delta, significant in self.ranking(metric)
                if not significant]

    def to_dict(self, metrics: Sequence[MetricSpec] = ()) -> dict[str, Any]:
        return {
            "reference": self.reference,
            "effects": [e.to_dict() for e in self.effects],
            "interactions": [i.to_dict() for i in self.interactions],
            "corrections": {k: v.to_dict() for k, v in self.corrections.items()},
            "matrix": self.matrix(metrics) if metrics else [],
            "notes": self.notes,
        }


def analyse_ablations(
    results: ResultSet,
    *,
    metrics: Sequence[MetricSpec],
    alpha: float = 0.05,
    correction: str = "holm",
    bootstrap_resamples: int = 5000,
    bootstrap_seed: int = 12345,
    reference_name: str = "",
) -> AblationAnalysis:
    """Measure the contribution of every ablated component."""
    analysis = AblationAnalysis()
    ablations = results.ablations
    if not ablations:
        return analysis

    primary = metrics[0].name if metrics else ""
    reference = (
        results.conditions.get(reference_name)
        if reference_name else results.primary_treatment(primary)
    )
    if reference is None:
        analysis.notes.append("No full-system condition to ablate against.")
        return analysis
    analysis.reference = reference.name

    for ablation in ablations:
        effect = ComponentEffect(ablation=ablation.name, removed=tuple(ablation.ablated))
        for spec in metrics:
            if not ablation.series(spec.name) or not reference.series(spec.name):
                continue
            left, right, seeds = results.aligned(ablation.name, reference.name, spec.name)
            if len(seeds) < 2:
                continue
            effect.comparisons[spec.name] = compare(
                left, right,
                metric=spec.name,
                treatment=ablation.name,
                baseline=reference.name,
                higher_is_better=spec.direction.sign >= 0,
                paired=results.can_pair(ablation.name, reference.name, spec.name),
                alpha=alpha,
                tolerance=spec.tolerance,
                bootstrap_resamples=bootstrap_resamples,
                bootstrap_seed=bootstrap_seed,
                group="ablation",
            )
        analysis.effects.append(effect)

    for spec in metrics:
        family = [(e.ablation, e.comparisons[spec.name].p_value)
                  for e in analysis.effects if spec.name in e.comparisons]
        if not family:
            continue
        result = adjust(family, method=correction, alpha=alpha)
        analysis.corrections[spec.name] = result
        for effect in analysis.effects:
            comparison = effect.comparisons.get(spec.name)
            entry = result.by_label(effect.ablation) if comparison else None
            if comparison and entry:
                effect.comparisons[spec.name] = comparison.with_adjustment(
                    entry.adjusted_p, correction
                )

    analysis.interactions = _find_interactions(analysis, metrics)
    _add_notes(analysis, metrics)
    return analysis


def _find_interactions(analysis: AblationAnalysis,
                       metrics: Sequence[MetricSpec]) -> list[Interaction]:
    """Compare joint ablations against the sum of their parts."""
    singles: dict[str, dict[str, float]] = {}
    for effect in analysis.effects:
        if len(effect.removed) == 1:
            singles[effect.removed[0]] = {
                spec.name: effect.delta(spec.name)
                for spec in metrics if spec.name in effect.comparisons
            }

    interactions: list[Interaction] = []
    for effect in analysis.effects:
        if len(effect.removed) < 2:
            continue
        if not all(component in singles for component in effect.removed):
            continue
        for spec in metrics:
            if spec.name not in effect.comparisons:
                continue
            predicted = sum(singles[c].get(spec.name, 0.0) for c in effect.removed)
            interactions.append(Interaction(
                components=tuple(effect.removed),
                joint_effect=effect.delta(spec.name),
                additive_prediction=predicted,
                metric=spec.name,
            ))
    return interactions


def _add_notes(analysis: AblationAnalysis, metrics: Sequence[MetricSpec]) -> None:
    if not metrics:
        return
    primary = metrics[0].name
    load_bearing = analysis.load_bearing(primary)
    decorative = analysis.decorative(primary)
    if load_bearing:
        analysis.notes.append(
            f"Load-bearing on {primary}: {', '.join(load_bearing)} "
            "(removing each causes a statistically detectable drop)."
        )
    if decorative:
        analysis.notes.append(
            f"Not measurably contributing to {primary}: {', '.join(decorative)}. "
            "With this many seeds a small real effect could still be hiding here, "
            "so this is a bound on the effect, not proof of its absence."
        )
