"""Parameter sweep analysis.

A sweep is only informative if the differences between settings are
larger than the differences between seeds.  Fabric therefore reports
every parameter's effect *in units of seed noise*, which is the number
that decides whether a hyperparameter table is signal or decoration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from fabric.analysis.results import ResultSet
from fabric.core.entities import MetricSpec
from fabric.statistics.descriptive import Summary, pooled_std, summarize
from fabric.statistics.hypothesis import TestResult, one_way_anova


@dataclass
class ParameterEffect:
    """One swept parameter's effect on one metric."""

    parameter: str
    metric: str
    by_value: dict[str, Summary] = field(default_factory=dict)
    omnibus: TestResult | None = None
    seed_noise: float = 0.0
    best_value: str = ""
    worst_value: str = ""

    @property
    def spread(self) -> float:
        """Difference between the best and worst setting's means."""
        if len(self.by_value) < 2:
            return 0.0
        means = [s.mean for s in self.by_value.values()]
        return max(means) - min(means)

    @property
    def noise_ratio(self) -> float:
        """Spread expressed in units of within-setting seed noise."""
        if self.seed_noise <= 0:
            return float("inf") if self.spread > 0 else 0.0
        return self.spread / self.seed_noise

    @property
    def verdict(self) -> str:
        if len(self.by_value) < 2:
            return "single_value"
        if self.noise_ratio < 1.0:
            return "within_noise"
        if self.omnibus and self.omnibus.p_value == self.omnibus.p_value \
                and self.omnibus.p_value < 0.05:
            return "matters"
        return "unclear"

    def describe(self) -> str:
        if self.verdict == "single_value":
            return f"{self.parameter} was not swept."
        if self.verdict == "within_noise":
            return (
                f"Varying {self.parameter} moves {self.metric} by {self.spread:.4f}, which is "
                f"{self.noise_ratio:.2f}x the seed-to-seed noise ({self.seed_noise:.4f}). "
                "This parameter is not doing anything you can distinguish from randomness."
            )
        if self.verdict == "matters":
            return (
                f"{self.parameter} matters: best setting '{self.best_value}' beats "
                f"'{self.worst_value}' by {self.spread:.4f} on {self.metric}, "
                f"{self.noise_ratio:.1f}x the seed noise."
            )
        return (
            f"{self.parameter} moves {self.metric} by {self.spread:.4f} "
            f"({self.noise_ratio:.1f}x seed noise) but the omnibus test does not reach "
            "significance; more seeds would settle it."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "metric": self.metric,
            "by_value": {k: v.to_dict(include_values=False) for k, v in self.by_value.items()},
            "omnibus": self.omnibus.to_dict() if self.omnibus else None,
            "seed_noise": self.seed_noise,
            "spread": self.spread,
            "noise_ratio": self.noise_ratio,
            "best_value": self.best_value,
            "worst_value": self.worst_value,
            "verdict": self.verdict,
            "description": self.describe(),
        }


@dataclass
class SweepAnalysis:
    """All swept parameters."""

    effects: list[ParameterEffect] = field(default_factory=list)
    parameters: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def for_metric(self, metric: str) -> list[ParameterEffect]:
        return [e for e in self.effects if e.metric == metric]

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameters": self.parameters,
            "effects": [e.to_dict() for e in self.effects],
            "notes": self.notes,
        }


def analyse_sweep(results: ResultSet, *, metrics: Sequence[MetricSpec],
                  alpha: float = 0.05) -> SweepAnalysis:
    """Measure how much each swept parameter actually moves each metric."""
    analysis = SweepAnalysis()
    treatments = results.treatments
    if len(treatments) < 2:
        return analysis

    swept: dict[str, set[Any]] = {}
    for condition in treatments:
        for key, value in condition.config.items():
            swept.setdefault(key, set()).add(_hashable(value))
    parameters = sorted(k for k, values in swept.items() if len(values) > 1)
    analysis.parameters = parameters
    if not parameters:
        return analysis

    for parameter in parameters:
        for spec in metrics:
            groups: dict[str, list[float]] = {}
            for condition in treatments:
                if parameter not in condition.config:
                    continue
                values = condition.values(spec.name)
                if values:
                    groups.setdefault(_format(condition.config[parameter]), []).extend(values)
            if len(groups) < 2:
                continue
            effect = ParameterEffect(parameter=parameter, metric=spec.name)
            effect.by_value = {k: summarize(v) for k, v in sorted(groups.items())}
            effect.seed_noise = pooled_std(list(groups.values()))
            effect.omnibus = one_way_anova(*groups.values(), alpha=alpha)
            sign = spec.direction.sign or 1
            ordered = sorted(effect.by_value.items(), key=lambda kv: kv[1].mean * sign)
            effect.worst_value, effect.best_value = ordered[0][0], ordered[-1][0]
            analysis.effects.append(effect)

    primary = metrics[0].name if metrics else ""
    inert = [e.parameter for e in analysis.for_metric(primary) if e.verdict == "within_noise"]
    if inert:
        analysis.notes.append(
            f"Parameters with no effect on {primary} beyond seed noise: {', '.join(inert)}. "
            "Sweeping them further is not a good use of compute."
        )
    return analysis


def _hashable(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    return value


def _format(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)
