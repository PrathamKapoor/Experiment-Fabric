"""Assembling a complete, honest comparison.

One call produces everything a reviewer needs to judge a claim: both
distributions, the chosen test and why, the effect size with its
interval, an equivalence verdict when a tolerance was declared, and a
plain-English reading that refuses to call a coin-flip an improvement.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from fabric.statistics.confidence import (
    Interval,
    bootstrap_difference_interval,
    paired_interval,
    welch_interval,
)
from fabric.statistics.descriptive import Summary, summarize
from fabric.statistics.effect_size import EffectSize, all_effect_sizes
from fabric.statistics.hypothesis import TestResult, tost_equivalence
from fabric.statistics.power import DetectableEffect, minimum_detectable_effect
from fabric.statistics.selection import Decision, run_selected, select_test


@dataclass(frozen=True)
class Comparison:
    """A treatment measured against a baseline on one metric."""

    metric: str
    treatment: str
    baseline: str
    higher_is_better: bool
    paired: bool
    treatment_summary: Summary
    baseline_summary: Summary
    decision: Decision
    test: TestResult
    effects: tuple[EffectSize, ...]
    difference_interval: Interval
    bootstrap_interval: Interval | None = None
    equivalence: TestResult | None = None
    detectable: DetectableEffect | None = None
    adjusted_p: float | None = None
    correction_method: str = ""
    alpha: float = 0.05
    group: str = "main"

    # -- convenience -------------------------------------------------------

    def effect(self, name: str) -> EffectSize | None:
        for item in self.effects:
            if item.name == name:
                return item
        return None

    @property
    def difference(self) -> float:
        """Signed difference in means, positive meaning better."""
        effect = self.effect("absolute_difference")
        return effect.value if effect else math.nan

    @property
    def relative(self) -> float:
        effect = self.effect("relative_improvement_pct")
        return effect.value if effect else math.nan

    @property
    def p_value(self) -> float:
        return self.test.p_value

    @property
    def effective_p(self) -> float:
        """The p-value that should be read: adjusted when a family exists."""
        return self.adjusted_p if self.adjusted_p is not None else self.test.p_value

    @property
    def significant(self) -> bool:
        p = self.effective_p
        return p == p and p < self.alpha

    @property
    def equivalent(self) -> bool:
        return bool(self.equivalence and self.equivalence.extra.get("equivalent"))

    def with_adjustment(self, adjusted_p: float, method: str) -> Comparison:
        """Return a copy carrying a multiple-comparison-adjusted p-value."""
        return replace(self, adjusted_p=adjusted_p, correction_method=method)

    # -- interpretation ----------------------------------------------------

    @property
    def verdict(self) -> str:
        """A short machine-readable verdict code."""
        if self.test.test == "none" or self.effective_p != self.effective_p:
            return "not_testable"
        if self.significant:
            return "improvement" if self.difference > 0 else "regression"
        if self.equivalent:
            return "equivalent"
        return "inconclusive"

    def narrative(self, digits: int = 4) -> str:
        """A sentence a paper could contain without embarrassing anyone."""
        t_mean = self.treatment_summary.mean
        b_mean = self.baseline_summary.mean
        n = self.treatment_summary.n
        g = self.effect("hedges_g")
        head = (
            f"{self.treatment} scored {t_mean:.{digits}f} against {self.baseline}'s "
            f"{b_mean:.{digits}f} on {self.metric} "
            f"({n} replications each, {self.difference:+.{digits}f} "
            f"{self.difference_interval.format(digits)} at {round((1 - self.alpha) * 100)}% confidence)"
        )
        p_label = "adjusted p" if self.adjusted_p is not None else "p"
        p_text = "n/a" if self.effective_p != self.effective_p else f"{self.effective_p:.4f}"
        if self.verdict == "not_testable":
            return head + ". No test was run: " + (self.test.reason or "insufficient data") + "."
        if self.verdict == "improvement":
            return (
                head + f". The difference is statistically detectable ({self.test.test}, "
                f"{p_label}={p_text})"
                + (f" with a {g.magnitude} effect size (Hedges' g={g.value:.2f})." if g and g.magnitude else ".")
            )
        if self.verdict == "regression":
            return (
                head + f". {self.treatment} is measurably worse ({self.test.test}, "
                f"{p_label}={p_text})."
            )
        if self.verdict == "equivalent":
            margin = self.equivalence.extra.get("margin") if self.equivalence else None
            return (
                head + f". The two are statistically equivalent within the declared "
                f"margin of {margin:g} (TOST p={self.equivalence.p_value:.4f}); "
                f"the difference test was not significant ({p_label}={p_text})."
            )
        tail = (
            head + f". This difference is not distinguishable from noise "
            f"({self.test.test}, {p_label}={p_text}). With n={n} per condition this is "
            "an absence of evidence, not evidence of absence."
        )
        if self.detectable and self.detectable.value == self.detectable.value:
            tail += (
                f" This design could only have detected a difference of "
                f"{self.detectable.value:.4f} or larger."
            )
        return tail

    def to_dict(self, *, include_values: bool = True) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "treatment": self.treatment,
            "baseline": self.baseline,
            "group": self.group,
            "higher_is_better": self.higher_is_better,
            "paired": self.paired,
            "alpha": self.alpha,
            "treatment_summary": self.treatment_summary.to_dict(include_values=include_values),
            "baseline_summary": self.baseline_summary.to_dict(include_values=include_values),
            "decision": self.decision.to_dict(),
            "test": self.test.to_dict(),
            "effects": [e.to_dict() for e in self.effects],
            "difference_interval": self.difference_interval.to_dict(),
            "bootstrap_interval": self.bootstrap_interval.to_dict() if self.bootstrap_interval else None,
            "equivalence": self.equivalence.to_dict() if self.equivalence else None,
            "detectable": self.detectable.to_dict() if self.detectable else None,
            "adjusted_p": self.adjusted_p,
            "correction_method": self.correction_method,
            "verdict": self.verdict,
            "narrative": self.narrative(),
        }


def compare(
    treatment_values: Sequence[float],
    baseline_values: Sequence[float],
    *,
    metric: str,
    treatment: str = "treatment",
    baseline: str = "baseline",
    higher_is_better: bool = True,
    paired: bool | None = None,
    alpha: float = 0.05,
    tolerance: float | None = None,
    bootstrap_resamples: int = 5000,
    bootstrap_seed: int = 12345,
    group: str = "main",
) -> Comparison:
    """Compare two sets of replications on one metric.

    ``paired=None`` means "decide from the shapes": equal-length samples
    are assumed to come from a shared seed set, which is how Fabric
    generates them.  Callers that know better should say so explicitly.
    """
    treatment_values = [float(v) for v in treatment_values]
    baseline_values = [float(v) for v in baseline_values]
    if paired is None:
        paired = len(treatment_values) == len(baseline_values) and len(treatment_values) > 1

    decision = select_test(treatment_values, baseline_values, paired=paired, alpha=alpha)
    test = run_selected(decision, treatment_values, baseline_values, alpha=alpha)

    if paired and len(treatment_values) == len(baseline_values) and len(treatment_values) > 1:
        difference_interval = paired_interval(treatment_values, baseline_values, 1 - alpha)
    else:
        difference_interval = welch_interval(treatment_values, baseline_values, 1 - alpha)

    boot: Interval | None = None
    if len(treatment_values) >= 3 and len(baseline_values) >= 3:
        boot = bootstrap_difference_interval(
            treatment_values, baseline_values, paired=paired,
            level=1 - alpha, n_resamples=bootstrap_resamples, seed=bootstrap_seed,
        )

    equivalence = None
    if tolerance:
        equivalence = tost_equivalence(treatment_values, baseline_values, tolerance,
                                       paired=paired, alpha=alpha)

    effects = tuple(all_effect_sizes(
        treatment_values, baseline_values, paired=paired,
        higher_is_better=higher_is_better, level=1 - alpha,
    ))

    detectable = minimum_detectable_effect(
        treatment_values, baseline_values, paired=paired, alpha=alpha, metric=metric,
    )

    if not higher_is_better:
        # Interval endpoints are in raw metric units; flip so that
        # "positive is better" holds everywhere a reader looks.
        difference_interval = Interval(
            -difference_interval.high, -difference_interval.low,
            difference_interval.level, difference_interval.method + " (sign-flipped)",
            -difference_interval.point,
            "signed so that positive means better, matching the metric's direction",
        )
        if boot:
            boot = Interval(-boot.high, -boot.low, boot.level,
                            boot.method + " (sign-flipped)", -boot.point,
                            "signed so that positive means better")

    return Comparison(
        metric=metric,
        treatment=treatment,
        baseline=baseline,
        higher_is_better=higher_is_better,
        paired=paired,
        treatment_summary=summarize(treatment_values),
        baseline_summary=summarize(baseline_values),
        decision=decision,
        test=test,
        effects=effects,
        difference_interval=difference_interval,
        bootstrap_interval=boot,
        equivalence=equivalence,
        detectable=detectable,
        alpha=alpha,
        group=group,
    )
