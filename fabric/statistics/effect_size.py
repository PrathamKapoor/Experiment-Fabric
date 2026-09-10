"""Effect sizes.

A p-value answers "could this be noise?".  It does not answer "does this
matter?".  Fabric reports both, always, and labels every magnitude with
the convention it came from so nobody has to guess whether 0.3 is big.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from fabric.statistics.confidence import Interval
from fabric.statistics.descriptive import mean, stdev, variance
from fabric.statistics.distributions import norm_ppf


@dataclass(frozen=True)
class EffectSize:
    """A standardised or unstandardised effect, with its interpretation."""

    name: str
    value: float
    magnitude: str = ""
    interval: Interval | None = None
    convention: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "magnitude": self.magnitude,
            "interval": self.interval.to_dict() if self.interval else None,
            "convention": self.convention,
            "note": self.note,
        }


def _label(value: float, thresholds: Sequence[tuple[float, str]]) -> str:
    magnitude = abs(value)
    for bound, name in thresholds:
        if magnitude < bound:
            return name
    return thresholds[-1][1]


_COHEN_THRESHOLDS = ((0.2, "negligible"), (0.5, "small"), (0.8, "medium"), (math.inf, "large"))
_CLIFF_THRESHOLDS = ((0.147, "negligible"), (0.33, "small"), (0.474, "medium"), (math.inf, "large"))


def cohens_d(a: Sequence[float], b: Sequence[float], *, paired: bool = False) -> EffectSize:
    """Cohen's d.

    For paired data this is d_z (difference scores divided by their own
    standard deviation), which is the quantity a paired t-test actually
    tests and is *not* comparable to an independent-groups d.
    """
    if paired:
        if len(a) != len(b):
            raise ValueError("paired samples must have equal length")
        diffs = [x - y for x, y in zip(a, b, strict=True)]
        sd = stdev(diffs)
        centre = mean(diffs)
        if sd == 0:
            # Every pair moved by exactly the same amount.  The standardised
            # effect is unbounded, not zero; saying "0.0" here would be a lie.
            value = 0.0 if centre == 0 else math.copysign(math.inf, centre)
            return EffectSize(
                "cohens_d_z", value, "" if centre == 0 else "unbounded",
                convention="Cohen (1988), paired d_z",
                note="all pairs differ by an identical amount, so the standardised "
                     "effect has no finite value; read the raw difference instead",
            )
        value = centre / sd
        return EffectSize(
            "cohens_d_z", value, _label(value, _COHEN_THRESHOLDS),
            convention="Cohen (1988), paired d_z",
            note="standardised by the SD of paired differences",
        )
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return EffectSize("cohens_d", math.nan, "", note="needs two observations per group")
    pooled = math.sqrt(((na - 1) * variance(a) + (nb - 1) * variance(b)) / (na + nb - 2))
    difference = mean(a) - mean(b)
    if pooled == 0:
        value = 0.0 if difference == 0 else math.copysign(math.inf, difference)
        return EffectSize("cohens_d", value, "" if difference == 0 else "unbounded",
                          convention="Cohen (1988), pooled SD",
                          note="both groups are constant, so the standardised effect "
                               "has no finite value; read the raw difference instead")
    value = difference / pooled
    return EffectSize("cohens_d", value, _label(value, _COHEN_THRESHOLDS),
                      convention="Cohen (1988), pooled SD")


def hedges_g(a: Sequence[float], b: Sequence[float], *, paired: bool = False,
             level: float = 0.95) -> EffectSize:
    """Hedges' g: Cohen's d with the small-sample bias correction.

    With ten seeds per arm, d overstates the effect by roughly 5%.  Since
    ten seeds is a *generous* ML experiment, g is the better default.
    """
    d = cohens_d(a, b, paired=paired)
    if not math.isfinite(d.value):
        return EffectSize("hedges_g", d.value, d.magnitude, note=d.note,
                          convention="Hedges (1981), bias-corrected d")
    df = (len(a) - 1) if paired else (len(a) + len(b) - 2)
    correction = 1.0 - 3.0 / (4.0 * df - 1.0) if df > 1 else 1.0
    value = d.value * correction
    if paired:
        n = len(a)
        se = math.sqrt(1.0 / n + value * value / (2.0 * n)) if n > 1 else math.nan
    else:
        na, nb = len(a), len(b)
        se = math.sqrt((na + nb) / (na * nb) + value * value / (2.0 * (na + nb))) if na and nb else math.nan
    interval = None
    if not math.isnan(se):
        z = norm_ppf(0.5 + level / 2.0)
        interval = Interval(value - z * se, value + z * se, level, "normal-approx", value)
    return EffectSize(
        "hedges_g", value, _label(value, _COHEN_THRESHOLDS), interval,
        convention="Hedges (1981), bias-corrected d",
        note=f"bias correction J = {correction:.4f}",
    )


def cliffs_delta(a: Sequence[float], b: Sequence[float], *, level: float = 0.95) -> EffectSize:
    """Cliff's delta: P(a > b) - P(a < b).

    Non-parametric, bounded in [-1, 1], and unaffected by the shape of
    the distribution.  Fabric's reproducibility stability score is built
    on it precisely because it makes no normality assumption and has a
    literal reading: the chance that a perturbed run beats an unperturbed
    one, minus the chance it loses.
    """
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return EffectSize("cliffs_delta", math.nan, "", note="empty sample")
    greater = 0
    less = 0
    for x in a:
        for y in b:
            if x > y:
                greater += 1
            elif x < y:
                less += 1
    value = (greater - less) / (na * nb)
    # Consistent variance estimator (Cliff 1993) via the per-observation
    # dominance means.
    di = [sum((1 if x > y else -1 if x < y else 0) for y in b) / nb for x in a]
    dj = [sum((1 if x > y else -1 if x < y else 0) for x in a) / na for y in b]
    interval = None
    if na > 1 and nb > 1:
        s_di = variance(di)
        s_dj = variance(dj)
        var = (nb * nb * s_di + na * na * s_dj) / (na * nb * na * nb)
        var = max(var, 0.0)
        if var > 0:
            z = norm_ppf(0.5 + level / 2.0)
            se = math.sqrt(var)
            interval = Interval(max(-1.0, value - z * se), min(1.0, value + z * se),
                                level, "cliff-consistent", value)
    return EffectSize("cliffs_delta", value, _label(value, _CLIFF_THRESHOLDS), interval,
                      convention="Romano et al. (2006) magnitude thresholds")


def common_language_effect(a: Sequence[float], b: Sequence[float]) -> EffectSize:
    """Vargha-Delaney A: the probability a random ``a`` exceeds a random ``b``.

    0.5 means indistinguishable.  This is the most readable effect size
    in the toolbox and the one worth putting in an abstract.
    """
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return EffectSize("vargha_delaney_a", math.nan, "", note="empty sample")
    wins = 0.0
    for x in a:
        for y in b:
            if x > y:
                wins += 1.0
            elif x == y:
                wins += 0.5
    value = wins / (na * nb)
    magnitude = _label(2 * value - 1, _CLIFF_THRESHOLDS)
    return EffectSize("vargha_delaney_a", value, magnitude,
                      convention="Vargha & Delaney (2000)",
                      note="P(treatment > baseline), ties counted as half")


def rank_biserial(a: Sequence[float], b: Sequence[float]) -> EffectSize:
    """Rank-biserial correlation, the effect size matching Mann-Whitney U."""
    delta = cliffs_delta(a, b)
    return EffectSize("rank_biserial", delta.value, delta.magnitude,
                      convention="equals Cliff's delta for two independent groups")


def relative_improvement(treatment: Sequence[float], baseline: Sequence[float],
                         *, higher_is_better: bool = True) -> EffectSize:
    """Percentage change of the treatment mean over the baseline mean.

    Signed so that positive always means "better", whichever direction the
    metric runs.  Undefined when the baseline mean is zero, and Fabric
    says so rather than printing infinity.
    """
    mb = mean(baseline)
    mt = mean(treatment)
    if mb == 0:
        return EffectSize("relative_improvement_pct", math.nan, "",
                          note="baseline mean is zero; relative change is undefined")
    raw = (mt - mb) / abs(mb) * 100.0
    value = raw if higher_is_better else -raw
    return EffectSize("relative_improvement_pct", value, "",
                      convention="signed so that positive means better",
                      note=f"absolute change {mt - mb:+.6g}")


def absolute_difference(treatment: Sequence[float], baseline: Sequence[float],
                        *, higher_is_better: bool = True) -> EffectSize:
    """Difference in means, signed so positive means better."""
    raw = mean(treatment) - mean(baseline)
    value = raw if higher_is_better else -raw
    return EffectSize("absolute_difference", value, "",
                      convention="signed so that positive means better",
                      note=f"raw difference {raw:+.6g}")


def all_effect_sizes(treatment: Sequence[float], baseline: Sequence[float], *,
                     paired: bool, higher_is_better: bool = True,
                     level: float = 0.95) -> list[EffectSize]:
    """The standard battery Fabric attaches to every comparison."""
    effects = [
        absolute_difference(treatment, baseline, higher_is_better=higher_is_better),
        relative_improvement(treatment, baseline, higher_is_better=higher_is_better),
        hedges_g(treatment, baseline, paired=paired, level=level),
        cohens_d(treatment, baseline, paired=paired),
        cliffs_delta(treatment, baseline, level=level),
        common_language_effect(treatment, baseline),
    ]
    return effects
