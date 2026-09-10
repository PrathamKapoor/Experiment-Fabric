"""Power: what an experiment could have detected.

Every non-significant result has a companion question that is almost
never asked -- *how big would the effect have had to be for this design to
find it?*  Without that number, "no significant difference" is
uninterpretable: it might mean the methods are equivalent, or it might
mean the experiment could not have detected a difference twice the size
of the one claimed in the abstract.

Fabric computes the minimum detectable effect for every comparison and
states it as a limitation, whether or not the result was significant.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from fabric.statistics.descriptive import mean, stdev, variance
from fabric.statistics.distributions import t_ppf

DEFAULT_POWER = 0.8


@dataclass(frozen=True)
class DetectableEffect:
    """The smallest difference a design could reliably find."""

    value: float
    power: float
    alpha: float
    paired: bool
    n: int
    standard_error: float
    observed_difference: float = math.nan
    metric: str = ""

    @property
    def observed_is_below_floor(self) -> bool:
        """Whether the measured difference is smaller than the design could resolve."""
        if self.observed_difference != self.observed_difference or self.value != self.value:
            return False
        return abs(self.observed_difference) < self.value

    def describe(self) -> str:
        if self.value != self.value:
            return "The minimum detectable effect could not be estimated."
        text = (
            f"With {self.n} replications and the observed spread, this design could "
            f"detect a difference of {self.value:.4f}"
            + (f" in {self.metric}" if self.metric else "")
            + f" or larger {self.power * 100:.0f}% of the time at alpha={self.alpha}."
        )
        if self.observed_is_below_floor:
            text += (
                f" The difference actually measured ({self.observed_difference:+.4f}) is "
                "smaller than that floor, so a null result here is a statement about "
                "the experiment's resolution, not about the methods."
            )
        return text

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum_detectable_effect": self.value,
            "power": self.power,
            "alpha": self.alpha,
            "paired": self.paired,
            "n": self.n,
            "standard_error": self.standard_error,
            "observed_difference": self.observed_difference,
            "observed_is_below_floor": self.observed_is_below_floor,
            "metric": self.metric,
            "description": self.describe(),
        }


def minimum_detectable_effect(
    treatment: Sequence[float],
    baseline: Sequence[float],
    *,
    paired: bool,
    alpha: float = 0.05,
    power: float = DEFAULT_POWER,
    metric: str = "",
) -> DetectableEffect:
    """The smallest true difference this design would usually detect.

    Uses the standard normal-theory approximation

        MDE = (t_{1-alpha/2, df} + t_{power, df}) * SE

    with the standard error taken from the data actually collected --
    paired differences when the design is paired, Welch's otherwise.  It
    is an estimate built on the observed spread, not a promise: the true
    variance is unknown, and at small n the estimate of it is itself
    noisy.
    """
    n_t, n_b = len(treatment), len(baseline)
    observed = (mean(treatment) - mean(baseline)) if n_t and n_b else math.nan

    if paired:
        if n_t != n_b or n_t < 2:
            return DetectableEffect(math.nan, power, alpha, paired, min(n_t, n_b),
                                    math.nan, observed, metric)
        diffs = [x - y for x, y in zip(treatment, baseline, strict=True)]
        standard_error = stdev(diffs) / math.sqrt(len(diffs))
        df = len(diffs) - 1
        n = len(diffs)
    else:
        if n_t < 2 or n_b < 2:
            return DetectableEffect(math.nan, power, alpha, paired, min(n_t, n_b),
                                    math.nan, observed, metric)
        va, vb = variance(treatment), variance(baseline)
        standard_error = math.sqrt(va / n_t + vb / n_b)
        if standard_error == 0:
            return DetectableEffect(0.0, power, alpha, paired, min(n_t, n_b), 0.0,
                                    observed, metric)
        df = (va / n_t + vb / n_b) ** 2 / (
            (va / n_t) ** 2 / (n_t - 1) + (vb / n_b) ** 2 / (n_b - 1))
        n = min(n_t, n_b)

    if standard_error == 0:
        return DetectableEffect(0.0, power, alpha, paired, n, 0.0, observed, metric)

    critical = t_ppf(1.0 - alpha / 2.0, df)
    power_quantile = t_ppf(power, df)
    return DetectableEffect(
        value=(critical + power_quantile) * standard_error,
        power=power,
        alpha=alpha,
        paired=paired,
        n=n,
        standard_error=standard_error,
        observed_difference=observed,
        metric=metric,
    )


def required_replications(
    effect: float,
    spread: float,
    *,
    alpha: float = 0.05,
    power: float = DEFAULT_POWER,
    paired: bool = True,
    maximum: int = 10000,
) -> int:
    """How many replications would be needed to detect ``effect``.

    The answer to "we found nothing; how many more seeds do we need?".
    Solved by search rather than a closed form, because the t quantiles
    depend on the sample size being solved for.
    """
    if effect <= 0 or spread <= 0:
        return 0
    for n in range(2, maximum + 1):
        df = n - 1 if paired else 2 * (n - 1)
        standard_error = spread / math.sqrt(n) if paired else spread * math.sqrt(2.0 / n)
        mde = (t_ppf(1.0 - alpha / 2.0, df) + t_ppf(power, df)) * standard_error
        if mde <= effect:
            return n
    return maximum
