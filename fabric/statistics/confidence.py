"""Interval estimation.

An effect without an interval is a rumour.  Every comparison Fabric
produces carries one, and every interval records *which method* produced
it so a reader can judge whether it was appropriate.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from fabric.statistics.descriptive import mean, sem, variance
from fabric.statistics.distributions import norm_cdf, norm_ppf, t_ppf


@dataclass(frozen=True)
class Interval:
    """A confidence interval, tagged with its construction method."""

    low: float
    high: float
    level: float = 0.95
    method: str = "t"
    point: float = math.nan
    note: str = ""

    def contains(self, value: float) -> bool:
        """Whether ``value`` lies inside the interval."""
        return self.low <= value <= self.high

    @property
    def width(self) -> float:
        return self.high - self.low

    def excludes_zero(self) -> bool:
        """Whether the interval rules out "no difference"."""
        return not self.contains(0.0)

    def overlaps(self, other: Interval) -> bool:
        """Whether two intervals overlap at all."""
        return not (self.high < other.low or other.high < self.low)

    def to_dict(self) -> dict[str, Any]:
        return {
            "low": self.low,
            "high": self.high,
            "level": self.level,
            "method": self.method,
            "point": self.point,
            "width": self.width,
            "note": self.note,
        }

    def format(self, digits: int = 4) -> str:
        return f"[{self.low:.{digits}f}, {self.high:.{digits}f}]"


def t_interval(values: Sequence[float], level: float = 0.95) -> Interval:
    """Student-t confidence interval for a mean."""
    n = len(values)
    mu = mean(values) if n else math.nan
    if n < 2:
        return Interval(math.nan, math.nan, level, "t", mu,
                        "fewer than two observations: no interval is defined")
    half = t_ppf(0.5 + level / 2.0, n - 1) * sem(values)
    return Interval(mu - half, mu + half, level, "t", mu)


def welch_interval(a: Sequence[float], b: Sequence[float], level: float = 0.95) -> Interval:
    """Welch interval for the difference of two independent means."""
    na, nb = len(a), len(b)
    diff = mean(a) - mean(b)
    if na < 2 or nb < 2:
        return Interval(math.nan, math.nan, level, "welch", diff,
                        "each group needs at least two observations")
    va, vb = variance(a), variance(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return Interval(diff, diff, level, "welch", diff, "zero variance in both groups")
    df = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    )
    half = t_ppf(0.5 + level / 2.0, df) * se
    return Interval(diff - half, diff + half, level, "welch", diff)


def paired_interval(a: Sequence[float], b: Sequence[float], level: float = 0.95) -> Interval:
    """Interval for the mean of paired differences."""
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    interval = t_interval(diffs, level)
    return Interval(interval.low, interval.high, level, "paired-t", interval.point, interval.note)


# --------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------


def _resample(values: Sequence[float], rng: random.Random) -> list[float]:
    n = len(values)
    return [values[rng.randrange(n)] for _ in range(n)]


def bootstrap_distribution(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float] = mean,
    *,
    n_resamples: int = 10000,
    seed: int = 12345,
) -> list[float]:
    """Bootstrap replicates of ``statistic``.

    The RNG is seeded explicitly: a confidence interval that changes when
    you rerun the analysis is not evidence, and Fabric records this seed
    in the analysis output.
    """
    rng = random.Random(seed)
    return [statistic(_resample(values, rng)) for _ in range(n_resamples)]


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return math.nan
    pos = (len(sorted_values) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return float(sorted_values[int(pos)])
    return float(sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (pos - low))


def bootstrap_interval(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float] = mean,
    *,
    level: float = 0.95,
    n_resamples: int = 10000,
    seed: int = 12345,
    method: str = "bca",
) -> Interval:
    """Bootstrap confidence interval (``bca`` or ``percentile``).

    BCa corrects for both bias and skewness in the bootstrap
    distribution, which matters for the bounded metrics (accuracy, F1)
    that dominate ML papers: a percentile interval on a mean accuracy of
    0.98 will happily extend past 1.0.
    """
    n = len(values)
    point = statistic(values) if n else math.nan
    if n < 2:
        return Interval(math.nan, math.nan, level, f"bootstrap-{method}", point,
                        "fewer than two observations: no interval is defined")
    replicates = sorted(bootstrap_distribution(values, statistic, n_resamples=n_resamples, seed=seed))
    alpha = 1.0 - level
    if method == "percentile":
        return Interval(
            _percentile(replicates, alpha / 2.0),
            _percentile(replicates, 1.0 - alpha / 2.0),
            level,
            "bootstrap-percentile",
            point,
        )

    n_below = sum(1 for r in replicates if r < point)
    if n_below in (0, len(replicates)):
        # Degenerate bootstrap (e.g. a constant sample): fall back rather
        # than divide by an infinite bias correction.
        return Interval(
            _percentile(replicates, alpha / 2.0),
            _percentile(replicates, 1.0 - alpha / 2.0),
            level,
            "bootstrap-percentile",
            point,
            "BCa undefined for a degenerate bootstrap distribution",
        )
    z0 = norm_ppf(n_below / len(replicates))

    jackknife = []
    for i in range(n):
        subset = list(values[:i]) + list(values[i + 1:])
        jackknife.append(statistic(subset))
    jack_mean = mean(jackknife)
    num = math.fsum((jack_mean - v) ** 3 for v in jackknife)
    den = 6.0 * (math.fsum((jack_mean - v) ** 2 for v in jackknife) ** 1.5)
    acceleration = num / den if den != 0 else 0.0

    def adjusted(prob: float) -> float:
        z = norm_ppf(prob)
        denom = 1.0 - acceleration * (z0 + z)
        if denom == 0:
            return prob
        return norm_cdf(z0 + (z0 + z) / denom)

    lo_q = adjusted(alpha / 2.0)
    hi_q = adjusted(1.0 - alpha / 2.0)
    return Interval(
        _percentile(replicates, lo_q),
        _percentile(replicates, hi_q),
        level,
        "bootstrap-bca",
        point,
    )


def bootstrap_difference_interval(
    a: Sequence[float],
    b: Sequence[float],
    *,
    paired: bool,
    level: float = 0.95,
    n_resamples: int = 10000,
    seed: int = 12345,
) -> Interval:
    """Bootstrap interval for a difference of means.

    Paired data is resampled by *pair*, not by observation: breaking the
    pairing throws away the very correlation that makes paired seeds
    powerful.
    """
    if paired:
        if len(a) != len(b):
            raise ValueError("paired samples must have equal length")
        diffs = [x - y for x, y in zip(a, b, strict=True)]
        interval = bootstrap_interval(diffs, level=level, n_resamples=n_resamples, seed=seed)
        return Interval(interval.low, interval.high, level, "bootstrap-bca-paired",
                        interval.point, interval.note)
    rng = random.Random(seed)
    point = mean(a) - mean(b)
    replicates = sorted(
        mean(_resample(a, rng)) - mean(_resample(b, rng)) for _ in range(n_resamples)
    )
    alpha = 1.0 - level
    return Interval(
        _percentile(replicates, alpha / 2.0),
        _percentile(replicates, 1.0 - alpha / 2.0),
        level,
        "bootstrap-percentile-unpaired",
        point,
    )


def proportion_interval(successes: int, trials: int, level: float = 0.95) -> Interval:
    """Wilson score interval for a proportion.

    Accuracy on a fixed test set is a proportion, and the normal
    approximation misbehaves exactly where ML papers live (near 1.0).
    """
    if trials <= 0:
        return Interval(math.nan, math.nan, level, "wilson", math.nan, "no trials")
    z = norm_ppf(0.5 + level / 2.0)
    phat = successes / trials
    denom = 1.0 + z * z / trials
    centre = (phat + z * z / (2 * trials)) / denom
    half = z * math.sqrt(phat * (1 - phat) / trials + z * z / (4 * trials * trials)) / denom
    return Interval(max(0.0, centre - half), min(1.0, centre + half), level, "wilson", phat)


__all__ = [
    "Interval",
    "bootstrap_difference_interval",
    "bootstrap_distribution",
    "bootstrap_interval",
    "paired_interval",
    "proportion_interval",
    "t_interval",
    "welch_interval",
]
