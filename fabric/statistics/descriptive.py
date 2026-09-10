"""Descriptive statistics.

Fabric never reports a bare mean.  A condition summarised here always
carries its spread and its sample size, because "91.4%" and "91.4% +/-
0.9 over 10 seeds" are different scientific statements.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class Summary:
    """Distribution summary for one condition's replications."""

    n: int
    mean: float
    median: float
    std: float
    variance: float
    sem: float
    minimum: float
    maximum: float
    q1: float
    q3: float
    iqr: float
    cv: float
    mad: float
    skewness: float
    kurtosis: float
    values: tuple[float, ...] = ()

    def to_dict(self, *, include_values: bool = True) -> dict[str, Any]:
        data = asdict(self)
        data["values"] = list(self.values) if include_values else []
        return data

    @property
    def range(self) -> float:
        return self.maximum - self.minimum


def quantile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile (the "type 7" definition, as in R/NumPy)."""
    if not values:
        raise ValueError("quantile of an empty sample")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return float(ordered[int(pos)])
    return float(ordered[low] + (ordered[high] - ordered[low]) * (pos - low))


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean."""
    if not values:
        raise ValueError("mean of an empty sample")
    return math.fsum(values) / len(values)


def variance(values: Sequence[float], ddof: int = 1) -> float:
    """Sample variance.

    ``ddof=1`` by default: seeds are a sample from the distribution of
    possible runs, not the population of them.
    """
    n = len(values)
    if n - ddof <= 0:
        return 0.0
    mu = mean(values)
    return math.fsum((v - mu) ** 2 for v in values) / (n - ddof)


def stdev(values: Sequence[float], ddof: int = 1) -> float:
    """Sample standard deviation."""
    return math.sqrt(variance(values, ddof))


def sem(values: Sequence[float]) -> float:
    """Standard error of the mean."""
    if len(values) < 2:
        return 0.0
    return stdev(values) / math.sqrt(len(values))


def median(values: Sequence[float]) -> float:
    """Median."""
    return quantile(values, 0.5)


def mad(values: Sequence[float], *, scale: float = 1.4826) -> float:
    """Median absolute deviation, scaled to be consistent with sigma."""
    if not values:
        return 0.0
    med = median(values)
    return scale * median([abs(v - med) for v in values])


def _moment(values: Sequence[float], order: int) -> float:
    mu = mean(values)
    return math.fsum((v - mu) ** order for v in values) / len(values)


def skewness(values: Sequence[float]) -> float:
    """Biased sample skewness ``g1`` (the estimator normality tests expect)."""
    if len(values) < 3:
        return 0.0
    m2 = _moment(values, 2)
    if m2 == 0:
        return 0.0
    return _moment(values, 3) / (m2 ** 1.5)


def kurtosis(values: Sequence[float], *, excess: bool = True) -> float:
    """Biased sample kurtosis ``b2``; excess (Fisher) by default."""
    if len(values) < 4:
        return 0.0
    m2 = _moment(values, 2)
    if m2 == 0:
        return 0.0
    b2 = _moment(values, 4) / (m2 * m2)
    return b2 - 3.0 if excess else b2


def summarize(values: Sequence[float]) -> Summary:
    """Compute the full descriptive summary of a sample."""
    vals = [float(v) for v in values]
    if not vals:
        return Summary(0, math.nan, math.nan, math.nan, math.nan, math.nan,
                       math.nan, math.nan, math.nan, math.nan, math.nan,
                       math.nan, math.nan, math.nan, math.nan, ())
    mu = mean(vals)
    sd = stdev(vals)
    q1 = quantile(vals, 0.25)
    q3 = quantile(vals, 0.75)
    return Summary(
        n=len(vals),
        mean=mu,
        median=median(vals),
        std=sd,
        variance=variance(vals),
        sem=sem(vals),
        minimum=min(vals),
        maximum=max(vals),
        q1=q1,
        q3=q3,
        iqr=q3 - q1,
        cv=(sd / abs(mu)) if mu != 0 else math.inf if sd else 0.0,
        mad=mad(vals),
        skewness=skewness(vals),
        kurtosis=kurtosis(vals),
        values=tuple(vals),
    )


def pooled_std(groups: Sequence[Sequence[float]]) -> float:
    """Pooled within-group standard deviation.

    This is Fabric's reference "noise floor": the spread you get from
    nothing but seed changes, against which every perturbation effect is
    measured.
    """
    num = 0.0
    den = 0
    for group in groups:
        if len(group) < 2:
            continue
        num += (len(group) - 1) * variance(group)
        den += len(group) - 1
    if den == 0:
        return 0.0
    return math.sqrt(num / den)
