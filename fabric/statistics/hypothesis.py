"""Hypothesis tests that show their work.

Every test returns not just a statistic and a p-value but the
assumptions it relies on, whether each was checked, and what the checks
found.  A ``TestResult`` is meant to be read by a reviewer, not just
formatted into a table.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any

from fabric.statistics.descriptive import mean, median, stdev, variance
from fabric.statistics.distributions import (
    chi2_sf,
    f_sf,
    norm_cdf,
    norm_sf,
    t_cdf,
    t_sf,
)

_EXACT_LIMIT = 20
"""Above this per-group size Fabric switches from exact to asymptotic.

Exact Mann-Whitney and Wilcoxon distributions are computed by dynamic
programming; at n = 20 that is instant, and by n = 30 the normal
approximation is accurate to well past the third decimal of a p-value.
"""


@dataclass(frozen=True)
class Assumption:
    """One precondition of a statistical test and its verdict."""

    name: str
    satisfied: bool | None
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def symbol(self) -> str:
        return {True: "ok", False: "violated", None: "unverified"}[self.satisfied]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "satisfied": self.satisfied,
            "status": self.symbol,
            "detail": self.detail,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class TestResult:
    """The outcome of a hypothesis test, with its audit trail."""

    test: str
    statistic: float
    p_value: float
    df: float | None = None
    alternative: str = "two-sided"
    n: tuple[int, ...] = ()
    method_detail: str = ""
    reason: str = ""
    assumptions: tuple[Assumption, ...] = ()
    warnings: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def significant(self, alpha: float = 0.05) -> bool:
        """Whether the p-value clears ``alpha``.

        This is a threshold, not a truth value; the report always prints
        the effect size beside it.
        """
        return self.p_value < alpha

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["n"] = list(self.n)
        data["assumptions"] = [a.to_dict() for a in self.assumptions]
        data["warnings"] = list(self.warnings)
        return data


# --------------------------------------------------------------------------
# assumption checks
# --------------------------------------------------------------------------


def anderson_darling_normality(values: Sequence[float]) -> tuple[float, float]:
    """Anderson-Darling test for normality with estimated parameters.

    Returns ``(A-squared-star, p_value)`` using the Stephens (1986)
    modification and p-value fit.  Chosen over Shapiro-Wilk because it is
    short enough to audit and behaves sensibly at the sample sizes seed
    studies actually use.
    """
    n = len(values)
    if n < 3:
        return math.nan, math.nan
    sd = stdev(values)
    if sd == 0:
        return math.nan, math.nan
    mu = mean(values)
    z = sorted((v - mu) / sd for v in values)
    total = 0.0
    for i, zi in enumerate(z):
        cdf_lo = norm_cdf(zi)
        cdf_hi = norm_cdf(z[n - 1 - i])
        cdf_lo = min(max(cdf_lo, 1e-15), 1 - 1e-15)
        cdf_hi = min(max(cdf_hi, 1e-15), 1 - 1e-15)
        total += (2 * (i + 1) - 1) * (math.log(cdf_lo) + math.log1p(-cdf_hi))
    a2 = -n - total / n
    a2_star = a2 * (1.0 + 0.75 / n + 2.25 / (n * n))
    if a2_star < 0.2:
        p = 1.0 - math.exp(-13.436 + 101.14 * a2_star - 223.73 * a2_star ** 2)
    elif a2_star < 0.34:
        p = 1.0 - math.exp(-8.318 + 42.796 * a2_star - 59.938 * a2_star ** 2)
    elif a2_star < 0.6:
        p = math.exp(0.9177 - 4.279 * a2_star - 1.38 * a2_star ** 2)
    elif a2_star < 10.0:
        p = math.exp(1.2937 - 5.709 * a2_star + 0.0186 * a2_star ** 2)
    else:
        p = 0.0
    return a2_star, min(1.0, max(0.0, p))


def brown_forsythe(*groups: Sequence[float]) -> TestResult:
    """Brown-Forsythe test for equality of variances.

    The median-centred variant of Levene's test; robust when the metric
    distribution is skewed, which loss curves reliably are.
    """
    valid = [list(g) for g in groups if len(g) >= 2]
    k = len(valid)
    if k < 2:
        return TestResult("brown-forsythe", math.nan, math.nan,
                          reason="needs at least two groups of size two")
    n_total = sum(len(g) for g in valid)
    z_groups = [[abs(v - median(g)) for v in g] for g in valid]
    grand = mean([v for g in z_groups for v in g])
    numerator = sum(len(g) * (mean(zg) - grand) ** 2
                    for g, zg in zip(valid, z_groups, strict=True))
    denominator = sum((v - mean(zg)) ** 2 for zg in z_groups for v in zg)
    if denominator == 0:
        return TestResult("brown-forsythe", math.nan, 1.0,
                          reason="all groups have identical spread")
    statistic = ((n_total - k) / (k - 1)) * numerator / denominator
    p = f_sf(statistic, k - 1, n_total - k)
    return TestResult("brown-forsythe", statistic, p, df=float(n_total - k),
                      n=tuple(len(g) for g in valid),
                      method_detail="Levene's test centred on group medians")


def _normality_assumption(label: str, values: Sequence[float], alpha: float = 0.05) -> Assumption:
    n = len(values)
    if n < 8:
        return Assumption(
            f"{label} is approximately normal", None,
            f"n={n} is too small to test normality; relying on the test's robustness "
            "or preferring a distribution-free alternative",
            {"n": n},
        )
    a2, p = anderson_darling_normality(values)
    if math.isnan(p):
        return Assumption(f"{label} is approximately normal", None,
                          "sample is constant; normality is undefined", {"n": n})
    return Assumption(
        f"{label} is approximately normal",
        p >= alpha,
        f"Anderson-Darling A2*={a2:.3f}, p={p:.4f}",
        {"n": n, "a2_star": a2, "p_value": p},
    )


def _size_warnings(*groups: Sequence[float]) -> list[str]:
    warnings: list[str] = []
    for i, group in enumerate(groups):
        if len(group) < 3:
            warnings.append(
                f"group {i + 1} has n={len(group)}: any p-value here is close to meaningless"
            )
        elif len(group) < 6:
            warnings.append(
                f"group {i + 1} has n={len(group)}: low power, a null result is not evidence of absence"
            )
    return warnings


# --------------------------------------------------------------------------
# parametric tests
# --------------------------------------------------------------------------


def _p_from_t(t: float, df: float, alternative: str) -> float:
    if alternative == "two-sided":
        return 2.0 * t_sf(abs(t), df)
    if alternative == "greater":
        return t_sf(t, df)
    if alternative == "less":
        return t_cdf(t, df)
    raise ValueError(f"unknown alternative {alternative!r}")


def welch_t_test(a: Sequence[float], b: Sequence[float], *,
                 alternative: str = "two-sided", alpha: float = 0.05) -> TestResult:
    """Welch's t-test for two independent samples with unequal variances.

    Welch is the default rather than Student's because equal variances
    are an assumption almost no ML comparison earns, and Welch costs
    almost no power when they happen to hold.
    """
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return TestResult("welch-t", math.nan, math.nan,
                          reason="each group needs at least two observations",
                          n=(na, nb))
    va, vb = variance(a), variance(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return TestResult("welch-t", math.nan, 1.0, n=(na, nb),
                          reason="both groups are constant; no variance to test")
    t = (mean(a) - mean(b)) / se
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return TestResult(
        "welch-t", t, _p_from_t(t, df, alternative), df=df, alternative=alternative,
        n=(na, nb),
        method_detail="Welch-Satterthwaite degrees of freedom",
        assumptions=(
            Assumption("observations are independent", None,
                       "structural: guaranteed by distinct seeds and isolated runs"),
            _normality_assumption("group 1", a, alpha),
            _normality_assumption("group 2", b, alpha),
            Assumption("equal variances", None, "not required by Welch's test"),
        ),
        warnings=tuple(_size_warnings(a, b)),
    )


def student_t_test(a: Sequence[float], b: Sequence[float], *,
                   alternative: str = "two-sided", alpha: float = 0.05) -> TestResult:
    """Student's pooled-variance t-test."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return TestResult("student-t", math.nan, math.nan, n=(na, nb),
                          reason="each group needs at least two observations")
    df = na + nb - 2
    pooled = ((na - 1) * variance(a) + (nb - 1) * variance(b)) / df
    se = math.sqrt(pooled * (1.0 / na + 1.0 / nb))
    if se == 0:
        return TestResult("student-t", math.nan, 1.0, n=(na, nb),
                          reason="both groups are constant")
    t = (mean(a) - mean(b)) / se
    variance_check = brown_forsythe(a, b)
    return TestResult(
        "student-t", t, _p_from_t(t, df, alternative), df=float(df),
        alternative=alternative, n=(na, nb),
        method_detail="pooled variance",
        assumptions=(
            Assumption("observations are independent", None, "structural"),
            _normality_assumption("group 1", a, alpha),
            _normality_assumption("group 2", b, alpha),
            Assumption("equal variances", bool(variance_check.p_value >= alpha)
                       if not math.isnan(variance_check.p_value) else None,
                       f"Brown-Forsythe p={variance_check.p_value:.4f}",
                       {"p_value": variance_check.p_value}),
        ),
        warnings=tuple(_size_warnings(a, b)),
    )


def paired_t_test(a: Sequence[float], b: Sequence[float], *,
                  alternative: str = "two-sided", alpha: float = 0.05) -> TestResult:
    """Paired t-test on within-seed differences.

    When two methods are evaluated on the same seeds, pairing removes
    seed-to-seed variance from the comparison and typically multiplies
    power several-fold.  Fabric pairs by default whenever the seed sets
    match.
    """
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    n = len(a)
    if n < 2:
        return TestResult("paired-t", math.nan, math.nan, n=(n,),
                          reason="needs at least two pairs")
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    sd = stdev(diffs)
    if sd == 0:
        return TestResult("paired-t", math.nan, 1.0 if mean(diffs) == 0 else 0.0, n=(n,),
                          reason="all pairs differ by exactly the same amount",
                          extra={"mean_difference": mean(diffs)})
    t = mean(diffs) / (sd / math.sqrt(n))
    df = n - 1
    return TestResult(
        "paired-t", t, _p_from_t(t, df, alternative), df=float(df),
        alternative=alternative, n=(n,),
        method_detail="one-sample t-test on within-pair differences",
        assumptions=(
            Assumption("pairs are matched", True,
                       "conditions were evaluated on an identical seed set"),
            Assumption("pairs are independent of each other", None, "structural"),
            _normality_assumption("paired differences", diffs, alpha),
        ),
        warnings=tuple(_size_warnings(diffs)),
        extra={"mean_difference": mean(diffs), "sd_difference": sd},
    )


def one_way_anova(*groups: Sequence[float], alpha: float = 0.05) -> TestResult:
    """One-way ANOVA across three or more conditions.

    Used as an omnibus gate: when it fails to reject, running every
    pairwise test afterwards is fishing.
    """
    valid = [list(g) for g in groups if len(g) >= 2]
    k = len(valid)
    if k < 2:
        return TestResult("one-way-anova", math.nan, math.nan,
                          reason="needs at least two groups of size two")
    n_total = sum(len(g) for g in valid)
    grand = mean([v for g in valid for v in g])
    ss_between = sum(len(g) * (mean(g) - grand) ** 2 for g in valid)
    ss_within = sum((v - mean(g)) ** 2 for g in valid for v in g)
    df_between, df_within = k - 1, n_total - k
    if ss_within == 0 or df_within <= 0:
        return TestResult("one-way-anova", math.nan, math.nan, n=tuple(len(g) for g in valid),
                          reason="no within-group variance")
    f = (ss_between / df_between) / (ss_within / df_within)
    eta_sq = ss_between / (ss_between + ss_within)
    variance_check = brown_forsythe(*valid)
    return TestResult(
        "one-way-anova", f, f_sf(f, df_between, df_within), df=float(df_within),
        n=tuple(len(g) for g in valid),
        method_detail=f"F({df_between}, {df_within})",
        assumptions=(
            Assumption("observations are independent", None, "structural"),
            Assumption("equal variances", bool(variance_check.p_value >= alpha)
                       if not math.isnan(variance_check.p_value) else None,
                       f"Brown-Forsythe p={variance_check.p_value:.4f}"),
            *[_normality_assumption(f"group {i + 1}", g, alpha) for i, g in enumerate(valid)],
        ),
        extra={"eta_squared": eta_sq, "ss_between": ss_between, "ss_within": ss_within,
               "df_between": df_between, "df_within": df_within},
    )


# --------------------------------------------------------------------------
# rank-based tests
# --------------------------------------------------------------------------


def _ranks(values: Sequence[float]) -> list[float]:
    """Average ranks, ties shared."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def _mann_whitney_exact_counts(n1: int, n2: int) -> list[int]:
    """Counts of rank arrangements giving each value of U (no ties).

    Uses the classical recurrence ``f(m,n,u) = f(m-1,n,u-n) + f(m,n-1,u)``,
    which enumerates the null distribution exactly rather than assuming
    normality -- the difference matters at the five-seed sample sizes that
    are common in practice.
    """
    table: list[list[list[int]]] = [[[1] for _ in range(n2 + 1)] for _ in range(n1 + 1)]
    for m in range(1, n1 + 1):
        for n in range(1, n2 + 1):
            size = m * n + 1
            current = [0] * size
            left = table[m - 1][n]
            down = table[m][n - 1]
            for u in range(size):
                total = 0
                if 0 <= u - n < len(left):
                    total += left[u - n]
                if u < len(down):
                    total += down[u]
                current[u] = total
            table[m][n] = current
    return table[n1][n2]


def mann_whitney_u(a: Sequence[float], b: Sequence[float], *,
                   alternative: str = "two-sided", alpha: float = 0.05) -> TestResult:
    """Mann-Whitney U test for two independent samples.

    Exact when both groups are small and free of ties, otherwise a
    normal approximation with tie and continuity corrections.  The result
    records which route it took.
    """
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return TestResult("mann-whitney-u", math.nan, math.nan, n=(na, nb),
                          reason="empty group")
    combined = list(a) + list(b)
    ranks = _ranks(combined)
    r1 = sum(ranks[:na])
    u1 = r1 - na * (na + 1) / 2.0
    u2 = na * nb - u1
    u = min(u1, u2)
    tie_groups: dict[float, int] = {}
    for value in combined:
        tie_groups[value] = tie_groups.get(value, 0) + 1
    has_ties = any(c > 1 for c in tie_groups.values())

    exact = (not has_ties) and na <= _EXACT_LIMIT and nb <= _EXACT_LIMIT
    if exact:
        counts = _mann_whitney_exact_counts(na, nb)
        total = float(sum(counts))
        cumulative = sum(counts[: int(u) + 1]) / total
        if alternative == "two-sided":
            p = min(1.0, 2.0 * cumulative)
        elif alternative == "greater":
            p = sum(counts[int(u1):]) / total
        else:
            p = sum(counts[: int(u1) + 1]) / total
        detail = "exact null distribution by dynamic programming"
        z = math.nan
    else:
        n = na + nb
        mu_u = na * nb / 2.0
        tie_term = sum(c ** 3 - c for c in tie_groups.values())
        sigma_sq = (na * nb / 12.0) * ((n + 1) - tie_term / (n * (n - 1))) if n > 1 else 0.0
        if sigma_sq <= 0:
            return TestResult("mann-whitney-u", u, 1.0, n=(na, nb),
                              reason="degenerate rank distribution")
        sigma = math.sqrt(sigma_sq)
        target = u1 if alternative != "two-sided" else u
        continuity = 0.5
        if alternative == "two-sided":
            z = (abs(target - mu_u) - continuity) / sigma
            p = 2.0 * norm_sf(max(z, 0.0))
        elif alternative == "greater":
            z = (target - mu_u - continuity) / sigma
            p = norm_sf(z)
        else:
            z = (target - mu_u + continuity) / sigma
            p = norm_cdf(z)
        p = min(1.0, max(0.0, p))
        detail = "normal approximation with tie and continuity correction"

    return TestResult(
        "mann-whitney-u", u, p, alternative=alternative, n=(na, nb),
        method_detail=detail,
        assumptions=(
            Assumption("observations are independent", None, "structural"),
            Assumption("metric is at least ordinal", True, "numeric metric"),
            Assumption("normality", None, "not required by this test"),
            Assumption("distributions have similar shape", None,
                       "required only to read the result as a shift in medians"),
        ),
        warnings=tuple(_size_warnings(a, b)),
        extra={"u1": u1, "u2": u2, "z": z, "exact": exact, "ties": has_ties},
    )


def _wilcoxon_exact_counts(n: int) -> list[int]:
    """Counts of signed-rank sums for ``n`` pairs."""
    total = n * (n + 1) // 2
    counts = [0] * (total + 1)
    counts[0] = 1
    for rank in range(1, n + 1):
        for s in range(total, rank - 1, -1):
            counts[s] += counts[s - rank]
    return counts


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float], *,
                         alternative: str = "two-sided", alpha: float = 0.05) -> TestResult:
    """Wilcoxon signed-rank test for paired samples."""
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    nonzero = [d for d in diffs if d != 0]
    dropped = len(diffs) - len(nonzero)
    n = len(nonzero)
    if n == 0:
        return TestResult("wilcoxon-signed-rank", math.nan, 1.0, n=(len(diffs),),
                          reason="every pair is identical")
    ranks = _ranks([abs(d) for d in nonzero])
    w_plus = sum(r for d, r in zip(nonzero, ranks, strict=True) if d > 0)
    w_minus = sum(r for d, r in zip(nonzero, ranks, strict=True) if d < 0)
    w = min(w_plus, w_minus)
    has_ties = len(set(abs(d) for d in nonzero)) != n

    exact = n <= _EXACT_LIMIT and not has_ties
    if exact:
        counts = _wilcoxon_exact_counts(n)
        total = float(sum(counts))
        if alternative == "two-sided":
            p = min(1.0, 2.0 * sum(counts[: int(w) + 1]) / total)
        elif alternative == "greater":
            p = sum(counts[: int(w_minus) + 1]) / total
        else:
            p = sum(counts[: int(w_plus) + 1]) / total
        detail = "exact null distribution by dynamic programming"
        z = math.nan
    else:
        mu_w = n * (n + 1) / 4.0
        tie_counts: dict[float, int] = {}
        for d in nonzero:
            tie_counts[abs(d)] = tie_counts.get(abs(d), 0) + 1
        tie_term = sum(c ** 3 - c for c in tie_counts.values())
        sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0)
        if sigma == 0:
            return TestResult("wilcoxon-signed-rank", w, 1.0, n=(n,),
                              reason="degenerate rank distribution")
        z = (abs(w - mu_w) - 0.5) / sigma
        if alternative == "two-sided":
            p = 2.0 * norm_sf(max(z, 0.0))
        elif alternative == "greater":
            # Evidence that a > b lives in a large positive rank sum.
            p = norm_sf((w_plus - mu_w - 0.5) / sigma)
        else:
            p = norm_cdf((w_plus - mu_w + 0.5) / sigma)
        p = min(1.0, max(0.0, p))
        detail = "normal approximation with tie and continuity correction"

    warnings = list(_size_warnings(nonzero))
    if dropped:
        warnings.append(f"{dropped} pair(s) with zero difference were dropped (Wilcoxon convention)")
    return TestResult(
        "wilcoxon-signed-rank", w, p, alternative=alternative, n=(n,),
        method_detail=detail,
        assumptions=(
            Assumption("pairs are matched", True, "identical seed set across conditions"),
            Assumption("differences are symmetric about the median", None,
                       "not checked; the test is otherwise distribution-free"),
            Assumption("normality", None, "not required by this test"),
        ),
        warnings=tuple(warnings),
        extra={"w_plus": w_plus, "w_minus": w_minus, "z": z, "exact": exact,
               "dropped_zero_pairs": dropped},
    )


# --------------------------------------------------------------------------
# resampling
# --------------------------------------------------------------------------


def permutation_test(a: Sequence[float], b: Sequence[float], *, paired: bool = False,
                     n_resamples: int = 10000, seed: int = 12345,
                     alternative: str = "two-sided") -> TestResult:
    """Permutation test on the difference of means.

    Makes no distributional assumption at all, which is why Fabric
    reaches for it when normality is rejected and the sample is too small
    to trust an asymptotic rank test.  Exact enumeration is used whenever
    the permutation space is small enough; the p-value uses the (r+1)/(B+1)
    convention so it can never be reported as exactly zero.
    """
    na, nb = len(a), len(b)
    if paired and na != nb:
        raise ValueError("paired samples must have equal length")
    if na == 0 or nb == 0:
        return TestResult("permutation", math.nan, math.nan, n=(na, nb), reason="empty group")

    observed = mean(a) - mean(b)

    def more_extreme(value: float) -> bool:
        if alternative == "two-sided":
            return abs(value) >= abs(observed) - 1e-15
        if alternative == "greater":
            return value >= observed - 1e-15
        return value <= observed + 1e-15

    if paired:
        diffs = [x - y for x, y in zip(a, b, strict=True)]
        n = len(diffs)
        exact = n <= 22 and 2 ** n <= max(n_resamples, 4096)
        count = 0
        total = 0
        if exact:
            for mask in range(2 ** n):
                signed = [d if (mask >> i) & 1 else -d for i, d in enumerate(diffs)]
                total += 1
                if more_extreme(mean(signed)):
                    count += 1
            p = count / total
            detail = f"exact sign-flip enumeration over {total} assignments"
        else:
            rng = random.Random(seed)
            for _ in range(n_resamples):
                signed = [d if rng.random() < 0.5 else -d for d in diffs]
                total += 1
                if more_extreme(mean(signed)):
                    count += 1
            p = (count + 1) / (total + 1)
            detail = f"Monte-Carlo sign-flip, {n_resamples} resamples, seed {seed}"
    else:
        pool = list(a) + list(b)
        n = len(pool)
        n_combinations = math.comb(n, na)
        exact = n_combinations <= max(n_resamples, 4096)
        count = 0
        total = 0
        if exact:
            indices = range(n)
            for chosen in combinations(indices, na):
                chosen_set = set(chosen)
                group_a = [pool[i] for i in chosen]
                group_b = [pool[i] for i in indices if i not in chosen_set]
                total += 1
                if more_extreme(mean(group_a) - mean(group_b)):
                    count += 1
            p = count / total
            detail = f"exact enumeration over {total} label assignments"
        else:
            rng = random.Random(seed)
            shuffled = list(pool)
            for _ in range(n_resamples):
                rng.shuffle(shuffled)
                total += 1
                if more_extreme(mean(shuffled[:na]) - mean(shuffled[na:])):
                    count += 1
            p = (count + 1) / (total + 1)
            detail = f"Monte-Carlo label shuffling, {n_resamples} resamples, seed {seed}"

    return TestResult(
        "permutation", observed, min(1.0, p), alternative=alternative,
        n=(na, nb) if not paired else (na,),
        method_detail=detail,
        assumptions=(
            Assumption("observations are exchangeable under the null", True,
                       "runs differ only in the condition label"),
            Assumption("normality", None, "not required by this test"),
        ),
        warnings=tuple(_size_warnings(a, b)),
        extra={"exact": exact, "resamples": total, "seed": seed},
    )


# --------------------------------------------------------------------------
# equivalence
# --------------------------------------------------------------------------


def tost_equivalence(a: Sequence[float], b: Sequence[float], margin: float, *,
                     paired: bool = False, alpha: float = 0.05) -> TestResult:
    """Two one-sided tests for practical equivalence.

    This is the test a *reproduction* actually needs.  "p > 0.05" means
    "we failed to detect a difference"; TOST lets you say "the difference
    is smaller than ``margin``", which is a positive claim and the only
    honest way to report a successful replication.
    """
    if margin <= 0:
        raise ValueError("equivalence margin must be positive")
    if paired:
        if len(a) != len(b):
            raise ValueError("paired samples must have equal length")
        diffs = [x - y for x, y in zip(a, b, strict=True)]
        n = len(diffs)
        if n < 2:
            return TestResult("tost", math.nan, math.nan, reason="needs at least two pairs")
        se = stdev(diffs) / math.sqrt(n)
        difference = mean(diffs)
        df = float(n - 1)
        sizes: tuple[int, ...] = (n,)
    else:
        na, nb = len(a), len(b)
        if na < 2 or nb < 2:
            return TestResult("tost", math.nan, math.nan,
                              reason="each group needs at least two observations")
        va, vb = variance(a), variance(b)
        se = math.sqrt(va / na + vb / nb)
        difference = mean(a) - mean(b)
        df = ((va / na + vb / nb) ** 2 /
              ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))) if se > 0 else 1.0
        sizes = (na, nb)

    if se == 0:
        equivalent = abs(difference) < margin
        return TestResult("tost", 0.0, 0.0 if equivalent else 1.0, n=sizes,
                          reason="zero variance; decided by the point difference alone",
                          extra={"equivalent": equivalent, "margin": margin,
                                 "difference": difference})
    t_lower = (difference + margin) / se
    t_upper = (difference - margin) / se
    p_lower = t_sf(t_lower, df)
    p_upper = t_cdf(t_upper, df)
    p = max(p_lower, p_upper)
    return TestResult(
        "tost", min(abs(t_lower), abs(t_upper)), p, df=df, alternative="equivalence",
        n=sizes,
        method_detail=f"two one-sided t-tests against a margin of +/-{margin:g}",
        reason="tests whether the difference is smaller than a declared practical margin",
        assumptions=(
            Assumption("the equivalence margin was declared in advance", None,
                       "taken from the metric's `tolerance` field in the specification"),
            _normality_assumption(
                "differences" if paired else "group 1",
                [x - y for x, y in zip(a, b, strict=True)] if paired else a, alpha),
        ),
        extra={"equivalent": p < alpha, "margin": margin, "difference": difference,
               "p_lower": p_lower, "p_upper": p_upper},
    )


def chi_square_independence(table: Sequence[Sequence[int]]) -> TestResult:
    """Chi-squared test of independence for a contingency table.

    Used by the failure analyser to ask whether failures cluster in a
    particular configuration rather than falling uniformly.
    """
    rows = len(table)
    cols = len(table[0]) if rows else 0
    if rows < 2 or cols < 2:
        return TestResult("chi-square", math.nan, math.nan, reason="table must be at least 2x2")
    total = sum(sum(row) for row in table)
    if total == 0:
        return TestResult("chi-square", math.nan, math.nan, reason="empty table")
    row_sums = [sum(row) for row in table]
    col_sums = [sum(table[r][c] for r in range(rows)) for c in range(cols)]
    statistic = 0.0
    small_expected = 0
    for r in range(rows):
        for c in range(cols):
            expected = row_sums[r] * col_sums[c] / total
            if expected == 0:
                continue
            if expected < 5:
                small_expected += 1
            statistic += (table[r][c] - expected) ** 2 / expected
    df = (rows - 1) * (cols - 1)
    warnings = []
    if small_expected:
        warnings.append(f"{small_expected} cell(s) have expected count below 5; "
                        "the chi-squared approximation is unreliable here")
    return TestResult("chi-square", statistic, chi2_sf(statistic, df), df=float(df),
                      method_detail=f"contingency table {rows}x{cols}",
                      warnings=tuple(warnings))
