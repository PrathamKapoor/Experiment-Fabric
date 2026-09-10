"""Choosing a statistical test, and saying why.

This module is the reason Fabric's statistics are auditable.  Most tools
either hard-code a t-test or make the researcher pick; Fabric inspects
the experimental structure, applies a written-down decision procedure,
and records the entire chain -- what it checked, what it found, what it
chose, and what it was worried about -- next to the p-value.

The procedure, in order:

1. Is the design paired?  Two conditions evaluated on the *same* seed set
   are paired; anything else is independent.
2. Is the sample big enough to say anything at all?  Below n=3 Fabric
   refuses to test and says so.
3. Does normality hold?  Anderson-Darling at n>=8; below that the
   question is unanswerable and Fabric treats it as unverified.
4. Choose:
   - normal (or unverifiable but n>=8): t-test (paired or Welch)
   - non-normal, small n: permutation test on the difference of means
   - non-normal, larger n: rank test (Wilcoxon or Mann-Whitney)
5. Record every assumption, including the ones that were not checkable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from fabric.statistics.hypothesis import (
    Assumption,
    TestResult,
    anderson_darling_normality,
    mann_whitney_u,
    paired_t_test,
    permutation_test,
    welch_t_test,
    wilcoxon_signed_rank,
)

PERMUTATION_THRESHOLD = 12
"""Below this many observations, prefer permutation over a rank test.

Rank tests lean on asymptotics that have not arrived yet at n=6; an
exact permutation test has no such problem and, at these sizes, runs in
milliseconds.
"""


@dataclass(frozen=True)
class Decision:
    """The chosen test and the reasoning that produced it."""

    test: str
    paired: bool
    reason: str
    considered: tuple[str, ...] = ()
    checks: tuple[Assumption, ...] = ()
    warnings: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "test": self.test,
            "paired": self.paired,
            "reason": self.reason,
            "considered": list(self.considered),
            "checks": [c.to_dict() for c in self.checks],
            "warnings": list(self.warnings),
            "parameters": self.parameters,
        }


def _normality(label: str, values: Sequence[float], alpha: float) -> tuple[bool | None, Assumption]:
    n = len(values)
    if n < 8:
        return None, Assumption(
            f"{label} is approximately normal", None,
            f"n={n}: too few observations for a normality test to have any power",
            {"n": n},
        )
    a2, p = anderson_darling_normality(values)
    if math.isnan(p):
        return None, Assumption(f"{label} is approximately normal", None,
                                "sample is constant", {"n": n})
    return p >= alpha, Assumption(
        f"{label} is approximately normal", p >= alpha,
        f"Anderson-Darling A2*={a2:.3f}, p={p:.4f}",
        {"n": n, "a2_star": a2, "p_value": p},
    )


def select_test(
    treatment: Sequence[float],
    baseline: Sequence[float],
    *,
    paired: bool,
    alpha: float = 0.05,
    permutation_resamples: int = 10000,
    permutation_seed: int = 12345,
) -> Decision:
    """Apply the decision procedure and return the chosen test."""
    n_t, n_b = len(treatment), len(baseline)
    considered = ("paired-t", "wilcoxon-signed-rank", "permutation") if paired else \
                 ("welch-t", "mann-whitney-u", "permutation")
    warnings: list[str] = []

    if n_t < 2 or n_b < 2:
        return Decision(
            "none", paired,
            f"cannot test with n={n_t} and n={n_b}: at least two replications per "
            "condition are required",
            considered, (), ("insufficient replications: no inference attempted",),
        )
    if paired and n_t != n_b:
        raise ValueError("paired comparison requires equal sample sizes")

    if n_t < 5 or n_b < 5:
        warnings.append(
            f"n={min(n_t, n_b)} per condition: this experiment has low power. "
            "A non-significant result here is not evidence of no effect."
        )

    checks: list[Assumption] = []
    if paired:
        diffs = [x - y for x, y in zip(treatment, baseline)]
        normal, check = _normality("paired differences", diffs, alpha)
        checks.append(check)
        sample_size = len(diffs)
    else:
        normal_t, check_t = _normality("treatment", treatment, alpha)
        normal_b, check_b = _normality("baseline", baseline, alpha)
        checks.extend([check_t, check_b])
        normal = None if (normal_t is None or normal_b is None) else (normal_t and normal_b)
        if normal_t is False or normal_b is False:
            normal = False
        sample_size = min(n_t, n_b)

    checks.append(Assumption(
        "observations are independent across replications", True,
        "each replication is a separate process with a distinct seed",
    ))

    if normal is False:
        if sample_size < PERMUTATION_THRESHOLD:
            test = "permutation"
            reason = (
                f"normality was rejected (see checks) and n={sample_size} is too small "
                "for a rank test's asymptotics, so an exact permutation test on the "
                "difference of means was used: it assumes only exchangeability."
            )
        else:
            test = "wilcoxon-signed-rank" if paired else "mann-whitney-u"
            reason = (
                f"normality was rejected (see checks) with n={sample_size}, so a "
                "distribution-free rank test was used instead of a t-test."
            )
    elif normal is None:
        if sample_size < PERMUTATION_THRESHOLD:
            test = "permutation"
            reason = (
                f"n={sample_size} is too small to verify normality, so Fabric did not "
                "assume it: an exact permutation test was used, which requires only "
                "that runs are exchangeable under the null."
            )
        else:
            test = "paired-t" if paired else "welch-t"
            reason = (
                "normality could not be verified but the sample is large enough for the "
                "t-test's robustness to mild departures to apply."
            )
    else:
        test = "paired-t" if paired else "welch-t"
        reason = (
            "normality was not rejected, so a t-test was used. "
            + ("Pairing by seed removes seed-to-seed variance from the comparison."
               if paired else
               "Welch's variant was chosen because equal variances were not assumed.")
        )

    parameters: dict[str, Any] = {}
    if test == "permutation":
        parameters = {"resamples": permutation_resamples, "seed": permutation_seed}

    floor = minimum_attainable_p(test, paired, n_t, n_b)
    if floor is not None:
        parameters["minimum_attainable_p"] = floor
        if floor > alpha:
            warnings.append(
                f"design limit: with these sample sizes the smallest two-sided p-value "
                f"{test} can produce is {floor:.4f}, which is above alpha={alpha}. "
                "No result of this experiment can be significant; add replications."
            )

    return Decision(test, paired, reason, considered, tuple(checks), tuple(warnings), parameters)


def minimum_attainable_p(test: str, paired: bool, n_t: int, n_b: int) -> float | None:
    """Smallest two-sided p-value the design could possibly produce.

    Exact tests draw from a finite null distribution, so a small
    experiment has a hard floor beneath which no p-value can go -- five
    paired runs cannot go below 0.0625 however large the effect.  A tool
    that lets someone run that experiment without warning them has wasted
    their compute, so Fabric computes the floor before execution.
    """
    if test in ("permutation", "wilcoxon-signed-rank") and paired:
        n = min(n_t, n_b)
        if n <= 0 or n > 30:
            return None
        return min(1.0, 2.0 / (2 ** n))
    if test in ("permutation", "mann-whitney-u") and not paired:
        if n_t <= 0 or n_b <= 0 or n_t + n_b > 60:
            return None
        return min(1.0, 2.0 / math.comb(n_t + n_b, n_t))
    return None


def run_selected(
    decision: Decision,
    treatment: Sequence[float],
    baseline: Sequence[float],
    *,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> TestResult:
    """Execute the test a :func:`select_test` decision chose."""
    if decision.test == "none":
        return TestResult("none", math.nan, math.nan, reason=decision.reason,
                          warnings=decision.warnings)
    if decision.test == "paired-t":
        result = paired_t_test(treatment, baseline, alternative=alternative, alpha=alpha)
    elif decision.test == "welch-t":
        result = welch_t_test(treatment, baseline, alternative=alternative, alpha=alpha)
    elif decision.test == "wilcoxon-signed-rank":
        result = wilcoxon_signed_rank(treatment, baseline, alternative=alternative, alpha=alpha)
    elif decision.test == "mann-whitney-u":
        result = mann_whitney_u(treatment, baseline, alternative=alternative, alpha=alpha)
    elif decision.test == "permutation":
        result = permutation_test(
            treatment, baseline, paired=decision.paired,
            n_resamples=int(decision.parameters.get("resamples", 10000)),
            seed=int(decision.parameters.get("seed", 12345)),
            alternative=alternative,
        )
    else:
        raise ValueError(f"unknown test {decision.test!r}")

    merged_assumptions = tuple(decision.checks) + tuple(
        a for a in result.assumptions
        if a.name not in {c.name for c in decision.checks}
    )
    return TestResult(
        test=result.test,
        statistic=result.statistic,
        p_value=result.p_value,
        df=result.df,
        alternative=result.alternative,
        n=result.n,
        method_detail=result.method_detail,
        reason=decision.reason,
        assumptions=merged_assumptions,
        warnings=tuple(dict.fromkeys(decision.warnings + result.warnings)),
        extra=result.extra,
    )
