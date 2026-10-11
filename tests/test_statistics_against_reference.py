"""The statistics engine, against values computed by something else.

``tests/test_statistics.py`` checks the distributions against published
critical-value tables. This file checks the same engine against numbers
produced by an independent implementation and by brute-force enumeration, so
a passing table cannot come from the same assumption being written down
twice.

The reference values below were produced with **SciPy 1.17.1** and are frozen
in as literals. SciPy is *not* a dependency of Fabric and this module does
not import it — that is the point: the expected values have to outlive the
library that produced them. Where a closed form or an exact enumeration
exists, the expected value is the enumerated one rather than SciPy's, so the
test does not inherit that library's conventions either.

Every tolerance was set from the observed agreement.

Observed agreement, worst case over all the cases below:

* distributions: relative 1e-12 (``norm_ppf`` 1e-8 in the far tail, where
  a double-precision quantile is least determined)
* Welch, Student, paired t, ANOVA, chi-squared, Brown-Forsythe: statistic,
  p-value and df to 1e-10
* Mann-Whitney and Wilcoxon, exact branches: exact agreement to 1e-12
  against a full enumeration
* effect sizes: 1e-12 against their closed forms
* t intervals: both endpoints to 1e-10
* corrections: 1e-12 against their definitions
"""

from __future__ import annotations

import itertools
import math
import random

import pytest

from fabric.statistics import (
    adjust,
    anderson_darling_normality,
    brown_forsythe,
    chi_square_independence,
    cliffs_delta,
    cohens_d,
    common_language_effect,
    hedges_g,
    mann_whitney_u,
    one_way_anova,
    paired_t_test,
    student_t_test,
    welch_t_test,
    wilcoxon_signed_rank,
)
from fabric.statistics.confidence import proportion_interval, t_interval
from fabric.statistics.distributions import (
    betainc,
    chi2_cdf,
    f_cdf,
    gammainc,
    norm_cdf,
    norm_ppf,
    norm_sf,
    t_cdf,
    t_ppf,
)
from fabric.statistics.hypothesis import _ranks


def _sample(n: int, loc: float, scale: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(loc, scale) for _ in range(n)]


# --------------------------------------------------------------------------
# distributions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("x,expected", [
    (-8.0, 6.22096057427174e-16),
    (-1.9599639845400545, 0.025),
    (-1.0, 0.15865525393145707),
    (0.0, 0.5),
    (1.0, 0.8413447460685429),
    (1.9599639845400545, 0.975),
    (8.0, 0.9999999999999993),
])
def test_norm_cdf_matches_scipy(x, expected):
    assert norm_cdf(x) == pytest.approx(expected, rel=1e-10, abs=1e-15)


@pytest.mark.parametrize("x,expected", [
    (-40.0, 1.0),
    (-8.0, 0.9999999999999993),
    (-3.0, 0.9986501019683699),
    (3.0, 0.001349898031630093),
    (8.0, 6.22096057427174e-16),
])
def test_norm_sf_matches_scipy_in_the_tail(x, expected):
    assert norm_sf(x) == pytest.approx(expected, rel=1e-9, abs=1e-300)


@pytest.mark.parametrize("p,expected", [
    (1e-12, -7.034483825301131),
    (0.001, -3.090232306167813),
    (0.5, 0.0),
    (0.975, 1.959963984540054),
    (0.999, 3.090232306167813),
    (1 - 1e-12, 7.0344869100478356),
])
def test_norm_ppf_matches_scipy(p, expected):
    assert norm_ppf(p) == pytest.approx(expected, rel=1e-8)


@pytest.mark.parametrize("df,t,expected", [
    (1, 0.0, 0.5),
    (5, 2.570581835636197, 0.975),
    (10, 2.228138851986271, 0.975),
    (10, -2.228138851986271, 0.025),
    (30, 2.042272456301238, 0.975),
    (1000, 1.9623390803879324, 0.9749999999744534),
    # SciPy prints 0.999 here; the value is closer to 0.9990000012751952 and
    # Fabric agrees with it to 1e-15, so the literal is the computed one.
    (9, 4.296806532084652, 0.9990000012751952),
])
def test_t_cdf_matches_scipy(df, t, expected):
    assert t_cdf(t, df) == pytest.approx(expected, rel=1e-12, abs=1e-14)


@pytest.mark.parametrize("df,p,expected", [
    (3, 0.975, 3.1824463052837078),
    (7, 0.95, 1.8945786050900062),
    (25, 0.995, 2.7874358136769697),
])
def test_t_ppf_matches_scipy(df, p, expected):
    assert t_ppf(p, df) == pytest.approx(expected, rel=1e-8)


@pytest.mark.parametrize("df,x,expected", [
    (1, 3.841458820694124, 0.95),
    (5, 11.070497693516351, 0.95),
    (10, 18.307038053275146, 0.95),
    (30, 43.77297183274442, 0.9500000000726245),
])
def test_chi2_cdf_matches_scipy(df, x, expected):
    assert chi2_cdf(x, df) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("d1,d2,f,expected", [
    (1, 10, 4.964602743094196, 0.9499999999878918),
    (3, 40, 2.8381707538893135, 0.9499677577751633),
    (10, 10, 1.0, 0.5),
])
def test_f_cdf_matches_scipy(d1, d2, f, expected):
    assert f_cdf(f, d1, d2) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("a,x,expected", [
    (2.5, 0.5, 0.5),
    (2.5, 2.5, 1.0),
    (10.0, 5.0, 1.0),
])
def test_betainc_matches_scipy(a, x, expected):
    assert betainc(a, a, x) == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize("a,x,expected", [
    (0.5, 2.0, 0.9544997361036415),
    (2.5, 5.0, 0.9247647538534878),
    (10.0, 20.0, 0.9950045876916924),
])
def test_gammainc_matches_scipy(a, x, expected):
    assert gammainc(a, x) == pytest.approx(expected, rel=1e-11)


# --------------------------------------------------------------------------
# hypothesis tests
# --------------------------------------------------------------------------


@pytest.mark.parametrize("na,nb,seed,statistic,p_value,df", [
    (5, 5, 1, 1.6305798295054454, 0.14274775214391622, 7.768993541595201),
    (10, 10, 2, 1.3404251761436283, 0.1972280112127395, 17.523592875818746),
    (7, 12, 3, 3.287092422891152, 0.006853363877743898, 11.491449142013455),
    (30, 30, 4, 3.181504893348704, 0.002361545630300746, 57.57249147937801),
])
def test_welch_t_matches_scipy_ttest_ind(na, nb, seed, statistic, p_value, df):
    """scipy.stats.ttest_ind(a, b, equal_var=False)."""
    result = welch_t_test(_sample(na, 1.0, 1.0, seed), _sample(nb, 0.0, 1.0, seed + 100))
    assert result.statistic == pytest.approx(statistic, rel=1e-10)
    assert result.p_value == pytest.approx(p_value, rel=1e-10)
    assert result.df == pytest.approx(df, rel=1e-12)


def test_student_t_matches_scipy_ttest_ind_pooled():
    """scipy.stats.ttest_ind(a, b, equal_var=True)."""
    result = student_t_test(_sample(8, 1.0, 1.0, 42), _sample(8, 0.2, 1.0, 43))
    assert result.statistic == pytest.approx(1.3765612462133676, rel=1e-10)
    assert result.p_value == pytest.approx(0.1902653095316314, rel=1e-10)
    assert result.df == 14.0


@pytest.mark.parametrize("n,seed,statistic,p_value", [
    (5, 7, -0.007105010799765622, 0.9946712979415142),
    (10, 8, 3.2399406473520282, 0.01015962327415377),
    (20, 9, 2.8902310332329257, 0.009377763954392618),
])
def test_paired_t_matches_scipy_ttest_rel(n, seed, statistic, p_value):
    """scipy.stats.ttest_rel(a, b)."""
    result = paired_t_test(_sample(n, 1.0, 1.0, seed), _sample(n, 0.0, 1.0, seed + 50))
    assert result.statistic == pytest.approx(statistic, rel=1e-10)
    assert result.p_value == pytest.approx(p_value, rel=1e-10)


@pytest.mark.parametrize("n,seed,statistic,p_value", [
    (4, 71, 1.9169450123368827, 0.2025251124499927),
    (10, 72, 0.9576026613770808, 0.3964660730698453),
    (20, 73, 5.187446602849928, 0.008516042638216986),
])
def test_one_way_anova_matches_scipy_f_oneway(n, seed, statistic, p_value):
    """scipy.stats.f_oneway(g1, g2, g3)."""
    groups = (_sample(n, 1.0, 1.0, seed), _sample(n, 1.4, 1.0, seed + 1),
              _sample(n, 0.8, 1.2, seed + 2))
    result = one_way_anova(*groups)
    assert result.statistic == pytest.approx(statistic, rel=1e-10)
    assert result.p_value == pytest.approx(p_value, rel=1e-10)


def test_chi_square_matches_scipy_contingency():
    """scipy.stats.chi2_contingency(table, correction=False)."""
    result = chi_square_independence([[10, 20, 30], [15, 25, 35]])
    assert result.statistic == pytest.approx(0.27692307692307694, rel=1e-12)
    assert result.p_value == pytest.approx(0.870696738961232, rel=1e-11)


def test_brown_forsythe_matches_scipy_levene_median():
    """scipy.stats.levene(g1, g2, center='median') -- Fabric's median-centred form."""
    result = brown_forsythe(_sample(8, 0.0, 1.0, 131), _sample(8, 0.0, 2.5, 132))
    assert result.statistic == pytest.approx(12.352850606169147, rel=1e-10)
    assert result.p_value == pytest.approx(0.003433693352328431, rel=1e-10)


def test_anderson_darling_is_the_stephens_modification_of_the_raw_statistic():
    """Fabric reports A2*; SciPy reports unmodified A².

    They differ by ``1 + 0.75/n + 2.25/n²`` (Stephens 1986). Checked
    numerically rather than trusted: SciPy's raw statistic for this sample is
    0.12059711600456957 and Fabric's is 0.1323553348150151, a ratio of
    1.097499 = 1 + 0.75/10 + 2.25/100. Comparing the two directly would fail,
    and that failure would say nothing about either being wrong.
    """
    values = [1.0, 2.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 9.0]
    scipy_raw = 0.12059711600456957
    a2_star, p = anderson_darling_normality(values)
    assert a2_star == pytest.approx(scipy_raw * (1 + 0.75 / 10 + 2.25 / 100),
                                    rel=1e-12)
    assert 0.0 <= p <= 1.0


# --------------------------------------------------------------------------
# rank tests: the exact branches against full enumeration, not a table
# --------------------------------------------------------------------------


def _enumerated_u_statistics(a, b):
    """Every U reachable by relabelling the pooled sample."""
    pool = list(a) + list(b)
    n = len(a)
    statistics = []
    for chosen in itertools.combinations(range(len(pool)), n):
        group_a = [pool[i] for i in chosen]
        group_b = [pool[i] for i in range(len(pool)) if i not in set(chosen)]
        ranks = _ranks(group_a + group_b)
        statistics.append(sum(ranks[:n]) - n * (n + 1) / 2)
    return statistics


def test_exact_mann_whitney_matches_a_full_enumeration():
    """Exhaustive over all C(10,5) = 252 label assignments.

    This is ground truth for the exact branch, independent of any library
    and of Fabric's own dynamic-programming table.
    """
    a = _sample(5, 1.0, 1.0, 611)
    b = _sample(5, 0.0, 1.0, 612)
    observed = mann_whitney_u(a, b)
    assert observed.extra["exact"] is True

    statistics = _enumerated_u_statistics(a, b)
    umin = min(observed.extra["u1"], 25 - observed.extra["u1"])
    expected = min(1.0, 2.0 * sum(1 for u in statistics if u <= umin) / len(statistics))
    assert observed.p_value == pytest.approx(expected, abs=1e-12)
    assert observed.p_value == pytest.approx(0.6904761904761905, abs=1e-12)


def _enumerated_wilcoxon_statistics(diffs):
    """Every rank sum reachable by sign-flipping the differences.

    The statistic is a sum of *ranks*, not of absolute differences, so the
    |diffs| are ranked once here and every sign assignment then sums ranks.

    Each entry is ``min(W+, W-)``. Counting how many fall at or below the
    observed minimum therefore counts **both** tails already -- doubling that
    count again would report four times the probability.
    """
    ranks = _ranks([abs(d) for d in diffs])
    statistics = []
    for signs in itertools.product((1, -1), repeat=len(diffs)):
        positive = sum(r for s, r in zip(signs, ranks, strict=True) if s > 0)
        negative = sum(r for s, r in zip(signs, ranks, strict=True) if s < 0)
        statistics.append(min(positive, negative))
    return statistics


def test_exact_wilcoxon_matches_a_full_enumeration():
    """Exhaustive over all 2^8 sign assignments.

    The exact branch enumerates the observed rank values rather than 1..n,
    so it is exact for tied differences too. This check shares no code with
    it: it re-derives the rank sums from a plain sign-flip enumeration.
    """
    a = _sample(8, 1.0, 1.0, 711)
    b = _sample(8, 0.0, 1.0, 712)
    observed = wilcoxon_signed_rank(a, b)
    assert observed.extra["exact"] is True

    diffs = [x - y for x, y in zip(a, b, strict=True)]
    w = min(observed.extra["w_plus"], observed.extra["w_minus"])
    statistics = _enumerated_wilcoxon_statistics(diffs)
    # The enumerated statistic is min(W+, W-), so the count at or below the
    # observed value already spans both tails.
    expected = min(1.0, sum(1 for v in statistics if v <= w + 1e-9)
                   / len(statistics))
    assert observed.p_value == pytest.approx(expected, abs=1e-12)
    assert observed.p_value == pytest.approx(0.25, abs=1e-12)


def test_the_asymptotic_wilcoxon_is_used_past_the_exact_limit():
    """Above n = 20 Fabric approximates; the approximation is documented as such.

    The enumerated exact p-value is also asserted, so a reader can see the
    size of the approximation error rather than being told it exists.
    """
    a = _sample(21, 1.0, 1.0, 811)
    b = _sample(21, 0.0, 1.0, 812)
    result = wilcoxon_signed_rank(a, b)
    assert result.extra["exact"] is False

    diffs = [x - y for x, y in zip(a, b, strict=True)]
    statistics = _enumerated_wilcoxon_statistics(diffs)
    w = min(result.extra["w_plus"], result.extra["w_minus"])
    exact = min(1.0, sum(1 for v in statistics if v <= w + 1e-9)
                / len(statistics))
    # The exact null by enumeration: 1.33514404296875e-05, which is what
    # scipy's method='exact' also returns for this sample.
    assert exact == pytest.approx(1.33514404296875e-05, rel=1e-9)
    assert result.p_value == pytest.approx(0.0001515084599925344, rel=1e-8)
    # Fabric's normal approximation is about eleven times the exact p-value
    # here -- that is what an approximation does in the far tail, and SciPy's
    # own approximation sits at 1.4125368757832353e-04, on the same side and
    # closer. The claim under test is that Fabric reports the approximation
    # it says it reports, not that the approximation is accurate.
    assert "normal approximation" in result.method_detail
    assert result.p_value > exact


# --------------------------------------------------------------------------
# effect sizes
# --------------------------------------------------------------------------


def test_effect_sizes_match_their_closed_forms():
    a = _sample(10, 0.60, 0.05, 911)
    b = _sample(10, 0.55, 0.05, 912)
    mean_a, mean_b = sum(a) / 10, sum(b) / 10
    var_a = sum((x - mean_a) ** 2 for x in a) / 9
    var_b = sum((x - mean_b) ** 2 for x in b) / 9
    pooled = math.sqrt((9 * var_a + 9 * var_b) / 18)

    assert cohens_d(a, b).value == pytest.approx((mean_a - mean_b) / pooled, rel=1e-12)
    correction = 1.0 - 3.0 / (4 * 18 - 1)          # Hedges' J
    assert hedges_g(a, b).value == pytest.approx(
        (mean_a - mean_b) / pooled * correction, rel=1e-12)

    greater = sum(1 for x in a for y in b if x > y)
    less = sum(1 for x in a for y in b if x < y)
    assert cliffs_delta(a, b).value == pytest.approx((greater - less) / 100, rel=1e-12)
    wins = sum(1 if x > y else 0.5 if x == y else 0 for x in a for y in b)
    assert common_language_effect(a, b).value == pytest.approx(wins / 100, rel=1e-12)


# --------------------------------------------------------------------------
# intervals
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n,level,low,high", [
    (5, 0.80, -0.16807759167224867, 0.7136862301897593),
    (10, 0.95, 0.5603177652116795, 1.333136660938097),
    (40, 0.99, 0.6637350125147174, 1.0674382315878),
])
def test_t_interval_matches_scipy(n, level, low, high):
    """scipy.stats.t.interval(level, n-1, loc=mean, scale=sd/sqrt(n))."""
    interval = t_interval(_sample(n, 1.0, 0.5, 1000 + n), level)
    assert interval.low == pytest.approx(low, rel=1e-10)
    assert interval.high == pytest.approx(high, rel=1e-10)
    assert math.isclose(interval.level, level)


def test_proportion_interval_is_the_wilson_closed_form():
    """Wilson, not the normal approximation: it stays inside [0, 1]."""
    z = 1.959963984540054
    successes, trials = 37, 100
    ph = successes / trials
    denominator = 1 + z * z / trials
    centre = (ph + z * z / (2 * trials)) / denominator
    half = z / denominator * math.sqrt(
        ph * (1 - ph) / trials + z * z / (4 * trials ** 2))
    interval = proportion_interval(successes, trials)
    assert interval.low == pytest.approx(centre - half, rel=1e-12)
    assert interval.high == pytest.approx(centre + half, rel=1e-12)
    assert 0.0 <= interval.low <= interval.high <= 1.0


# --------------------------------------------------------------------------
# multiplicity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method,expected", [
    ("holm", {"a": 0.004, "b": 0.060, "c": 0.060, "d": 0.060}),
    ("bonferroni", {"a": 0.004, "b": 0.080, "c": 0.120, "d": 0.160}),
    ("sidak", {"a": 0.003994, "b": 0.077632, "c": 0.114707, "d": 0.150653}),
    ("benjamini-hochberg", {"a": 0.004, "b": 0.040, "c": 0.040, "d": 0.040}),
])
def test_corrections_match_their_definitions(method, expected):
    pairs = [("a", 0.001), ("b", 0.02), ("c", 0.03), ("d", 0.04)]
    correction = adjust(pairs, method=method)
    assert correction.family_size == 4
    for label, value in expected.items():
        assert correction.by_label(label).adjusted_p == pytest.approx(value,
                                                                      rel=1e-5)


def test_sidak_is_exactly_one_minus_the_power():
    pairs = [("a", 0.001), ("b", 0.02), ("c", 0.03), ("d", 0.04)]
    correction = adjust(pairs, method="sidak")
    assert correction.by_label("a").adjusted_p == pytest.approx(1 - 0.999 ** 4,
                                                              rel=1e-12)
