"""The statistics engine, checked against published reference values.

These are the tests that matter most: everything Fabric reports rests on
this module, and a subtly wrong incomplete-beta would produce plausible
p-values that are simply untrue.  Reference values come from published
critical-value tables and textbook worked examples, not from another
implementation of the same code.
"""

from __future__ import annotations

import math

import pytest

from fabric.statistics import (
    adjust,
    anderson_darling_normality,
    brown_forsythe,
    chi_square_independence,
    cliffs_delta,
    cohens_d,
    common_language_effect,
    compare,
    hedges_g,
    mann_whitney_u,
    one_way_anova,
    paired_t_test,
    permutation_test,
    select_test,
    student_t_test,
    summarize,
    tost_equivalence,
    welch_t_test,
    wilcoxon_signed_rank,
)
from fabric.statistics.confidence import bootstrap_interval, proportion_interval, t_interval
from fabric.statistics.descriptive import pooled_std, quantile
from fabric.statistics.distributions import (
    betainc,
    chi2_cdf,
    f_cdf,
    gammainc,
    norm_cdf,
    norm_ppf,
    t_cdf,
    t_ppf,
)
from fabric.statistics.selection import minimum_attainable_p

# --------------------------------------------------------------------------
# distributions
# --------------------------------------------------------------------------


def test_normal_matches_stdlib():
    from statistics import NormalDist

    nd = NormalDist()
    for x in (-4.0, -1.96, -0.5, 0.0, 0.5, 1.2837, 1.96, 3.5):
        assert norm_cdf(x) == pytest.approx(nd.cdf(x), abs=1e-13)
    for p in (1e-8, 0.001, 0.025, 0.3, 0.5, 0.975, 0.999999):
        assert norm_ppf(p) == pytest.approx(nd.inv_cdf(p), abs=1e-10)


@pytest.mark.parametrize("df,critical", [
    (1, 12.7062047), (2, 4.30265273), (5, 2.57058184),
    (10, 2.22813885), (30, 2.04227246), (100, 1.98397152), (1000, 1.96233908),
])
def test_t_critical_values(df, critical):
    """Two-sided 5% critical values from published t tables."""
    assert t_ppf(0.975, df) == pytest.approx(critical, abs=1e-6)
    assert t_cdf(critical, df) == pytest.approx(0.975, abs=1e-9)


@pytest.mark.parametrize("df,critical", [
    (1, 3.841459), (2, 5.991465), (5, 11.070498), (10, 18.307038),
])
def test_chi2_critical_values(df, critical):
    assert chi2_cdf(critical, df) == pytest.approx(0.95, abs=1e-6)


@pytest.mark.parametrize("df1,df2,critical", [(1, 1, 161.4476), (2, 10, 4.102821), (5, 20, 2.71089)])
def test_f_critical_values(df1, df2, critical):
    assert f_cdf(critical, df1, df2) == pytest.approx(0.95, abs=1e-5)


def test_special_functions():
    assert betainc(2, 3, 0.5) == pytest.approx(0.6875, abs=1e-12)
    assert betainc(1, 1, 0.37) == pytest.approx(0.37, abs=1e-12)
    assert gammainc(1, 1) == pytest.approx(1 - math.exp(-1), abs=1e-12)


# --------------------------------------------------------------------------
# descriptive
# --------------------------------------------------------------------------


def test_summary_values():
    summary = summarize([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    assert summary.n == 8
    assert summary.mean == pytest.approx(5.0)
    assert summary.std == pytest.approx(2.13808993, abs=1e-8)  # ddof=1
    assert summary.median == pytest.approx(4.5)
    assert summary.minimum == 2.0 and summary.maximum == 9.0


def test_quantile_matches_type_seven():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert quantile(values, 0.0) == 1
    assert quantile(values, 0.25) == pytest.approx(3.25)
    assert quantile(values, 0.5) == pytest.approx(5.5)
    assert quantile(values, 1.0) == 10


def test_pooled_std_is_the_within_group_noise():
    pooled = pooled_std([[1.0, 2.0, 3.0], [11.0, 12.0, 13.0]])
    assert pooled == pytest.approx(1.0)


def test_summary_of_empty_sample_is_not_an_exception():
    summary = summarize([])
    assert summary.n == 0 and summary.mean != summary.mean


# --------------------------------------------------------------------------
# hypothesis tests
# --------------------------------------------------------------------------


def test_mann_whitney_exact_small_sample():
    result = mann_whitney_u([1, 2, 3], [4, 5, 6])
    assert result.statistic == 0
    assert result.extra["exact"] is True
    # Two of the C(6,3)=20 arrangements are at least this extreme.
    assert result.p_value == pytest.approx(0.1)
    assert mann_whitney_u([1, 2, 3, 4], [5, 6, 7, 8]).p_value == pytest.approx(2 / 70)


def test_mann_whitney_falls_back_when_tied():
    result = mann_whitney_u([1, 2, 2, 3], [2, 4, 5, 6])
    assert result.extra["ties"] is True
    assert result.extra["exact"] is False
    assert 0.0 <= result.p_value <= 1.0


def test_wilcoxon_exact():
    result = wilcoxon_signed_rank([2, 3, 4, 5, 6], [1, 1, 1, 1, 1])
    assert result.statistic == 0
    assert result.p_value == pytest.approx(2 / 32)


def test_paired_t_equals_one_sample_t_on_differences():
    a = [1.2, 2.3, 3.1, 4.8, 5.2]
    b = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = paired_t_test(a, b)
    diffs = [x - y for x, y in zip(a, b)]
    mean = sum(diffs) / len(diffs)
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1))
    assert result.statistic == pytest.approx(mean / (sd / math.sqrt(len(diffs))))


def test_welch_equals_student_when_balanced_and_equal_variance():
    a, b = [1, 2, 3, 4, 5], [2, 3, 4, 5, 6]
    welch, student = welch_t_test(a, b), student_t_test(a, b)
    assert welch.statistic == pytest.approx(student.statistic)
    assert welch.p_value == pytest.approx(student.p_value, abs=1e-9)
    assert welch.statistic == pytest.approx(-1.0)


def test_anova_textbook_example():
    result = one_way_anova([6, 8, 4, 5, 3, 4], [8, 12, 9, 11, 6, 8], [13, 9, 11, 8, 7, 12])
    assert result.statistic == pytest.approx(9.2647, abs=1e-4)
    assert result.p_value < 0.01
    assert result.extra["df_between"] == 2 and result.extra["df_within"] == 15


def test_permutation_exact_enumeration():
    result = permutation_test([1, 2, 3], [4, 5, 6])
    assert result.extra["exact"] is True
    assert result.p_value == pytest.approx(0.1)


def test_permutation_paired_sign_flip():
    result = permutation_test([2, 3, 4, 5, 6], [1, 1, 1, 1, 1], paired=True)
    assert result.extra["exact"] is True
    assert result.p_value == pytest.approx(2 / 32)


def test_chi_square_two_by_two():
    result = chi_square_independence([[20, 30], [30, 20]])
    assert result.statistic == pytest.approx(4.0)
    assert result.p_value == pytest.approx(0.0455, abs=1e-4)


def test_brown_forsythe_detects_unequal_spread():
    equal = brown_forsythe([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
    unequal = brown_forsythe([1, 2, 3, 4, 5], [-50, 2, 3, 4, 60])
    assert equal.p_value > 0.5
    assert unequal.p_value < equal.p_value


def test_anderson_darling_separates_normal_from_not():
    normal = [-1.2, -0.5, -0.1, 0.0, 0.15, 0.4, 0.6, 1.1, -0.8, 0.9, 0.25, -0.35]
    skewed = [1, 1, 1, 1, 1, 1, 1, 1, 2, 50, 100]
    assert anderson_darling_normality(normal)[1] > 0.05
    assert anderson_darling_normality(skewed)[1] < 0.05


def test_tost_establishes_equivalence_only_when_earned():
    close = tost_equivalence([10.0, 10.1, 9.9, 10.05], [10.02, 9.98, 10.0, 10.01], 0.5)
    far = tost_equivalence([10.0, 10.1, 9.9, 10.05], [12.0, 12.1, 11.9, 12.05], 0.5)
    assert close.extra["equivalent"] is True
    assert far.extra["equivalent"] is False


# --------------------------------------------------------------------------
# effect sizes
# --------------------------------------------------------------------------


def test_cliffs_delta_is_bounded_and_signed():
    assert cliffs_delta([1, 2, 3], [4, 5, 6]).value == -1.0
    assert cliffs_delta([4, 5, 6], [1, 2, 3]).value == 1.0
    assert cliffs_delta([1, 2, 3], [1, 2, 3]).value == 0.0


def test_common_language_effect_reads_as_a_probability():
    assert common_language_effect([4, 5, 6], [1, 2, 3]).value == 1.0
    assert common_language_effect([1, 2, 3], [1, 2, 3]).value == 0.5


def test_hedges_g_shrinks_cohens_d():
    a, b = [10, 11, 12, 13], [8, 9, 10, 11]
    assert abs(hedges_g(a, b).value) < abs(cohens_d(a, b).value)


def test_zero_variance_effect_is_unbounded_not_zero():
    """A constant offset is an infinite standardised effect, not a null one."""
    effect = cohens_d([2.0, 3.0, 4.0], [1.0, 2.0, 3.0], paired=True)
    assert math.isinf(effect.value)
    assert effect.magnitude == "unbounded"
    assert "identical amount" in effect.note


# --------------------------------------------------------------------------
# intervals
# --------------------------------------------------------------------------


def test_t_interval_covers_the_mean():
    interval = t_interval([1.0, 2.0, 3.0, 4.0, 5.0])
    assert interval.low < 3.0 < interval.high
    assert interval.method == "t"


def test_bootstrap_interval_is_deterministic():
    values = [0.91, 0.92, 0.90, 0.93, 0.915, 0.925, 0.905, 0.935]
    first = bootstrap_interval(values, n_resamples=500, seed=7)
    second = bootstrap_interval(values, n_resamples=500, seed=7)
    assert (first.low, first.high) == (second.low, second.high)


def test_wilson_interval_stays_inside_zero_one():
    interval = proportion_interval(100, 100)
    assert 0.0 <= interval.low <= 1.0 and interval.high <= 1.0


# --------------------------------------------------------------------------
# multiplicity
# --------------------------------------------------------------------------


def test_holm_is_monotone_and_stops_at_first_failure():
    # 3*0.001 = 0.003 rejects; 2*0.03 = 0.06 does not, and Holm's step-down
    # rule stops there, so 0.04 is accepted even though 1*0.04 < alpha.
    correction = adjust([("a", 0.001), ("b", 0.03), ("c", 0.04)], method="holm")
    adjusted = [r.adjusted_p for r in correction.results]
    assert adjusted == sorted(adjusted)
    assert correction.by_label("a").rejected is True
    assert correction.by_label("b").rejected is False
    assert correction.by_label("c").rejected is False
    assert correction.by_label("c").adjusted_p == pytest.approx(0.06)


def test_holm_rejects_everything_when_every_step_clears():
    correction = adjust([("a", 0.001), ("b", 0.02), ("c", 0.04)], method="holm")
    assert all(r.rejected for r in correction.results)


def test_bonferroni_multiplies_by_family_size():
    correction = adjust([("a", 0.01), ("b", 0.02)], method="bonferroni")
    assert correction.by_label("a").adjusted_p == pytest.approx(0.02)


def test_benjamini_hochberg_is_less_conservative_than_bonferroni():
    pairs = [("a", 0.01), ("b", 0.02), ("c", 0.03), ("d", 0.04)]
    bh = adjust(pairs, method="benjamini-hochberg")
    bonferroni = adjust(pairs, method="bonferroni")
    assert sum(r.rejected for r in bh.results) >= sum(r.rejected for r in bonferroni.results)


def test_correction_carries_nan_through_without_rejecting():
    correction = adjust([("a", 0.01), ("b", float("nan"))], method="holm")
    assert correction.family_size == 1
    assert correction.by_label("b").rejected is False


# --------------------------------------------------------------------------
# test selection
# --------------------------------------------------------------------------


def test_selection_prefers_permutation_when_normality_is_unverifiable():
    decision = select_test([1.0, 2.0, 3.0, 4.0], [1.5, 2.5, 3.5, 4.5], paired=True)
    assert decision.test == "permutation"
    assert "too small to verify normality" in decision.reason


def test_selection_uses_paired_t_for_larger_normal_samples():
    import random

    rng = random.Random(4)
    base = [rng.gauss(0.9, 0.01) for _ in range(14)]
    treatment = [b + 0.005 for b in base]
    decision = select_test(treatment, base, paired=True)
    assert decision.test == "paired-t"


def test_selection_warns_when_no_p_value_can_reach_alpha():
    decision = select_test([1.0, 2, 3, 4, 5], [2.0, 3, 4, 5, 6], paired=True)
    assert any("design limit" in w for w in decision.warnings)
    assert decision.parameters["minimum_attainable_p"] == pytest.approx(2 / 32)


def test_minimum_attainable_p_for_unpaired_designs():
    assert minimum_attainable_p("permutation", False, 3, 3) == pytest.approx(2 / 20)
    assert minimum_attainable_p("welch-t", False, 3, 3) is None


def test_selection_refuses_to_test_a_single_run():
    decision = select_test([1.0], [2.0], paired=False)
    assert decision.test == "none"


# --------------------------------------------------------------------------
# assembled comparison
# --------------------------------------------------------------------------


def test_comparison_reports_improvement_with_narrative():
    import random

    rng = random.Random(0)
    base = [rng.gauss(0.912, 0.006) for _ in range(10)]
    proposed = [b + 0.004 for b in base]
    comparison = compare(proposed, base, metric="accuracy", treatment="ours",
                         baseline="theirs", tolerance=0.005)
    assert comparison.verdict == "improvement"
    assert comparison.difference == pytest.approx(0.004, abs=1e-9)
    assert "ours" in comparison.narrative()
    assert comparison.difference_interval.excludes_zero()


def test_comparison_on_a_minimised_metric_signs_positive_as_better():
    faster = [100.0, 102, 99, 101, 98, 103, 97, 100, 101, 99]
    slower = [120.0, 119, 121, 118, 122, 120, 119, 121, 120, 118]
    comparison = compare(faster, slower, metric="latency_ms", higher_is_better=False)
    assert comparison.difference > 0  # faster is better, so the difference is positive
    assert comparison.verdict == "improvement"


def test_comparison_says_inconclusive_rather_than_equal():
    import random

    rng = random.Random(11)
    a = [rng.gauss(0.9, 0.05) for _ in range(6)]
    b = [rng.gauss(0.9, 0.05) for _ in range(6)]
    comparison = compare(a, b, metric="accuracy")
    assert comparison.verdict in ("inconclusive", "equivalent")
    if comparison.verdict == "inconclusive":
        assert "absence of evidence" in comparison.narrative()


# --------------------------------------------------------------------------
# power
# --------------------------------------------------------------------------


def test_minimum_detectable_effect_shrinks_with_more_replications():
    import random

    from fabric.statistics.power import minimum_detectable_effect

    rng = random.Random(2)
    # The pairs need their own noise: a constant offset makes the paired
    # standard error exactly zero, and the floor with it.
    small_a = [rng.gauss(0.9, 0.02) for _ in range(5)]
    small_b = [a + rng.gauss(0.001, 0.004) for a in small_a]
    large_a = [rng.gauss(0.9, 0.02) for _ in range(40)]
    large_b = [a + rng.gauss(0.001, 0.004) for a in large_a]

    small = minimum_detectable_effect(small_b, small_a, paired=True)
    large = minimum_detectable_effect(large_b, large_a, paired=True)
    assert large.value < small.value
    assert small.value > 0


def test_a_difference_below_the_floor_is_reported_as_such():
    from fabric.statistics.power import minimum_detectable_effect

    a = [0.90, 0.92, 0.88, 0.94, 0.86, 0.91]
    b = [x + 0.0001 for x in a]
    # Paired differences here are constant, so use an unpaired framing where
    # the spread is real and the shift is negligible.
    effect = minimum_detectable_effect(b, a, paired=False, metric="accuracy")
    assert effect.observed_is_below_floor is True
    assert "resolution" in effect.describe()


def test_required_replications_answers_how_many_more_seeds():
    from fabric.statistics.power import required_replications

    few = required_replications(effect=0.05, spread=0.02)
    many = required_replications(effect=0.005, spread=0.02)
    assert 2 <= few < many
