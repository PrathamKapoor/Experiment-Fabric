"""Fabric's statistical engine.

Dependency-free implementations of the descriptive statistics, interval
estimators, hypothesis tests, effect sizes and multiplicity corrections
that a computational experiment needs -- plus, unusually, a documented
procedure for *choosing* between them that records its own reasoning.
"""

from fabric.statistics.compare import Comparison, compare
from fabric.statistics.confidence import (
    Interval,
    bootstrap_difference_interval,
    bootstrap_interval,
    paired_interval,
    proportion_interval,
    t_interval,
    welch_interval,
)
from fabric.statistics.descriptive import Summary, pooled_std, summarize
from fabric.statistics.effect_size import (
    EffectSize,
    all_effect_sizes,
    cliffs_delta,
    cohens_d,
    common_language_effect,
    hedges_g,
    relative_improvement,
)
from fabric.statistics.hypothesis import (
    Assumption,
    TestResult,
    anderson_darling_normality,
    brown_forsythe,
    chi_square_independence,
    mann_whitney_u,
    one_way_anova,
    paired_t_test,
    permutation_test,
    student_t_test,
    tost_equivalence,
    welch_t_test,
    wilcoxon_signed_rank,
)
from fabric.statistics.multiple import Correction, adjust
from fabric.statistics.selection import Decision, run_selected, select_test

__all__ = [
    "Comparison", "compare", "Interval", "Summary", "summarize", "pooled_std",
    "t_interval", "welch_interval", "paired_interval", "bootstrap_interval",
    "bootstrap_difference_interval", "proportion_interval",
    "EffectSize", "all_effect_sizes", "cohens_d", "hedges_g", "cliffs_delta",
    "common_language_effect", "relative_improvement",
    "Assumption", "TestResult", "welch_t_test", "student_t_test", "paired_t_test",
    "mann_whitney_u", "wilcoxon_signed_rank", "permutation_test", "one_way_anova",
    "tost_equivalence", "brown_forsythe", "chi_square_independence",
    "anderson_darling_normality",
    "Correction", "adjust", "Decision", "select_test", "run_selected",
]
