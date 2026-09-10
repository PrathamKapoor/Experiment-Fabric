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
from fabric.statistics.power import (
    DetectableEffect,
    minimum_detectable_effect,
    required_replications,
)
from fabric.statistics.selection import Decision, run_selected, select_test

__all__ = [
    "Assumption",
    "Comparison",
    "Correction",
    "Decision",
    "DetectableEffect",
    "EffectSize",
    "Interval",
    "Summary",
    "TestResult",
    "adjust",
    "all_effect_sizes",
    "anderson_darling_normality",
    "bootstrap_difference_interval",
    "bootstrap_interval",
    "brown_forsythe",
    "chi_square_independence",
    "cliffs_delta",
    "cohens_d",
    "common_language_effect",
    "compare",
    "hedges_g",
    "mann_whitney_u",
    "minimum_detectable_effect",
    "one_way_anova",
    "paired_interval",
    "paired_t_test",
    "permutation_test",
    "pooled_std",
    "proportion_interval",
    "relative_improvement",
    "required_replications",
    "run_selected",
    "select_test",
    "student_t_test",
    "summarize",
    "t_interval",
    "tost_equivalence",
    "welch_interval",
    "welch_t_test",
    "wilcoxon_signed_rank",
]
