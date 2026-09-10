"""Statistical analysis of executed experiments."""

from fabric.analysis.ablation import AblationAnalysis, analyse_ablations
from fabric.analysis.baseline import BaselineAnalysis, analyse_baselines, cost_effectiveness
from fabric.analysis.determinism import DeterminismReport, analyse_determinism
from fabric.analysis.engine import ExperimentAnalysis, analyse
from fabric.analysis.failure import FailureAnalysis, analyse_failures
from fabric.analysis.reproducibility import (
    AxisStability,
    ReproducibilityProfile,
    analyse_reproducibility,
)
from fabric.analysis.results import ConditionResults, ResultSet
from fabric.analysis.sweep import SweepAnalysis, analyse_sweep

__all__ = [
    "ExperimentAnalysis", "analyse",
    "ResultSet", "ConditionResults",
    "BaselineAnalysis", "analyse_baselines", "cost_effectiveness",
    "AblationAnalysis", "analyse_ablations",
    "SweepAnalysis", "analyse_sweep",
    "FailureAnalysis", "analyse_failures",
    "DeterminismReport", "analyse_determinism",
    "ReproducibilityProfile", "AxisStability", "analyse_reproducibility",
]
