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
    "AblationAnalysis",
    "AxisStability",
    "BaselineAnalysis",
    "ConditionResults",
    "DeterminismReport",
    "ExperimentAnalysis",
    "FailureAnalysis",
    "ReproducibilityProfile",
    "ResultSet",
    "SweepAnalysis",
    "analyse",
    "analyse_ablations",
    "analyse_baselines",
    "analyse_determinism",
    "analyse_failures",
    "analyse_reproducibility",
    "analyse_sweep",
    "cost_effectiveness",
]
