"""Experiment Fabric: reproducible, provenance-aware scientific experimentation.

An experiment is not a script that produces a number. It is a structured
scientific claim backed by a reproducible chain of evidence -- and this
package is the machinery that keeps the chain intact from the research
question to the published figure.

The short version::

    from fabric import Workspace, ExperimentSpec, plan_experiment, Runner, analyse

    workspace = Workspace.discover(".")
    spec = ExperimentSpec.from_file("experiment.yaml")
    plan = plan_experiment(spec, workspace)
    Runner(workspace, plan).execute()
    analysis = analyse(plan, workspace.load_runs(plan.experiment.id))
    print(analysis.headline)

Most people will use the command line instead: ``fabric run experiment.yaml``.
"""

__version__ = "0.3.0"

from fabric.analysis.engine import ExperimentAnalysis, analyse
from fabric.core.entities import (
    Algorithm,
    Condition,
    Dataset,
    Experiment,
    MetricSpec,
    Model,
    Role,
    Run,
    RunStatus,
)
from fabric.core.graph import ExperimentGraph
from fabric.core.registry import Registry
from fabric.core.spec import ExperimentSpec, SpecError
from fabric.core.store import Workspace
from fabric.execution.adapters import RunContext
from fabric.execution.planner import ExperimentPlan, plan_experiment
from fabric.execution.runner import ExecutionOptions, Runner

__all__ = [
    "Algorithm",
    "Condition",
    "Dataset",
    "ExecutionOptions",
    "Experiment",
    "ExperimentAnalysis",
    "ExperimentGraph",
    "ExperimentPlan",
    "ExperimentSpec",
    "MetricSpec",
    "Model",
    "Registry",
    "Role",
    "Run",
    "RunContext",
    "RunStatus",
    "Runner",
    "SpecError",
    "Workspace",
    "__version__",
    "analyse",
    "plan_experiment",
]
