"""``fabric diff``: what is actually different between two experiments.

The value of this command is entirely in what it does *not* bury. Two
experiments differ in a dozen fields and one of them explains the gap, so
the tests below are mostly about the classification: does the thing that
can move a number get separated from the thing that cannot.
"""

from __future__ import annotations

import os

import pytest

from fabric.analysis.engine import analyse
from fabric.core.diff import MATERIAL, diff_experiments, diff_plans
from fabric.core.spec import ExperimentSpec
from fabric.core.store import Workspace
from fabric.execution.planner import plan_experiment
from fabric.execution.runner import ExecutionOptions, Runner

BASE = """
experiment:
  name: {name}
research_question: Does the component do anything?
algorithm:
  name: method
  entrypoint: python:tests.helpers:ablatable
components: [alpha, beta]
metrics:
  - name: accuracy
    direction: maximize
    primary: true
    tolerance: 0.02
baselines:
  - name: control
    algorithm:
      name: control
      entrypoint: python:tests.helpers:deterministic
seeds:
  values: [1, 2, 3, 4, 5, 6]
configuration:
  scale: {scale}
analysis:
  alpha: {alpha}
execution:
  isolation: inline
"""


def _run(root, text):
    root = str(root)
    os.makedirs(root, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(root)
    try:
        workspace = Workspace.initialise(root, force=True)
        spec = ExperimentSpec.from_yaml(text)
        plan = plan_experiment(spec, workspace, root=root)
        workspace.create_experiment_dir(plan.experiment.id)
        workspace.write_text(plan.experiment.id, "experiment.yaml", text)
        Runner(workspace, plan, ExecutionOptions(isolation="inline"),
               root=root).execute()
        runs = workspace.load_runs(plan.experiment.id)
        return plan, runs, analyse(plan, runs)
    finally:
        os.chdir(cwd)


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    root = tmp_path_factory.mktemp("diff")
    left = _run(root, BASE.format(name="left", scale=1.0, alpha=0.05))
    right = _run(root, BASE.format(name="right", scale=3.0, alpha=0.01))
    return left, right


def test_an_experiment_is_identical_to_itself(pair):
    (plan, _runs, _analysis), _right = pair
    result = diff_plans(plan, plan)
    assert result.identical
    assert not result.differences
    assert "identical in every field" in result.describe()


def test_a_configuration_change_is_reported_as_material(pair):
    (left_plan, left_runs, left_analysis), (right_plan, right_runs, right_analysis) = pair
    result = diff_experiments(left_plan, left_runs, left_analysis,
                              right_plan, right_runs, right_analysis)
    scale = [d for d in result.for_category("configuration") if d.field == "scale"]
    assert len(scale) == 1
    assert scale[0].left == 1.0
    assert scale[0].right == 3.0
    assert scale[0].material is True


SWEEP = """
experiment:
  name: {name}
research_question: does a later swept value matter?
algorithm:
  name: method
  entrypoint: python:tests.helpers:ablatable
metrics:
  - name: accuracy
    direction: maximize
configuration:
  scale: {scale}
seeds:
  values: [1, 2, 3]
execution:
  isolation: inline
"""


def test_a_change_to_a_later_swept_value_is_reported(tmp_path):
    """The reference configuration cannot reveal a change to a later value.

    ``expand_configurations`` puts the first declared value of each swept
    parameter first, so configurations[0] is identical here while the sweep
    itself changed from [1.0, 2.0] to [1.0, 9.0] -- and n_configurations is
    unchanged too. Comparing only the reference configuration, which is all
    this module used to do, calls these two experiments identical.
    """
    left = ExperimentSpec.from_yaml(SWEEP.format(name="left", scale="[1.0, 2.0]"))
    right = ExperimentSpec.from_yaml(SWEEP.format(name="right", scale="[1.0, 9.0]"))
    assert left.configurations[0] == right.configurations[0]

    root = str(tmp_path)
    workspace = Workspace.initialise(root, force=True)
    result = diff_plans(plan_experiment(left, workspace, root=root),
                        plan_experiment(right, workspace, root=root))
    assert not result.identical
    swept = [d for d in result.for_category("configuration")
             if d.field == "swept configurations"]
    assert len(swept) == 1
    assert swept[0].left == [{"scale": 2.0}]
    assert swept[0].right == [{"scale": 9.0}]
    assert swept[0].material is True


def test_two_specifications_with_the_same_sweep_are_identical(tmp_path):
    """The set comparison must not report a difference where there is none."""
    left = ExperimentSpec.from_yaml(SWEEP.format(name="left", scale="[1.0, 2.0]"))
    right = ExperimentSpec.from_yaml(SWEEP.format(name="right", scale="[1.0, 2.0]"))
    root = str(tmp_path)
    workspace = Workspace.initialise(root, force=True)
    result = diff_plans(plan_experiment(left, workspace, root=root),
                        plan_experiment(right, workspace, root=root))
    assert not [d for d in result.for_category("configuration")
                if d.field == "swept configurations"]


def test_an_added_configuration_is_reported(tmp_path):
    left = ExperimentSpec.from_yaml(SWEEP.format(name="left", scale="1.0"))
    right = ExperimentSpec.from_yaml(SWEEP.format(name="right", scale="[1.0, 2.0]"))
    root = str(tmp_path)
    workspace = Workspace.initialise(root, force=True)
    result = diff_plans(plan_experiment(left, workspace, root=root),
                        plan_experiment(right, workspace, root=root))
    swept = [d for d in result.for_category("configuration")
             if d.field == "swept configurations"]
    assert len(swept) == 1
    assert swept[0].left == []
    assert swept[0].right == [{"scale": 2.0}]


def test_an_analysis_setting_is_reported_but_not_as_material(pair):
    (left_plan, left_runs, left_analysis), (right_plan, right_runs, right_analysis) = pair
    result = diff_experiments(left_plan, left_runs, left_analysis,
                              right_plan, right_runs, right_analysis)
    alpha = [d for d in result.for_category("analysis") if d.field == "alpha"]
    assert len(alpha) == 1
    assert alpha[0].material is False, (
        "alpha decides how a number is read, not what was measured"
    )


def test_the_summary_leads_with_what_could_explain_a_gap(pair):
    (left_plan, left_runs, left_analysis), (right_plan, right_runs, right_analysis) = pair
    result = diff_experiments(left_plan, left_runs, left_analysis,
                              right_plan, right_runs, right_analysis)
    summary = result.describe()
    assert "could move a number" in summary
    assert "configuration" in summary
    assert result.material
    assert all(d.category in MATERIAL for d in result.material)


def test_the_measured_difference_is_reported_as_an_outcome(pair):
    (left_plan, left_runs, left_analysis), (right_plan, right_runs, right_analysis) = pair
    result = diff_experiments(left_plan, left_runs, left_analysis,
                              right_plan, right_runs, right_analysis)
    moved = [d for d in result.for_category("results") if "accuracy" in d.field]
    assert moved, "the configuration change should have moved the metric"
    assert all("moved by" in d.note for d in moved)
    assert all(d.material is False for d in moved), (
        "a result is the outcome, not an input that explains one"
    )


def test_categories_come_back_in_reading_order(pair):
    (left_plan, left_runs, left_analysis), (right_plan, right_runs, right_analysis) = pair
    result = diff_experiments(left_plan, left_runs, left_analysis,
                              right_plan, right_runs, right_analysis)
    order = result.categories
    assert order.index("configuration") < order.index("analysis")
    assert order[-1] == "results"


def test_the_json_shape_is_grouped_and_self_describing(pair):
    (left_plan, left_runs, left_analysis), (right_plan, right_runs, right_analysis) = pair
    payload = diff_experiments(left_plan, left_runs, left_analysis,
                               right_plan, right_runs, right_analysis).to_dict()
    assert payload["schema_version"]
    assert payload["identical"] is False
    assert payload["left"]["name"] == "left"
    assert payload["right"]["name"] == "right"
    assert set(payload["by_category"]) <= set(payload["by_category"])
    for entry in payload["differences"]:
        assert {"category", "field", "left", "right", "material"} <= set(entry)


def test_a_different_seed_set_is_material(tmp_path):
    left = _run(tmp_path / "a", BASE.format(name="six", scale=1.0, alpha=0.05))
    text = BASE.format(name="three", scale=1.0, alpha=0.05).replace(
        "values: [1, 2, 3, 4, 5, 6]", "values: [1, 2, 3]")
    right = _run(tmp_path / "b", text)
    result = diff_plans(left[0], right[0])
    seeds = [d for d in result.for_category("seeds") if d.field == "values"]
    assert len(seeds) == 1
    assert seeds[0].material is True


def test_a_benchmark_change_shows_up_as_its_own_category(tmp_path):
    """A benchmark that changed underneath is the difference that matters."""
    directory = tmp_path / "benchmarks"
    directory.mkdir()
    definition = directory / "b.yaml"
    definition.write_text(
        "benchmark: {name: b, version: v1}\n"
        "metrics:\n  - name: accuracy\n    direction: maximize\n"
        "    primary: true\n    tolerance: 0.02\n",
        encoding="utf-8",
    )
    # The experiment declares no metrics of its own: the benchmark supplies
    # them, which is the whole point of the arrangement under test.
    text = (
        BASE.format(name="claims-a-benchmark", scale=1.0, alpha=0.05)
        .replace("research_question:", "benchmark: b@v1\nresearch_question:")
        .replace("    tolerance: 0.02\n", "")
    )
    left = _run(tmp_path, text)

    definition.write_text(
        "benchmark: {name: b, version: v1}\n"
        "metrics:\n  - name: accuracy\n    direction: maximize\n"
        "    primary: true\n    tolerance: 0.04\n",
        encoding="utf-8",
    )
    right = _run(tmp_path, text.replace("claims-a-benchmark", "same-spec-later"))

    result = diff_plans(left[0], right[0])
    digests = [d for d in result.for_category("benchmark") if d.field == "digest"]
    assert len(digests) == 1
    assert digests[0].material is True
    assert "definition itself changed" in digests[0].note
