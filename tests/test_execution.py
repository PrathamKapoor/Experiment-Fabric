"""Planning, execution, caching, failures and provenance, end to end."""

from __future__ import annotations

import os

import pytest

from fabric.core.entities import FailureClass, Role, RunStatus
from fabric.core.provenance import build_provenance, explain, find_metric_node, impact
from fabric.core.spec import ExperimentSpec
from fabric.core.store import Workspace
from fabric.execution.adapters import (
    EntrypointError,
    RunContext,
    entrypoint_digest,
    parse_entrypoint,
    resolve,
)
from fabric.execution.cache import CachePolicy, cache_key, is_cacheable
from fabric.execution.failures import classify, from_process
from fabric.execution.planner import experiment_id_for, plan_experiment
from fabric.execution.runner import ExecutionOptions, Runner

SPEC = """
experiment:
  name: integration
research_question: Does the harness work end to end?
algorithm:
  name: method
  entrypoint: python:tests.helpers:{entry}
components: [alpha, beta]
metrics:
  - name: accuracy
    tolerance: 0.02
  - latency_ms
configuration:
  scale: {scale}
seeds:
  values: [1, 2, 3, 4, 5, 6]
{extra}
execution:
  isolation: inline
  max_workers: 1
"""


def make_spec(entry: str = "noop", scale: str = "1.0", extra: str = "") -> ExperimentSpec:
    return ExperimentSpec.from_yaml(
        SPEC.format(entry=entry, scale=scale, extra=extra))


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return Workspace.initialise(str(tmp_path))


# --------------------------------------------------------------------------
# entrypoints
# --------------------------------------------------------------------------


def test_entrypoint_forms_are_recognised():
    assert parse_entrypoint("python:pkg.mod:fn") == ("python", "pkg.mod:fn")
    assert parse_entrypoint("script:a/b.py:fn") == ("script", "a/b.py:fn")
    assert parse_entrypoint("command:./train.sh --x 1") == ("command", "./train.sh --x 1")
    assert parse_entrypoint("pkg.mod:fn") == ("python", "pkg.mod:fn")
    assert parse_entrypoint("./train.sh") == ("command", "./train.sh")


def test_unresolvable_entrypoint_says_what_to_check():
    with pytest.raises(EntrypointError, match="cannot import"):
        resolve("python:no.such.module:fn")
    with pytest.raises(EntrypointError, match="has no attribute"):
        resolve("python:tests.helpers:not_a_function")


def test_entrypoint_digest_tracks_the_source_file():
    digest = entrypoint_digest("python:tests.helpers:noop")
    assert digest.startswith("sha256:")


def test_run_context_rng_streams_are_independent_and_reproducible():
    ctx = RunContext(run_id="R", experiment_id="E", condition="c", role="treatment", seed=7)
    assert ctx.rng("a").random() == RunContext(
        run_id="R", experiment_id="E", condition="c", role="treatment", seed=7
    ).rng("a").random()
    assert ctx.rng("a").random() != ctx.rng("b").random()


def test_run_context_reports_ablated_components():
    ctx = RunContext(run_id="R", experiment_id="E", condition="c", role="ablation",
                     seed=1, ablated=("alpha",))
    assert ctx.enabled("beta") is True
    assert ctx.enabled("alpha") is False


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------


def test_plan_expands_arms_and_seeds(workspace):
    spec = make_spec(extra="""
baselines:
  - name: control
    algorithm:
      name: control
      entrypoint: python:tests.helpers:deterministic
ablations: [remove_alpha, remove_beta]
""")
    plan = plan_experiment(spec, workspace, root=workspace.root)
    roles = {c.name: c.role for c in plan.conditions}
    assert roles["proposed"] == Role.TREATMENT
    assert roles["control"] == Role.BASELINE
    assert roles["remove_alpha"] == Role.ABLATION
    assert len(plan.conditions) == 4
    assert len(plan.runs) == 24


def test_identifiers_are_deterministic_across_planning(workspace):
    spec = make_spec()
    first = plan_experiment(spec, workspace, root=workspace.root)
    second = plan_experiment(spec, workspace, root=workspace.root)
    assert first.experiment.id == second.experiment.id
    assert [r.id for r in first.runs] == [r.id for r in second.runs]
    assert first.experiment.id == experiment_id_for(spec)


def test_changing_the_spec_changes_the_experiment_id(workspace):
    a = plan_experiment(make_spec(), workspace, root=workspace.root)
    b = plan_experiment(make_spec(scale="2.0"), workspace, root=workspace.root)
    assert a.experiment.id != b.experiment.id


def test_sweep_produces_one_condition_per_configuration(workspace):
    spec = make_spec(scale="[1.0, 2.0, 3.0]")
    plan = plan_experiment(spec, workspace, root=workspace.root)
    treatments = [c for c in plan.conditions if c.role == Role.TREATMENT]
    assert len(treatments) == 3
    assert {c.config["scale"] for c in treatments} == {1.0, 2.0, 3.0}


def test_ablations_stay_at_the_reference_configuration(workspace):
    spec = make_spec(scale="[1.0, 2.0]", extra="ablations: [remove_alpha]\n")
    plan = plan_experiment(spec, workspace, root=workspace.root)
    ablations = [c for c in plan.conditions if c.role == Role.ABLATION]
    assert len(ablations) == 1
    assert ablations[0].config["scale"] == 1.0


def test_graph_is_built_with_every_run(workspace):
    plan = plan_experiment(make_spec(), workspace, root=workspace.root)
    stats = plan.graph.stats()
    assert stats["kinds"]["run"] == len(plan.runs)
    assert stats["kinds"]["condition"] == len(plan.conditions)


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------


def test_runs_execute_and_persist(workspace):
    plan = plan_experiment(make_spec(), workspace, root=workspace.root)
    workspace.create_experiment_dir(plan.experiment.id)
    report = Runner(workspace, plan, ExecutionOptions(isolation="inline"),
                    root=workspace.root).execute()
    assert len(report.completed) == 6
    stored = workspace.load_runs(plan.experiment.id)
    assert len(stored) == 6
    assert all("accuracy" in r.metrics for r in stored)
    assert all(r.environment_id and r.hardware_id for r in stored)


def test_execution_is_resumable(workspace):
    plan = plan_experiment(make_spec(), workspace, root=workspace.root)
    workspace.create_experiment_dir(plan.experiment.id)
    options = ExecutionOptions(isolation="inline")
    runner = Runner(workspace, plan, options, root=workspace.root)
    runner.execute(list(plan.runs)[:3])
    remaining = Runner(workspace, plan, options, root=workspace.root).pending_runs()
    assert len(remaining) == 3


def test_same_seed_gives_the_same_number(workspace):
    spec = make_spec(entry="deterministic")
    plan = plan_experiment(spec, workspace, root=workspace.root)
    workspace.create_experiment_dir(plan.experiment.id)
    Runner(workspace, plan, ExecutionOptions(isolation="inline"),
           root=workspace.root).execute()
    runs = {r.seed: r.metrics["accuracy"] for r in workspace.load_runs(plan.experiment.id)}
    assert runs[1] == pytest.approx(0.51)
    assert runs[3] == pytest.approx(0.53)


def test_process_isolation_produces_the_same_results_as_inline(workspace):
    spec = make_spec(entry="deterministic")
    plan = plan_experiment(spec, workspace, root=workspace.root)
    workspace.create_experiment_dir(plan.experiment.id)
    Runner(workspace, plan, ExecutionOptions(isolation="process"),
           root=workspace.root).execute()
    runs = {r.seed: r.metrics["accuracy"] for r in workspace.load_runs(plan.experiment.id)}
    assert runs[1] == pytest.approx(0.51)


def test_artifacts_are_hashed_and_recorded(workspace):
    spec = make_spec(entry="ablatable")
    plan = plan_experiment(spec, workspace, root=workspace.root)
    workspace.create_experiment_dir(plan.experiment.id)
    Runner(workspace, plan, ExecutionOptions(isolation="inline"),
           root=workspace.root).execute()
    run = workspace.load_runs(plan.experiment.id)[0]
    assert run.artifacts
    artifact = run.artifacts[0]
    assert artifact.digest.startswith("sha256:") and artifact.bytes > 0
    path = os.path.join(workspace.artifacts_dir(plan.experiment.id, run.id), artifact.path)
    assert os.path.isfile(path)


def test_a_run_with_no_metrics_is_a_failure_not_an_empty_success(workspace):
    spec = make_spec(entry="returns_nothing")
    plan = plan_experiment(spec, workspace, root=workspace.root)
    workspace.create_experiment_dir(plan.experiment.id)
    report = Runner(workspace, plan, ExecutionOptions(isolation="inline"),
                    root=workspace.root).execute()
    assert len(report.failed) == 6
    assert report.failed[0].failure.failure_class == FailureClass.INVALID_OUTPUT


def test_failures_are_recorded_and_classified(workspace):
    spec = make_spec(entry="always_fails")
    plan = plan_experiment(spec, workspace, root=workspace.root)
    workspace.create_experiment_dir(plan.experiment.id)
    report = Runner(workspace, plan, ExecutionOptions(isolation="inline"),
                    root=workspace.root).execute()
    assert len(report.failed) == 6
    assert all(r.failure.failure_class == FailureClass.OOM for r in report.failed)
    assert workspace.load_runs(plan.experiment.id)[0].status == RunStatus.FAILED


def test_a_failure_does_not_stop_the_other_runs(workspace):
    spec = make_spec(entry="fails_above_threshold", scale="[1.0, 4.0]")
    plan = plan_experiment(spec, workspace, root=workspace.root)
    workspace.create_experiment_dir(plan.experiment.id)
    report = Runner(workspace, plan, ExecutionOptions(isolation="inline"),
                    root=workspace.root).execute()
    assert len(report.failed) == 6 and len(report.completed) == 6


# --------------------------------------------------------------------------
# caching
# --------------------------------------------------------------------------


def test_cache_hit_reuses_a_previous_run(workspace):
    spec = make_spec(entry="deterministic")
    plan = plan_experiment(spec, workspace, root=workspace.root)
    workspace.create_experiment_dir(plan.experiment.id)
    options = ExecutionOptions(isolation="inline", cache=CachePolicy(enabled=True))
    Runner(workspace, plan, options, root=workspace.root).execute()

    second = Runner(workspace, plan,
                    ExecutionOptions(isolation="inline", rerun=True,
                                     cache=CachePolicy(enabled=True)),
                    root=workspace.root)
    report = second.execute(list(plan.runs))
    assert len(report.cached) == 0  # rerun bypasses the cache by design

    third = Runner(workspace, plan, options, root=workspace.root)
    report = third.execute(list(plan.runs))
    assert len(report.cached) == 6
    assert all(r.cached_from for r in report.cached)


def test_cached_runs_carry_their_artifacts(workspace):
    spec = make_spec(entry="ablatable")
    plan = plan_experiment(spec, workspace, root=workspace.root)
    workspace.create_experiment_dir(plan.experiment.id)
    options = ExecutionOptions(isolation="inline", cache=CachePolicy(enabled=True))
    Runner(workspace, plan, options, root=workspace.root).execute()
    report = Runner(workspace, plan, options, root=workspace.root).execute(list(plan.runs))
    cached = report.cached[0]
    assert cached.artifacts
    for artifact in cached.artifacts:
        path = os.path.join(workspace.artifacts_dir(plan.experiment.id, cached.id),
                            artifact.path)
        assert os.path.isfile(path), "a cached run must own its artifacts"


def test_cache_key_responds_to_what_matters(workspace):
    from fabric.environments.capture import capture_environment

    plan = plan_experiment(make_spec(), workspace, root=workspace.root)
    environment = capture_environment(workspace.root)
    policy = CachePolicy(enabled=True)
    run = plan.runs[0]
    base = cache_key(run, algorithm_id="alg", source_digest="sha256:a",
                     environment=environment, policy=policy)
    assert base == cache_key(run, algorithm_id="alg", source_digest="sha256:a",
                             environment=environment, policy=policy)
    assert base != cache_key(run, algorithm_id="alg", source_digest="sha256:CHANGED",
                             environment=environment, policy=policy)
    other = plan.runs[1]
    assert base != cache_key(other, algorithm_id="alg", source_digest="sha256:a",
                             environment=environment, policy=policy)
    assert base != cache_key(run, algorithm_id="alg", source_digest="sha256:a",
                             environment=environment, policy=policy,
                             env_overlay={"OMP_NUM_THREADS": "1"})


def test_determinism_replicates_never_come_from_cache(workspace):
    plan = plan_experiment(make_spec(), workspace, root=workspace.root)
    run = plan.runs[0]
    policy = CachePolicy(enabled=True)
    assert is_cacheable(run, policy) is True
    from dataclasses import replace

    assert is_cacheable(replace(run, replicate=1), policy) is False


# --------------------------------------------------------------------------
# failure classification
# --------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("RuntimeError: CUDA out of memory", FailureClass.OOM),
    ("TimeoutError: exceeded the 60s timeout", FailureClass.TIMEOUT),
    ("ModuleNotFoundError: No module named 'torch'", FailureClass.DEPENDENCY),
    ("ConnectionResetError: [Errno 104]", FailureClass.NETWORK),
    ("Segmentation fault (core dumped)", FailureClass.SEGFAULT),
    ("ValueError: something went wrong", FailureClass.USER_ERROR),
    ("a totally unremarkable message", FailureClass.UNKNOWN),
])
def test_failure_classification(text, expected):
    assert classify(text) == expected


def test_exit_codes_classify_signals():
    assert from_process("", "", 137).failure_class == FailureClass.OOM
    assert from_process("", "", -11).failure_class == FailureClass.SEGFAULT


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


def test_provenance_traces_a_metric_to_its_inputs(workspace):
    plan = plan_experiment(make_spec(), workspace, root=workspace.root)
    workspace.create_experiment_dir(plan.experiment.id)
    Runner(workspace, plan, ExecutionOptions(isolation="inline"),
           root=workspace.root).execute()
    runs = workspace.load_runs(plan.experiment.id)
    graph = build_provenance(plan, runs)

    node = find_metric_node(graph, condition="proposed", metric="accuracy", seed=3)
    assert node is not None
    explanation = explain(graph, node.id)
    kinds = [item["kind"] for item in explanation.chain]
    assert kinds[0] == "metric"
    for expected in ("run", "condition", "algorithm", "environment", "hardware"):
        assert expected in kinds
    assert "accuracy" in explanation.render()


def test_impact_lists_what_depends_on_an_input(workspace):
    plan = plan_experiment(make_spec(), workspace, root=workspace.root)
    workspace.create_experiment_dir(plan.experiment.id)
    Runner(workspace, plan, ExecutionOptions(isolation="inline"),
           root=workspace.root).execute()
    graph = build_provenance(plan, workspace.load_runs(plan.experiment.id))
    algorithm_id = plan.algorithm.id
    result = impact(graph, algorithm_id)
    assert result["affected"] >= len(plan.runs)
    assert result["by_kind"]["run"] == len(plan.runs)
