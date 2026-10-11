"""Entities, hashing, YAML, the specification, the graph and the registry."""

from __future__ import annotations

import json
import os

import pytest

from fabric.core.entities import (
    Algorithm,
    Dataset,
    Direction,
    Model,
    Run,
    RunStatus,
    metric_specs,
)
from fabric.core.graph import ExperimentGraph, GraphError
from fabric.core.registry import Registry, RegistryError
from fabric.core.spec import ExperimentSpec, SpecError
from fabric.core.store import Workspace, WorkspaceError
from fabric.util.hashing import canonical_json, content_id, tree_digest
from fabric.util.jsonio import CorruptRecordError, atomic_write_text, read_json, write_json
from fabric.util.locking import FileLock
from fabric.util.yamlio import dump_yaml, loads_yaml

# --------------------------------------------------------------------------
# hashing
# --------------------------------------------------------------------------


def test_canonical_json_is_order_independent():
    assert canonical_json({"a": 1, "b": 2}) == canonical_json({"b": 2, "a": 1})


def test_canonical_json_absorbs_float_noise():
    assert canonical_json({"x": 0.1 + 0.2}) == canonical_json({"x": 0.3})


def test_content_id_is_stable_and_prefixed():
    first = content_id({"a": [1, 2, {"b": None}]}, prefix="x-")
    second = content_id({"a": [1, 2, {"b": None}]}, prefix="x-")
    assert first == second and first.startswith("x-")


def test_tree_digest_changes_with_content_and_with_names(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "a.txt").write_text("hello", encoding="utf-8")
    original = tree_digest(root)
    (root / "a.txt").write_text("hello!", encoding="utf-8")
    assert tree_digest(root) != original
    (root / "a.txt").write_text("hello", encoding="utf-8")
    assert tree_digest(root) == original
    os.rename(root / "a.txt", root / "b.txt")
    assert tree_digest(root) != original


# --------------------------------------------------------------------------
# entity identity
# --------------------------------------------------------------------------


def test_dataset_identity_is_its_content_not_its_annotations():
    bare = Dataset(name="bench", version="v1", digest="sha256:aa")
    annotated = Dataset(name="bench", version="v1", digest="sha256:aa",
                        source="somewhere", notes="a note", license="MIT")
    assert bare.id == annotated.id
    assert Dataset(name="bench", version="v1", digest="sha256:bb").id != bare.id


def test_dataset_without_a_digest_falls_back_to_every_field():
    a = Dataset(name="bench", version="v1", n_examples=10)
    b = Dataset(name="bench", version="v1", n_examples=20)
    assert a.id != b.id


def test_model_identity_covers_inference_relevant_fields():
    base = Model(name="m", version="v1", weights_digest="sha256:1", precision="bf16")
    assert base.id == Model(name="m", version="v1", weights_digest="sha256:1",
                            precision="bf16", architecture="described later").id
    assert base.id != Model(name="m", version="v1", weights_digest="sha256:1",
                            precision="fp32").id


def test_algorithm_identity_ignores_description_and_dirtiness():
    a = Algorithm(name="alg", entrypoint="python:m:f", commit="abc", description="one")
    b = Algorithm(name="alg", entrypoint="python:m:f", commit="abc", description="two",
                  dirty=True)
    assert a.id == b.id
    assert a.id != Algorithm(name="alg", entrypoint="python:m:g", commit="abc").id


def test_identity_ignores_registration_bookkeeping():
    a = Dataset(name="d", digest="sha256:x")
    b = Dataset(name="d", digest="sha256:x", registered_at=999.0, local_path="/tmp/d")
    assert a.id == b.id


def test_run_round_trips_through_json():
    run = Run(id="RUN-1", experiment_id="E", condition_id="C", condition_name="arm",
              seed=3, metrics={"accuracy": 0.5}, status=RunStatus.COMPLETED,
              ablated=("retrieval",))
    restored = Run.from_dict(json.loads(json.dumps(run.to_dict())))
    assert restored.metrics == run.metrics
    assert restored.ablated == ("retrieval",)
    assert restored.status == RunStatus.COMPLETED


def test_metric_direction_is_inferred_from_the_name():
    specs = metric_specs(["accuracy", "latency_ms", "val_loss"])
    directions = {s.name: s.direction for s in specs}
    assert directions["accuracy"] == Direction.MAXIMIZE
    assert directions["latency_ms"] == Direction.MINIMIZE
    assert directions["val_loss"] == Direction.MINIMIZE


def test_first_metric_becomes_primary_when_none_is_declared():
    specs = metric_specs(["accuracy", "f1"])
    assert specs[0].primary is True and specs[1].primary is False


# --------------------------------------------------------------------------
# YAML
# --------------------------------------------------------------------------


DOC = """
# a comment
experiment:
  name: demo
values: [1, 2.5, true, null, "quoted: colon"]
nested:
  - name: first
    inner: {a: 1, b: two}
  - name: second
matrix:
  - [1, 2]
  - [3, 4]
block: |
  line one
  line two
empty_map: {}
empty_list: []
"""


def test_yaml_parses_the_subset_specifications_need():
    data = loads_yaml(DOC)
    assert data["experiment"]["name"] == "demo"
    assert data["values"] == [1, 2.5, True, None, "quoted: colon"]
    assert data["nested"][0]["inner"] == {"a": 1, "b": "two"}
    assert data["nested"][1]["name"] == "second"
    assert data["matrix"] == [[1, 2], [3, 4]]
    assert data["block"] == "line one\nline two\n"
    assert data["empty_map"] == {} and data["empty_list"] == []


def test_yaml_round_trips_through_the_emitter():
    data = loads_yaml(DOC)
    assert loads_yaml(dump_yaml(data)) == data


def test_yaml_rejects_tabs_with_a_useful_message():
    from fabric.util.yamlio import YamlError, _pyyaml

    if _pyyaml is not None:
        pytest.skip("PyYAML is installed and raises its own error")
    with pytest.raises(YamlError, match="tab"):
        loads_yaml("a:\n\tb: 1\n")


# --------------------------------------------------------------------------
# json io
# --------------------------------------------------------------------------


def test_missing_json_returns_the_default(tmp_path):
    assert read_json(tmp_path / "nope.json", {"x": 1}) == {"x": 1}


def test_corrupt_json_is_named_not_swallowed(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(CorruptRecordError, match=r"broken\.json"):
        read_json(path)


def test_write_json_is_atomic_and_leaves_no_partials(tmp_path):
    path = tmp_path / "out" / "value.json"
    write_json(path, {"a": 1})
    assert read_json(path) == {"a": 1}
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["value.json"]


def test_a_transient_sharing_violation_does_not_fail_the_reader(tmp_path, monkeypatch):
    """Windows refuses a rename onto a file a reader holds open.

    The cache index is read by every worker while the runner writes it, so
    the reader can catch a sharing violation it did nothing wrong about.
    Retrying is what the writer already does; the reader has to as well, or
    a working experiment fails for a reason that has nothing to do with it.
    """
    path = tmp_path / "index.json"
    write_json(path, {"a": 1})

    real_open = open
    state = {"left": 2}

    def flaky_open(file, *args, **kwargs):
        if str(file) == str(path) and state["left"] > 0:
            state["left"] -= 1
            raise PermissionError(13, "Permission denied", str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr("fabric.util.jsonio.open", flaky_open, raising=False)
    monkeypatch.setattr("builtins.open", flaky_open)

    assert read_json(path) == {"a": 1}
    assert state["left"] == 0, "the reader should have retried through both violations"


def test_a_persistent_sharing_violation_is_still_reported(tmp_path, monkeypatch):
    """Retrying must not turn a real lock into an infinite hang or a silent default."""
    path = tmp_path / "locked.json"
    write_json(path, {"a": 1})

    def always_denied(file, *args, **kwargs):
        if str(file) == str(path):
            raise PermissionError(13, "Permission denied", str(file))
        return open(file, *args, **kwargs)

    monkeypatch.setattr("builtins.open", always_denied)
    monkeypatch.setattr("fabric.util.jsonio.time.sleep", lambda _s: None)

    with pytest.raises(PermissionError):
        read_json(path)


def test_a_missing_file_is_still_a_missing_file_not_a_sharing_violation(tmp_path):
    assert read_json(tmp_path / "never-written.json", {"fallback": True}) == {"fallback": True}


def test_reading_a_file_before_rewriting_it_works(tmp_path):
    """Copying a specification onto itself must not hold the destination open.

    Windows refuses ``os.replace`` onto a file that still has an open reader
    handle, and it does so deterministically rather than as a race. A
    read-and-copy that keeps its handle alive across the write fails every
    time on Windows and passes everywhere else, which is the worst possible
    shape for a bug.
    """
    path = tmp_path / "experiment.yaml"
    write_json(path, {"experiment": {"name": "demo"}})

    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    atomic_write_text(path, text)   # must be outside the `with`

    assert read_json(path) == {"experiment": {"name": "demo"}}


def test_file_lock_is_reentrant_within_a_thread(tmp_path):
    target = str(tmp_path / "index.json")
    with FileLock(target), FileLock(target):
        pass
    assert not os.path.exists(target + ".lock")


# --------------------------------------------------------------------------
# specification
# --------------------------------------------------------------------------


MINIMAL = """
experiment:
  name: demo
research_question: Does it work?
algorithm:
  name: method
  entrypoint: python:tests.helpers:noop
metrics:
  - accuracy
seeds:
  values: [1, 2, 3, 4, 5]
"""


def test_minimal_specification_parses():
    spec = ExperimentSpec.from_yaml(MINIMAL)
    assert spec.name == "demo"
    assert spec.primary_metric == "accuracy"
    assert len(spec.seeds.values) == 5


def test_specification_requires_a_research_question():
    with pytest.raises(SpecError, match="research question"):
        ExperimentSpec.from_yaml(MINIMAL.replace("research_question: Does it work?", ""))


def test_specification_requires_metrics():
    text = MINIMAL.replace("metrics:\n  - accuracy\n", "")
    with pytest.raises(SpecError, match="at least one metric"):
        ExperimentSpec.from_yaml(text)


def test_specification_rejects_an_unknown_correction():
    with pytest.raises(SpecError, match="unknown correction"):
        ExperimentSpec.from_yaml(MINIMAL + "\nanalysis:\n  correction: magic\n")


def test_specification_rejects_duplicate_arm_names():
    text = MINIMAL + "\nbaselines:\n  - name: a\n  - name: a\n"
    with pytest.raises(SpecError, match="duplicate arm"):
        ExperimentSpec.from_yaml(text)


def test_specification_rejects_an_ablation_of_an_unknown_component():
    text = MINIMAL + "\ncomponents: [retrieval]\nablations:\n  - remove_memory\n"
    with pytest.raises(SpecError, match="cannot infer"):
        ExperimentSpec.from_yaml(text)


def test_specification_advises_about_thin_seed_counts():
    spec = ExperimentSpec.from_yaml(MINIMAL.replace("[1, 2, 3, 4, 5]", "[1, 2]"))
    advisories = spec.validate()
    assert any("variance estimate" in a for a in advisories)


def test_grid_expansion_is_deterministic_and_complete():
    text = MINIMAL + """
configuration:
  lr: [0.1, 0.01]
  batch: [16, 32]
  fixed_thing: 7
"""
    spec = ExperimentSpec.from_yaml(text)
    configs = spec.configurations
    assert len(configs) == 4
    assert all(c["fixed_thing"] == 7 for c in configs)
    assert configs == ExperimentSpec.from_yaml(text).configurations


def test_log_sweep_generates_the_declared_count():
    text = MINIMAL + """
configuration:
  lr: {sweep: log, low: 0.0001, high: 0.1, count: 4}
"""
    values = ExperimentSpec.from_yaml(text).parameters[0].values
    assert len(values) == 4
    assert values[0] == pytest.approx(1e-4) and values[-1] == pytest.approx(0.1)


def test_fixed_wrapper_keeps_a_list_as_a_value():
    text = MINIMAL + "\nconfiguration:\n  layers: {fixed: [64, 64]}\n"
    parameter = ExperimentSpec.from_yaml(text).parameters[0]
    assert parameter.swept is False and parameter.values == ([64, 64],)


def test_random_sweep_subsamples_reproducibly():
    text = MINIMAL + """
configuration:
  a: [1, 2, 3, 4]
  b: [1, 2, 3, 4]
sweep:
  strategy: random
  samples: 5
  seed: 3
"""
    spec = ExperimentSpec.from_yaml(text)
    assert len(spec.configurations) == 5
    assert spec.configurations == ExperimentSpec.from_yaml(text).configurations


def test_spec_digest_ignores_the_file_it_came_from():
    a = ExperimentSpec.from_yaml(MINIMAL, source_path="/one.yaml")
    b = ExperimentSpec.from_yaml(MINIMAL, source_path="/two.yaml")
    assert a.digest == b.digest


def test_run_count_accounts_for_arms_and_seeds():
    text = MINIMAL + """
components: [x, y]
ablations: [remove_x, remove_y]
baselines:
  - name: b1
configuration:
  lr: [0.1, 0.01]
"""
    spec = ExperimentSpec.from_yaml(text)
    # 2 treatment configs + 2 ablations (at the reference config) + 1 baseline
    assert spec.n_conditions == 5
    assert spec.n_runs == 25


# --------------------------------------------------------------------------
# graph
# --------------------------------------------------------------------------


def build_graph() -> ExperimentGraph:
    graph = ExperimentGraph("demo")
    graph.add_node("ds", "dataset", "data@v1")
    graph.add_node("cond", "condition", "arm")
    graph.add_node("run", "run", "arm seed=1")
    graph.add_node("metric", "metric", "accuracy=0.9")
    graph.add_edge("ds", "cond")
    graph.add_edge("cond", "run", "replicates")
    graph.add_edge("run", "metric", "produces")
    return graph


def test_ancestors_walk_backwards_to_the_root():
    graph = build_graph()
    kinds = [n.kind for n in graph.ancestors("metric")]
    assert kinds == ["run", "condition", "dataset"]


def test_descendants_answer_what_would_break():
    graph = build_graph()
    assert {n.id for n in graph.descendants("ds")} == {"cond", "run", "metric"}


def test_topological_order_detects_cycles():
    graph = build_graph()
    assert len(graph.topological_order()) == 4
    graph.add_edge("metric", "ds", "uses")
    with pytest.raises(GraphError, match="cycle"):
        graph.topological_order()


def test_graph_round_trips_and_renders():
    graph = build_graph()
    restored = ExperimentGraph.from_dict(graph.to_dict())
    assert restored.stats() == graph.stats()
    assert "digraph fabric" in graph.to_dot()
    assert "flowchart LR" in graph.to_mermaid()
    assert "[metric]" in graph.to_ascii("metric")


def test_adding_a_node_twice_merges_rather_than_duplicates():
    graph = build_graph()
    graph.add_node("ds", "dataset", "data@v1", digest="sha256:x")
    assert len(graph) == 4
    assert graph.node("ds").attributes["digest"] == "sha256:x"


# --------------------------------------------------------------------------
# workspace and registry
# --------------------------------------------------------------------------


def test_workspace_discovery_walks_upwards(tmp_path):
    Workspace.initialise(str(tmp_path))
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert Workspace.discover(str(nested)).root == str(tmp_path)


def test_workspace_discovery_fails_with_advice(tmp_path):
    with pytest.raises(WorkspaceError, match="fabric init"):
        Workspace.discover(str(tmp_path))


def test_registry_refuses_a_conflicting_version(tmp_path):
    workspace = Workspace.initialise(str(tmp_path))
    registry = Registry(workspace)
    registry.register_dataset(Dataset(name="d", version="v1", digest="sha256:a"))
    with pytest.raises(RegistryError) as excinfo:
        registry.register_dataset(Dataset(name="d", version="v1", digest="sha256:b"))
    message = str(excinfo.value)
    assert "differing fields" in message and "digest" in message


def test_registry_accepts_an_identical_re_registration(tmp_path):
    workspace = Workspace.initialise(str(tmp_path))
    registry = Registry(workspace)
    first = registry.register_dataset(Dataset(name="d", version="v1", digest="sha256:a"))
    second = registry.register_dataset(Dataset(name="d", version="v1", digest="sha256:a"))
    assert first.id == second.id


def test_registry_verifies_stored_records(tmp_path):
    workspace = Workspace.initialise(str(tmp_path))
    registry = Registry(workspace)
    dataset = registry.register_dataset(Dataset(name="d", version="v1", digest="sha256:a"))
    assert registry.verify("dataset", dataset.id)[0] is True

    path = os.path.join(workspace.registry_dir, "dataset", f"{dataset.id}.json")
    record = read_json(path)
    record["digest"] = "sha256:tampered"
    write_json(path, record)
    ok, message = registry.verify("dataset", dataset.id)
    assert ok is False and "modified" in message


def test_experiment_reference_resolution(tmp_path):
    workspace = Workspace.initialise(str(tmp_path))
    workspace.register_experiment({"id": "EXP-abcdef1234", "name": "demo", "created_at": 1})
    assert workspace.resolve_experiment("EXP-abcdef1234") == "EXP-abcdef1234"
    assert workspace.resolve_experiment("EXP-abc") == "EXP-abcdef1234"
    assert workspace.resolve_experiment("demo") == "EXP-abcdef1234"
    with pytest.raises(WorkspaceError, match="no experiment matches"):
        workspace.resolve_experiment("nothing")
