"""The benchmark registry, and the contract it enforces.

A benchmark exists so that "accuracy on X" means one thing. These tests
are mostly about the ways that guarantee can be broken: a definition that
changes underneath an experiment, an experiment that redefines a pinned
metric, a constraint that goes unchecked.
"""

from __future__ import annotations

import os
from typing import ClassVar

import pytest

from fabric.core.benchmark import (
    BenchmarkError,
    BenchmarkSpec,
    apply,
    available,
    check_constraints,
    resolve,
    search_paths,
    unsatisfied,
)
from fabric.core.spec import ExperimentSpec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFINITION = """
benchmark:
  name: tiny-task
  version: v2
  description: A benchmark used only by the tests.
  license: CC0-1.0

tasks:
  - name: main
    dataset: tiny
    metrics: [accuracy]

datasets:
  - name: tiny
    version: v3
    digest: "sha256:abc"
    n_examples: 100

metrics:
  - name: accuracy
    direction: maximize
    primary: true
    tolerance: 0.01
  - name: latency_ms
    direction: minimize
    unit: ms
    required: false
    pinned: false

output_schema:
  accuracy: float

baselines:
  - name: floor
    description: The trivial arm.
    algorithm:
      name: floor
      entrypoint: python:tests.helpers:deterministic

constraints:
  - kind: min_seeds
    value: 5
    reason: Fewer than five gives no usable variance estimate.
  - kind: require_baselines
    value: true
"""


EXPERIMENT = """
experiment:
  name: uses-a-benchmark
research_question: Does the benchmark contribute what it should?
benchmark: {reference}
algorithm:
  name: method
  entrypoint: python:tests.helpers:ablatable
components: [alpha]
seeds:
  values: [1, 2, 3, 4, 5, 6]
execution:
  isolation: inline
"""


@pytest.fixture()
def definition(tmp_path):
    directory = tmp_path / "benchmarks"
    directory.mkdir()
    path = directory / "tiny-task.yaml"
    path.write_text(DEFINITION, encoding="utf-8")
    return path


@pytest.fixture()
def loaded(definition):
    return BenchmarkSpec.from_file(str(definition))


# --------------------------------------------------------------------------
# identity
# --------------------------------------------------------------------------


def test_a_benchmark_has_a_content_addressed_identity(loaded):
    assert loaded.ref == "tiny-task@v2"
    assert loaded.digest.startswith("sha256:")
    again = BenchmarkSpec.from_dict(
        {"benchmark": {"name": "tiny-task", "version": "v2"},
         "metrics": [{"name": "accuracy", "direction": "maximize",
                      "primary": True, "tolerance": 0.01},
                     {"name": "latency_ms", "direction": "minimize",
                      "unit": "ms", "required": False, "pinned": False}],
         "datasets": [{"name": "tiny", "version": "v3", "digest": "sha256:abc",
                       "n_examples": 100}],
         "tasks": [{"name": "main", "dataset": "tiny", "metrics": ["accuracy"]}],
         "output_schema": {"accuracy": "float"},
         "baselines": [{"name": "floor", "description": "The trivial arm.",
                        "algorithm": {"name": "floor",
                                      "entrypoint": "python:tests.helpers:deterministic"}}],
         "constraints": [{"kind": "min_seeds", "value": 5,
                          "reason": "Fewer than five gives no usable variance estimate."},
                         {"kind": "require_baselines", "value": True}]},
    )
    assert again.digest == loaded.digest


def test_the_digest_moves_when_the_contract_moves(definition, loaded):
    changed = definition.read_text(encoding="utf-8").replace(
        "tolerance: 0.01", "tolerance: 0.05")
    definition.write_text(changed, encoding="utf-8")
    assert BenchmarkSpec.from_file(str(definition)).digest != loaded.digest


def test_the_digest_ignores_prose(definition, loaded):
    """A clearer description is not a different benchmark."""
    changed = definition.read_text(encoding="utf-8").replace(
        "A benchmark used only by the tests.", "Rewritten for clarity.")
    definition.write_text(changed, encoding="utf-8")
    assert BenchmarkSpec.from_file(str(definition)).digest == loaded.digest


def test_a_pinned_digest_that_no_longer_matches_is_refused(tmp_path, definition, loaded):
    stale = "sha256:" + "0" * 64
    with pytest.raises(BenchmarkError, match="has changed since"):
        resolve({"name": "tiny-task", "version": "v2", "digest": stale},
                root=str(tmp_path))
    # the matching pin is accepted
    assert resolve({"name": "tiny-task", "version": "v2", "digest": loaded.digest},
                   root=str(tmp_path)).ref == "tiny-task@v2"


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def test_a_benchmark_without_metrics_is_refused():
    with pytest.raises(BenchmarkError, match="at least one metric"):
        BenchmarkSpec.from_dict({"benchmark": {"name": "empty"}})


def test_two_primary_metrics_are_refused():
    with pytest.raises(BenchmarkError, match="only one metric may be primary"):
        BenchmarkSpec.from_dict({
            "benchmark": {"name": "b"},
            "metrics": [{"name": "a", "primary": True},
                        {"name": "b", "primary": True}],
        })


def test_a_task_naming_an_undefined_dataset_is_refused():
    with pytest.raises(BenchmarkError, match="does not define"):
        BenchmarkSpec.from_dict({
            "benchmark": {"name": "b"},
            "metrics": [{"name": "accuracy"}],
            "tasks": [{"name": "t", "dataset": "nowhere"}],
        })


def test_a_task_naming_an_undefined_metric_is_refused():
    with pytest.raises(BenchmarkError, match="does not define"):
        BenchmarkSpec.from_dict({
            "benchmark": {"name": "b"},
            "metrics": [{"name": "accuracy"}],
            "tasks": [{"name": "t", "metrics": ["bleu"]}],
        })


def test_a_future_schema_version_is_refused_rather_than_guessed():
    """Section 58: old data must not be silently reinterpreted."""
    with pytest.raises(BenchmarkError, match="schema version"):
        BenchmarkSpec.from_dict({
            "benchmark": {"name": "b", "schema_version": "9.0"},
            "metrics": [{"name": "accuracy"}],
        })


def test_an_unknown_constraint_kind_is_refused_with_the_known_ones():
    with pytest.raises(BenchmarkError) as raised:
        BenchmarkSpec.from_dict({
            "benchmark": {"name": "b"},
            "metrics": [{"name": "accuracy"}],
            "constraints": [{"kind": "must_be_good"}],
        })
    assert "min_seeds" in str(raised.value)


# --------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------


def test_a_benchmark_is_found_by_reference(tmp_path, definition):
    found = resolve("tiny-task@v2", root=str(tmp_path))
    assert found.ref == "tiny-task@v2"
    assert found.source_path == str(definition)


def test_a_benchmark_is_found_without_a_version(tmp_path, definition):
    assert resolve("tiny-task", root=str(tmp_path)).version == "v2"


def test_a_missing_benchmark_names_where_it_looked(tmp_path, definition):
    with pytest.raises(BenchmarkError) as raised:
        resolve("nonexistent@v1", root=str(tmp_path))
    message = str(raised.value)
    assert "Searched:" in message
    assert "tiny-task@v2" in message          # what it did find
    assert "FABRIC_BENCHMARK_PATH" in message  # how to point it elsewhere


def test_an_explicit_path_resolves(tmp_path, definition):
    assert resolve(str(definition), root=str(tmp_path)).ref == "tiny-task@v2"


def test_the_environment_variable_extends_the_search_path(tmp_path, definition,
                                                          monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    monkeypatch.setenv("FABRIC_BENCHMARK_PATH", str(definition.parent))
    assert any(str(definition.parent) == path
               for path in search_paths(root=str(elsewhere)))
    assert resolve("tiny-task", root=str(elsewhere)).ref == "tiny-task@v2"


def test_a_malformed_definition_does_not_break_the_listing(tmp_path, definition):
    """One unparseable file must not hide the definitions that are readable.

    The listing spans every directory on the search path, which includes the
    definitions that ship inside the package. So the assertion is that the
    local one is present and that nothing raised -- not that it is the only
    one, which would be false the moment the shipped definitions become
    resolvable.
    """
    (definition.parent / "broken.yaml").write_text("not: [a, benchmark\n",
                                                   encoding="utf-8")
    refs = [b.ref for b in available(root=str(tmp_path))]
    assert "tiny-task@v2" in refs
    assert "broken" not in " ".join(refs)


def test_the_definitions_that_ship_with_fabric_are_on_the_search_path():
    """The last search-path entry has to resolve to a real directory.

    It is the entry that makes a benchmark resolvable from an installed
    wheel rather than only from a checkout, so a wrong join here is silent:
    the search path still returns five entries, one of which is empty.
    """
    from fabric.core import benchmark as module

    paths = search_paths()
    packaged = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(module.__file__))), "benchmarks")
    assert os.path.isdir(packaged), f"expected shipped definitions at {packaged}"
    assert packaged in paths


def test_a_shipped_definition_resolves_from_an_installed_package(tmp_path):
    """Resolution must not depend on the working directory containing a copy."""
    assert resolve("sentiment-small", root=str(tmp_path)).ref == "sentiment-small@v1"
    assert resolve("spiral", root=str(tmp_path)).ref == "spiral@v1"
    assert resolve("agent-qa", root=str(tmp_path)).ref == "agent-qa@v1"


def test_an_unversioned_reference_resolves_to_the_newest_version(tmp_path):
    """`v10` is later than `v2`; a lexicographic sort serves the wrong contract."""
    directory = tmp_path / "benchmarks"
    directory.mkdir()
    for version in ("v2", "v10", "v1"):
        (directory / f"ordered-{version}.yaml").write_text(
            f"benchmark:\n  name: ordered\n  version: {version}\n"
            "metrics:\n  - name: accuracy\n    direction: maximize\n",
            encoding="utf-8")

    assert resolve("ordered", root=str(tmp_path)).version == "v10"

    listing = [b.version for b in available(root=str(tmp_path))
               if b.name == "ordered"]
    assert listing == ["v1", "v2", "v10"], listing


def test_an_explicitly_versioned_reference_still_wins_over_a_newer_one(tmp_path):
    directory = tmp_path / "benchmarks"
    directory.mkdir()
    for version in ("v2", "v10"):
        (directory / f"pinned-{version}.yaml").write_text(
            f"benchmark:\n  name: pinned\n  version: {version}\n"
            "metrics:\n  - name: accuracy\n    direction: maximize\n",
            encoding="utf-8")

    assert resolve("pinned@v2", root=str(tmp_path)).version == "v2"


def test_a_non_numeric_version_is_still_orderable(tmp_path):
    directory = tmp_path / "benchmarks"
    directory.mkdir()
    for version in ("draft-b", "draft-a", "v3"):
        (directory / f"mixed-{version}.yaml").write_text(
            f"benchmark:\n  name: mixed\n  version: {version}\n"
            "metrics:\n  - name: accuracy\n    direction: maximize\n",
            encoding="utf-8")

    listing = [b.version for b in available(root=str(tmp_path)) if b.name == "mixed"]
    assert listing == ["draft-a", "draft-b", "v3"], listing
    assert resolve("mixed", root=str(tmp_path)).version == "v3"


# --------------------------------------------------------------------------
# what the benchmark contributes
# --------------------------------------------------------------------------


def _spec(tmp_path, reference="tiny-task@v2", extra=""):
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        return ExperimentSpec.from_yaml(
            EXPERIMENT.format(reference=reference) + extra)
    finally:
        os.chdir(cwd)


def test_the_benchmark_supplies_metrics_dataset_and_baselines(tmp_path, definition):
    spec = _spec(tmp_path)
    assert [m.name for m in spec.metrics] == ["accuracy", "latency_ms"]
    assert spec.primary_metric == "accuracy"
    assert spec.metric("accuracy").tolerance == 0.01
    assert [b.name for b in spec.baselines] == ["floor"]
    assert spec.dataset is not None
    assert spec.dataset.name == "tiny"
    assert spec.dataset.inline["digest"] == "sha256:abc"
    assert any("tiny-task@v2" in note for note in spec.benchmark_notes)


def test_the_benchmark_digest_is_part_of_the_experiment_identity(tmp_path, definition):
    before = _spec(tmp_path).digest
    definition.write_text(
        definition.read_text(encoding="utf-8").replace("tolerance: 0.01",
                                                       "tolerance: 0.02"),
        encoding="utf-8",
    )
    after = _spec(tmp_path).digest
    assert before != after, (
        "a benchmark change must make the experiment a different experiment"
    )


def test_the_experiment_may_add_its_own_metrics(tmp_path, definition):
    spec = _spec(tmp_path, extra="metrics:\n  - name: extra\n    direction: maximize\n")
    names = [m.name for m in spec.metrics]
    assert names == ["accuracy", "latency_ms", "extra"]


def test_the_experiment_may_add_its_own_baselines(tmp_path, definition):
    spec = _spec(tmp_path, extra=(
        "baselines:\n"
        "  - name: mine\n"
        "    algorithm: {name: mine, entrypoint: 'python:tests.helpers:deterministic'}\n"
    ))
    assert [b.name for b in spec.baselines] == ["floor", "mine"]


def test_the_experiment_may_override_a_baseline_of_the_same_name(tmp_path, definition):
    spec = _spec(tmp_path, extra=(
        "baselines:\n"
        "  - name: floor\n"
        "    description: my own version\n"
        "    algorithm: {name: floor2, entrypoint: 'python:tests.helpers:noop'}\n"
    ))
    assert [b.name for b in spec.baselines] == ["floor"]
    assert spec.baselines[0].description == "my own version"


def test_redefining_a_pinned_metric_is_refused_with_a_reason(tmp_path, definition):
    with pytest.raises(Exception) as raised:
        _spec(tmp_path, extra=(
            "metrics:\n  - name: accuracy\n    direction: maximize\n"
            "    tolerance: 0.5\n"
        ))
    message = str(raised.value)
    assert "pins accuracy.tolerance" in message
    assert "not measuring the same thing" in message


def test_an_unpinned_metric_may_be_redefined(tmp_path, definition):
    spec = _spec(tmp_path, extra=(
        "metrics:\n  - name: latency_ms\n    direction: minimize\n"
        "    tolerance: 20.0\n"
    ))
    assert spec.metric("latency_ms").tolerance == 20.0


def test_the_experiments_own_dataset_wins(tmp_path, definition):
    spec = _spec(tmp_path, extra=(
        "dataset:\n  name: mine\n  version: v9\n  digest: 'sha256:def'\n"
    ))
    assert spec.dataset is not None
    assert spec.dataset.name == "mine"


def test_applying_a_benchmark_does_not_mutate_the_input(loaded):
    raw = {"metrics": [{"name": "accuracy"}], "baselines": []}
    snapshot = {"metrics": [{"name": "accuracy"}], "baselines": []}
    apply(loaded, raw)
    assert raw == snapshot


# --------------------------------------------------------------------------
# constraints
# --------------------------------------------------------------------------


def test_a_met_constraint_is_reported_as_met(tmp_path, definition):
    spec = _spec(tmp_path)
    results = {r.kind: r for r in spec.benchmark_constraints}
    assert results["min_seeds"].satisfied is True
    assert results["require_baselines"].satisfied is True
    assert results["benchmark_metrics"].satisfied is True


def test_an_unmet_constraint_becomes_an_advisory_not_an_error(tmp_path, definition):
    spec = _spec(tmp_path, extra="seeds:\n  values: [1, 2]\n")
    unmet = unsatisfied(spec.benchmark_constraints)
    assert [r.kind for r in unmet] == ["min_seeds"]
    assert any("min_seeds" in advisory for advisory in spec.validate())
    # and it is advisory: the specification still parsed and is runnable
    assert len(spec.seeds.values) == 2


def test_a_required_metric_the_experiment_dropped_is_caught(loaded, tmp_path):
    """The benchmark's own metric list is a constraint even without one."""
    spec = _spec(tmp_path)

    class _Trimmed:
        metrics: ClassVar[list] = [m for m in spec.metrics if m.name != "accuracy"]
        seeds = spec.seeds
        analysis = spec.analysis
        baselines = spec.baselines
        dataset = spec.dataset
        reproducibility = spec.reproducibility
        execution = spec.execution

    results = {r.kind: r for r in check_constraints(loaded, _Trimmed())}
    assert results["benchmark_metrics"].satisfied is False
    assert "accuracy" in results["benchmark_metrics"].detail


@pytest.mark.parametrize("kind,extra", [
    ("max_alpha", "analysis:\n  alpha: 0.2\n"),
    ("require_determinism_check", ""),
])
def test_each_constraint_kind_evaluates(tmp_path, definition, kind, extra):
    definition.write_text(
        definition.read_text(encoding="utf-8")
        + f"  - kind: {kind}\n    value: "
        + ("0.05\n" if kind == "max_alpha" else "true\n"),
        encoding="utf-8",
    )
    spec = _spec(tmp_path, extra=extra)
    results = {r.kind: r for r in spec.benchmark_constraints}
    assert kind in results
    assert results[kind].satisfied is False


def test_require_dataset_digest_passes_when_the_data_is_identified(tmp_path, definition):
    definition.write_text(
        definition.read_text(encoding="utf-8")
        + "  - kind: require_dataset_digest\n    value: true\n",
        encoding="utf-8",
    )
    results = {r.kind: r for r in _spec(tmp_path).benchmark_constraints}
    assert results["require_dataset_digest"].satisfied is True


# --------------------------------------------------------------------------
# the definitions that ship with the project
# --------------------------------------------------------------------------


def shipped() -> list[str]:
    directory = os.path.join(ROOT, "benchmarks")
    if not os.path.isdir(directory):
        return []
    return sorted(os.path.join(directory, name) for name in os.listdir(directory)
                  if name.endswith((".yaml", ".yml")))


@pytest.mark.parametrize("path", shipped(),
                         ids=lambda p: os.path.basename(p))
def test_every_shipped_benchmark_is_valid(path):
    benchmark = BenchmarkSpec.from_file(path)
    assert benchmark.metrics
    assert benchmark.digest.startswith("sha256:")
    assert benchmark.description.strip(), "a benchmark must say what it is"
    # every shipped definition is synthetic; none may look like a citation
    assert benchmark.citation.strip(), "a benchmark must say where it came from"


@pytest.mark.parametrize("path", [
    os.path.join(ROOT, "examples", name, "experiment.yaml")
    for name in ("agent_pipeline", "text_classification", "torch_mlp")
], ids=["agent_pipeline", "text_classification", "torch_mlp"])
def test_every_shipped_example_conforms_to_the_benchmark_it_claims(path):
    """Not that every constraint passes -- that none of them errors.

    One example deliberately fails a constraint, which is the behaviour
    worth having: the advisory appears, and the experiment still runs.
    """
    spec = ExperimentSpec.from_file(path)
    assert spec.benchmark is not None, f"{path} should claim a benchmark"
    results = spec.benchmark_constraints
    assert results, "a claimed benchmark with no constraints checked is suspicious"
    for result in results:
        assert isinstance(result.satisfied, bool)
        assert result.detail
