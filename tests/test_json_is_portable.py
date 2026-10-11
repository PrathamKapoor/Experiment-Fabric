"""Everything Fabric writes must be JSON that other languages can read.

Python's ``json`` module writes bare ``NaN``, ``Infinity`` and
``-Infinity`` by default, and reads them back, so a round trip inside
Python succeeds and hides the problem completely. Nothing else accepts
them: they are not in the JSON grammar, and ``JSON.parse``, ``jq``, Go and
Rust reject the *whole document* over one of them.

That matters here more than it would in most projects. Fabric's analysis
is full of legitimately non-finite numbers -- a p-value that could not be
computed, an effect size that is unbounded because a condition had zero
variance -- and its whole claim is that the evidence it writes outlives
the tool that wrote it. A ``results.json`` only Python can read does not
outlive anything.

The bug these tests were written for: the API's analysis endpoint returned
``"value": NaN``, and every browser refused to parse the response.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import pytest

from fabric.analysis.engine import analyse
from fabric.core.spec import ExperimentSpec
from fabric.core.store import Workspace
from fabric.execution.planner import plan_experiment
from fabric.execution.runner import ExecutionOptions, Runner
from fabric.util.jsonio import dumps, read_json, sanitise, write_json

# A specification built to produce non-finite numbers. `deterministic`
# returns the same value for a given seed, so the control arm has zero
# variance -- which makes a standardised effect size against it unbounded
# and leaves several tests with no computable statistic.
DEGENERATE = """
experiment:
  name: degenerate
research_question: Does the analysis survive a zero-variance condition?
algorithm:
  name: method
  entrypoint: python:tests.helpers:deterministic
metrics:
  - name: accuracy
    direction: maximize
    primary: true
    tolerance: 0.01
baselines:
  - name: identical
    algorithm:
      name: identical
      entrypoint: python:tests.helpers:deterministic
seeds:
  values: [1, 2, 3, 4, 5]
execution:
  isolation: inline
"""


def strict(text: str):
    """Parse with a parser that refuses the Python extensions."""
    return json.loads(text, parse_constant=_reject)


def _reject(name: str):
    raise AssertionError(
        f"the document contains the bare literal {name!r}, which is not JSON; "
        "no parser outside Python will accept this file"
    )


# --------------------------------------------------------------------------
# the serialiser
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_number_becomes_null(value):
    assert strict(dumps({"x": value})) == {"x": None}


def test_finite_numbers_are_untouched():
    payload = {"a": 0.1, "b": -2.5e-8, "c": 0, "d": 1e300, "e": True, "f": None}
    assert strict(dumps(payload)) == payload


def test_non_finite_numbers_are_found_however_deeply_they_are_buried():
    payload = {
        "list": [1.0, float("nan"), {"deep": [float("inf")]}],
        "tuple": (float("-inf"), 2.0),
        "map": {"k": {"k2": float("nan")}},
    }
    parsed = strict(dumps(payload))
    assert parsed["list"] == [1.0, None, {"deep": [None]}]
    assert parsed["tuple"] == [None, 2.0]
    assert parsed["map"]["k"]["k2"] is None


def test_entities_are_unwrapped_through_to_dict():
    class _Thing:
        def to_dict(self):
            return {"value": float("nan"), "name": "thing"}

    assert strict(dumps({"t": _Thing()})) == {"t": {"value": None, "name": "thing"}}


def test_sets_become_sorted_lists():
    assert sanitise({3, 1, 2}) == [1, 2, 3]


def test_a_non_finite_value_can_never_escape_through_dumps():
    """``allow_nan=False`` is the backstop if the walk ever misses one."""
    class _Sneaky:
        def to_dict(self):
            return {"x": float("nan")}

    # sanitise recurses into the result of to_dict, so even this is caught
    assert "NaN" not in dumps({"s": _Sneaky()})


def test_written_files_are_strictly_valid(tmp_path):
    path = tmp_path / "out.json"
    write_json(path, {"p": float("nan"), "g": float("inf"), "ok": 1.25})
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    assert strict(text) == {"p": None, "g": None, "ok": 1.25}
    assert read_json(path) == {"p": None, "g": None, "ok": 1.25}


# --------------------------------------------------------------------------
# a real analysis, which is where the non-finite numbers actually come from
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def degenerate(tmp_path_factory):
    root = tmp_path_factory.mktemp("degenerate")
    cwd = os.getcwd()
    os.chdir(root)
    try:
        workspace = Workspace.initialise(str(root))
        spec = ExperimentSpec.from_yaml(DEGENERATE)
        plan = plan_experiment(spec, workspace, root=str(root))
        workspace.create_experiment_dir(plan.experiment.id)
        workspace.write_text(plan.experiment.id, "experiment.yaml", DEGENERATE)
        workspace.write(plan.experiment.id, "plan.json", plan.to_dict())
        workspace.register_experiment({
            "id": plan.experiment.id,
            "name": plan.experiment.name,
            "research_question": plan.experiment.research_question,
            "spec_digest": spec.digest,
            "created_at": plan.experiment.created_at,
            "runs": len(plan.runs),
        })
        Runner(workspace, plan, ExecutionOptions(isolation="inline"),
               root=str(root)).execute()
        runs = workspace.load_runs(plan.experiment.id)
        yield workspace, plan, runs, analyse(plan, runs)
    finally:
        os.chdir(cwd)


def test_the_fixture_really_does_produce_non_finite_numbers(degenerate):
    """Otherwise the tests below prove nothing."""
    _workspace, _plan, _runs, analysis = degenerate
    found = _find_non_finite(analysis.to_dict())
    assert found, (
        "this fixture is supposed to produce NaN or infinity somewhere in the "
        "analysis; if it no longer does, the tests below are vacuous"
    )


def _find_non_finite(value, path="") -> list[str]:
    if isinstance(value, float) and (value != value or math.isinf(value)):
        return [path or "<root>"]
    if isinstance(value, dict):
        return [p for k, v in value.items()
                for p in _find_non_finite(v, f"{path}.{k}")]
    if isinstance(value, (list, tuple)):
        return [p for i, v in enumerate(value)
                for p in _find_non_finite(v, f"{path}[{i}]")]
    return []


def test_the_analysis_serialises_to_strictly_valid_json(degenerate):
    _workspace, _plan, _runs, analysis = degenerate
    parsed = strict(dumps(analysis.to_dict()))
    assert parsed["headline"]


def test_every_file_an_experiment_writes_is_strictly_valid(degenerate):
    workspace, plan, _runs, _analysis = degenerate
    directory = workspace.experiment_dir(plan.experiment.id)
    checked = 0
    for folder, _subdirs, files in os.walk(directory):
        for name in files:
            if not name.endswith(".json"):
                continue
            path = os.path.join(folder, name)
            with open(path, encoding="utf-8") as handle:
                strict(handle.read())
            checked += 1
    assert checked > 0


def test_an_exported_bundle_is_strictly_valid(degenerate, tmp_path):
    from fabric import bundle as bundle_module
    from fabric.core.provenance import build_provenance
    from fabric.core.registry import Registry
    from fabric.reporting import builder, markdown
    from fabric.reporting.certificate import issue

    workspace, plan, runs, analysis = degenerate
    cwd = os.getcwd()
    os.chdir(workspace.root)
    try:
        graph = build_provenance(plan, runs, analysis)
        registry = Registry(workspace)
        certificate = issue(analysis, plan, runs, registry,
                            workspace_root=workspace.root)
        document = builder.build_report(analysis, plan, runs, graph=graph)
        destination = str(tmp_path / "bundle")
        bundle_module.export(
            workspace, plan, runs, destination,
            analysis=analysis, provenance=graph, certificate=certificate,
            report_markdown=markdown.render(document),
        )
    finally:
        os.chdir(cwd)

    checked = 0
    for folder, _subdirs, files in os.walk(destination):
        for name in files:
            if name.endswith(".json"):
                with open(os.path.join(folder, name), encoding="utf-8") as handle:
                    strict(handle.read())
                checked += 1
    assert checked >= 4


# --------------------------------------------------------------------------
# the command line
# --------------------------------------------------------------------------


def test_the_cli_json_output_is_strictly_valid(degenerate):
    """A pipeline downstream of ``--json`` is usually not Python."""
    workspace, plan, _runs, _analysis = degenerate
    result = subprocess.run(
        [sys.executable, "-m", "fabric", "analyze", plan.experiment.id, "--json"],
        cwd=workspace.root, capture_output=True, text=True, timeout=300,
        env={**os.environ, "PYTHONPATH": os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))},
    )
    assert result.returncode == 0, result.stderr
    parsed = strict(result.stdout)
    assert parsed["experiment_id"] == plan.experiment.id


# --------------------------------------------------------------------------
# the shipped evidence
# --------------------------------------------------------------------------


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def shipped_json() -> list[str]:
    found: list[str] = []
    for directory in ("paper", "benchmarks", "examples"):
        base = os.path.join(ROOT, directory)
        for folder, _subdirs, files in os.walk(base):
            found += [os.path.join(folder, name) for name in files
                      if name.endswith(".json")]
    return sorted(found)


@pytest.mark.parametrize("path", shipped_json(),
                         ids=lambda p: os.path.relpath(p, ROOT).replace(os.sep, "/"))
def test_every_json_file_in_the_repository_is_strictly_valid(path):
    """Including the pilot study's bundles, which are the shipped evidence."""
    with open(path, encoding="utf-8") as handle:
        strict(handle.read())
