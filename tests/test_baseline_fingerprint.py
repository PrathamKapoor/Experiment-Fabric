"""Every algorithm a plan resolves must fingerprint identically across a lifecycle.

`plan_experiment` fingerprints the treatment algorithm with the specification
excluded, because the spec is hashed separately as `spec_digest` and counting
it twice made the algorithm's identity depend on which copy of the YAML was
used. The *baseline* and *implementation-perturbation* arms were initially
missed: they went on including it.

The failure is silent until it is loud. A specification whose baseline shares
the treatment's entrypoint ran fine, exported fine, and then

    fabric reproduce <bundle>
    error b@v1 is already registered as alg-5aa36ae83f40343e, but the
    incoming algorithm has different contents (alg-9b4ed47eb43cdc29)

exited 2, because by then the bundle's copy of the spec sat inside the
snapshot root and the baseline's fingerprint changed while the treatment's
did not.

Reproduced on Windows against a wheel installed into a clean virtualenv, from
a workspace outside the source checkout.
"""

from __future__ import annotations

import os

from fabric.core.entities import Algorithm
from fabric.core.registry import Registry
from fabric.core.spec import ExperimentSpec
from fabric.core.store import Workspace
from fabric.execution.planner import plan_experiment

SPEC = """
experiment:
  name: baseline-fingerprint
  research_question: Do all arms fingerprint the same way?
algorithm:
  name: m
  entrypoint: script:HERE/e.py:main
metrics:
  - name: accuracy
    direction: maximize
    primary: true
seeds:
  values: [1, 2]
baselines:
  - name: control
    algorithm:
      name: b
      entrypoint: script:HERE/e.py:main
ablations: [remove_alpha]
reproducibility:
  enabled: true
  perturbations:
    implementation:
      values:
        - name: reordered
          entrypoint: script:HERE/e2.py:main
execution:
  isolation: inline
"""


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _every_algorithm(plan, workspace: Workspace) -> dict[str, Algorithm]:
    """Every distinct algorithm the plan's runs depend on, by reference."""
    registry = Registry(workspace)
    found: dict[str, Algorithm] = {}
    for planned in plan.runs:
        if not planned.algorithm_id:
            continue
        entity = registry.get("algorithm", planned.algorithm_id)
        found[entity.ref] = entity
    return found


def test_every_resolved_algorithm_excludes_the_specification(tmp_path):
    """Editing the spec must not change any arm's fingerprint."""
    here = str(tmp_path).replace("\\", "/")
    _write(str(tmp_path / "e.py"), "def main(ctx):\n    ctx.record('accuracy', 0.5)\n")
    _write(str(tmp_path / "e2.py"), "def main(ctx):\n    ctx.record('accuracy', 0.4)\n")

    workspace = Workspace.initialise(str(tmp_path / "ws"))
    spec_path = str(tmp_path / "e.yaml")
    _write(spec_path, SPEC.replace("HERE", here))

    first = plan_experiment(ExperimentSpec.from_file(spec_path),
                            workspace, root=workspace.root)
    before = {ref: a.source_digest for ref, a in _every_algorithm(first, workspace).items()}
    assert len(before) >= 2, (
        "the specification must resolve more than one algorithm, or this test "
        "cannot see the bug"
    )

    # The spec's bytes are already part of the experiment id via spec_digest,
    # so re-identifying any algorithm because the spec changed double-counts
    # the same edit under two different names.
    _write(spec_path, SPEC.replace("HERE", here).replace("values: [1, 2]",
                                                     "values: [1, 2, 3]"))
    second = plan_experiment(ExperimentSpec.from_file(spec_path),
                             workspace, root=workspace.root)
    after = {ref: a.source_digest for ref, a in _every_algorithm(second, workspace).items()}

    assert set(before) == set(after)
    for ref in before:
        assert before[ref] == after[ref], (
            f"{ref}'s fingerprint changed when the specification was edited. "
            f"Every arm must exclude the specification, not just the treatment."
        )


def test_replanning_an_older_specification_still_registers(tmp_path):
    """The registry must accept a re-plan of a spec it has already planned."""
    here = str(tmp_path).replace("\\", "/")
    _write(str(tmp_path / "e.py"), "def main(ctx):\n    ctx.record('accuracy', 0.5)\n")
    _write(str(tmp_path / "e2.py"), "def main(ctx):\n    ctx.record('accuracy', 0.4)\n")
    workspace = Workspace.initialise(str(tmp_path / "ws"))
    spec_path = str(tmp_path / "e.yaml")
    _write(spec_path, SPEC.replace("HERE", here))

    first = plan_experiment(ExperimentSpec.from_file(spec_path),
                            workspace, root=workspace.root)
    # Re-plan the original spec again. If any arm's fingerprint drifted with
    # the file's mtime or with a file written since, this raises RegistryError.
    again = plan_experiment(ExperimentSpec.from_file(spec_path),
                            workspace, root=workspace.root)

    assert {r.ref: r.source_digest for r in _every_algorithm(first, workspace).values()} == \
           {r.ref: r.source_digest for r in _every_algorithm(again, workspace).values()}
    assert again.experiment.id == first.experiment.id


