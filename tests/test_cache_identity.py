"""Cache identity: what must invalidate it, and what must not.

Two rules, both of which were violated in ways that broke real workflows
rather than only theory:

1. A change to the code -- the entrypoint file, or a module it imports --
   must change the fingerprint. (D-015)
2. A change to anything that is not code must not. (D-016, D-017)

Rule 2 is not cosmetic. Fabric's workspace is where experiments *write*,
so a snapshot taken over the workspace changes every time Fabric or the
user's own scripts produce output. That re-identified the algorithm
mid-experiment and made the registry refuse to re-plan a specification it
had just successfully run.
"""

from __future__ import annotations

import os

from fabric.bundle import REPRODUCE_SH
from fabric.execution.adapters import entrypoint_digest

ENTRY = "def main(ctx):\n    ctx.record('accuracy', 0.5)\n"
IMPORTER = ("import helper\n"
            "def main(ctx):\n"
            "    ctx.record('accuracy', float(helper.X))\n")


def _digest(workspace: str, script: str = "entry.py") -> str:
    return entrypoint_digest(
        f"script:{os.path.join(workspace, script).replace(os.sep, '/')}:main",
        root=workspace,
    )


def _write(path: str | os.PathLike[str], text: str) -> None:
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def test_the_fingerprint_is_stable_when_nothing_changes(tmp_path):
    _write(str(tmp_path / "entry.py"), ENTRY)
    assert _digest(str(tmp_path)) == _digest(str(tmp_path))


def test_editing_the_entrypoint_invalidates(tmp_path):
    _write(str(tmp_path / "entry.py"), ENTRY)
    before = _digest(str(tmp_path))
    _write(str(tmp_path / "entry.py"), ENTRY + "\n# a change\n")
    assert _digest(str(tmp_path)) != before


def test_editing_an_imported_module_invalidates(tmp_path):
    _write(str(tmp_path / "entry.py"), IMPORTER)
    _write(str(tmp_path / "helper.py"), "X = 1\n")
    before = _digest(str(tmp_path))
    _write(str(tmp_path / "helper.py"), "X = 2\n")
    assert _digest(str(tmp_path)) != before


def test_generated_output_does_not_invalidate(tmp_path):
    """The workspace is where experiments write; writing is not a code change."""
    _write(str(tmp_path / "entry.py"), ENTRY)
    baseline = _digest(str(tmp_path))

    _write(str(tmp_path / ".fabric" / "experiments" / "EXP-1" / "run.json"), '{"x": 1}')
    _write(tmp_path / "results.json", '{"y": 2}')
    _write(tmp_path / "analysis.json", '{"z": 3}')
    _write(tmp_path / "report.md", "# report")
    _write(tmp_path / "report.html", "<html></html>")
    _write(tmp_path / "data.csv", "a,b\n1,2\n")
    _write(tmp_path / "jobs.sqlite3", "not really sqlite")
    _write(tmp_path / "bundle" / "certificate.json", "{}")
    _write(tmp_path / "bundle" / "manifest.json", "{}")

    assert _digest(str(tmp_path)) == baseline


def test_a_generated_reproduction_script_does_not_invalidate(tmp_path):
    """`fabric export` writes reproduce.sh; exporting must not re-identify the code.

    Reproduced: after `fabric export`, `fabric reproduce` died with

        m@v1 is already registered as alg-d6aa955189573acb, but the incoming
        algorithm has different contents

    because the bundle's `reproduce.sh` sat inside the snapshot. Its extension
    makes it look like source; only its contents identify it as Fabric's.
    """
    _write(tmp_path / "entry.py", ENTRY)
    baseline = _digest(str(tmp_path))

    _write(tmp_path / "bundle" / "reproduce.sh",
           REPRODUCE_SH.replace("{name}", "x").replace("{experiment_id}", "EXP-1"))
    _write(tmp_path / "bundle" / "README.md", "# bundle\n")
    _write(tmp_path / "bundle" / "results.json", '{"runs": []}')
    assert _digest(str(tmp_path)) == baseline


def test_a_hand_written_script_does_invalidate(tmp_path):
    """The counterpart: the user's own shell script is source and must count."""
    _write(tmp_path / "entry.py", ENTRY)
    baseline = _digest(str(tmp_path))

    _write(tmp_path / "run.sh", "#!/usr/bin/env bash\npython entry.py\n")
    after_first = _digest(str(tmp_path))
    assert after_first != baseline, "a hand-written script is source"

    _write(tmp_path / "run.sh", "#!/usr/bin/env bash\npython entry.py --fast\n")
    assert _digest(str(tmp_path)) != after_first


def test_editing_the_specification_does_not_reidentify_the_algorithm(tmp_path):
    """The spec is hashed as spec_digest; counting it twice conflicts with itself.

    Reproduced as a real failure: re-running a workspace with an amended
    seed list died with

        method@v1 is already registered as alg-f463afd391f24ba4, but the
        incoming algorithm has different contents (alg-1a84292b3ce44033)

    because the edited YAML was inside the snapshot root.
    """
    _write(str(tmp_path / "entry.py"), ENTRY)
    spec = str(tmp_path / "experiment.yaml")
    _write(spec, "seeds:\n  values: [1, 2]\n")

    before = entrypoint_digest(
        f"script:{os.path.join(str(tmp_path), 'entry.py').replace(os.sep, '/')}:main",
        root=str(tmp_path), exclude=[spec],
    )
    _write(spec, "seeds:\n  values: [1, 2, 3, 4]\n")
    after = entrypoint_digest(
        f"script:{os.path.join(str(tmp_path), 'entry.py').replace(os.sep, '/')}:main",
        root=str(tmp_path), exclude=[spec],
    )
    assert before == after


def test_the_specification_is_still_hashed_when_it_is_the_only_input(tmp_path):
    """Guard against over-correcting into 'the spec is never a source change'.

    The spec remains part of the experiment's identity through
    ``spec_digest``; this test only records that excluding it from *this*
    fingerprint did not remove it from the record.
    """
    from fabric.core.spec import ExperimentSpec

    spec = str(tmp_path / "experiment.yaml")
    _write(str(tmp_path / "entry.py"), ENTRY)
    _write(spec, ("experiment:\n  name: n\n"
                  "research_question: q\n"
                  "algorithm:\n  name: m\n  entrypoint: python:m:f\n"
                  "metrics:\n  - accuracy\n"
                  "seeds:\n  values: [1]\n"))

    before = ExperimentSpec.from_file(spec).digest
    _write(spec, open(spec, encoding="utf-8").read().replace("[1]", "[1, 2]"))
    assert ExperimentSpec.from_file(spec).digest != before