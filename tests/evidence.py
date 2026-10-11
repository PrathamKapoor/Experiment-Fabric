"""Shared helpers for the failure-truthfulness tests.

Reading results through ``Workspace.load_runs`` rather than parsing
``results.json`` keeps these tests honest about where evidence actually
lives: one file per run, written as each run completes.
"""

from __future__ import annotations

import json

from fabric.core.store import Workspace


def load_runs(workspace_path, experiment: str):
    # Workspace.root is the directory that *contains* .fabric.
    return Workspace(str(workspace_path)).load_runs(experiment)


def analysis(workspace_path, experiment: str) -> dict:
    path = (workspace_path / ".fabric" / "experiments" / experiment
            / "analysis.json")
    return json.loads(path.read_text(encoding="utf-8"))


def integrity_findings(workspace_path, experiment: str) -> dict[str, dict]:
    # IntegrityReport serialises findings with `code`, not `name`.
    return {f["code"]: f
            for f in analysis(workspace_path, experiment)["integrity"]["findings"]}
