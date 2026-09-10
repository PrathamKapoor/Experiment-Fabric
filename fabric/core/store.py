"""The workspace: Fabric's on-disk layout.

Everything Fabric knows lives in a ``.fabric`` directory beside the
project, in plain JSON.  That is a deliberate constraint: a results
database you cannot read without the tool that wrote it is a liability
for a scientific record that has to outlive the tool.

::

    .fabric/
      config.json                  workspace settings
      registry/<kind>/<id>.json    content-addressed entities
      registry/index.json          name@version -> id
      experiments/index.json       every experiment ever planned
      experiments/<EXP>/           one experiment
        spec.yaml  spec.json       the claim, as declared
        plan.json  graph.json      what will run, and how it relates
        runs/<RUN>/run.json        one execution
        runs/<RUN>/artifacts/      immutable outputs
        analysis.json              statistics
        provenance.json            the full evidence graph
        reports/                   markdown / html / latex / json
        certificate.json           the signed summary
      cache/index.json             cache key -> completed run
"""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from fabric.core.entities import Run
from fabric.util.jsonio import atomic_write_text, read_json, write_json
from fabric.util.locking import FileLock

WORKSPACE_DIRNAME = ".fabric"
LAYOUT_VERSION = 1


class WorkspaceError(RuntimeError):
    """Raised when a workspace is missing or unusable."""


@dataclass(frozen=True)
class Workspace:
    """A handle on a Fabric workspace."""

    root: str
    """The project directory containing ``.fabric``."""

    # -- lifecycle ---------------------------------------------------------

    @property
    def path(self) -> str:
        return os.path.join(self.root, WORKSPACE_DIRNAME)

    @property
    def config_path(self) -> str:
        return os.path.join(self.path, "config.json")

    @property
    def registry_dir(self) -> str:
        return os.path.join(self.path, "registry")

    @property
    def experiments_dir(self) -> str:
        return os.path.join(self.path, "experiments")

    @property
    def cache_dir(self) -> str:
        return os.path.join(self.path, "cache")

    @property
    def exists(self) -> bool:
        return os.path.isdir(self.path)

    @staticmethod
    def initialise(root: str, *, name: str = "", force: bool = False) -> Workspace:
        """Create a workspace at ``root``."""
        workspace = Workspace(os.path.abspath(root))
        if workspace.exists and not force:
            raise WorkspaceError(
                f"a workspace already exists at {workspace.path}; pass force to reinitialise"
            )
        for directory in (workspace.path, workspace.registry_dir,
                          workspace.experiments_dir, workspace.cache_dir):
            os.makedirs(directory, exist_ok=True)
        write_json(workspace.config_path, {
            "layout_version": LAYOUT_VERSION,
            "name": name or os.path.basename(workspace.root),
            "created_at": time.time(),
        })
        if not os.path.exists(workspace.experiment_index_path):
            write_json(workspace.experiment_index_path, {"experiments": []})
        if not os.path.exists(os.path.join(workspace.registry_dir, "index.json")):
            write_json(os.path.join(workspace.registry_dir, "index.json"), {})
        if not os.path.exists(workspace.cache_index_path):
            write_json(workspace.cache_index_path, {})
        return workspace

    @staticmethod
    def discover(start: str | None = None) -> Workspace:
        """Find the nearest workspace at or above ``start``."""
        current = os.path.abspath(start or os.getcwd())
        while True:
            if os.path.isdir(os.path.join(current, WORKSPACE_DIRNAME)):
                return Workspace(current)
            parent = os.path.dirname(current)
            if parent == current:
                raise WorkspaceError(
                    "no Fabric workspace found in this directory or any parent. "
                    "Run 'fabric init' first."
                )
            current = parent

    @property
    def config(self) -> dict[str, Any]:
        return read_json(self.config_path, {}) or {}

    # -- experiments -------------------------------------------------------

    @property
    def experiment_index_path(self) -> str:
        return os.path.join(self.experiments_dir, "index.json")

    def experiment_dir(self, experiment_id: str) -> str:
        return os.path.join(self.experiments_dir, experiment_id)

    def runs_dir(self, experiment_id: str) -> str:
        return os.path.join(self.experiment_dir(experiment_id), "runs")

    def run_dir(self, experiment_id: str, run_id: str) -> str:
        return os.path.join(self.runs_dir(experiment_id), run_id)

    def artifacts_dir(self, experiment_id: str, run_id: str) -> str:
        return os.path.join(self.run_dir(experiment_id, run_id), "artifacts")

    def reports_dir(self, experiment_id: str) -> str:
        return os.path.join(self.experiment_dir(experiment_id), "reports")

    def create_experiment_dir(self, experiment_id: str) -> str:
        directory = self.experiment_dir(experiment_id)
        os.makedirs(os.path.join(directory, "runs"), exist_ok=True)
        os.makedirs(os.path.join(directory, "reports"), exist_ok=True)
        return directory

    def register_experiment(self, record: dict[str, Any]) -> None:
        """Add or update an entry in the experiment index."""
        with FileLock(self.experiment_index_path):
            index = read_json(self.experiment_index_path, {"experiments": []}) or {"experiments": []}
            entries = [e for e in index.get("experiments", []) if e.get("id") != record.get("id")]
            entries.append(record)
            entries.sort(key=lambda e: e.get("created_at", 0), reverse=True)
            write_json(self.experiment_index_path, {"experiments": entries})

    def list_experiments(self) -> list[dict[str, Any]]:
        index = read_json(self.experiment_index_path, {"experiments": []}) or {}
        return list(index.get("experiments", []))

    def resolve_experiment(self, reference: str) -> str:
        """Resolve an id, id prefix, or experiment name to an experiment id.

        Names are ambiguous by nature -- the same spec can be run many
        times -- so a name resolves to the most recent match and Fabric
        says which one it picked.
        """
        experiments = self.list_experiments()
        for entry in experiments:
            if entry.get("id") == reference:
                return str(entry["id"])
        prefix_matches = [e for e in experiments if str(e.get("id", "")).startswith(reference)]
        if len(prefix_matches) == 1:
            return str(prefix_matches[0]["id"])
        if len(prefix_matches) > 1:
            raise WorkspaceError(
                f"'{reference}' matches {len(prefix_matches)} experiments: "
                + ", ".join(str(e["id"]) for e in prefix_matches[:5])
            )
        name_matches = [e for e in experiments if e.get("name") == reference]
        if name_matches:
            return str(name_matches[0]["id"])
        raise WorkspaceError(
            f"no experiment matches '{reference}'. Run 'fabric ls' to see what exists."
        )

    # -- runs --------------------------------------------------------------

    def save_run(self, experiment_id: str, run: Run) -> str:
        directory = self.run_dir(experiment_id, run.id)
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "run.json")
        write_json(path, run.to_dict())
        return path

    def load_run(self, experiment_id: str, run_id: str) -> Run | None:
        data = read_json(os.path.join(self.run_dir(experiment_id, run_id), "run.json"))
        return Run.from_dict(data) if data else None

    def iter_runs(self, experiment_id: str) -> Iterator[Run]:
        directory = self.runs_dir(experiment_id)
        if not os.path.isdir(directory):
            return
        for name in sorted(os.listdir(directory)):
            data = read_json(os.path.join(directory, name, "run.json"))
            if data:
                yield Run.from_dict(data)

    def load_runs(self, experiment_id: str) -> list[Run]:
        return list(self.iter_runs(experiment_id))

    # -- cache -------------------------------------------------------------

    @property
    def cache_index_path(self) -> str:
        return os.path.join(self.cache_dir, "index.json")

    def cache_lookup(self, key: str) -> dict[str, Any] | None:
        index = read_json(self.cache_index_path, {}) or {}
        entry = index.get(key)
        if not entry:
            return None
        run_path = os.path.join(self.run_dir(entry["experiment_id"], entry["run_id"]), "run.json")
        if not os.path.exists(run_path):
            # The cached run was deleted; forget the entry rather than
            # returning a promise Fabric cannot keep.
            self.cache_forget(key)
            return None
        return entry

    def cache_store(self, key: str, experiment_id: str, run_id: str,
                    metrics: dict[str, float]) -> None:
        with FileLock(self.cache_index_path):
            index = read_json(self.cache_index_path, {}) or {}
            index[key] = {
                "experiment_id": experiment_id,
                "run_id": run_id,
                "metrics": metrics,
                "stored_at": time.time(),
            }
            write_json(self.cache_index_path, index)

    def cache_forget(self, key: str) -> None:
        with FileLock(self.cache_index_path):
            index = read_json(self.cache_index_path, {}) or {}
            if key in index:
                del index[key]
                write_json(self.cache_index_path, index)

    def cache_clear(self) -> int:
        index = read_json(self.cache_index_path, {}) or {}
        count = len(index)
        write_json(self.cache_index_path, {})
        return count

    # -- generic files -----------------------------------------------------

    def write(self, experiment_id: str, relative: str, value: Any) -> str:
        path = os.path.join(self.experiment_dir(experiment_id), relative)
        write_json(path, value)
        return path

    def write_text(self, experiment_id: str, relative: str, text: str) -> str:
        path = os.path.join(self.experiment_dir(experiment_id), relative)
        atomic_write_text(path, text)
        return path

    def read(self, experiment_id: str, relative: str, default: Any = None) -> Any:
        return read_json(os.path.join(self.experiment_dir(experiment_id), relative), default)

    def delete_experiment(self, experiment_id: str) -> None:
        directory = self.experiment_dir(experiment_id)
        if os.path.isdir(directory):
            shutil.rmtree(directory)
        index = read_json(self.experiment_index_path, {"experiments": []}) or {}
        entries = [e for e in index.get("experiments", []) if e.get("id") != experiment_id]
        write_json(self.experiment_index_path, {"experiments": entries})
