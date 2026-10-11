"""The service layer: one implementation of every operation.

This exists so that the HTTP API cannot drift from the command line
(section 31).  Both call the functions here; neither contains business
logic of its own.  The rule is easy to check and easy to break, so it is
worth stating: **nothing in this module imports anything from
``fabric.server.app`` or ``fabric.cli``, and nothing in either of those
reimplements what is here.**

Everything returns plain JSON-ready dictionaries, because that is what
the API serialises and what the CLI's ``--json`` flags print.  The
human-readable rendering stays in the CLI, where it belongs.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from fabric.analysis.engine import ExperimentAnalysis, analyse
from fabric.core.benchmark import available as benchmarks_available
from fabric.core.diff import diff_experiments
from fabric.core.entities import Run, RunStatus
from fabric.core.graph import ExperimentGraph
from fabric.core.provenance import build_provenance, explain, find_metric_node, impact
from fabric.core.registry import Registry, RegistryError
from fabric.core.spec import ExperimentSpec, SpecError
from fabric.core.store import Workspace, WorkspaceError
from fabric.execution.planner import ExperimentPlan, PlanError, plan_experiment
from fabric.reporting import builder, html, latex, markdown
from fabric.reporting.certificate import issue

SERVICE_SCHEMA_VERSION = "1.0"


class ServiceError(Exception):
    """An operation failed in a way the caller should be told about.

    ``status`` is an HTTP status code so the API can map it directly;
    the CLI ignores it and prints the message.
    """

    def __init__(self, message: str, status: int = 400, *,
                 remedy: str = "") -> None:
        self.status = status
        self.remedy = remedy
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": str(self),
            "status": self.status,
            "remedy": self.remedy,
        }


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def resolve_experiment(workspace: Workspace, reference: str) -> str:
    try:
        return workspace.resolve_experiment(reference)
    except WorkspaceError as exc:
        raise ServiceError(str(exc), 404) from exc


def specification_text(workspace: Workspace, experiment_id: str) -> str:
    """The exact YAML the experiment was created from."""
    path = os.path.join(workspace.experiment_dir(experiment_id), "experiment.yaml")
    if not os.path.isfile(path):
        raise ServiceError(
            f"{experiment_id} has no stored specification", 404,
            remedy="The experiment directory is incomplete; re-run its YAML file.",
        )
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def load_plan(workspace: Workspace, experiment_id: str) -> ExperimentPlan:
    """Rebuild the stored plan for an experiment."""
    text = specification_text(workspace, experiment_id)
    try:
        spec = ExperimentSpec.from_yaml(text)
        return plan_experiment(spec, workspace, root=workspace.root, strict=False)
    except (SpecError, PlanError) as exc:
        raise ServiceError(str(exc), 500) from exc


def load_analysis(workspace: Workspace, experiment_id: str) -> tuple[
        ExperimentPlan, list[Run], ExperimentAnalysis]:
    plan = load_plan(workspace, experiment_id)
    runs = workspace.load_runs(experiment_id)
    if not runs:
        raise ServiceError(
            f"{experiment_id} has no runs yet", 409,
            remedy="Execute it first: POST /experiments/{id}/run, or 'fabric run'.",
        )
    stored = workspace.read(experiment_id, "plan.json", {}) or {}
    analysis = analyse(plan, runs,
                       stored_spec_digest=str(stored.get("spec_digest", "")))
    return plan, runs, analysis


def provenance_graph(plan: ExperimentPlan, runs: Sequence[Run],
                     analysis: ExperimentAnalysis | None) -> ExperimentGraph:
    return build_provenance(plan, list(runs), analysis)


# --------------------------------------------------------------------------
# status
# --------------------------------------------------------------------------


def experiment_status(planned: int, completed: int, failed: int) -> str:
    """The lifecycle state of an experiment, as one word (section 78)."""
    if planned == 0 or (completed == 0 and failed == 0):
        return "planned"
    if completed + failed < planned:
        return "partial"
    if completed == 0:
        return "failed"
    return "completed" if failed == 0 else "partial"


def list_experiments(workspace: Workspace, *, limit: int = 50) -> list[dict[str, Any]]:
    listing = []
    for entry in workspace.list_experiments()[:limit]:
        experiment_id = str(entry["id"])
        runs = workspace.load_runs(experiment_id)
        completed = sum(1 for r in runs if r.succeeded)
        failed = sum(1 for r in runs if r.status == RunStatus.FAILED)
        planned = int(entry.get("runs", len(runs)) or len(runs))
        listing.append({
            "experiment_id": experiment_id,
            "name": entry.get("name", ""),
            "status": experiment_status(planned, completed, failed),
            "runs": {"planned": planned, "completed": completed, "failed": failed},
            "created_at": entry.get("created_at", 0),
            "spec_digest": entry.get("spec_digest", ""),
        })
    return listing


def describe_experiment(workspace: Workspace, experiment_id: str) -> dict[str, Any]:
    """Everything about an experiment that does not need the analysis."""
    plan = load_plan(workspace, experiment_id)
    runs = workspace.load_runs(experiment_id)
    completed = sum(1 for r in runs if r.succeeded)
    failed = sum(1 for r in runs if r.status == RunStatus.FAILED)
    spec = plan.spec
    benchmark = spec.benchmark
    return {
        "schema_version": SERVICE_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "name": spec.name,
        "research_question": spec.research_question,
        "hypothesis": spec.hypothesis,
        "status": experiment_status(len(plan.runs), completed, failed),
        "spec_digest": spec.digest,
        "primary_metric": spec.primary_metric,
        "metrics": [m.to_dict() for m in spec.metrics],
        "seeds": list(spec.seeds.values),
        "runs": {"planned": len(plan.runs), "completed": completed, "failed": failed},
        "conditions": [
            {"name": c.name, "role": c.role.value, "id": c.id,
             "removed": list(c.ablated), "config": c.config,
             "perturbation": c.perturbation}
            for c in plan.conditions
        ],
        "benchmark": (
            {"ref": benchmark.ref, "digest": benchmark.digest,
             "description": " ".join(benchmark.description.split()),
             "constraints": [r.to_dict() for r in spec.benchmark_constraints]}
            if benchmark else None
        ),
        "inputs": {
            "dataset": plan.dataset.to_dict() if plan.dataset else None,
            "model": plan.model.to_dict() if plan.model else None,
            "algorithm": plan.algorithm.to_dict() if plan.algorithm else None,
        },
        "advisories": list(plan.advisories),
    }


def list_runs(workspace: Workspace, experiment_id: str, *,
              condition: str = "", limit: int = 1000) -> list[dict[str, Any]]:
    runs = workspace.load_runs(experiment_id)
    if condition:
        runs = [r for r in runs if r.condition_name == condition]
    return [_run_summary(r) for r in runs[:limit]]


def describe_run(workspace: Workspace, experiment_id: str,
                 run_id: str) -> dict[str, Any]:
    for run in workspace.load_runs(experiment_id):
        if run.id == run_id or run.id.endswith(run_id):
            data = run.to_dict()
            data["artifacts"] = [a.to_dict() for a in run.artifacts]
            return data
    raise ServiceError(f"no run {run_id} in {experiment_id}", 404)


def _run_summary(run: Run) -> dict[str, Any]:
    return {
        "id": run.id,
        "condition": run.condition_name,
        "seed": run.seed,
        "replicate": run.replicate,
        "status": run.status.value,
        "metrics": run.metrics,
        "artifacts": len(run.artifacts),
        "wall_seconds": run.resources.wall_seconds,
        "environment_id": run.environment_id,
        "hardware_id": run.hardware_id,
        "failure": run.failure.to_dict() if run.failure else None,
    }


# --------------------------------------------------------------------------
# analysis, reporting, provenance
# --------------------------------------------------------------------------


def analysis_payload(workspace: Workspace, experiment_id: str) -> dict[str, Any]:
    _plan, _runs, analysis = load_analysis(workspace, experiment_id)
    return analysis.to_dict()


def graph_payload(workspace: Workspace, experiment_id: str, *,
                  node: str = "") -> dict[str, Any]:
    plan = load_plan(workspace, experiment_id)
    runs = workspace.load_runs(experiment_id)
    graph = provenance_graph(plan, runs, None)
    if node:
        if node not in graph:
            raise ServiceError(f"no node {node} in the provenance graph", 404)
        graph = graph.lineage(node)
    return graph.to_dict()


def report_payload(workspace: Workspace, experiment_id: str, *,
                   fmt: str = "json") -> tuple[str, str]:
    """Render a report. Returns ``(content, media type)``."""
    plan, runs, analysis = load_analysis(workspace, experiment_id)
    graph = provenance_graph(plan, runs, analysis)
    registry = Registry(workspace)
    environment = _first(registry, "environment", [r.environment_id for r in runs])
    hardware = _first(registry, "hardware", [r.hardware_id for r in runs])
    document = builder.build_report(analysis, plan, runs, graph=graph,
                                    environment=environment, hardware=hardware)
    if fmt == "html":
        return html.render(document), "text/html; charset=utf-8"
    if fmt in ("md", "markdown"):
        return markdown.render(document), "text/markdown; charset=utf-8"
    if fmt in ("tex", "latex"):
        return latex.render(document), "application/x-tex; charset=utf-8"
    if fmt == "json":
        from fabric.util.jsonio import dumps as json_dumps

        return json_dumps(document.to_dict()), "application/json; charset=utf-8"
    raise ServiceError(
        f"unknown report format {fmt!r}", 400,
        remedy="Use one of: html, md, tex, json.",
    )


def certificate_payload(workspace: Workspace, experiment_id: str) -> dict[str, Any]:
    plan, runs, analysis = load_analysis(workspace, experiment_id)
    registry = Registry(workspace)
    certificate = issue(analysis, plan, runs, registry,
                        workspace_root=workspace.root,
                        reproduction=workspace.read(experiment_id, "reproduction.json"))
    return certificate.to_dict()


def why_payload(workspace: Workspace, experiment_id: str, *,
                metric: str = "", seed: int | None = None,
                condition: str = "") -> dict[str, Any]:
    plan, runs, analysis = load_analysis(workspace, experiment_id)
    graph = provenance_graph(plan, runs, analysis)
    metric = metric or analysis.primary_metric
    node = find_metric_node(graph, metric=metric, seed=seed, condition=condition)
    if node is None:
        raise ServiceError(
            f"no recorded {metric} matches that run", 404,
            remedy="Check the metric name and the seed against GET "
                   f"/experiments/{experiment_id}/runs.",
        )
    return explain(graph, node.id)


def impact_payload(workspace: Workspace, experiment_id: str, *,
                   node: str = "") -> dict[str, Any]:
    plan = load_plan(workspace, experiment_id)
    runs = workspace.load_runs(experiment_id)
    graph = provenance_graph(plan, runs, None)
    if not node:
        if plan.dataset is None:
            raise ServiceError(
                "this experiment has no dataset; name a node explicitly", 400)
        node = plan.dataset.id
    if node not in graph:
        raise ServiceError(f"no node {node} in the provenance graph", 404)
    return {"experiment_id": experiment_id, "node": node, **impact(graph, node)}


def diff_payload(workspace: Workspace, left: str, right: str) -> dict[str, Any]:
    left_id = resolve_experiment(workspace, left)
    right_id = resolve_experiment(workspace, right)
    if left_id == right_id:
        raise ServiceError(f"{left} and {right} are the same experiment", 400)
    left_parts = load_analysis(workspace, left_id)
    right_parts = load_analysis(workspace, right_id)
    return diff_experiments(*left_parts, *right_parts).to_dict()


def benchmarks_payload(root: str = "") -> list[dict[str, Any]]:
    return [b.to_dict() for b in benchmarks_available(root=root or os.getcwd())]


def _first(registry: Registry, kind: str, ids: Sequence[str]) -> Any:
    for entity_id in ids:
        if not entity_id:
            continue
        try:
            return registry.get(kind, entity_id)
        except RegistryError:
            continue
    return None


# --------------------------------------------------------------------------
# submission
# --------------------------------------------------------------------------


def submit_specification(workspace: Workspace, text: str) -> dict[str, Any]:
    """Validate and store a specification without executing anything.

    Planning is safe: it parses, expands and hashes, and the one thing it
    would otherwise do that touches user code -- importing the entrypoint
    to check it exists -- is switched off here deliberately. Importing a
    module runs its top level, and a server must not do that on the
    strength of an HTTP request. The check happens in the worker instead,
    inside the same subprocess isolation the run itself gets.
    """
    try:
        spec = ExperimentSpec.from_yaml(text)
    except SpecError as exc:
        raise ServiceError(
            str(exc), 422,
            remedy="Fix the specification and submit it again.",
        ) from exc
    try:
        plan = plan_experiment(spec, workspace, root=workspace.root, strict=False)
    except PlanError as exc:
        raise ServiceError(str(exc), 422) from exc

    workspace.create_experiment_dir(plan.experiment.id)
    workspace.write_text(plan.experiment.id, "experiment.yaml", text)
    workspace.write(plan.experiment.id, "plan.json", plan.to_dict())
    workspace.write(plan.experiment.id, "graph.json", plan.graph.to_dict())
    workspace.register_experiment({
        "id": plan.experiment.id,
        "name": plan.experiment.name,
        "research_question": plan.experiment.research_question,
        "spec_digest": plan.spec.digest,
        "created_at": plan.experiment.created_at,
        "source": "submitted over the API",
        "runs": len(plan.runs),
    })
    return {
        "experiment_id": plan.experiment.id,
        "name": spec.name,
        "spec_digest": spec.digest,
        "conditions": len(plan.conditions),
        "runs": len(plan.runs),
        "status": "planned",
        "advisories": list(plan.advisories),
        "benchmark": spec.benchmark.ref if spec.benchmark else None,
    }
