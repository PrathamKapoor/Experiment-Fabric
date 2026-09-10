"""Provenance: connecting every number to what produced it.

The planner builds the *structural* graph -- experiment, entities,
conditions, runs.  This module completes it with what execution and
analysis added: the environment and hardware each run actually used, the
artifacts it wrote, the metric values it produced, the comparisons those
values fed, and the claims those comparisons support.

Once that graph exists, the question "why does this number exist?" is a
backwards traversal, and "what does changing this dataset invalidate?" is
a forwards one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from fabric.core import graph as g
from fabric.core.entities import Run
from fabric.core.graph import ExperimentGraph, Node
from fabric.execution.planner import ExperimentPlan


def metric_node_id(run_id: str, metric: str) -> str:
    return f"{run_id}:{metric}"


def build_provenance(
    plan: ExperimentPlan,
    runs: Sequence[Run],
    analysis: Any = None,
    *,
    environments: Mapping[str, Any] | None = None,
    hardware: Mapping[str, Any] | None = None,
    include_metrics: bool = True,
) -> ExperimentGraph:
    """Extend the plan graph with everything execution and analysis added."""
    graph = plan.graph
    by_id = {r.id: r for r in runs}

    for run in runs:
        if not graph.has_node(run.id):
            graph.add_node(run.id, "run", f"{run.condition_name} seed={run.seed}",
                           seed=run.seed)
            if graph.has_node(run.condition_id):
                graph.add_edge(run.condition_id, run.id, g.REPLICATES)
        graph.add_node(run.id, "run", "", status=run.status.value,
                       metrics=dict(run.metrics),
                       failure=run.failure.failure_class.value if run.failure else "",
                       wall_seconds=run.resources.wall_seconds,
                       cached_from=run.cached_from)

        if run.environment_id:
            label = (environments or {}).get(run.environment_id, run.environment_id)
            graph.add_node(run.environment_id, "environment",
                           _short_label("env", run.environment_id, label))
            graph.add_edge(run.environment_id, run.id, g.USES)
        if run.hardware_id:
            label = (hardware or {}).get(run.hardware_id, run.hardware_id)
            graph.add_node(run.hardware_id, "hardware",
                           _short_label("hw", run.hardware_id, label))
            graph.add_edge(run.hardware_id, run.id, g.USES)

        for artifact in run.artifacts:
            node_id = f"{run.id}/artifact/{artifact.name}"
            graph.add_node(node_id, "artifact", artifact.name,
                           digest=artifact.digest, bytes=artifact.bytes,
                           media_type=artifact.media_type, role=artifact.role)
            graph.add_edge(run.id, node_id, g.PRODUCES)

        if include_metrics:
            for name, value in run.metrics.items():
                node_id = metric_node_id(run.id, name)
                graph.add_node(node_id, "metric", f"{name}={_format(value)}",
                               metric=name, value=value, run=run.id, seed=run.seed,
                               condition=run.condition_name)
                graph.add_edge(run.id, node_id, g.PRODUCES)

        if run.cached_from:
            source_id = run.cached_from.split("/")[-1]
            if source_id in by_id and source_id != run.id:
                graph.add_edge(source_id, run.id, g.DERIVES_FROM, reason="cache hit")

    if analysis is not None:
        _attach_analysis(graph, plan, analysis, runs, include_metrics=include_metrics)
    return graph


def _attach_analysis(graph: ExperimentGraph, plan: ExperimentPlan, analysis: Any,
                     runs: Sequence[Run], *, include_metrics: bool) -> None:
    """Add comparison and claim nodes, wired to the values that fed them."""
    by_condition: dict[str, list[Run]] = {}
    for run in runs:
        if run.succeeded:
            by_condition.setdefault(run.condition_name, []).append(run)

    analysis_id = f"{plan.experiment.id}/analysis"
    graph.add_node(analysis_id, "analysis", "statistical analysis",
                   version=getattr(analysis, "analysis_version", ""),
                   alpha=getattr(analysis, "alpha", None),
                   correction=getattr(analysis, "correction", ""))

    comparisons = list(getattr(getattr(analysis, "baselines", None), "comparisons", []) or [])
    for effect in getattr(getattr(analysis, "ablations", None), "effects", []) or []:
        comparisons.extend(effect.comparisons.values())

    for index, comparison in enumerate(comparisons):
        node_id = f"{analysis_id}/comparison/{index}"
        graph.add_node(
            node_id, "comparison",
            f"{comparison.treatment} vs {comparison.baseline} on {comparison.metric}",
            metric=comparison.metric, test=comparison.test.test,
            p_value=comparison.p_value, adjusted_p=comparison.adjusted_p,
            difference=comparison.difference, verdict=comparison.verdict,
        )
        graph.add_edge(analysis_id, node_id, g.CONTAINS)
        for side in (comparison.treatment, comparison.baseline):
            for run in by_condition.get(side, []):
                if include_metrics:
                    value_id = metric_node_id(run.id, comparison.metric)
                    if graph.has_node(value_id):
                        graph.add_edge(value_id, node_id, g.SUPPORTS)
                elif graph.has_node(run.id):
                    graph.add_edge(run.id, node_id, g.SUPPORTS)

    headline = getattr(analysis, "headline", "")
    if headline:
        claim_id = f"{plan.experiment.id}/claim"
        graph.add_node(claim_id, "claim", _truncate(headline, 90), statement=headline)
        for index in range(len(comparisons)):
            node_id = f"{analysis_id}/comparison/{index}"
            if graph.has_node(node_id):
                graph.add_edge(node_id, claim_id, g.SUPPORTS)


# --------------------------------------------------------------------------
# explanation
# --------------------------------------------------------------------------


@dataclass
class Explanation:
    """The chain of custody for one number."""

    target: str
    value: float | None = None
    metric: str = ""
    chain: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "metric": self.metric,
            "value": self.value,
            "chain": self.chain,
            "artifacts": self.artifacts,
            "missing": self.missing,
        }

    def render(self) -> str:
        """The terminal rendering used by ``fabric why``."""
        lines: list[str] = []
        if self.metric and self.value is not None:
            lines.append(f"{self.metric} = {_format(self.value)}")
        else:
            lines.append(self.target)
        lines.append("")
        width = max((len(item["kind"]) for item in self.chain), default=8)
        for item in self.chain:
            lines.append(f"  {item['kind']:<{width}}  {item['label']}")
            for key, value in item.get("detail", {}).items():
                if value not in (None, "", [], {}):
                    lines.append(f"  {'':<{width}}    {key}: {value}")
        if self.artifacts:
            lines.append("")
            lines.append("  raw artifacts")
            for artifact in self.artifacts:
                lines.append(
                    f"    {artifact['name']}  {artifact['digest'][:19]}  "
                    f"{artifact['bytes']} bytes"
                )
        if self.missing:
            lines.append("")
            lines.append("  not recorded: " + ", ".join(self.missing))
        return "\n".join(lines)


_CHAIN_ORDER = ("metric", "run", "condition", "algorithm", "model", "dataset",
                "environment", "hardware", "experiment")


def explain(graph: ExperimentGraph, target: str) -> Explanation:
    """Walk backwards from a node to everything that produced it."""
    if not graph.has_node(target):
        raise KeyError(f"{target} is not in the provenance graph")
    node = graph.node(target)
    explanation = Explanation(
        target=target,
        metric=str(node.attributes.get("metric", "")),
        value=node.attributes.get("value"),
    )

    ancestors = [node] + graph.ancestors(target)
    seen_kinds: dict[str, list[Node]] = {}
    for item in ancestors:
        seen_kinds.setdefault(item.kind, []).append(item)

    for kind in _CHAIN_ORDER:
        for item in seen_kinds.get(kind, []):
            explanation.chain.append({
                "kind": kind,
                "id": item.id,
                "label": item.label,
                "detail": _detail(item),
            })

    for kind, items in sorted(seen_kinds.items()):
        if kind in _CHAIN_ORDER or kind == "artifact":
            continue
        for item in items:
            explanation.chain.append({
                "kind": kind, "id": item.id, "label": item.label,
                "detail": _detail(item),
            })

    run_nodes = [n for n in ancestors if n.kind == "run"]
    for run_node in run_nodes:
        for child in graph.successors(run_node.id, g.PRODUCES):
            if child.kind == "artifact":
                explanation.artifacts.append({
                    "name": child.label,
                    "digest": str(child.attributes.get("digest", "")),
                    "bytes": child.attributes.get("bytes", 0),
                })

    for kind in ("dataset", "model", "algorithm", "environment", "hardware"):
        if kind not in seen_kinds:
            explanation.missing.append(kind)
    return explanation


def find_metric_node(graph: ExperimentGraph, *, condition: str = "", metric: str = "",
                     seed: int | None = None) -> Node | None:
    """Locate a metric value node by condition, metric name and seed."""
    for node in graph.nodes_of_kind("metric"):
        attributes = node.attributes
        if metric and attributes.get("metric") != metric:
            continue
        if condition and attributes.get("condition") != condition:
            continue
        if seed is not None and attributes.get("seed") != seed:
            continue
        return node
    return None


def impact(graph: ExperimentGraph, node_id: str) -> dict[str, Any]:
    """What depends on a node -- the "what breaks if this changes?" query."""
    descendants = graph.descendants(node_id)
    by_kind: dict[str, int] = {}
    for node in descendants:
        by_kind[node.kind] = by_kind.get(node.kind, 0) + 1
    claims = [n.attributes.get("statement", n.label)
              for n in descendants if n.kind == "claim"]
    return {
        "node": node_id,
        "label": graph.node(node_id).label if graph.has_node(node_id) else "",
        "affected": len(descendants),
        "by_kind": dict(sorted(by_kind.items())),
        "claims": claims,
    }


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


_DETAIL_KEYS = {
    "run": ("seed", "status", "wall_seconds", "cached_from"),
    "condition": ("role", "config", "ablated", "group"),
    "dataset": ("digest", "n_examples"),
    "model": ("weights_digest", "precision"),
    "algorithm": ("commit", "entrypoint"),
    "environment": (),
    "hardware": (),
    "experiment": ("research_question",),
    "metric": ("value", "seed"),
    "comparison": ("test", "p_value", "adjusted_p", "verdict"),
}


def _detail(node: Node) -> dict[str, Any]:
    keys = _DETAIL_KEYS.get(node.kind, ())
    detail: dict[str, Any] = {}
    for key in keys:
        value = node.attributes.get(key)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, float):
            detail[key] = _format(value)
        else:
            detail[key] = value
    return detail


def _format(value: Any) -> str:
    if isinstance(value, float):
        if value != value:
            return "nan"
        return f"{value:.6g}"
    return str(value)


def _short_label(prefix: str, entity_id: str, label: Any) -> str:
    if isinstance(label, str) and label != entity_id:
        return label
    return f"{prefix} {entity_id[-8:]}"


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."
