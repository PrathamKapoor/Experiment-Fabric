"""The experiment graph.

This is the defining abstraction.  An experiment is not a script that
produces a number; it is a directed acyclic graph whose nodes are
addressable scientific objects -- datasets, models, algorithms,
environments, conditions, runs, analyses, claims -- and whose edges say
how each thing came from the others.

Because the graph is explicit, two questions become mechanical that are
normally archaeology:

* *Why does this number exist?*  Walk backwards from the metric.
* *What breaks if this dataset changes?*  Walk forwards from the dataset.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping

# Relations are typed so a traversal can ask for meaningful paths rather
# than "anything connected to anything".
USES = "uses"
PRODUCES = "produces"
DERIVES_FROM = "derives_from"
COMPARES = "compares"
REPLICATES = "replicates"
PERTURBS = "perturbs"
SUPPORTS = "supports"
CONTAINS = "contains"

RELATIONS = (USES, PRODUCES, DERIVES_FROM, COMPARES, REPLICATES, PERTURBS, SUPPORTS, CONTAINS)


@dataclass(frozen=True)
class Node:
    """A vertex: one addressable scientific object."""

    id: str
    kind: str
    label: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "label": self.label,
                "attributes": self.attributes}


@dataclass(frozen=True)
class Edge:
    """A typed, directed relationship."""

    source: str
    target: str
    relation: str = USES
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "target": self.target,
                "relation": self.relation, "attributes": self.attributes}


class GraphError(ValueError):
    """Raised on an invalid graph operation."""


class ExperimentGraph:
    """A directed acyclic graph of experimental provenance."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._nodes: dict[str, Node] = {}
        self._out: dict[str, list[Edge]] = {}
        self._in: dict[str, list[Edge]] = {}

    # -- construction ------------------------------------------------------

    def add_node(self, node_id: str, kind: str, label: str = "",
                 **attributes: Any) -> Node:
        """Add or merge a node.

        Merging matters: the same dataset is referenced by hundreds of
        runs, and each reference should land on one node, not a hundred.
        """
        existing = self._nodes.get(node_id)
        if existing is not None:
            merged = dict(existing.attributes)
            merged.update(attributes)
            node = Node(node_id, existing.kind, label or existing.label, merged)
        else:
            node = Node(node_id, kind, label or node_id, dict(attributes))
            self._out.setdefault(node_id, [])
            self._in.setdefault(node_id, [])
        self._nodes[node_id] = node
        return node

    def add_edge(self, source: str, target: str, relation: str = USES,
                 **attributes: Any) -> Edge:
        """Connect two existing nodes."""
        if source not in self._nodes:
            raise GraphError(f"unknown source node {source!r}")
        if target not in self._nodes:
            raise GraphError(f"unknown target node {target!r}")
        edge = Edge(source, target, relation, dict(attributes))
        if any(e.target == target and e.relation == relation for e in self._out[source]):
            return edge
        self._out[source].append(edge)
        self._in[target].append(edge)
        return edge

    # -- access ------------------------------------------------------------

    def __contains__(self, node_id: object) -> bool:
        return node_id in self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self) -> Iterator[Node]:
        return iter(self._nodes.values())

    @property
    def nodes(self) -> list[Node]:
        return list(self._nodes.values())

    @property
    def edges(self) -> list[Edge]:
        return [edge for edges in self._out.values() for edge in edges]

    def node(self, node_id: str) -> Node:
        try:
            return self._nodes[node_id]
        except KeyError:
            raise GraphError(f"unknown node {node_id!r}") from None

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def nodes_of_kind(self, kind: str) -> list[Node]:
        return [n for n in self._nodes.values() if n.kind == kind]

    def successors(self, node_id: str, relation: str | None = None) -> list[Node]:
        return [self._nodes[e.target] for e in self._out.get(node_id, [])
                if relation is None or e.relation == relation]

    def predecessors(self, node_id: str, relation: str | None = None) -> list[Node]:
        return [self._nodes[e.source] for e in self._in.get(node_id, [])
                if relation is None or e.relation == relation]

    # -- traversal ---------------------------------------------------------

    def ancestors(self, node_id: str, relations: Iterable[str] | None = None) -> list[Node]:
        """Everything this node depends on, breadth-first.

        This is the "why does this number exist?" traversal.
        """
        allowed = set(relations) if relations else None
        seen: set[str] = set()
        order: list[Node] = []
        queue = deque([node_id])
        while queue:
            current = queue.popleft()
            for edge in self._in.get(current, []):
                if allowed and edge.relation not in allowed:
                    continue
                if edge.source in seen:
                    continue
                seen.add(edge.source)
                order.append(self._nodes[edge.source])
                queue.append(edge.source)
        return order

    def descendants(self, node_id: str, relations: Iterable[str] | None = None) -> list[Node]:
        """Everything that depends on this node.

        This is the "what does changing the dataset invalidate?" traversal.
        """
        allowed = set(relations) if relations else None
        seen: set[str] = set()
        order: list[Node] = []
        queue = deque([node_id])
        while queue:
            current = queue.popleft()
            for edge in self._out.get(current, []):
                if allowed and edge.relation not in allowed:
                    continue
                if edge.target in seen:
                    continue
                seen.add(edge.target)
                order.append(self._nodes[edge.target])
                queue.append(edge.target)
        return order

    def path(self, source: str, target: str) -> list[Node]:
        """A shortest directed path, or an empty list when none exists."""
        if source not in self._nodes or target not in self._nodes:
            return []
        previous: dict[str, str] = {}
        queue = deque([source])
        seen = {source}
        while queue:
            current = queue.popleft()
            if current == target:
                chain = [current]
                while chain[-1] != source:
                    chain.append(previous[chain[-1]])
                return [self._nodes[n] for n in reversed(chain)]
            for edge in self._out.get(current, []):
                if edge.target not in seen:
                    seen.add(edge.target)
                    previous[edge.target] = current
                    queue.append(edge.target)
        return []

    @property
    def roots(self) -> list[Node]:
        return [n for n in self._nodes.values() if not self._in.get(n.id)]

    @property
    def leaves(self) -> list[Node]:
        return [n for n in self._nodes.values() if not self._out.get(n.id)]

    def topological_order(self) -> list[Node]:
        """Kahn's algorithm; raises if the graph has a cycle.

        Provenance must be acyclic -- a result cannot be its own
        ancestor -- so a cycle is a bug worth failing loudly on.
        """
        indegree = {n: len(self._in.get(n, [])) for n in self._nodes}
        queue = deque(sorted(n for n, d in indegree.items() if d == 0))
        order: list[Node] = []
        while queue:
            current = queue.popleft()
            order.append(self._nodes[current])
            for edge in self._out.get(current, []):
                indegree[edge.target] -= 1
                if indegree[edge.target] == 0:
                    queue.append(edge.target)
        if len(order) != len(self._nodes):
            remaining = sorted(set(self._nodes) - {n.id for n in order})
            raise GraphError(f"the provenance graph contains a cycle involving {remaining[:5]}")
        return order

    def subgraph(self, node_ids: Iterable[str]) -> "ExperimentGraph":
        """Induced subgraph over the given nodes."""
        keep = set(node_ids)
        result = ExperimentGraph(self.name)
        for node_id in keep:
            if node_id in self._nodes:
                node = self._nodes[node_id]
                result.add_node(node.id, node.kind, node.label, **node.attributes)
        for edge in self.edges:
            if edge.source in keep and edge.target in keep:
                result.add_edge(edge.source, edge.target, edge.relation, **edge.attributes)
        return result

    def lineage(self, node_id: str) -> "ExperimentGraph":
        """The node plus everything it descends from."""
        ids = {node_id} | {n.id for n in self.ancestors(node_id)}
        return self.subgraph(ids)

    # -- statistics --------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        kinds: dict[str, int] = {}
        for node in self._nodes.values():
            kinds[node.kind] = kinds.get(node.kind, 0) + 1
        relations: dict[str, int] = {}
        for edge in self.edges:
            relations[edge.relation] = relations.get(edge.relation, 0) + 1
        return {
            "nodes": len(self._nodes),
            "edges": len(self.edges),
            "kinds": dict(sorted(kinds.items())),
            "relations": dict(sorted(relations.items())),
            "roots": len(self.roots),
            "leaves": len(self.leaves),
        }

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
            "stats": self.stats(),
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "ExperimentGraph":
        graph = ExperimentGraph(str(data.get("name", "")))
        for node in data.get("nodes", []):
            graph.add_node(node["id"], node.get("kind", "unknown"),
                           node.get("label", ""), **(node.get("attributes") or {}))
        for edge in data.get("edges", []):
            graph.add_edge(edge["source"], edge["target"],
                           edge.get("relation", USES), **(edge.get("attributes") or {}))
        return graph

    # -- rendering ---------------------------------------------------------

    _DOT_STYLE = {
        "dataset": ("box", "#0e7490"),
        "model": ("box", "#7c3aed"),
        "algorithm": ("box", "#b45309"),
        "environment": ("note", "#475569"),
        "hardware": ("note", "#475569"),
        "condition": ("ellipse", "#15803d"),
        "run": ("ellipse", "#334155"),
        "metric": ("diamond", "#be123c"),
        "analysis": ("hexagon", "#1d4ed8"),
        "comparison": ("hexagon", "#1d4ed8"),
        "claim": ("doubleoctagon", "#9f1239"),
        "experiment": ("folder", "#0f172a"),
        "artifact": ("cylinder", "#0891b2"),
    }

    def to_dot(self) -> str:
        """Graphviz DOT source."""
        lines = ["digraph fabric {", '  rankdir=LR;', '  node [fontname="Helvetica"];']
        for node in self._nodes.values():
            shape, colour = self._DOT_STYLE.get(node.kind, ("box", "#333333"))
            label = node.label.replace('"', "'")
            lines.append(
                f'  "{node.id}" [label="{label}", shape={shape}, color="{colour}"];'
            )
        for edge in self.edges:
            lines.append(f'  "{edge.source}" -> "{edge.target}" [label="{edge.relation}"];')
        lines.append("}")
        return "\n".join(lines)

    def to_mermaid(self, *, max_nodes: int = 200) -> str:
        """Mermaid flowchart source, safe to embed in Markdown or HTML."""
        lines = ["flowchart LR"]
        safe: dict[str, str] = {}
        for index, node in enumerate(self._nodes.values()):
            if index >= max_nodes:
                lines.append(f"  more[\"... {len(self._nodes) - max_nodes} more nodes\"]")
                break
            key = f"n{index}"
            safe[node.id] = key
            label = node.label.replace('"', "'").replace("\n", " ")
            shape_open, shape_close = {
                "dataset": ("[(", ")]"),
                "model": ("[", "]"),
                "run": ("(", ")"),
                "condition": ("([", "])"),
                "claim": ("{{", "}}"),
                "comparison": ("{{", "}}"),
            }.get(node.kind, ("[", "]"))
            lines.append(f'  {key}{shape_open}"{label}"{shape_close}')
        for edge in self.edges:
            if edge.source in safe and edge.target in safe:
                lines.append(f"  {safe[edge.source]} -->|{edge.relation}| {safe[edge.target]}")
        return "\n".join(lines)

    def to_ascii(self, node_id: str, *, depth: int = 6) -> str:
        """An ASCII ancestry tree, for terminals.

        This is what ``fabric why`` prints: the chain of custody for one
        number, readable without any rendering tool at all.
        """
        lines: list[str] = []

        def walk(current: str, prefix: str, level: int, last: bool) -> None:
            node = self._nodes[current]
            connector = "" if level == 0 else ("`-- " if last else "|-- ")
            lines.append(f"{prefix}{connector}[{node.kind}] {node.label}")
            if level >= depth:
                return
            parents = self.predecessors(current)
            child_prefix = prefix + ("" if level == 0 else ("    " if last else "|   "))
            for i, parent in enumerate(parents):
                walk(parent.id, child_prefix, level + 1, i == len(parents) - 1)

        walk(node_id, "", 0, True)
        return "\n".join(lines)
