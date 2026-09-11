"""An interactive provenance graph, as inline SVG and vanilla JavaScript.

The graph is the project's core abstraction and, until this module, the
only thing that could not be *seen*: it was exportable as DOT or Mermaid
and rendered nowhere. A picture of an evidence chain is the fastest way
to understand what the tool does, so the report draws one.

What makes it worth drawing rather than decorative: **click any node and
its lineage lights up.** That is the ``fabric why`` query, performed with
a mouse. Everything not on the path recedes, so the chain of custody for
one number is visible at a glance.

Mechanics, all dependency-free:

* a layered (Sugiyama-style) layout computed in Python -- longest-path
  layering, then barycentre ordering to reduce crossings
* inline SVG, themed by the same CSS custom properties as the charts
* about eighty lines of vanilla JavaScript for pan, zoom, hover and the
  lineage highlight; no library, no network, no build step

The view is an *abstraction* -- replications are folded per condition --
and the caption says so, with the full node count beside it.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from typing import Any

from fabric.core.graph import ExperimentGraph, Node

# Kind -> (fill token, short label). Colours reuse the validated chart
# palette; shape and position carry the meaning, colour reinforces it.
KIND_STYLE: dict[str, tuple[str, str]] = {
    "experiment": ("var(--viz-ink-2, #52514e)", "experiment"),
    "dataset": ("var(--viz-s3, #1baf7a)", "dataset"),
    "model": ("var(--viz-s2, #eb6834)", "model"),
    "algorithm": ("var(--viz-s1, #2a78d6)", "algorithm"),
    "environment": ("var(--viz-muted, #898781)", "environment"),
    "hardware": ("var(--viz-muted, #898781)", "hardware"),
    "condition": ("var(--viz-s1, #2a78d6)", "condition"),
    "runs": ("var(--viz-ink-2, #52514e)", "runs"),
    "run": ("var(--viz-ink-2, #52514e)", "run"),
    "artifact": ("var(--viz-s3, #1baf7a)", "artifact"),
    "metric": ("var(--viz-neg, #d03b3b)", "metric"),
    "comparison": ("var(--viz-s1, #2a78d6)", "comparison"),
    "analysis": ("var(--viz-s1, #2a78d6)", "analysis"),
    "claim": ("var(--viz-neg, #d03b3b)", "claim"),
}

NODE_WIDTH = 168
NODE_HEIGHT = 32
LAYER_GAP = 220
ROW_GAP = 42
MARGIN = 22


@dataclass
class Placed:
    """A node with a position."""

    node: Node
    layer: int
    row: int
    x: float = 0.0
    y: float = 0.0


@dataclass
class Layout:
    """A computed layout."""

    placed: dict[str, Placed] = field(default_factory=dict)
    width: float = 0.0
    height: float = 0.0


def _layer_assignment(graph: ExperimentGraph) -> dict[str, int]:
    """Longest path from a source, so every edge points rightwards."""
    order = graph.topological_order()
    layers: dict[str, int] = {}
    for node in order:
        parents = graph.predecessors(node.id)
        layers[node.id] = (
            max((layers.get(p.id, 0) for p in parents), default=-1) + 1
        )
    return layers


def _order_within_layers(graph: ExperimentGraph,
                         layers: dict[str, int]) -> dict[int, list[str]]:
    """Barycentre ordering: two passes, which removes most crossings."""
    buckets: dict[int, list[str]] = {}
    for node_id, layer in layers.items():
        buckets.setdefault(layer, []).append(node_id)
    for layer in buckets:
        buckets[layer].sort(key=lambda n: (graph.node(n).kind, graph.node(n).label))

    for _sweep in range(2):
        for layer in sorted(buckets)[1:]:
            previous = {n: i for i, n in enumerate(buckets[layer - 1])}

            def barycentre(node_id: str, previous=previous) -> float:
                positions = [previous[p.id] for p in graph.predecessors(node_id)
                             if p.id in previous]
                return sum(positions) / len(positions) if positions else 1e9

            buckets[layer].sort(key=barycentre)
        for layer in sorted(buckets, reverse=True)[1:]:
            following = {n: i for i, n in enumerate(buckets[layer + 1])} \
                if (layer + 1) in buckets else {}
            if not following:
                continue

            def reverse_barycentre(node_id: str, following=following) -> float:
                positions = [following[s.id] for s in graph.successors(node_id)
                             if s.id in following]
                return sum(positions) / len(positions) if positions else 1e9

            buckets[layer].sort(key=reverse_barycentre)
    return buckets


def compute_layout(graph: ExperimentGraph) -> Layout:
    """Place every node on a left-to-right layered grid."""
    layout = Layout()
    if not len(graph):
        return layout
    layers = _layer_assignment(graph)
    buckets = _order_within_layers(graph, layers)
    tallest = max((len(nodes) for nodes in buckets.values()), default=1)

    for layer, node_ids in buckets.items():
        # Centre each layer vertically against the tallest one.
        offset = (tallest - len(node_ids)) * ROW_GAP / 2.0
        for row, node_id in enumerate(node_ids):
            placed = Placed(graph.node(node_id), layer, row)
            placed.x = MARGIN + layer * LAYER_GAP
            placed.y = MARGIN + offset + row * ROW_GAP
            layout.placed[node_id] = placed

    layout.width = MARGIN * 2 + (max(buckets) + 1) * LAYER_GAP - (LAYER_GAP - NODE_WIDTH)
    layout.height = MARGIN * 2 + tallest * ROW_GAP
    return layout


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _embed_json(payload: Any) -> str:
    """Serialise JSON for embedding inside a <script> element.

    A node label is spec-controlled -- it comes from a dataset name, a
    condition name, a metric name -- and a label containing ``</script>``
    would close the element early and let the rest of the label execute as
    HTML in anyone who opened the report. Escaping the three characters
    that can start a tag or an entity keeps the payload valid JSON while
    making that impossible.
    """
    escaped = json.dumps(payload)
    for character, sequence in (
        ("<", "\\u003c"),
        (">", "\\u003e"),
        ("&", "\\u0026"),
        ("\u2028", "\\u2028"),
        ("\u2029", "\\u2029"),
    ):
        escaped = escaped.replace(character, sequence)
    return escaped


def _truncate(text: str, limit: int = 24) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _detail(node: Node) -> str:
    """The tooltip body: the handful of attributes worth showing."""
    interesting = ("role", "digest", "weights_digest", "commit", "entrypoint",
                   "precision", "seeds", "count", "succeeded", "failed", "value",
                   "n", "test", "p_value", "verdict", "research_question",
                   "n_examples", "status")
    parts = [f"{node.kind}: {node.label}"]
    for key in interesting:
        value = node.attributes.get(key)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, float):
            value = f"{value:.6g}"
        elif isinstance(value, list):
            value = ", ".join(str(v) for v in value[:8])
        parts.append(f"{key}: {str(value)[:120]}")
    return "\n".join(parts)


STYLE = """
.fabric-graph { position: relative; border: 1px solid var(--line, #e1e0d9);
  border-radius: 10px; background: var(--viz-surface, #fcfcfb); overflow: hidden; }
.fabric-graph .controls { display: flex; flex-wrap: wrap; gap: 6px; align-items: center;
  padding: 8px 10px; border-bottom: 1px solid var(--line, #e1e0d9);
  font: 12px system-ui, -apple-system, 'Segoe UI', sans-serif;
  color: var(--ink-2, #52514e); }
.fabric-graph .controls button { font: inherit; cursor: pointer; padding: 3px 9px;
  border: 1px solid var(--line, #e1e0d9); border-radius: 6px;
  background: var(--surface, #fff); color: inherit; }
.fabric-graph .controls button:hover { border-color: var(--viz-s1, #2a78d6);
  color: var(--viz-s1, #2a78d6); }
.fabric-graph .controls .hint { margin-left: auto; color: var(--muted, #898781); }
.fabric-graph .stage { cursor: grab; display: block; touch-action: none; }
.fabric-graph .stage.dragging { cursor: grabbing; }
.fabric-graph .node rect { stroke: var(--viz-surface, #fcfcfb); stroke-width: 1.5;
  rx: 7; transition: opacity .12s ease; }
.fabric-graph .node text { font: 11px system-ui, -apple-system, 'Segoe UI', sans-serif;
  fill: #fff; pointer-events: none; }
.fabric-graph .node .kind { font-size: 9px; opacity: .82; }
.fabric-graph .node { cursor: pointer; }
.fabric-graph .edge { fill: none; stroke: var(--viz-axis, #c3c2b7); stroke-width: 1.4;
  transition: opacity .12s ease, stroke .12s ease; }
.fabric-graph.has-selection .node { opacity: .16; }
.fabric-graph.has-selection .edge { opacity: .07; }
.fabric-graph.has-selection .node.lit { opacity: 1; }
.fabric-graph.has-selection .edge.lit { opacity: 1; stroke: var(--viz-s1, #2a78d6);
  stroke-width: 2.1; }
.fabric-graph.has-selection .node.root rect { stroke: var(--viz-ink, #0b0b0b);
  stroke-width: 2.5; }
.fabric-graph .tip { position: absolute; pointer-events: none; opacity: 0;
  transition: opacity .1s ease; max-width: 320px; white-space: pre-line;
  background: var(--ink, #0b0b0b); color: var(--viz-surface, #fcfcfb);
  padding: 7px 9px; border-radius: 7px; font: 11.5px/1.45 ui-monospace,
  SFMono-Regular, Menlo, Consolas, monospace; z-index: 5; }
.fabric-graph .caption { padding: 8px 10px; border-top: 1px solid var(--line, #e1e0d9);
  font: 12px system-ui, -apple-system, 'Segoe UI', sans-serif;
  color: var(--ink-2, #52514e); }
"""

SCRIPT = """
(function () {
  var root = document.getElementById(ROOT_ID);
  if (!root || root.dataset.ready) { return; }
  root.dataset.ready = '1';
  var stage = root.querySelector('.stage');
  var canvas = root.querySelector('.canvas');
  var tip = root.querySelector('.tip');
  var data = JSON.parse(root.querySelector('.graph-data').textContent);

  var view = { x: 0, y: 0, k: 1 };
  function apply() {
    canvas.setAttribute('transform',
      'translate(' + view.x + ',' + view.y + ') scale(' + view.k + ')');
  }

  var dragging = false, lastX = 0, lastY = 0;
  stage.addEventListener('pointerdown', function (e) {
    dragging = true; lastX = e.clientX; lastY = e.clientY;
    stage.classList.add('dragging'); stage.setPointerCapture(e.pointerId);
  });
  stage.addEventListener('pointermove', function (e) {
    if (!dragging) { return; }
    view.x += e.clientX - lastX; view.y += e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY; apply();
  });
  function endDrag(e) {
    dragging = false; stage.classList.remove('dragging');
    if (e.pointerId !== undefined && stage.hasPointerCapture(e.pointerId)) {
      stage.releasePointerCapture(e.pointerId);
    }
  }
  stage.addEventListener('pointerup', endDrag);
  stage.addEventListener('pointercancel', endDrag);
  stage.addEventListener('wheel', function (e) {
    e.preventDefault();
    var box = stage.getBoundingClientRect();
    var mx = e.clientX - box.left, my = e.clientY - box.top;
    var factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
    var next = Math.min(3, Math.max(0.25, view.k * factor));
    view.x = mx - (mx - view.x) * (next / view.k);
    view.y = my - (my - view.y) * (next / view.k);
    view.k = next; apply();
  }, { passive: false });

  function clear() {
    root.classList.remove('has-selection');
    root.querySelectorAll('.lit').forEach(function (el) { el.classList.remove('lit'); });
    root.querySelectorAll('.root').forEach(function (el) { el.classList.remove('root'); });
  }

  // The fabric why query, with a mouse: light the clicked node, everything it
  // descends from, and everything that descends from it.
  function select(id) {
    clear();
    var lineage = {};
    lineage[id] = true;
    (data.ancestors[id] || []).forEach(function (n) { lineage[n] = true; });
    (data.descendants[id] || []).forEach(function (n) { lineage[n] = true; });
    root.classList.add('has-selection');
    root.querySelectorAll('.node').forEach(function (el) {
      if (lineage[el.dataset.id]) { el.classList.add('lit'); }
    });
    root.querySelector('.node[data-id="' + CSS.escape(id) + '"]').classList.add('root');
    root.querySelectorAll('.edge').forEach(function (el) {
      if (lineage[el.dataset.source] && lineage[el.dataset.target]) {
        el.classList.add('lit');
      }
    });
    var counted = Object.keys(lineage).length;
    root.querySelector('.selection').textContent =
      data.labels[id] + ' \\u2014 ' + counted + ' of ' + data.total + ' nodes on this path';
  }

  root.querySelectorAll('.node').forEach(function (el) {
    el.addEventListener('click', function (e) {
      e.stopPropagation();
      select(el.dataset.id);
    });
    el.addEventListener('pointerenter', function (e) {
      tip.textContent = data.details[el.dataset.id] || '';
      tip.style.opacity = '1';
      var box = root.getBoundingClientRect();
      tip.style.left = Math.min(e.clientX - box.left + 14, box.width - 330) + 'px';
      tip.style.top = (e.clientY - box.top + 14) + 'px';
    });
    el.addEventListener('pointerleave', function () { tip.style.opacity = '0'; });
  });

  stage.addEventListener('click', function () {
    clear();
    root.querySelector('.selection').textContent = SELECTION_HINT;
  });
  root.querySelector('.reset').addEventListener('click', function () {
    view = { x: 0, y: 0, k: 1 }; apply(); clear();
    root.querySelector('.selection').textContent = SELECTION_HINT;
  });
  root.querySelector('.fit').addEventListener('click', function () {
    var box = stage.getBoundingClientRect();
    var k = Math.min(box.width / data.width, box.height / data.height, 1);
    view = { x: (box.width - data.width * k) / 2, y: (box.height - data.height * k) / 2, k: k };
    apply();
  });
  apply();
})();
"""


def render(graph: ExperimentGraph, *, element_id: str = "fabric-graph",
           caption: str = "", height: int = 520,
           full_node_count: int | None = None) -> str:
    """Render an interactive provenance graph as a self-contained fragment."""
    if not len(graph):
        return ""
    layout = compute_layout(graph)
    if not layout.placed:
        return ""

    ancestors = {n.id: [a.id for a in graph.ancestors(n.id)] for n in graph}
    descendants = {n.id: [d.id for d in graph.descendants(n.id)] for n in graph}
    payload = {
        "ancestors": ancestors,
        "descendants": descendants,
        "labels": {n.id: f"{n.kind}: {n.label}" for n in graph},
        "details": {n.id: _detail(n) for n in graph},
        "total": len(graph),
        "width": layout.width,
        "height": layout.height,
    }

    parts: list[str] = [f'<div class="fabric-graph" id="{_esc(element_id)}">']
    parts.append('<div class="controls">')
    parts.append('<button type="button" class="fit">Fit</button>')
    parts.append('<button type="button" class="reset">Reset</button>')
    hint = "Click a node to light up its lineage. Drag to pan, scroll to zoom."
    parts.append(f'<span class="selection">{_esc(hint)}</span>')
    parts.append('<span class="hint">interactive</span>')
    parts.append("</div>")

    # The viewBox follows the content so nothing is clipped, and the element
    # height is clamped so a tall graph does not push the rest of the report
    # off the page. Pan and zoom cover the difference.
    drawn_height = max(280.0, min(float(height), layout.height + 24))
    parts.append(
        f'<svg class="stage" viewBox="0 0 {layout.width:.0f} {layout.height:.0f}" '
        f'preserveAspectRatio="xMidYMid meet" '
        f'width="100%" height="{drawn_height:.0f}" '
        f'xmlns="http://www.w3.org/2000/svg" role="img" '
        f'aria-label="Provenance graph with {len(graph)} nodes">'
    )
    parts.append('<g class="canvas">')

    for edge in graph.edges:
        source = layout.placed.get(edge.source)
        target = layout.placed.get(edge.target)
        if source is None or target is None:
            continue
        x1 = source.x + NODE_WIDTH
        y1 = source.y + NODE_HEIGHT / 2
        x2 = target.x
        y2 = target.y + NODE_HEIGHT / 2
        midpoint = (x1 + x2) / 2
        path = f"M{x1:.1f},{y1:.1f} C{midpoint:.1f},{y1:.1f} {midpoint:.1f},{y2:.1f} {x2:.1f},{y2:.1f}"
        parts.append(
            f'<path class="edge" d="{path}" data-source="{_esc(edge.source)}" '
            f'data-target="{_esc(edge.target)}"><title>{_esc(edge.relation)}</title></path>'
        )

    for node_id, placed in layout.placed.items():
        colour, kind_label = KIND_STYLE.get(placed.node.kind,
                                            ("var(--viz-muted, #898781)", placed.node.kind))
        parts.append(
            f'<g class="node" data-id="{_esc(node_id)}" data-kind="{_esc(placed.node.kind)}">'
            f'<rect x="{placed.x:.1f}" y="{placed.y:.1f}" width="{NODE_WIDTH}" '
            f'height="{NODE_HEIGHT}" fill="{colour}"/>'
            f'<text x="{placed.x + 9:.1f}" y="{placed.y + 14:.1f}" class="kind">'
            f'{_esc(kind_label)}</text>'
            f'<text x="{placed.x + 9:.1f}" y="{placed.y + 26:.1f}">'
            f'{_esc(_truncate(placed.node.label))}</text>'
            f"</g>"
        )

    parts.append("</g></svg>")
    parts.append('<div class="tip"></div>')
    parts.append(
        '<script type="application/json" class="graph-data" style="display:none">'
        + _embed_json(payload)
        + "</script>"
    )

    if caption:
        note = caption
        if full_node_count and full_node_count > len(graph):
            note += (f" This view folds each condition's replications into one node: "
                     f"{len(graph)} shown of {full_node_count} in the full graph, which "
                     "stays queryable with fabric why and exportable with fabric graph.")
        parts.append(f'<div class="caption">{_esc(note)}</div>')
    parts.append("</div>")

    script = (
        SCRIPT
        .replace("ROOT_ID", json.dumps(element_id))
        .replace("SELECTION_HINT", json.dumps(hint))
    )
    parts.append(f"<script>{script}</script>")
    return "".join(parts)


def render_page(graph: ExperimentGraph, *, title: str = "Provenance graph",
                caption: str = "", full_node_count: int | None = None) -> str:
    """A standalone HTML page, for ``fabric graph --format html``."""
    from fabric.reporting.html import STYLESHEET

    body = render(graph, caption=caption, height=640, full_node_count=full_node_count)
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_esc(title)}</title>\n"
        f"<style>{STYLESHEET}{STYLE}</style>\n</head>\n<body>\n"
        f'<div class="wrap"><h1>{_esc(title)}</h1>{body}</div>\n</body>\n</html>\n'
    )
