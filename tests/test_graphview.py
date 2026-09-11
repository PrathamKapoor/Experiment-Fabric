"""Collapsing the provenance graph, and drawing it.

The graph is the project's core abstraction, so the view of it has to be
honest in two specific ways: it must not draw a relationship it cannot
support, and it must say what it left out.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from fabric.core.graph import ExperimentGraph
from fabric.reporting import graphview


def experiment_shaped_graph(conditions: int = 3, runs_each: int = 4) -> ExperimentGraph:
    """A graph with the shape a real experiment produces."""
    graph = ExperimentGraph("demo")
    graph.add_node("exp", "experiment", "demo")
    graph.add_node("ds", "dataset", "data@v1", digest="sha256:abc")
    for index in range(conditions):
        condition = f"cond{index}"
        graph.add_node(condition, "condition", f"arm{index}", role="treatment")
        graph.add_edge("exp", condition, "contains")
        graph.add_edge("ds", condition, "uses")
        for seed in range(runs_each):
            run = f"{condition}-run{seed}"
            graph.add_node(run, "run", f"arm{index} seed={seed}",
                           status="completed", seed=seed)
            graph.add_edge(condition, run, "replicates")
            for metric, value in (("accuracy", 0.90 + 0.01 * seed), ("latency", 10.0)):
                node = f"{run}:{metric}"
                graph.add_node(node, "metric", f"{metric}={value}", metric=metric,
                               value=value, run=run)
                graph.add_edge(run, node, "produces")
            artifact = f"{run}/artifact/out.json"
            graph.add_node(artifact, "artifact", "out.json", digest="sha256:x", bytes=10)
            graph.add_edge(run, artifact, "produces")
    return graph


# --------------------------------------------------------------------------
# collapsing
# --------------------------------------------------------------------------


def test_collapsing_folds_replications_into_one_node():
    summary = experiment_shaped_graph().collapsed()
    assert not summary.nodes_of_kind("run")
    groups = summary.nodes_of_kind("runs")
    assert len(groups) == 3
    assert all(group.attributes["count"] == 4 for group in groups)
    assert all("4 runs" in group.label for group in groups)


def test_collapsing_keeps_the_structural_relationships():
    summary = experiment_shaped_graph().collapsed()
    reachable = {node.kind for node in summary.descendants("ds")}
    assert {"condition", "runs", "metric"} <= reachable


def test_collapsing_can_restrict_which_metrics_survive():
    graph = experiment_shaped_graph()
    assert len(graph.collapsed().nodes_of_kind("metric")) == 6  # 3 arms x 2 metrics
    restricted = graph.collapsed(metrics=["accuracy"])
    assert len(restricted.nodes_of_kind("metric")) == 3
    assert all(node.attributes["metric"] == "accuracy"
               for node in restricted.nodes_of_kind("metric"))


def test_collapsing_averages_the_folded_metric():
    summary = experiment_shaped_graph(conditions=1, runs_each=4).collapsed(
        metrics=["accuracy"])
    node = summary.nodes_of_kind("metric")[0]
    assert node.attributes["n"] == 4
    assert node.attributes["value"] == pytest.approx(0.915)  # mean of .90 .91 .92 .93


def test_artifacts_are_summarised_only_when_asked_for():
    graph = experiment_shaped_graph()
    assert not graph.collapsed().nodes_of_kind("artifact")
    with_artifacts = graph.collapsed(include_artifacts=True)
    nodes = with_artifacts.nodes_of_kind("artifact")
    assert len(nodes) == 3 and all(n.attributes["count"] == 4 for n in nodes)


def test_collapsing_drops_orphans_rather_than_drawing_them():
    graph = experiment_shaped_graph()
    graph.add_node("analysis", "analysis", "statistical analysis")
    summary = graph.collapsed()
    assert "analysis" not in summary


def test_collapsing_stays_acyclic():
    summary = experiment_shaped_graph().collapsed(metrics=["accuracy"])
    assert len(summary.topological_order()) == len(summary)


def test_collapsing_is_a_large_reduction():
    graph = experiment_shaped_graph(conditions=12, runs_each=10)
    summary = graph.collapsed(metrics=["accuracy"])
    assert len(graph) > 350
    assert len(summary) < 45, "the whole point is that the picture stays legible"


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------


def small_graph() -> ExperimentGraph:
    graph = ExperimentGraph("demo")
    graph.add_node("exp", "experiment", "demo")
    graph.add_node("ds", "dataset", "data@v1", digest="sha256:abc")
    graph.add_node("cond", "condition", "arm", role="treatment")
    graph.add_node("runs", "runs", "10 runs", count=10)
    graph.add_node("metric", "metric", "accuracy=0.9", metric="accuracy", value=0.9)
    graph.add_edge("exp", "cond", "contains")
    graph.add_edge("ds", "cond", "uses")
    graph.add_edge("cond", "runs", "replicates")
    graph.add_edge("runs", "metric", "produces")
    return graph


def test_layout_places_every_node_and_points_edges_rightwards():
    graph = small_graph()
    layout = graphview.compute_layout(graph)
    assert set(layout.placed) == {node.id for node in graph}
    for edge in graph.edges:
        assert layout.placed[edge.source].layer < layout.placed[edge.target].layer, (
            "a left-pointing edge would misrepresent the direction of causation")


def test_layout_of_an_empty_graph_is_empty():
    assert graphview.compute_layout(ExperimentGraph("empty")).placed == {}


def test_layers_do_not_overlap_vertically():
    layout = graphview.compute_layout(experiment_shaped_graph().collapsed())
    by_layer: dict[int, list[float]] = {}
    for placed in layout.placed.values():
        by_layer.setdefault(placed.layer, []).append(placed.y)
    for positions in by_layer.values():
        ordered = sorted(positions)
        gaps = [b - a for a, b in pairwise(ordered)]
        assert all(gap >= graphview.NODE_HEIGHT for gap in gaps)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def test_the_view_is_self_contained_and_escapes_labels():
    graph = small_graph()
    graph.add_node("evil", "metric", "<script>alert(1)</script>", metric="x", value=1.0)
    graph.add_edge("runs", "evil", "produces")
    fragment = graphview.render(graph, caption="c", full_node_count=100)
    assert "<script>alert(1)</script>" not in fragment
    assert "&lt;script&gt;" in fragment
    assert "http://" not in fragment.replace("http://www.w3.org/2000/svg", "")


def test_the_lineage_data_precedes_the_script_that_reads_it():
    fragment = graphview.render(small_graph(), caption="c")
    assert fragment.index("graph-data") < fragment.rindex("<script>")


def test_the_view_says_what_it_folded():
    fragment = graphview.render(small_graph(), caption="c", full_node_count=913)
    assert "5 shown of 913" in fragment
    assert "fabric why" in fragment


def test_the_view_carries_the_lineage_of_every_node():
    import json
    import re

    graph = small_graph()
    fragment = graphview.render(graph, caption="c")
    payload = json.loads(
        re.search(r'class="graph-data" style="display:none">(.*?)</script>',
                  fragment, re.S).group(1))
    # Clicking the metric must light the whole chain back to the dataset.
    assert set(payload["ancestors"]["metric"]) == {"runs", "cond", "ds", "exp"}
    assert payload["descendants"]["ds"] == payload["descendants"]["ds"]
    assert "metric" in payload["descendants"]["ds"]
    assert payload["total"] == len(graph)


def test_an_empty_graph_renders_nothing_rather_than_a_broken_box():
    assert graphview.render(ExperimentGraph("empty")) == ""


def test_the_standalone_page_is_a_whole_document():
    page = graphview.render_page(small_graph(), title="P", caption="c")
    assert page.startswith("<!doctype html>")
    assert page.rstrip().endswith("</html>")
    assert "fabric-graph" in page
    assert "prefers-color-scheme" in page
