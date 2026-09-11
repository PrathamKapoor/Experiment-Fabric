"""Analysis, reproducibility, reporting, certificates and bundles."""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ElementTree

import pytest

from fabric import bundle as bundle_module
from fabric.analysis.engine import analyse
from fabric.analysis.reproducibility import (
    calibrated_stability,
    decision_stability,
    null_calibration,
)
from fabric.analysis.results import ResultSet
from fabric.core.provenance import build_provenance
from fabric.core.registry import Registry
from fabric.core.spec import ExperimentSpec
from fabric.core.store import Workspace
from fabric.execution.planner import plan_experiment
from fabric.execution.runner import ExecutionOptions, Runner
from fabric.reporting import builder, html, latex, markdown, plots
from fabric.reporting.certificate import issue, verify_certificate

SPEC = """
experiment:
  name: analysis-fixture
research_question: Do the components contribute anything measurable?
algorithm:
  name: method
  entrypoint: python:tests.helpers:ablatable
components: [alpha, beta]
metrics:
  - name: accuracy
    direction: maximize
    primary: true
    tolerance: 0.02
baselines:
  - name: control
    algorithm:
      name: control
      entrypoint: python:tests.helpers:deterministic
ablations: [remove_alpha, remove_beta]
seeds:
  values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
reproducibility:
  enabled: true
  determinism_replicates: 2
  perturbations:
    environment:
      values:
        - label: single-threaded
          OMP_NUM_THREADS: "1"
    configuration:
      values:
        - label: shifted
          scale: 2.0
execution:
  isolation: inline
  max_workers: 1
"""


@pytest.fixture(scope="module")
def executed(tmp_path_factory):
    """Run the fixture experiment once and reuse it across the module."""
    root = tmp_path_factory.mktemp("analysis")
    cwd = os.getcwd()
    os.chdir(root)
    try:
        workspace = Workspace.initialise(str(root))
        spec = ExperimentSpec.from_yaml(SPEC)
        plan = plan_experiment(spec, workspace, root=str(root))
        workspace.create_experiment_dir(plan.experiment.id)
        workspace.write_text(plan.experiment.id, "experiment.yaml", SPEC)
        Runner(workspace, plan, ExecutionOptions(isolation="inline"),
               root=str(root)).execute()
        runs = workspace.load_runs(plan.experiment.id)
        analysis = analyse(plan, runs, stored_spec_digest=spec.digest)
        yield workspace, plan, runs, analysis
    finally:
        os.chdir(cwd)


# --------------------------------------------------------------------------
# result alignment
# --------------------------------------------------------------------------


def test_conditions_are_aligned_by_seed(executed):
    _workspace, plan, runs, analysis = executed
    results = analysis.results
    left, right, seeds = results.aligned("proposed", "control", "accuracy")
    assert len(left) == len(right) == len(seeds) == 10
    assert seeds == sorted(seeds)
    assert results.can_pair("proposed", "control", "accuracy") is True


def test_a_missing_seed_drops_from_both_sides(executed):
    _workspace, plan, runs, analysis = executed
    from dataclasses import replace

    from fabric.core.entities import RunStatus

    damaged = [replace(r, status=RunStatus.FAILED, metrics={})
               if (r.condition_name == "control" and r.seed == 5) else r
               for r in runs]
    results = ResultSet.build(damaged, plan.conditions, plan.spec.metrics)
    left, right, seeds = results.aligned("proposed", "control", "accuracy")
    assert len(seeds) == 9 and 5 not in seeds
    assert len(left) == len(right) == 9


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------


def test_headline_does_not_overstate(executed):
    *_rest, analysis = executed
    assert analysis.primary_metric == "accuracy"
    assert "proposed" in analysis.headline
    assert "seeds" in analysis.headline


def test_baseline_comparison_is_tested_and_corrected(executed):
    *_rest, analysis = executed
    comparisons = analysis.baselines.for_metric("accuracy")
    assert len(comparisons) == 1
    comparison = comparisons[0]
    assert comparison.test.test != "none"
    assert comparison.adjusted_p is not None
    assert comparison.decision.reason


def test_every_test_records_its_assumptions(executed):
    *_rest, analysis = executed
    for comparison in analysis.baselines.comparisons:
        assert comparison.test.assumptions
        names = {a.name for a in comparison.test.assumptions}
        assert any("independent" in n for n in names)


def test_ablation_ranking_finds_the_load_bearing_component(executed):
    *_rest, analysis = executed
    ranking = analysis.ablations.ranking("accuracy")
    assert ranking, "ablations should have been analysed"
    names = [name for name, _delta, _sig in ranking]
    assert names[0] == "alpha"  # alpha contributes 0.20, beta only 0.05
    assert analysis.ablations.load_bearing("accuracy")[0] == "alpha"


def test_ablation_deltas_have_the_right_sign(executed):
    *_rest, analysis = executed
    for name, delta, _significant in analysis.ablations.ranking("accuracy"):
        assert delta > 0, f"removing {name} should reduce accuracy"


def test_determinism_is_actually_tested(executed):
    *_rest, analysis = executed
    assert analysis.determinism.tested is True
    assert analysis.determinism.deterministic is True
    assert "Deterministic" in analysis.determinism.describe()


def test_reproducibility_measures_every_declared_axis(executed):
    *_rest, analysis = executed
    axes = {a.axis for a in analysis.reproducibility.axes}
    assert {"environment", "configuration"} <= axes
    environment = [a for a in analysis.reproducibility.axes if a.axis == "environment"][0]
    configuration = [a for a in analysis.reproducibility.axes if a.axis == "configuration"][0]
    # Thread counts do not change this pipeline; the scale parameter does.
    assert environment.stability == pytest.approx(1.0)
    assert environment.tolerance_verdict == "within"
    assert configuration.stability < environment.stability


def test_seed_axis_is_a_noise_floor_not_a_score(executed):
    *_rest, analysis = executed
    seed = analysis.reproducibility.seed
    assert seed is not None
    assert seed.n_seeds == 10
    assert seed.std > 0
    assert "not scored" in seed.to_dict()["role"]


def test_limitations_are_generated(executed):
    *_rest, analysis = executed
    assert analysis.limitations
    assert any(any(word in text.lower() for word in ("replication", "seed", "tolerance"))
               for text in analysis.limitations)


def test_integrity_checks_run(executed):
    *_rest, analysis = executed
    codes = {f.code for f in analysis.integrity.findings}
    assert "seeds.aligned" in codes
    assert analysis.integrity.passed is True


def test_analysis_serialises_to_json(executed):
    *_rest, analysis = executed
    text = json.dumps(analysis.to_dict(), default=str)
    assert len(text) > 1000
    restored = json.loads(text)
    assert restored["primary_metric"] == "accuracy"
    assert restored["reproducibility"]["method"]


# --------------------------------------------------------------------------
# the stability metric itself
# --------------------------------------------------------------------------


def test_identical_distributions_score_near_one():
    import random
    import statistics

    scores = []
    for trial in range(40):
        rng = random.Random(trial)
        a = [rng.gauss(0.9, 0.02) for _ in range(12)]
        b = [rng.gauss(0.9, 0.02) for _ in range(12)]
        from fabric.statistics.effect_size import cliffs_delta

        delta = cliffs_delta(a, b).value
        calibration = null_calibration(a, b, observed=delta, resamples=150, seed=5)
        scores.append(calibrated_stability(delta, calibration.expectation))
    assert statistics.median(scores) > 0.9
    assert statistics.mean(scores) > 0.8


def test_a_large_shift_scores_near_zero():
    import random

    rng = random.Random(3)
    a = [rng.gauss(0.9, 0.02) for _ in range(12)]
    b = [rng.gauss(0.96, 0.02) for _ in range(12)]
    from fabric.statistics.effect_size import cliffs_delta

    delta = cliffs_delta(b, a).value
    calibration = null_calibration(b, a, observed=delta, resamples=150, seed=5)
    assert calibrated_stability(delta, calibration.expectation) < 0.15
    assert calibration.p_value < 0.05


def test_stability_is_bounded():
    assert calibrated_stability(0.0, 0.3) == 1.0
    assert calibrated_stability(1.0, 0.3) == 0.0
    assert 0.0 <= calibrated_stability(0.6, 0.3) <= 1.0


def test_decision_stability_flags_a_floor_limited_design():
    treatment = [0.75 + i * 0.001 for i in range(10)]
    baseline = [0.50 + i * 0.001 for i in range(10)]
    stability, detail = decision_stability(treatment, baseline, paired=True, resamples=60)
    assert detail["full_verdict"] == "better"
    assert detail["floor_limited"] is True
    assert detail["minimum_attainable_p"] > 0.05


# --------------------------------------------------------------------------
# charts
# --------------------------------------------------------------------------


def test_every_chart_is_well_formed_svg():
    charts = {
        "forest": plots.forest([
            {"label": "a & b", "estimate": 0.2, "low": 0.1, "high": 0.3,
             "verdict": "improvement"},
            {"label": "c", "estimate": -0.01, "low": -0.05, "high": 0.03,
             "verdict": "inconclusive"},
        ]),
        "strip": plots.strip([{"label": "x <tag>", "values": [1.0, 1.1, 0.9],
                               "emphasise": True}]),
        "diverging": plots.diverging_bars([
            {"label": "alpha", "value": 0.2, "significant": True},
            {"label": "beta", "value": -0.01, "significant": False},
        ]),
        "stability": plots.stability_bars([
            {"label": "env", "stability": 1.0, "low": 0.9, "high": 1.0,
             "verdict": "within"},
        ]),
        "sweep": plots.sweep_line([{"label": "1", "mean": 0.5, "std": 0.01},
                                   {"label": "2", "mean": 0.6, "std": 0.02}]),
        "scatter": plots.cost_scatter([
            {"label": "a", "x": 1.0, "y": 0.5, "role": "treatment"},
            {"label": "b", "x": 2.0, "y": 0.4, "role": "baseline"},
        ]),
    }
    for name, svg in charts.items():
        assert svg.startswith("<svg"), name
        ElementTree.fromstring(svg)  # raises if the SVG is malformed


def test_charts_degrade_to_nothing_rather_than_crashing_on_empty_data():
    assert plots.forest([]) == ""
    assert plots.strip([]) == ""
    assert plots.sweep_line([{"label": "1", "mean": 0.5}]) == ""


def test_chart_labels_are_escaped():
    svg = plots.forest([{"label": '<script>&"', "estimate": 0.1, "low": 0.0, "high": 0.2}])
    assert "<script>" not in svg
    ElementTree.fromstring(svg)


# --------------------------------------------------------------------------
# reports
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def document(executed):
    workspace, plan, runs, analysis = executed
    graph = build_provenance(plan, runs, analysis)
    return builder.build_report(analysis, plan, runs, graph=graph)


def test_report_covers_every_required_section(document):
    titles = {section.title for _depth, section in document.walk()}
    for expected in ("Summary", "Experimental setup", "Results", "Statistical analysis",
                     "Ablation study", "Failure analysis", "Reproducibility",
                     "Evaluation integrity", "Limitations", "Provenance"):
        assert expected in titles


def test_markdown_report_renders(document):
    text = markdown.render(document)
    assert text.startswith("# analysis-fixture")
    assert "## Contents" in text
    assert "| :---" in text  # a table was rendered
    assert "Limitations" in text


def test_html_report_is_self_contained(document):
    text = html.render(document)
    assert text.startswith("<!doctype html>")
    assert "<svg" in text
    assert "prefers-color-scheme: dark" in text
    assert "http://" not in text.replace("http://www.w3.org/2000/svg", "")


def test_latex_report_renders_and_escapes(document):
    text = latex.render(document)
    assert r"\documentclass" in text and r"\end{document}" in text
    assert r"\toprule" in text
    assert "_" not in text.replace(r"\_", "").replace("\n", " ").split(r"\begin{verbatim}")[0]


def test_latex_tables_are_individually_extractable(document):
    snippets = latex.render_tables(document)
    assert snippets
    for label, snippet in snippets.items():
        assert snippet.startswith(r"\begin{table}")
        assert rf"\label{{tab:{label}}}" in snippet


def test_json_report_round_trips(document):
    data = json.loads(json.dumps(document.to_dict(), default=str))
    assert data["title"] == "analysis-fixture"
    assert data["sections"]


# --------------------------------------------------------------------------
# certificates
# --------------------------------------------------------------------------


def test_certificate_records_what_was_verified(executed):
    workspace, plan, runs, analysis = executed
    certificate = issue(analysis, plan, runs, Registry(workspace),
                        workspace_root=workspace.root)
    names = {c.name for c in certificate.checks}
    for expected in ("code", "dataset", "environment", "hardware", "seeds",
                     "baselines", "ablations", "statistics", "artifacts",
                     "provenance", "determinism", "reproducibility", "integrity"):
        assert expected in names
    assert certificate.verdict in ("PASS", "PASS WITH GAPS")
    assert certificate.certificate_id.startswith("EXP-CERT-")


def test_certificate_detects_tampering(executed):
    workspace, plan, runs, analysis = executed
    certificate = issue(analysis, plan, runs, Registry(workspace),
                        workspace_root=workspace.root)
    data = certificate.to_dict()
    assert verify_certificate(data)[0] is True
    data["verdict"] = "PASS"
    data["checks"][0]["status"] = "verified"
    ok, message = verify_certificate(data)
    assert ok is False and "modified" in message


# --------------------------------------------------------------------------
# bundles
# --------------------------------------------------------------------------


def test_bundle_exports_verifies_and_detects_edits(executed, tmp_path):
    workspace, plan, runs, analysis = executed
    graph = build_provenance(plan, runs, analysis)
    certificate = issue(analysis, plan, runs, Registry(workspace),
                        workspace_root=workspace.root)
    destination = tmp_path / "bundle"
    manifest = bundle_module.export(
        workspace, plan, runs, str(destination),
        analysis=analysis, provenance=graph, certificate=certificate,
        report_markdown="# report\n", include_artifacts=True,
    )
    assert manifest.counts["runs"] == len(runs)
    for name in ("experiment.yaml", "results.json", "analysis.json", "provenance.json",
                 "certificate.json", "reproduce.sh", "README.md", "manifest.json"):
        assert (destination / name).is_file(), name

    result = bundle_module.verify(str(destination))
    assert result.ok is True and result.altered == [] and result.missing == []

    (destination / "results.json").write_text("{}", encoding="utf-8")
    tampered = bundle_module.verify(str(destination))
    assert tampered.ok is False
    assert "results.json" in tampered.altered


def test_bundle_load_exposes_the_recorded_runs(executed, tmp_path):
    workspace, plan, runs, analysis = executed
    destination = tmp_path / "bundle2"
    bundle_module.export(workspace, plan, runs, str(destination), analysis=analysis,
                         include_artifacts=False)
    loaded = bundle_module.load(str(destination))
    assert len(loaded["results"]["runs"]) == len(runs)
    assert loaded["manifest"]["experiment_id"] == plan.experiment.id


# --------------------------------------------------------------------------
# reproduction
# --------------------------------------------------------------------------


def test_reproduction_of_identical_runs_passes(executed):
    from fabric.reproduce import compare_runs

    _workspace, plan, runs, _analysis = executed
    report = compare_runs(plan, runs, runs, alpha=0.05)
    assert report.verdict == "PASS"
    headline = report.headline
    assert headline is not None
    assert headline.difference == pytest.approx(0.0)


def test_reproduction_reports_a_real_deviation(executed):
    from dataclasses import replace

    from fabric.reproduce import compare_runs

    _workspace, plan, runs, _analysis = executed
    shifted = [
        replace(r, metrics={k: v + 0.2 for k, v in r.metrics.items()}) if r.metrics else r
        for r in runs
    ]
    report = compare_runs(plan, runs, shifted, alpha=0.05)
    assert report.verdict in ("FAIL", "PASS WITH DEVIATION")
    assert report.headline.difference == pytest.approx(0.2, abs=1e-6)
    assert "REPRODUCTION COMPLETE" in report.render()


def test_an_experiment_without_baselines_still_reports_its_result(tmp_path, monkeypatch):
    """A headline must never claim there are no results when there are.

    With no baselines to compare against, the treatment resolution used to
    be skipped entirely, and the headline announced "No treatment condition
    produced results" above a table of successful runs.
    """
    from fabric.core.spec import ExperimentSpec
    from fabric.core.store import Workspace
    from fabric.execution.planner import plan_experiment
    from fabric.execution.runner import ExecutionOptions, Runner

    monkeypatch.chdir(tmp_path)
    workspace = Workspace.initialise(str(tmp_path))
    spec = ExperimentSpec.from_yaml("""
experiment:
  name: no-baselines
research_question: Does it report a result with nothing to compare against?
algorithm:
  name: solo
  entrypoint: python:tests.helpers:deterministic
metrics: [accuracy]
seeds: {values: [1, 2, 3, 4, 5]}
execution: {isolation: inline}
""")
    plan = plan_experiment(spec, workspace, root=str(tmp_path))
    workspace.create_experiment_dir(plan.experiment.id)
    Runner(workspace, plan, ExecutionOptions(isolation="inline"),
           root=str(tmp_path)).execute()
    analysis = analyse(plan, workspace.load_runs(plan.experiment.id))

    assert "No treatment condition" not in analysis.headline
    assert "proposed" in analysis.headline
    assert "0.5" in analysis.headline
    assert "no baseline to compare against" in analysis.headline


def test_exact_reproduction_passes_without_a_declared_tolerance(executed):
    """Bit-identical values are conclusive on their own.

    A metric with no tolerance used to be reported INCONCLUSIVE even when
    every seed reproduced its value exactly -- the strongest possible
    evidence recorded as the most uncertain verdict.
    """
    from dataclasses import replace

    from fabric.reproduce import compare_runs

    _workspace, plan, runs, _analysis = executed
    # Strip the tolerance from every metric, then reproduce identical runs.
    bare = replace(plan.spec, metrics=tuple(
        replace(m, tolerance=None) for m in plan.spec.metrics))
    bare_plan = replace(plan, spec=bare)

    report = compare_runs(bare_plan, runs, runs, alpha=0.05)
    assert report.verdict == "PASS"
    assert all(m.exact for m in report.metrics)
    assert "exact" in report.render()
    assert not any("tolerance was declared" in note for note in report.notes)


def test_a_non_exact_reproduction_without_tolerance_stays_honest(executed):
    from dataclasses import replace

    from fabric.reproduce import compare_runs

    _workspace, plan, runs, _analysis = executed
    bare = replace(plan.spec, metrics=tuple(
        replace(m, tolerance=None) for m in plan.spec.metrics))
    bare_plan = replace(plan, spec=bare)
    nudged = [
        replace(r, metrics={k: v + 1e-6 for k, v in r.metrics.items()}) if r.metrics else r
        for r in runs
    ]
    report = compare_runs(bare_plan, runs, nudged, alpha=0.05)
    assert not any(m.exact for m in report.metrics)
    # The numbers genuinely differ, so this is not a reproduction; with no
    # declared tolerance nobody ever said how much drift was acceptable, and
    # the note has to say so rather than the verdict implying a threshold
    # that was never set.
    assert report.verdict == "FAIL"
    assert any("tolerance was declared" in note for note in report.notes)
