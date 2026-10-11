"""Every declared configuration field must actually change behaviour.

An audit found four fields that the specification parsed, stored and
serialised, and that nothing downstream ever read: ``confidence_level``,
``permutation_resamples``, ``baselines[].reference_value`` and
``execution.energy_watts``. A field like that is worse than a missing
one -- it silently tells the user their study was analysed at 99% when it
was analysed at 95%.

So each test here changes one field and asserts that a number visible to
the user changes with it. The point is not that the code path exists; it
is that the value arrives at the end of it.
"""

from __future__ import annotations

import os
from typing import ClassVar

import pytest

from fabric.analysis.engine import analyse
from fabric.analysis.reference import analyse_references
from fabric.core.spec import ExperimentSpec, SpecError
from fabric.core.store import Workspace
from fabric.execution.planner import plan_experiment
from fabric.execution.runner import ExecutionOptions, Runner
from fabric.reporting import builder, markdown
from fabric.statistics.compare import compare

SPEC = """
experiment:
  name: live-configuration
research_question: Do the declared analysis settings reach the analysis?
algorithm:
  name: method
  entrypoint: python:tests.helpers:ablatable
components: [alpha, beta]
metrics:
  - name: accuracy
    direction: maximize
    primary: true
baselines:
  - name: control
    algorithm:
      name: control
      entrypoint: python:tests.helpers:deterministic
  - name: published
    description: An arm whose number was also reported by someone else.
    source: "Fictional et al. 2024, Table 3"
    reference_value: {accuracy: 0.9}
    algorithm:
      name: control
      entrypoint: python:tests.helpers:deterministic
ablations: [remove_alpha]
seeds:
  values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
analysis:
  primary_metric: accuracy
  alpha: 0.05
  confidence_level: __LEVEL__
execution:
  isolation: inline
"""


def _analyse(root, level: float):
    """Run the fixture once at a given confidence level."""
    cwd = os.getcwd()
    os.chdir(root)
    try:
        workspace = Workspace.initialise(str(root))
        text = SPEC.replace("__LEVEL__", repr(float(level)))
        spec = ExperimentSpec.from_yaml(text)
        plan = plan_experiment(spec, workspace, root=str(root))
        workspace.create_experiment_dir(plan.experiment.id)
        workspace.write_text(plan.experiment.id, "experiment.yaml", text)
        Runner(workspace, plan, ExecutionOptions(isolation="inline"),
               root=str(root)).execute()
        runs = workspace.load_runs(plan.experiment.id)
        return plan, runs, analyse(plan, runs, stored_spec_digest=spec.digest)
    finally:
        os.chdir(cwd)


@pytest.fixture(scope="module")
def at_95(tmp_path_factory):
    return _analyse(tmp_path_factory.mktemp("level95"), 0.95)


@pytest.fixture(scope="module")
def at_99(tmp_path_factory):
    return _analyse(tmp_path_factory.mktemp("level99"), 0.99)


# --------------------------------------------------------------------------
# analysis.confidence_level
# --------------------------------------------------------------------------


def test_confidence_level_widens_every_interval(at_95, at_99):
    """The whole point: 99% intervals must be wider than 95% ones."""
    _p95, _r95, narrow = at_95
    _p99, _r99, wide = at_99

    assert narrow.confidence_level == 0.95
    assert wide.confidence_level == 0.99

    narrow_by_baseline = {c.baseline: c for c in narrow.baselines.for_metric("accuracy")}
    wide_by_baseline = {c.baseline: c for c in wide.baselines.for_metric("accuracy")}
    assert narrow_by_baseline, "fixture produced no comparisons"

    for name, tight in narrow_by_baseline.items():
        loose = wide_by_baseline[name]
        assert tight.confidence_level == 0.95
        assert loose.confidence_level == 0.99
        assert tight.difference_interval.level == 0.95
        assert loose.difference_interval.level == 0.99
        assert loose.difference_interval.width > tight.difference_interval.width
        # the point estimate must be untouched: only the interval moves
        assert loose.difference == pytest.approx(tight.difference)


def test_confidence_level_reaches_the_bootstrap_and_the_effect_sizes(at_95, at_99):
    tight = at_95[2].baselines.for_metric("accuracy")[0]
    loose = at_99[2].baselines.for_metric("accuracy")[0]
    assert tight.bootstrap_interval is not None
    assert loose.bootstrap_interval is not None
    assert loose.bootstrap_interval.level == 0.99
    assert loose.bootstrap_interval.width > tight.bootstrap_interval.width

    tight_g = tight.effect("hedges_g")
    loose_g = loose.effect("hedges_g")
    assert tight_g is not None and loose_g is not None
    if tight_g.interval and loose_g.interval:
        assert loose_g.interval.level == 0.99


def test_confidence_level_reaches_the_ablation_comparisons(at_99):
    effects = at_99[2].ablations.effects
    assert effects, "fixture produced no ablations"
    comparison = effects[0].comparisons["accuracy"]
    assert comparison.confidence_level == 0.99
    assert comparison.difference_interval.level == 0.99


def test_the_report_states_the_level_it_used(at_99):
    plan, runs, analysis = at_99
    document = builder.build_report(analysis, plan, runs)
    rendered = markdown.render(document)
    assert "99%" in rendered
    # and it must not still be labelling things with 1 - alpha
    assert "95% CI" not in rendered


def test_alpha_still_drives_the_decision_not_the_interval(at_99):
    """A 99% interval must not silently become a 0.01 test."""
    analysis = at_99[2]
    assert analysis.alpha == 0.05
    for comparison in analysis.baselines.comparisons:
        assert comparison.alpha == 0.05
        assert comparison.confidence_level == 0.99


def test_confidence_level_defaults_to_one_minus_alpha():
    spec = ExperimentSpec.from_yaml(
        "experiment: {name: d}\nresearch_question: q\n"
        "algorithm: {name: a, entrypoint: 'python:m:f'}\nmetrics: [accuracy]\n"
        "seeds: {values: [1, 2]}\nanalysis: {alpha: 0.01}\n"
    )
    assert spec.analysis.confidence_level == pytest.approx(0.99)


@pytest.mark.parametrize("bad", ["0", "1", "1.7", "-0.5"])
def test_an_impossible_confidence_level_is_refused_with_its_location(bad):
    text = (
        "experiment: {name: d}\nresearch_question: q\n"
        "algorithm: {name: a, entrypoint: 'python:m:f'}\nmetrics: [accuracy]\n"
        f"seeds: {{values: [1, 2]}}\nanalysis: {{confidence_level: {bad}}}\n"
    )
    with pytest.raises(SpecError) as raised:
        ExperimentSpec.from_yaml(text)
    message = str(raised.value)
    assert "confidence_level" in message
    assert "between 0 and 1" in message


def test_compare_rejects_an_impossible_level_directly():
    with pytest.raises(ValueError, match="confidence_level"):
        compare([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], metric="accuracy",
                confidence_level=1.5)


# --------------------------------------------------------------------------
# analysis.permutation_resamples
# --------------------------------------------------------------------------


# Two unpaired groups of eight, heavy-tailed enough that the normality
# check rejects and the selector reaches for a permutation test. Eight per
# side gives C(16,8) = 12870 label assignments -- above the 4096 free
# enumeration budget, so the configured resample count is what decides how
# the null actually gets explored.
UNPAIRED_A = [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 9.00]
UNPAIRED_B = [0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 9.10]


def test_the_configured_resample_count_reaches_the_permutation_procedure():
    """Not "is it stored" -- is it the number the test actually ran."""
    comparison = compare(UNPAIRED_A, UNPAIRED_B, metric="accuracy", paired=False,
                         permutation_resamples=1234, permutation_seed=7)
    assert comparison.decision.test == "permutation"
    assert comparison.decision.parameters["resamples"] == 1234
    assert comparison.decision.parameters["seed"] == 7
    # the detail line is written by the procedure itself, so finding the
    # number there proves it arrived rather than merely being recorded
    assert "1234 resamples" in comparison.test.method_detail
    assert comparison.test.extra["resamples"] == 1234
    assert comparison.test.extra["seed"] == 7


def test_a_large_enough_budget_promotes_the_test_to_exact_enumeration():
    """The budget is not decoration: it decides how the null is explored."""
    sampled = compare(UNPAIRED_A, UNPAIRED_B, metric="accuracy", paired=False,
                      permutation_resamples=200, permutation_seed=7)
    exact = compare(UNPAIRED_A, UNPAIRED_B, metric="accuracy", paired=False,
                    permutation_resamples=20000, permutation_seed=7)
    assert sampled.test.extra["exact"] is False
    assert "Monte-Carlo" in sampled.test.method_detail
    assert exact.test.extra["exact"] is True
    assert "exact enumeration over 12870" in exact.test.method_detail
    assert sampled.p_value != exact.p_value


def test_the_permutation_seed_makes_the_p_value_reproducible():
    kwargs = {"metric": "accuracy", "paired": False, "permutation_resamples": 500}
    first = compare(UNPAIRED_A, UNPAIRED_B, permutation_seed=99, **kwargs)
    again = compare(UNPAIRED_A, UNPAIRED_B, permutation_seed=99, **kwargs)
    other = compare(UNPAIRED_A, UNPAIRED_B, permutation_seed=100, **kwargs)
    assert first.p_value == again.p_value
    # a different stream may agree by chance, so the claim under test is
    # that the seed is honoured, not that the two disagree
    assert other.decision.parameters["seed"] == 100
    assert other.test.extra["seed"] == 100


def test_the_specification_value_survives_into_the_engine(tmp_path):
    # Five seeds keeps the sample below the size at which Fabric is willing
    # to assume normality, so the selector reaches for a permutation test
    # and the configured parameters have somewhere to land.
    text = (
        SPEC.replace("__LEVEL__", "0.95")
        .replace("  values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
                 "  values: [1, 2, 3, 4, 5]")
        .replace("  confidence_level: 0.95",
                 "  confidence_level: 0.95\n  permutation_resamples: 321\n"
                 "  permutation_seed: 4242")
    )
    spec = ExperimentSpec.from_yaml(text)
    assert spec.analysis.permutation_resamples == 321
    assert spec.analysis.permutation_seed == 4242

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        workspace = Workspace.initialise(str(tmp_path))
        plan = plan_experiment(spec, workspace, root=str(tmp_path))
        workspace.create_experiment_dir(plan.experiment.id)
        Runner(workspace, plan, ExecutionOptions(isolation="inline"),
               root=str(tmp_path)).execute()
        runs = workspace.load_runs(plan.experiment.id)
        analysis = analyse(plan, runs)
    finally:
        os.chdir(cwd)

    permutations = [c for c in analysis.baselines.comparisons
                    if c.decision.test == "permutation"]
    assert permutations, "fixture chose no permutation test to check"
    for comparison in permutations:
        assert comparison.decision.parameters["resamples"] == 321
        assert comparison.decision.parameters["seed"] == 4242


@pytest.mark.parametrize("field,bad", [("permutation_resamples", 0),
                                       ("bootstrap_resamples", -5)])
def test_a_non_positive_resample_count_is_refused(field, bad):
    text = (
        "experiment: {name: d}\nresearch_question: q\n"
        "algorithm: {name: a, entrypoint: 'python:m:f'}\nmetrics: [accuracy]\n"
        f"seeds: {{values: [1, 2]}}\nanalysis: {{{field}: {bad}}}\n"
    )
    with pytest.raises(SpecError, match=field):
        ExperimentSpec.from_yaml(text)


# --------------------------------------------------------------------------
# baselines[].reference_value
# --------------------------------------------------------------------------


def test_a_published_value_is_compared_but_never_tested(at_95):
    analysis = at_95[2]
    references = analysis.references.for_metric("accuracy")
    assert len(references) == 1
    item = references[0]
    assert item.condition == "published"
    assert item.published == 0.9
    assert item.source == "Fictional et al. 2024, Table 3"
    assert item.observed.n == 10
    assert item.difference == pytest.approx(item.observed.mean - 0.9)
    assert item.verdict in ("consistent", "diverges")
    # no p-value is invented for a comparison against a scalar
    assert "p_value" not in item.to_dict()
    assert "test" not in item.to_dict()


def test_the_published_value_is_labelled_as_a_citation_not_a_run(at_95):
    plan, runs, analysis = at_95
    rendered = markdown.render(builder.build_report(analysis, plan, runs))
    assert "Published reference values" in rendered
    assert "These are citations, not runs" in rendered
    assert "no hypothesis test is reported here" in rendered.lower()


def test_a_reference_value_that_diverges_becomes_a_limitation(at_95):
    analysis = at_95[2]
    item = analysis.references.for_metric("accuracy")[0]
    if item.verdict != "diverges":
        pytest.skip("fixture happened to agree with the published value")
    assert any("published" in text and "differs" in text
               for text in analysis.limitations)
    assert any(claim["kind"] == "reference" for claim in analysis.claims)


def test_the_gap_is_expressed_in_units_of_the_measured_seed_noise(at_95):
    item = at_95[2].references.for_metric("accuracy")[0]
    assert item.seed_noise == pytest.approx(item.observed.std)
    if item.seed_noise > 0:
        assert item.noise_multiple == pytest.approx(
            abs(item.difference) / item.seed_noise)


def test_a_reference_value_for_an_arm_that_never_ran_is_not_silently_dropped():
    """Section 20: "reference value supplied but not independently reproduced"."""
    class _Spec:
        name = "never-ran"
        reference_value: ClassVar[dict] = {"accuracy": 0.91}
        source = "Somebody 2020"

    from fabric.analysis.results import ResultSet
    from fabric.core.entities import MetricSpec

    empty = ResultSet.build([], [], [MetricSpec(name="accuracy")])
    analysis = analyse_references(empty, [_Spec()],
                                  metrics=[MetricSpec(name="accuracy")])
    assert not analysis.comparisons
    assert len(analysis.missing) == 1
    assert analysis.missing[0]["condition"] == "never-ran"
    assert any("not independently" in note for note in analysis.notes)


def test_a_reference_value_for_an_undeclared_metric_is_reported_not_ignored():
    class _Spec:
        name = "arm"
        reference_value: ClassVar[dict] = {"bleu": 0.4}
        source = ""

    from fabric.analysis.results import ResultSet
    from fabric.core.entities import MetricSpec

    empty = ResultSet.build([], [], [MetricSpec(name="accuracy")])
    analysis = analyse_references(empty, [_Spec()],
                                  metrics=[MetricSpec(name="accuracy")])
    assert not analysis.comparisons and not analysis.missing
    assert any("bleu" in note for note in analysis.notes)


def test_an_experiment_without_reference_values_adds_no_section(tmp_path):
    from fabric.analysis.results import ResultSet
    from fabric.core.entities import MetricSpec

    empty = ResultSet.build([], [], [MetricSpec(name="accuracy")])
    analysis = analyse_references(empty, [], metrics=[MetricSpec(name="accuracy")])
    assert not analysis
    assert analysis.to_dict()["comparisons"] == []


# --------------------------------------------------------------------------
# execution.energy_watts
# --------------------------------------------------------------------------


def test_energy_is_not_reported_unless_a_wattage_was_declared(tmp_path):
    from fabric.execution.resources import ResourceMonitor, aggregate

    with ResourceMonitor() as monitor:
        pass
    assert monitor.result.energy_joules is None
    assert monitor.result.energy_watts is None
    totals = aggregate([monitor.result])
    assert totals["energy_joules"] is None
    assert "not measured" in totals["energy_source"]


def test_a_declared_wattage_produces_an_estimate_that_says_it_is_one():
    from fabric.execution.resources import ResourceMonitor, aggregate

    with ResourceMonitor(energy_watts=65.0) as monitor:
        pass
    result = monitor.result
    assert result.energy_watts == 65.0
    assert result.energy_joules == pytest.approx(65.0 * result.wall_seconds)
    assert monitor.measured["energy_joules"] is True

    totals = aggregate([result])
    assert totals["energy_watts"] == 65.0
    assert "estimated" in totals["energy_source"]
    assert "65" in totals["energy_source"]


def test_the_specification_can_set_the_wattage():
    text = (
        "experiment: {name: d}\nresearch_question: q\n"
        "algorithm: {name: a, entrypoint: 'python:m:f'}\nmetrics: [accuracy]\n"
        "seeds: {values: [1, 2]}\nexecution: {energy_watts: 120}\n"
    )
    spec = ExperimentSpec.from_yaml(text)
    assert spec.execution.energy_watts == 120.0


def test_an_absent_wattage_stays_absent_rather_than_defaulting():
    text = (
        "experiment: {name: d}\nresearch_question: q\n"
        "algorithm: {name: a, entrypoint: 'python:m:f'}\nmetrics: [accuracy]\n"
        "seeds: {values: [1, 2]}\n"
    )
    assert ExperimentSpec.from_yaml(text).execution.energy_watts is None


@pytest.mark.parametrize("bad", ["0", "-40"])
def test_a_non_positive_wattage_is_refused(bad):
    text = (
        "experiment: {name: d}\nresearch_question: q\n"
        "algorithm: {name: a, entrypoint: 'python:m:f'}\nmetrics: [accuracy]\n"
        f"seeds: {{values: [1, 2]}}\nexecution: {{energy_watts: {bad}}}\n"
    )
    with pytest.raises(SpecError, match="energy_watts"):
        ExperimentSpec.from_yaml(text)


def test_the_cli_exposes_a_flag_for_it():
    """A spec field with no command-line equivalent is half a feature."""
    from fabric.cli.main import build_parser

    parsed = build_parser().parse_args(["run", "spec.yaml", "--energy-watts", "42"])
    assert parsed.energy_watts == 42.0


def test_the_report_labels_energy_as_an_estimate(at_95):
    plan, runs, analysis = at_95
    analysis.resources = {
        "runs": 3, "wall_seconds": 10.0, "cpu_seconds": 9.0, "gpu_seconds": 0.0,
        "peak_rss_bytes": 0, "cost_usd": None, "energy_joules": 650.0,
        "energy_watts": 65.0,
        "energy_source": "estimated: declared 65 W x wall time",
    }
    rendered = markdown.render(builder.build_report(analysis, plan, runs))
    assert "ESTIMATED, not measured" in rendered
    assert "65 W" in rendered
