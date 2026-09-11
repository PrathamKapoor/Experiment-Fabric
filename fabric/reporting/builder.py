"""Building the scientific report.

Assembles one :class:`~fabric.reporting.document.Document` from an
analysis, a plan and the runs.  Every renderer consumes this same
document, so the Markdown, HTML and LaTeX versions cannot drift apart.

The section order is the order a reviewer reads in: what was claimed,
how it was set up, what came out, how it was tested, what failed, how far
it travels, what could be wrong with it, and where the raw evidence is.
Limitations are a section, not a footnote.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Any

from fabric.analysis.engine import ExperimentAnalysis
from fabric.core.entities import MetricSpec, Role, Run
from fabric.core.graph import ExperimentGraph
from fabric.environments.capture import describe as describe_environment
from fabric.environments.hardware import describe as describe_hardware
from fabric.execution.planner import ExperimentPlan
from fabric.reporting.document import (
    Callout,
    Chart,
    Code,
    Document,
    GraphView,
    KeyValues,
    Paragraph,
    Section,
)


def _num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:
        return "n/a"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    if number == int(number) and abs(number) < 1e6 and digits == 0:
        return str(int(number))
    return f"{number:.{digits}f}"


def _p(value: float | None) -> str:
    if value is None or value != value:
        return "n/a"
    if value < 0.0001:
        return "<0.0001"
    return f"{value:.4f}"


def _duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 90:
        return f"{seconds:.1f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"


def build_report(
    analysis: ExperimentAnalysis,
    plan: ExperimentPlan,
    runs: Sequence[Run],
    *,
    graph: ExperimentGraph | None = None,
    environment: Any = None,
    hardware: Any = None,
) -> Document:
    """Assemble the full report document."""
    metrics = list(plan.spec.metrics)
    primary = analysis.primary_metric

    document = Document(
        title=f"{analysis.name}",
        subtitle=analysis.research_question,
        metadata={
            "experiment_id": analysis.experiment_id,
            "spec_digest": analysis.spec_digest,
            "generated_at": time.time(),
            "primary_metric": primary,
            "alpha": analysis.alpha,
            "correction": analysis.correction,
            "analysis_version": analysis.analysis_version,
            "n_runs": len(runs),
        },
    )

    _summary_section(document, analysis, plan, runs)
    _setup_section(document, analysis, plan, runs, environment, hardware)
    _results_section(document, analysis, metrics, primary)
    _statistics_section(document, analysis, metrics, primary)
    _ablation_section(document, analysis, metrics, primary)
    _sweep_section(document, analysis, primary)
    _cost_section(document, analysis, primary)
    _failure_section(document, analysis)
    _reproducibility_section(document, analysis)
    _integrity_section(document, analysis)
    _limitations_section(document, analysis)
    _provenance_section(document, analysis, plan, runs, graph)
    return document


# --------------------------------------------------------------------------
# 1. executive summary
# --------------------------------------------------------------------------


def _summary_section(document: Document, analysis: ExperimentAnalysis,
                     plan: ExperimentPlan, runs: Sequence[Run]) -> None:
    section = document.section("Summary")
    # The headline itself is a paragraph below, so every renderer shows it
    # (LaTeX does not render section summaries).
    section.summary = "The claim this experiment supports, and what it rests on."
    section.add(Callout("info", "Research question", analysis.research_question))
    if analysis.hypothesis:
        section.add(Callout("info", "Hypothesis, stated before the runs",
                            analysis.hypothesis))
    section.add(Paragraph(analysis.headline))

    succeeded = sum(1 for r in runs if r.succeeded)
    section.add(KeyValues([
        ("Experiment", analysis.experiment_id),
        ("Runs", f"{succeeded} completed of {len(runs)} planned"),
        ("Conditions", str(len(plan.conditions))),
        ("Seeds", str(len(plan.spec.seeds.values))),
        ("Primary metric", analysis.primary_metric),
        ("Significance level", f"{analysis.alpha} with {analysis.correction} correction"),
        ("Specification digest", analysis.spec_digest[:31]),
    ], caption="At a glance"))

    verdicts = []
    for claim in analysis.claims:
        if claim["kind"] == "comparison":
            mark = {"improvement": "supported", "regression": "contradicted",
                    "equivalent": "equivalent", "inconclusive": "undecided",
                    "not_testable": "untestable"}.get(claim["verdict"], claim["verdict"])
            verdicts.append([claim["baseline"], mark, _num(claim["difference"]),
                             _p(claim.get("adjusted_p") or claim.get("p_value")),
                             claim["test"]])
    if verdicts:
        section.table(
            ["Compared against", "Verdict", "Difference", "p", "Test"],
            verdicts,
            caption=f"Claims about {analysis.primary_metric}, "
                    f"{analysis.correction}-corrected",
            aligns=["left", "left", "right", "right", "left"],
            note="Difference is signed so that positive always means better, "
                 "whichever direction the metric runs.",
        )

    if analysis.determinism.tested:
        level = "good" if analysis.determinism.deterministic else "warn"
        section.add(Callout(level, "Determinism", analysis.determinism.describe()))
    if analysis.reproducibility.measured:
        verdict = analysis.reproducibility.verdict
        level = {"robust": "good", "sensitive": "warn",
                 "fragile": "bad", "fails_tolerance": "warn"}.get(verdict, "info")
        section.add(Callout(level, "Reproducibility",
                            analysis.reproducibility.describe()))
    if analysis.integrity.failures:
        section.add(Callout("bad", "Integrity",
                            "; ".join(f.title for f in analysis.integrity.failures)))


# --------------------------------------------------------------------------
# 2. setup
# --------------------------------------------------------------------------


def _setup_section(document: Document, analysis: ExperimentAnalysis,
                   plan: ExperimentPlan, runs: Sequence[Run],
                   environment: Any, hardware: Any) -> None:
    section = document.section("Experimental setup")
    section.summary = "Exactly what was run, and on what."

    if plan.dataset:
        dataset = plan.dataset
        section.subsection("Dataset").add(KeyValues([
            ("Name", dataset.ref),
            ("Content digest", dataset.digest or "not hashed"),
            ("Examples", str(dataset.n_examples) if dataset.n_examples else "unknown"),
            ("Source", dataset.source or "not declared"),
            ("Preprocessing", dataset.preprocessing or "not declared"),
            ("Splits", ", ".join(dataset.splits) if dataset.splits else "not declared"),
            ("Identifier", dataset.id),
        ]))
    if plan.model:
        model = plan.model
        section.subsection("Model").add(KeyValues([
            ("Name", model.ref),
            ("Architecture", model.architecture or "not declared"),
            ("Weights digest", model.weights_digest or "not declared"),
            ("Precision", model.precision or "not declared"),
            ("Quantization", model.quantization),
            ("Tokenizer", model.tokenizer or "not declared"),
            ("Identifier", model.id),
        ]))
    if plan.algorithm:
        algorithm = plan.algorithm
        section.subsection("Method").add(KeyValues([
            ("Name", algorithm.name),
            ("Entrypoint", algorithm.entrypoint),
            ("Source commit", algorithm.commit or "not a git repository"),
            ("Source digest", algorithm.source_digest or "not hashed"),
            ("Identifier", algorithm.id),
        ]))

    conditions = section.subsection("Conditions")
    conditions.paragraph(
        f"The specification expanded to {len(plan.conditions)} conditions across "
        f"{len(plan.spec.configurations)} configuration(s) and "
        f"{len(plan.spec.seeds.values)} seeds, giving {len(plan.runs)} runs."
    )
    rows = []
    for condition in plan.conditions:
        rows.append([
            condition.name,
            condition.role.value,
            ", ".join(condition.ablated) or "-",
            _config_text(condition.config),
        ])
    conditions.table(["Condition", "Role", "Components removed", "Configuration"], rows,
                     aligns=["left", "left", "left", "left"],
                     caption="Every experimental arm")

    seeds = plan.spec.seeds
    conditions.add(KeyValues([
        ("Seed strategy", seeds.strategy),
        ("Seed values", ", ".join(str(s) for s in seeds.values)),
        ("Sweep strategy", plan.spec.sweep.strategy),
        ("Ablation scope", plan.spec.ablation_scope),
    ]))

    if environment is not None:
        section.subsection("Software environment").add(
            KeyValues(describe_environment(environment),
                      caption=f"Environment {environment.id}")
        )
    if hardware is not None:
        section.subsection("Hardware").add(
            KeyValues(describe_hardware(hardware), caption=f"Hardware {hardware.id}")
        )


def _config_text(config: dict[str, Any]) -> str:
    if not config:
        return "-"
    parts = []
    for key, value in sorted(config.items()):
        parts.append(f"{key}={value:g}" if isinstance(value, float) else f"{key}={value}")
    return ", ".join(parts)


# --------------------------------------------------------------------------
# 3. results
# --------------------------------------------------------------------------


def _results_section(document: Document, analysis: ExperimentAnalysis,
                     metrics: Sequence[MetricSpec], primary: str) -> None:
    section = document.section("Results")
    results = analysis.results
    if results is None:
        section.paragraph("No results were produced.")
        return
    section.summary = "Every condition, every metric, with its spread."

    conditions = [c for c in results.conditions.values()
                  if c.successful and not c.perturbation]
    conditions.sort(key=lambda c: (
        {"treatment": 0, "baseline": 1, "ablation": 2, "reference": 3}[c.role.value],
        -c.summary(primary).mean * (results.metric_spec(primary).direction.sign or 1),
    ))

    headers = ["Condition", "Role", "n"]
    for spec in metrics:
        headers.append(f"{spec.name}{f' ({spec.unit})' if spec.unit else ''}")
    rows = []
    emphasise = []
    for index, condition in enumerate(conditions):
        row: list[Any] = [condition.name, condition.role.value, len(condition.successful)]
        for spec in metrics:
            summary = condition.summary(spec.name)
            row.append("-" if summary.n == 0
                       else f"{_num(summary.mean)} ± {_num(summary.std)}")
        rows.append(row)
        if condition.role == Role.TREATMENT:
            emphasise.append(index)
    section.table(headers, rows,
                  caption="Mean ± standard deviation across seeds",
                  aligns=["left", "left", "right"] + ["right"] * len(metrics),
                  emphasise=emphasise,
                  note="A standard deviation is across seeds, not across examples: "
                       "it is the run-to-run variability of the whole pipeline.")

    strip_rows = [
        {"label": c.name, "values": c.values(primary),
         "emphasise": c.role == Role.TREATMENT}
        for c in conditions if c.series(primary)
    ]
    if strip_rows:
        section.add(Chart(
            "strip", {"rows": strip_rows, "title": f"{primary} by condition"},
            caption=f"Every run of every condition on {primary}. "
                    "One dot per seed; the vertical rule is the mean.",
            alt=f"Strip plot of {primary} for {len(strip_rows)} conditions.",
        ))

    interval_rows = []
    for condition in conditions:
        values = condition.values(primary)
        if len(values) < 2:
            continue
        from fabric.statistics.confidence import t_interval

        interval = t_interval(values, 1 - analysis.alpha)
        interval_rows.append([
            condition.name, _num(interval.point), interval.format(),
            _num(condition.summary(primary).std), str(len(values)),
        ])
    if interval_rows:
        section.table(
            ["Condition", "Mean", f"{round((1 - analysis.alpha) * 100)}% interval",
             "SD", "n"],
            interval_rows,
            caption=f"Confidence intervals on the mean {primary}",
            note="Intervals are Student-t on the seed distribution.",
        )


# --------------------------------------------------------------------------
# 4. statistics
# --------------------------------------------------------------------------


def _statistics_section(document: Document, analysis: ExperimentAnalysis,
                        metrics: Sequence[MetricSpec], primary: str) -> None:
    section = document.section("Statistical analysis")
    baselines = analysis.baselines
    if not baselines.comparisons:
        section.paragraph(
            "No baselines were declared, so there is no comparison to test. "
            "The results above describe the method; they do not establish that it "
            "is better than an alternative."
        )
        return
    section.summary = (
        f"{len(baselines.comparisons)} comparisons, {analysis.correction}-corrected "
        f"at alpha={analysis.alpha}."
    )

    if baselines.treatment_selection_note:
        section.add(Callout("warn", "Treatment selection",
                            baselines.treatment_selection_note))

    for spec in metrics:
        comparisons = baselines.for_metric(spec.name)
        if not comparisons:
            continue
        subsection = section.subsection(f"{spec.name}")
        omnibus = baselines.omnibus.get(spec.name)
        if omnibus and omnibus.p_value == omnibus.p_value:
            subsection.paragraph(
                f"Omnibus test across all arms: {omnibus.test} F={_num(omnibus.statistic)}, "
                f"p={_p(omnibus.p_value)}"
                + (". At least one arm differs, so the pairwise tests below are "
                   "confirmatory." if omnibus.p_value < analysis.alpha else
                   ". Nothing differs overall, so the pairwise tests below should be "
                   "read as exploratory.")
            )

        rows = []
        for comparison in comparisons:
            g = comparison.effect("hedges_g")
            rows.append([
                comparison.baseline,
                _num(comparison.treatment_summary.mean),
                _num(comparison.baseline_summary.mean),
                _num(comparison.difference),
                comparison.difference_interval.format(),
                comparison.test.test,
                _p(comparison.p_value),
                _p(comparison.adjusted_p),
                f"{_num(g.value, 2)} ({g.magnitude})" if g and g.magnitude else _num(g.value, 2) if g else "-",
                comparison.verdict,
            ])
        subsection.table(
            ["Baseline", "Treatment", "Baseline mean", "Difference",
             f"{round((1 - analysis.alpha) * 100)}% CI", "Test", "p", "adjusted p",
             "Effect size", "Verdict"],
            rows,
            caption=f"Treatment versus each baseline on {spec.name}",
            aligns=["left", "right", "right", "right", "left", "left", "right",
                    "right", "left", "left"],
        )

        forest_rows = [
            {"label": c.baseline, "estimate": c.difference,
             "low": c.difference_interval.low, "high": c.difference_interval.high,
             "verdict": c.verdict, "note": f"p={_p(c.effective_p)}"}
            for c in comparisons
            if c.difference_interval.low == c.difference_interval.low
        ]
        if forest_rows:
            subsection.add(Chart(
                "forest",
                {"rows": forest_rows,
                 "title": f"Difference in {spec.name} against each baseline"},
                caption=f"Estimated difference in {spec.name} with "
                        f"{round((1 - analysis.alpha) * 100)}% intervals. "
                        "An interval crossing zero means the difference is unresolved.",
                alt=f"Forest plot of {len(forest_rows)} comparisons on {spec.name}.",
            ))

        detectables = [c for c in comparisons
                       if c.detectable and c.detectable.value == c.detectable.value]
        if detectables:
            subsection.table(
                ["Baseline", "Smallest detectable difference", "Measured difference",
                 "Below the floor"],
                [[c.baseline, _num(c.detectable.value), _num(c.difference),
                  "yes" if c.detectable.observed_is_below_floor else "no"]
                 for c in detectables],
                caption=f"What this design could resolve on {spec.name}",
                note="The smallest difference the design would detect 80% of the time "
                     "at this alpha, given the spread actually observed. A measured "
                     "difference below that floor cannot be separated from noise by "
                     "this experiment however the test comes out.",
            )

        if spec.name == primary:
            _test_rationale(subsection, comparisons)

    correction = baselines.corrections.get(primary)
    if correction:
        section.subsection("Multiple comparisons").add(
            Paragraph(
                f"{correction.family_size} comparisons on {primary} were corrected "
                f"with the {correction.method} method. {correction.rationale}"
            )
        )


def _test_rationale(section: Section, comparisons: Sequence[Any]) -> None:
    """Explain, per comparison, why that test and what it assumed."""
    subsection = section.subsection("Why these tests")
    subsection.paragraph(
        "Fabric selects the test from the experimental structure rather than "
        "letting the author pick after seeing the data. The reasoning for each "
        "comparison is recorded below, including the assumptions it could not check."
    )
    for comparison in comparisons:
        block = subsection.subsection(f"{comparison.treatment} vs {comparison.baseline}")
        block.paragraph(comparison.decision.reason)
        rows = []
        for assumption in comparison.test.assumptions:
            rows.append([assumption.name, assumption.symbol, assumption.detail or "-"])
        if rows:
            block.table(["Assumption", "Status", "Evidence"], rows,
                        aligns=["left", "left", "left"],
                        caption="Assumptions of the chosen test")
        if comparison.test.warnings:
            block.bullets(comparison.test.warnings)
        if comparison.equivalence:
            equivalence = comparison.equivalence
            block.paragraph(
                f"Equivalence test (TOST) against a margin of "
                f"{equivalence.extra.get('margin')}: p={_p(equivalence.p_value)}, "
                + ("equivalence established."
                   if equivalence.extra.get("equivalent")
                   else "equivalence not established (which is not the same as a "
                        "difference having been shown).")
            )


# --------------------------------------------------------------------------
# 5. ablations
# --------------------------------------------------------------------------


def _ablation_section(document: Document, analysis: ExperimentAnalysis,
                      metrics: Sequence[MetricSpec], primary: str) -> None:
    ablations = analysis.ablations
    if not ablations.effects:
        return
    section = document.section("Ablation study")
    section.summary = f"Contribution of each component, measured against {ablations.reference}."
    section.paragraph(
        f"Each row removes one or more components from '{ablations.reference}' and "
        "measures the change. A component that can be removed without a measurable "
        "loss is not carrying the result."
    )

    headers = ["Configuration", "Removed"]
    for spec in metrics:
        headers.extend([spec.name, "Δ"])
    rows = []
    for effect in ablations.effects:
        row: list[Any] = [effect.ablation, ", ".join(effect.removed) or "-"]
        for spec in metrics:
            comparison = effect.comparisons.get(spec.name)
            if comparison is None:
                row.extend(["-", "-"])
                continue
            delta = effect.delta(spec.name)
            marker = "" if comparison.significant else " ns"
            row.extend([_num(comparison.treatment_summary.mean),
                        f"{-delta:+.4f}{marker}"])
        rows.append(row)
    section.table(headers, rows,
                  caption="Ablation matrix",
                  aligns=["left", "left"] + ["right"] * (len(metrics) * 2),
                  note="Δ is the change the removal caused, signed so that positive "
                       "always means better: a negative Δ means removing the component "
                       "made the metric worse. 'ns' marks a change that is not "
                       "distinguishable from noise after correction.")

    ranking = ablations.ranking(primary)
    if ranking:
        section.add(Chart(
            "diverging_bars",
            {"rows": [{"label": name, "value": delta, "significant": significant}
                      for name, delta, significant in ranking],
             "title": f"Cost of removing each component ({primary})",
             "positive_label": "removing it hurts",
             "negative_label": "removing it helps"},
            caption=f"Change in {primary} caused by removing each component. "
                    "Grey bars are not distinguishable from noise.",
            alt=f"Diverging bar chart of {len(ranking)} component removals.",
        ))
        section.table(
            ["Component", f"Drop in {primary} when removed", "Detectable"],
            [[name, f"{delta:+.4f}", "yes" if significant else "no"]
             for name, delta, significant in ranking],
            caption="Components ranked by how much removing them costs",
            note="This column is the negation of the matrix above: a positive drop "
                 "means the component was contributing, which is the natural way to "
                 "read a component's value.",
        )

    section.bullets(ablations.notes)
    interactions = [i for i in ablations.interactions if i.metric == primary]
    if interactions:
        subsection = section.subsection("Component interactions")
        subsection.paragraph(
            "Removing two components together does not have to cost what removing "
            "each separately costs. Where it does not, the components either "
            "reinforce or substitute for each other."
        )
        subsection.table(
            ["Components", "Joint effect", "If independent", "Excess", "Reading"],
            [[", ".join(i.components), _num(i.joint_effect),
              _num(i.additive_prediction), f"{i.excess:+.4f}", i.kind]
             for i in interactions],
            caption=f"Interaction analysis on {primary}",
        )
        subsection.bullets([i.describe() for i in interactions])


# --------------------------------------------------------------------------
# 6. sweep
# --------------------------------------------------------------------------


def _sweep_section(document: Document, analysis: ExperimentAnalysis,
                   primary: str) -> None:
    sweep = analysis.sweep
    effects = sweep.for_metric(primary)
    if not effects:
        return
    section = document.section("Parameter sensitivity")
    section.summary = "Whether swept parameters move the metric more than the seed does."
    section.paragraph(
        "A parameter is only worth tuning if changing it moves the metric further "
        "than changing the random seed does. The ratio below is exactly that "
        "comparison."
    )
    section.table(
        ["Parameter", "Best", "Worst", "Spread", "Seed noise", "Spread / noise",
         "Omnibus p", "Verdict"],
        [[e.parameter, e.best_value, e.worst_value, _num(e.spread),
          _num(e.seed_noise), _num(e.noise_ratio, 2),
          _p(e.omnibus.p_value if e.omnibus else None), e.verdict]
         for e in effects],
        caption=f"Effect of each swept parameter on {primary}",
    )
    for effect in effects:
        points = [{"label": value, "mean": summary.mean, "std": summary.std}
                  for value, summary in effect.by_value.items()]
        if len(points) >= 2:
            section.add(Chart(
                "sweep_line",
                {"points": points, "title": f"{primary} against {effect.parameter}",
                 "x_label": effect.parameter, "y_label": primary},
                caption=effect.describe(),
                alt=f"Line chart of {primary} across {len(points)} values of "
                    f"{effect.parameter}.",
            ))
    section.bullets([e.describe() for e in effects])
    section.bullets(sweep.notes)


# --------------------------------------------------------------------------
# 7. cost
# --------------------------------------------------------------------------


def _cost_section(document: Document, analysis: ExperimentAnalysis,
                  primary: str) -> None:
    if not analysis.cost:
        return
    section = document.section("Cost of the result")
    section.summary = "What each condition's quality cost in compute."
    section.paragraph(
        "Accuracy without cost is half a comparison. The table below puts the "
        "quality of each condition beside the compute it took to get it."
    )
    rows = []
    for row in analysis.cost:
        rows.append([
            row["condition"], row["role"], _num(row.get(primary)),
            _duration(row.get("wall_seconds", 0.0)),
            _num(row.get("compute_multiple"), 2) if row.get("compute_multiple") else "-",
            _num(row.get("metric_delta")) if row.get("metric_delta") is not None else "-",
            _num(row.get("cost_usd"), 2) if row.get("cost_usd") is not None else "not measured",
        ])
    anchor = analysis.cost[0].get("vs_cheapest_baseline", "the cheapest baseline")
    section.table(
        ["Condition", "Role", primary, "Wall time", "Compute x", f"Δ vs {anchor}",
         "Cost (USD)"],
        rows,
        caption="Quality against compute, summed over all runs of each condition",
        note="Wall time is the total across every run of the condition, measured on "
             "the machine in the setup section. Cost is only shown when a price was "
             "declared; Fabric does not invent one.",
    )
    points = [
        {"label": row["condition"], "x": row.get("wall_seconds", 0.0),
         "y": row.get(primary), "role": row["role"]}
        for row in analysis.cost if row.get(primary) is not None
    ]
    if len(points) >= 2:
        section.add(Chart(
            "cost_scatter",
            {"points": points, "title": f"{primary} against compute",
             "x_label": "wall-clock seconds (all runs)", "y_label": primary},
            caption="Up and to the left is better. A condition far to the right has "
                    "to earn its position with height.",
            alt=f"Scatter plot of {primary} against compute for {len(points)} conditions.",
        ))


# --------------------------------------------------------------------------
# 8. failures
# --------------------------------------------------------------------------


def _failure_section(document: Document, analysis: ExperimentAnalysis) -> None:
    failures = analysis.failures
    section = document.section("Failure analysis")
    section.summary = failures.summary_line()
    if not failures.failed_runs:
        section.add(Callout(
            "good", "No failures",
            f"All {failures.total_runs} runs completed and produced every declared "
            "metric."))
        return

    section.paragraph(
        f"{failures.failed_runs} of {failures.total_runs} runs failed "
        f"({failures.failure_rate * 100:.1f}%). Failures are reported rather than "
        "dropped: a condition that cannot complete has not scored badly, it has "
        "failed to produce a score, and the difference matters for any comparison "
        "that excludes it."
    )
    section.table(["Failure class", "Runs"],
                  [[name, str(count)] for name, count in failures.by_class.items()],
                  caption="Failures by cause")
    if failures.by_condition:
        section.table(
            ["Condition", "Failed", "Of", "Rate"],
            [[name, str(data["n_failed"]), str(data["n_runs"]),
              f"{data['failure_rate'] * 100:.1f}%"]
             for name, data in sorted(failures.by_condition.items())],
            caption="Failures by condition")
    if failures.associations:
        section.table(
            ["Parameter", "Value", "Failure rate", "Runs", "Lift", "chi-square p"],
            [[a.parameter, a.value, f"{a.failure_rate * 100:.1f}%", str(a.n_runs),
              _num(a.lift, 2), _p(a.p_value)]
             for a in failures.associations[:10]],
            caption="Configuration values that failures cluster around",
            note="Lift is the failure rate here divided by the overall rate.")
    if failures.examples:
        section.table(["Class", "Condition", "Seed", "Message"],
                      [[e["class"], e["condition"], str(e["seed"]), e["message"]]
                       for e in failures.examples],
                      aligns=["left", "left", "right", "left"],
                      caption="One example per failure class")
    section.bullets(failures.notes)


# --------------------------------------------------------------------------
# 9. reproducibility
# --------------------------------------------------------------------------


def _reproducibility_section(document: Document, analysis: ExperimentAnalysis) -> None:
    profile = analysis.reproducibility
    section = document.section("Reproducibility")
    section.summary = profile.describe()

    determinism = analysis.determinism
    subsection = section.subsection("Determinism")
    subsection.paragraph(determinism.describe())
    if determinism.tested:
        subsection.table(
            ["Condition", "Metric", "Seed", "Replicates", "Identical",
             "Max deviation", "Counts toward the verdict"],
            [[r.condition, r.metric, str(r.seed), str(r.n_replicates),
              "yes" if r.identical else "no", _num(r.max_deviation, 8),
              "yes" if r.expected_deterministic else "no (measures the machine)"]
             for r in determinism.results],
            caption="Repeat executions of identical runs",
            note="A determinism replicate is the same run executed again: same seed, "
                 "same code, same data, same machine. Any difference on a result "
                 "metric is a floor on the effects this pipeline can measure. Timing "
                 "and memory metrics vary between identical runs by construction and "
                 "are reported but excluded from the verdict.")
    subsection.bullets(determinism.notes)

    if profile.seed:
        seed_section = section.subsection("Seed noise (the reference)")
        seed_section.paragraph(profile.seed.describe())
        seed_section.add(KeyValues([
            ("Condition", profile.seed.condition),
            ("Seeds", str(profile.seed.n_seeds)),
            ("Mean", _num(profile.seed.mean)),
            ("Standard deviation", _num(profile.seed.std)),
            ("Coefficient of variation",
             f"{profile.seed.cv * 100:.2f}%" if profile.seed.cv == profile.seed.cv else "n/a"),
            ("Interval on the mean",
             profile.seed.interval.format() if profile.seed.interval else "n/a"),
            ("Decision stability",
             f"{profile.seed.decision_stability * 100:.0f}%"
             if profile.seed.decision_stability == profile.seed.decision_stability
             else "not estimated"),
        ]))
        seed_section.bullets(profile.seed.warnings)

    if not profile.measured:
        section.add(Callout(
            "warn", "No perturbation axes were measured",
            "Only the seed noise floor is known. Whether these results survive a "
            "different environment, a different accelerator, a different dataset "
            "version, a different model version or a different implementation has "
            "not been tested, and untested is not the same as fine."))
        return

    axes = section.subsection("Perturbation axes")
    axes.paragraph(
        "Each axis below changes exactly one thing and leaves everything else "
        "identical. Stability is calibrated against random relabelling of the same "
        "runs, so 1.00 means the perturbed runs are indistinguishable from the "
        "reference ones."
    )
    axes.add(Code(profile.method, caption="Stability, defined"))
    rows = []
    for axis in sorted(profile.axes, key=lambda a: a.stability):
        rows.append([
            axis.axis, axis.label, f"{axis.delta_mean:+.4f}",
            axis.interval.format() if axis.interval else "-",
            _num(axis.noise_multiples, 2), _num(axis.stability, 2),
            f"[{_num(axis.stability_low, 2)}, {_num(axis.stability_high, 2)}]"
            if axis.stability_low == axis.stability_low else "-",
            _p(axis.null_p_value), axis.tolerance_verdict,
        ])
    axes.table(
        ["Axis", "Perturbation", "Δ mean", "Interval", "× seed noise", "Stability",
         "Stability CI", "Permutation p", "Tolerance"],
        rows,
        caption=f"Effect of each perturbation on {profile.metric}",
        note="'× seed noise' is the shift divided by the seed standard deviation: "
             "above 1 means the axis moves the result more than the randomness "
             "already reported does. The tolerance column is a three-way verdict; "
             "'inconclusive' means the interval straddles the declared margin, "
             "which is different from exceeding it.")
    axes.add(Chart(
        "stability",
        {"rows": [{"label": f"{a.axis}: {a.label}"[:44], "stability": a.stability,
                   "low": a.stability_low, "high": a.stability_high,
                   "verdict": a.tolerance_verdict}
                  for a in sorted(profile.axes, key=lambda x: x.stability)],
         "title": "Stability by perturbation axis"},
        caption="Calibrated stability per axis, with its interval. "
                "The chip on the right is the tolerance verdict.",
        alt=f"Bar chart of stability for {len(profile.axes)} perturbation axes.",
    ))
    axes.bullets([a.describe() for a in sorted(profile.axes, key=lambda a: a.stability)])
    for axis in profile.axes:
        for warning in axis.warnings:
            axes.add(Callout("warn", f"{axis.axis}", warning))

    summary = section.subsection("Fingerprint")
    summary.add(KeyValues([
        ("Metric", profile.metric),
        ("Axes measured", str(len(profile.measured))),
        ("Worst-case stability", _num(profile.overall, 2)),
        ("Most fragile axis", profile.worst_axis.axis if profile.worst_axis else "-"),
        ("Geometric mean", _num(profile.geometric_mean, 2)),
        ("Verdict", profile.verdict),
    ]))
    summary.paragraph(
        "The worst case is the headline because a result is only as reproducible as "
        "its most fragile dependency. Neither number should be quoted without the "
        "per-axis table above."
    )
    summary.bullets(profile.notes)


# --------------------------------------------------------------------------
# 10. integrity
# --------------------------------------------------------------------------


_LEVEL_TO_CALLOUT = {"pass": "good", "info": "info", "warn": "warn", "fail": "bad"}


def _integrity_section(document: Document, analysis: ExperimentAnalysis) -> None:
    integrity = analysis.integrity
    if not integrity.findings:
        return
    section = document.section("Evaluation integrity")
    section.summary = integrity.summary_line()
    section.table(
        ["Check", "Result", "Finding"],
        [[f.code, f.level, f.title] for f in integrity.findings],
        aligns=["left", "left", "left"],
        caption="Automated integrity checks",
    )
    for finding in integrity.findings:
        if finding.level in ("warn", "fail"):
            text = finding.detail
            if finding.remedy:
                text += f" Remedy: {finding.remedy}"
            section.add(Callout(_LEVEL_TO_CALLOUT[finding.level], finding.title, text))


# --------------------------------------------------------------------------
# 11. limitations
# --------------------------------------------------------------------------


def _limitations_section(document: Document, analysis: ExperimentAnalysis) -> None:
    section = document.section("Limitations")
    section.summary = "What this experiment does not establish."
    if not analysis.limitations:
        section.paragraph(
            "No automatic limitations were detected. That is a statement about the "
            "checks Fabric runs, not a guarantee about the science."
        )
        return
    section.paragraph(
        "These are generated from the experiment's own structure and results. They "
        "are the objections a careful reviewer would raise, stated before the "
        "reviewer has to."
    )
    section.bullets(analysis.limitations, ordered=True)


# --------------------------------------------------------------------------
# 12. provenance and artifacts
# --------------------------------------------------------------------------


def _provenance_section(document: Document, analysis: ExperimentAnalysis,
                        plan: ExperimentPlan, runs: Sequence[Run],
                        graph: ExperimentGraph | None) -> None:
    section = document.section("Provenance")
    section.summary = "Where every number came from."

    if graph is not None:
        from fabric.reporting import graphview

        # The primary metric only: every metric of every arm is a hairball.
        summary = graph.collapsed(metrics=[analysis.primary_metric])
        rendered = graphview.render(
            summary,
            element_id=f"provenance-{analysis.experiment_id}",
            caption="Every arm of this experiment and the inputs it came from. "
                    "Click a node to light up its lineage; drag to pan, scroll to zoom.",
            full_node_count=len(graph),
        )
        if rendered:
            section.add(GraphView(
                html=rendered,
                caption="The provenance graph. Clicking a node performs the same "
                        "query as 'fabric why' and highlights the chain of custody.",
                mermaid=summary.to_mermaid(),
                stats=summary.stats(),
                alt=f"Provenance graph with {len(summary)} nodes.",
            ))

        stats = graph.stats()
        section.add(KeyValues([
            ("Graph nodes", str(stats["nodes"])),
            ("Graph edges", str(stats["edges"])),
            ("Node kinds", ", ".join(f"{k}: {v}" for k, v in stats["kinds"].items())),
        ], caption="Provenance graph"))
        section.paragraph(
            "Every metric in this report is a node in that graph, connected "
            "backwards to the run, condition, configuration, seed, algorithm "
            "version, model version, dataset version, environment and hardware that "
            "produced it. Ask for any one of them with:"
        )
        section.add(Code(
            f"fabric why {analysis.experiment_id} --metric {analysis.primary_metric}\n"
            f"fabric graph {analysis.experiment_id} --format mermaid",
            language="console",
        ))

    artifacts = [(r, a) for r in runs for a in r.artifacts if a.role != "context"]
    if artifacts:
        total_bytes = sum(a.bytes for _, a in artifacts)
        section.add(KeyValues([
            ("Artifacts", str(len(artifacts))),
            ("Total size", f"{total_bytes / 1e6:.2f} MB"),
            ("Location", f".fabric/experiments/{analysis.experiment_id}/runs/"),
        ], caption="Artifact manifest"))
        sample = artifacts[:12]
        section.table(
            ["Run", "Condition", "Seed", "Artifact", "Digest", "Bytes"],
            [[r.id, r.condition_name, str(r.seed), a.name, a.digest[7:19], str(a.bytes)]
             for r, a in sample],
            aligns=["left", "left", "right", "left", "left", "right"],
            caption=f"First {len(sample)} of {len(artifacts)} artifacts",
            note="Every artifact is hashed at the moment it is written; the digest "
                 "is what makes it citable evidence rather than a file that happens "
                 "to be there.")

    section.add(KeyValues([
        ("Experiment identifier", analysis.experiment_id),
        ("Specification digest", analysis.spec_digest),
        ("Analysis version", analysis.analysis_version),
        ("Runs recorded", str(len(runs))),
    ], caption="Identity"))
