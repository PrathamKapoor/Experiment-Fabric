"""The analysis engine.

Composes the whole scientific analysis from a plan and its runs:
descriptive summaries, baseline comparisons, the ablation matrix, sweep
sensitivity, failures, determinism, reproducibility, integrity, cost --
and then, from all of it, a headline claim and an explicit list of the
limitations that claim carries.

The headline is generated conservatively.  If the evidence does not
support "better", it does not say better.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from fabric.analysis.ablation import AblationAnalysis, analyse_ablations
from fabric.analysis.baseline import BaselineAnalysis, analyse_baselines, cost_effectiveness
from fabric.analysis.determinism import DeterminismReport, analyse_determinism
from fabric.analysis.failure import FailureAnalysis, analyse_failures
from fabric.analysis.reference import ReferenceAnalysis, analyse_references
from fabric.analysis.reproducibility import ReproducibilityProfile, analyse_reproducibility
from fabric.analysis.results import ResultSet
from fabric.analysis.sweep import SweepAnalysis, analyse_sweep
from fabric.core.entities import Run
from fabric.execution.planner import ExperimentPlan
from fabric.execution.resources import aggregate
from fabric.integrity.checks import IntegrityReport, run_checks

ANALYSIS_VERSION = "1.0"


@dataclass
class ExperimentAnalysis:
    """Everything Fabric can say about one experiment."""

    experiment_id: str
    name: str
    research_question: str
    hypothesis: str = ""
    primary_metric: str = ""
    alpha: float = 0.05
    confidence_level: float = 0.95
    correction: str = "holm"
    spec_digest: str = ""
    analysis_version: str = ANALYSIS_VERSION
    created_at: float = field(default_factory=time.time)

    results: ResultSet | None = None
    baselines: BaselineAnalysis = field(default_factory=BaselineAnalysis)
    ablations: AblationAnalysis = field(default_factory=AblationAnalysis)
    sweep: SweepAnalysis = field(default_factory=SweepAnalysis)
    references: ReferenceAnalysis = field(default_factory=ReferenceAnalysis)
    failures: FailureAnalysis = field(default_factory=FailureAnalysis)
    determinism: DeterminismReport = field(default_factory=DeterminismReport)
    reproducibility: ReproducibilityProfile = field(default_factory=ReproducibilityProfile)
    integrity: IntegrityReport = field(default_factory=IntegrityReport)
    cost: list[dict[str, Any]] = field(default_factory=list)
    resources: dict[str, Any] = field(default_factory=dict)
    headline: str = ""
    claims: list[dict[str, Any]] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "research_question": self.research_question,
            "hypothesis": self.hypothesis,
            "primary_metric": self.primary_metric,
            "alpha": self.alpha,
            "confidence_level": self.confidence_level,
            "correction": self.correction,
            "spec_digest": self.spec_digest,
            "analysis_version": self.analysis_version,
            "created_at": self.created_at,
            "headline": self.headline,
            "claims": self.claims,
            "limitations": self.limitations,
            "conditions": self.results.to_dict()["conditions"] if self.results else [],
            "baselines": self.baselines.to_dict(),
            "references": self.references.to_dict(),
            "ablations": self.ablations.to_dict(self.results.metrics if self.results else ()),
            "sweep": self.sweep.to_dict(),
            "failures": self.failures.to_dict(),
            "determinism": self.determinism.to_dict(),
            "reproducibility": self.reproducibility.to_dict(),
            "integrity": self.integrity.to_dict(),
            "cost": self.cost,
            "resources": self.resources,
        }


def analyse(plan: ExperimentPlan, runs: list[Run], *,
            stored_spec_digest: str = "") -> ExperimentAnalysis:
    """Run the full analysis for one experiment."""
    spec = plan.spec
    analysis_spec = spec.analysis
    metrics = list(spec.metrics)
    primary = spec.primary_metric

    results = ResultSet.build(runs, plan.conditions, metrics)

    analysis = ExperimentAnalysis(
        experiment_id=plan.experiment.id,
        name=spec.name,
        research_question=spec.research_question,
        hypothesis=spec.hypothesis,
        primary_metric=primary,
        alpha=analysis_spec.alpha,
        confidence_level=analysis_spec.confidence_level,
        correction=analysis_spec.correction,
        spec_digest=spec.digest,
        results=results,
    )

    analysis.baselines = analyse_baselines(
        results, metrics=metrics, alpha=analysis_spec.alpha,
        correction=analysis_spec.correction, paired_policy=analysis_spec.paired,
        confidence_level=analysis_spec.confidence_level,
        bootstrap_resamples=analysis_spec.bootstrap_resamples,
        bootstrap_seed=analysis_spec.bootstrap_seed,
        permutation_resamples=analysis_spec.permutation_resamples,
        permutation_seed=analysis_spec.permutation_seed,
        omnibus=analysis_spec.omnibus,
    )
    analysis.ablations = analyse_ablations(
        results, metrics=metrics, alpha=analysis_spec.alpha,
        correction=analysis_spec.correction,
        confidence_level=analysis_spec.confidence_level,
        bootstrap_resamples=analysis_spec.bootstrap_resamples,
        bootstrap_seed=analysis_spec.bootstrap_seed,
        permutation_resamples=analysis_spec.permutation_resamples,
        permutation_seed=analysis_spec.permutation_seed,
    )
    analysis.references = analyse_references(
        results, spec.baselines, metrics=metrics,
        confidence_level=analysis_spec.confidence_level,
    )
    analysis.sweep = analyse_sweep(results, metrics=metrics, alpha=analysis_spec.alpha)
    analysis.failures = analyse_failures(results)
    analysis.determinism = analyse_determinism(results, metrics=metrics)
    # The decision-stability estimate needs a comparator: use the strongest
    # baseline, since that is the comparison a replication would care about.
    strongest_baseline = _strongest_baseline(analysis.baselines, primary)
    analysis.reproducibility = analyse_reproducibility(
        results, metric=primary, metrics=metrics,
        determinism=analysis.determinism, alpha=analysis_spec.alpha,
        confidence_level=analysis_spec.confidence_level,
        reference_name=spec.reproducibility.reference_condition,
        decision_baseline=strongest_baseline,
    )
    analysis.integrity = run_checks(plan, results, stored_spec_digest=stored_spec_digest)
    analysis.cost = cost_effectiveness(results, primary)
    analysis.resources = aggregate([r.resources for r in runs if r.succeeded])

    analysis.headline = _headline(analysis)
    analysis.claims = _claims(analysis)
    analysis.limitations = _limitations(analysis)
    return analysis


def _strongest_baseline(baselines: BaselineAnalysis, metric: str) -> str:
    """The baseline the treatment beat by the least, i.e. the hardest one."""
    comparisons = [c for c in baselines.for_metric(metric)
                   if c.difference == c.difference]
    if not comparisons:
        return ""
    return min(comparisons, key=lambda c: c.difference).baseline


# --------------------------------------------------------------------------
# interpretation
# --------------------------------------------------------------------------


def _headline(analysis: ExperimentAnalysis) -> str:
    """One sentence that does not overstate what was shown."""
    results = analysis.results
    metric = analysis.primary_metric
    if results is None:
        return "No results."
    treatment_name = analysis.baselines.treatment
    if not treatment_name or treatment_name not in results.conditions:
        return "No treatment condition produced results."
    treatment = results[treatment_name]
    summary = treatment.summary(metric)
    if summary.n == 0:
        return f"No {metric} values were produced."

    head = (
        f"{treatment_name} scored {summary.mean:.4f} +/- {summary.std:.4f} on {metric} "
        f"over {summary.n} seeds"
    )

    comparisons = analysis.baselines.for_metric(metric)
    if not comparisons:
        return head + ", with no baseline to compare against."

    improvements = [c for c in comparisons if c.verdict == "improvement"]
    regressions = [c for c in comparisons if c.verdict == "regression"]
    inconclusive = [c for c in comparisons if c.verdict in ("inconclusive", "equivalent")]

    if regressions:
        worst = min(regressions, key=lambda c: c.difference)
        return (
            head + f", which is measurably worse than {worst.baseline} "
            f"({worst.difference:+.4f}, {worst.test.test} "
            f"{'adjusted ' if worst.adjusted_p is not None else ''}p="
            f"{worst.effective_p:.4f})."
        )
    if improvements and not inconclusive:
        weakest = min(improvements, key=lambda c: c.difference)
        return (
            head + f", beating every baseline; the narrowest margin is "
            f"{weakest.difference:+.4f} over {weakest.baseline} "
            f"{weakest.difference_interval.format()} "
            f"({weakest.test.test}, {'adjusted ' if weakest.adjusted_p is not None else ''}"
            f"p={weakest.effective_p:.4f})."
        )
    if improvements:
        undecided = ", ".join(c.baseline for c in inconclusive)
        return (
            head + f", beating {len(improvements)} of {len(comparisons)} baselines. "
            f"Against {undecided} the difference is not distinguishable from noise."
        )
    return (
        head + f", which is not distinguishable from any of the {len(comparisons)} "
        "baselines at this sample size."
    )


def _claims(analysis: ExperimentAnalysis) -> list[dict[str, Any]]:
    """Structured, individually checkable claims."""
    claims: list[dict[str, Any]] = []
    metric = analysis.primary_metric

    for comparison in analysis.baselines.for_metric(metric):
        claims.append({
            "kind": "comparison",
            "statement": comparison.narrative(),
            "verdict": comparison.verdict,
            "metric": metric,
            "treatment": comparison.treatment,
            "baseline": comparison.baseline,
            "difference": comparison.difference,
            "interval": comparison.difference_interval.to_dict(),
            "p_value": comparison.p_value,
            "adjusted_p": comparison.adjusted_p,
            "test": comparison.test.test,
            "supported": comparison.verdict in ("improvement", "equivalent"),
        })

    for component, delta, significant in analysis.ablations.ranking(metric):
        claims.append({
            "kind": "ablation",
            "statement": (
                f"Removing '{component}' changes {metric} by {-delta:+.4f}"
                + (" (statistically detectable)." if significant
                   else " (not distinguishable from noise).")
            ),
            "verdict": "load_bearing" if (significant and delta > 0) else "not_detected",
            "component": component,
            "delta": delta,
            "supported": significant,
        })

    for reference in analysis.references.comparisons:
        claims.append({
            "kind": "reference",
            "statement": reference.describe(),
            "verdict": reference.verdict,
            "metric": reference.metric,
            "condition": reference.condition,
            "published": reference.published,
            "observed": reference.observed.mean,
            "difference": reference.difference,
            "source": reference.source,
            "evidence": "published scalar, not an independent run",
            "supported": reference.verdict == "consistent",
        })

    for axis in analysis.reproducibility.measured:
        claims.append({
            "kind": "reproducibility",
            "statement": axis.describe(),
            "verdict": axis.verdict,
            "axis": axis.axis,
            "stability": axis.stability,
            "supported": axis.tolerance_verdict != "beyond",
        })

    if analysis.determinism.tested:
        claims.append({
            "kind": "determinism",
            "statement": analysis.determinism.describe(),
            "verdict": analysis.determinism.verdict,
            "supported": analysis.determinism.deterministic,
        })
    return claims


def _limitations(analysis: ExperimentAnalysis) -> list[str]:
    """The caveats a careful reviewer would raise, stated up front."""
    limitations: list[str] = []
    results = analysis.results
    metric = analysis.primary_metric

    if results is not None:
        counts = [len(c.successful) for c in results.conditions.values() if c.runs]
        if counts and min(counts) < 10:
            limitations.append(
                f"The smallest arm has {min(counts)} successful replications. Effect "
                "sizes at this sample size have wide intervals, and non-significant "
                "results are weak evidence of no effect."
            )

    if analysis.baselines.treatment_selection_note:
        limitations.append(analysis.baselines.treatment_selection_note)

    # Every design has a resolution limit, significant result or not. Report
    # the weakest comparison's, since that is the one that bounds the claim.
    detectables = [c.detectable for c in analysis.baselines.for_metric(metric)
                   if c.detectable and c.detectable.value == c.detectable.value]
    if detectables:
        worst = max(detectables, key=lambda d: d.value)
        limitations.append(worst.describe())
    elif results is not None and results.treatments:
        treatment = results.primary_treatment(metric)
        if treatment and len(treatment.values(metric)) > 1:
            from fabric.statistics.power import minimum_detectable_effect

            values = treatment.values(metric)
            estimate = minimum_detectable_effect(
                values, values, paired=True, alpha=analysis.alpha, metric=metric)
            if estimate.value == estimate.value:
                limitations.append(
                    f"With {len(values)} replications and a standard deviation of "
                    f"{treatment.summary(metric).std:.4f}, differences smaller than "
                    f"roughly {2 * estimate.standard_error * 2.3:.4f} in {metric} "
                    "would not be reliably detectable by this design."
                )

    if analysis.failures.failed_runs:
        limitations.append(
            f"{analysis.failures.failed_runs} of {analysis.failures.total_runs} runs "
            f"failed ({analysis.failures.failure_rate * 100:.1f}%). Conditions are "
            "compared on the runs that completed, which is only unbiased if the "
            "failures were unrelated to the conditions."
        )

    if analysis.determinism.tested and not analysis.determinism.deterministic:
        worst = analysis.determinism.worst
        if worst:
            limitations.append(
                f"The pipeline is not deterministic: identical runs differ by up to "
                f"{worst.max_deviation:.3g} on {worst.metric}. Differences smaller than "
                "that are not measurable here."
            )
    elif not analysis.determinism.tested:
        limitations.append(
            "Determinism was not tested, so it is not known whether these numbers "
            "reproduce when the same run is executed twice."
        )

    measured_axes = analysis.reproducibility.measured
    if not measured_axes:
        limitations.append(
            "Only seed variation was measured. The effect of a different environment, "
            "different hardware, a different dataset version, or a different "
            "implementation on these conclusions is unknown."
        )
    else:
        for axis in analysis.reproducibility.beyond_tolerance:
            limitations.append(
                f"Changing the {axis.axis} moves {axis.metric} by {axis.delta_mean:+.4f}, "
                f"beyond the declared tolerance of {axis.tolerance:g}. Results should not "
                f"be expected to reproduce across {axis.axis} changes."
            )
        for axis in analysis.reproducibility.inconclusive:
            limitations.append(
                f"Whether a change of {axis.axis} keeps {axis.metric} within its declared "
                f"tolerance of {axis.tolerance:g} is undecided: the interval "
                f"{axis.interval.format() if axis.interval else ''} straddles that "
                "boundary. More runs on that axis would resolve it."
            )

    for entry in analysis.references.missing:
        limitations.append(
            f"A published {entry['metric']} of {entry['published']:g} was cited for "
            f"'{entry['condition']}' but never reproduced here ({entry['reason']}). "
            "Citing a number is not the same as measuring it, and nothing in this "
            "report verifies it."
        )
    for reference in analysis.references.diverging:
        limitations.append(
            f"The observed {reference.metric} for '{reference.condition}' differs from "
            f"the published {reference.published:g} by {reference.difference:+.4f}"
            + (f", {reference.noise_multiple:.1f}x the seed-to-seed spread"
               if reference.noise_multiple == reference.noise_multiple else "")
            + ". The cause is not established by this experiment: it could be method, "
            "data version, implementation, hardware, or the original protocol."
        )

    for finding in analysis.integrity.failures:
        limitations.append(f"Integrity check failed: {finding.title}. {finding.detail}")
    for finding in analysis.integrity.warnings:
        limitations.append(f"{finding.title}. {finding.detail}")

    if results is not None and not results.baselines:
        limitations.append(
            "No baseline was run, so nothing here establishes that the method is better "
            "than an alternative."
        )

    if analysis.sweep.effects:
        inert = [e.parameter for e in analysis.sweep.for_metric(metric)
                 if e.verdict == "within_noise"]
        if inert:
            limitations.append(
                f"Swept parameters {', '.join(inert)} had no effect beyond seed noise; "
                "any configuration recommendation for them is not supported by this data."
            )
    return limitations
