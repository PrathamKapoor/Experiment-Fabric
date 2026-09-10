"""Reproducibility as a measured quantity.

The claim "this experiment is reproducible" is not falsifiable as
stated.  Fabric replaces it with measurements: for each axis of an
experiment -- the software environment, the hardware, the dataset
version, the model version, the implementation, the configuration --
*how far does the result move when only that axis changes, compared to
how far it moves when nothing changes?*

The seed axis is the yardstick
------------------------------

Changing the random seed is not a perturbation to be scored: it *is* the
experiment's own noise, and it is the reference against which every other
axis is measured.  So the seed row reports the noise floor -- the
standard deviation across seeds, the coefficient of variation, the
interval on the mean -- plus one genuine stability number, **decision
stability**: the probability that an independent replication using half
as many seeds would reach the same conclusion about the headline
comparison.  That is estimated by subsampling the seeds actually run.

The stability score for every other axis
----------------------------------------

For an axis ``A`` we have two samples of the same metric: the reference
runs ``R`` and the perturbed runs ``P``, which differ in ``A`` and in
nothing else.  Start from Cliff's delta,

    delta = P(p > r) - P(p < r)

for a random perturbed value ``p`` and reference value ``r``.  It is
non-parametric, bounded in [-1, 1], scale-free, and has published
magnitude thresholds -- everything a metric distribution across seeds
needs, since those distributions are bounded, often skewed, and
occasionally bimodal.

Raw ``|delta|`` will not do on its own, because two samples drawn from
the *same* distribution do not give ``delta = 0``.  With five runs per
side, relabelling noise alone produces ``|delta|`` around 0.3.  Scoring
against that would report a perfectly reproducible axis as 30%
irreproducible.

So the score is calibrated against its own null.  Pool the reference and
perturbed values, relabel them at random into groups of the observed
sizes many times, and take ``e_null``, the mean ``|delta|`` under
relabelling.  Then

    excess     = max(0, (|delta| - e_null) / (1 - e_null))
    stability  = 1 - excess

``stability = 1`` means the perturbed runs are indistinguishable from a
random relabelling of the reference runs: knowing which set a value came
from tells you nothing.  ``stability = 0`` means complete separation
beyond anything relabelling could produce.

What the score deliberately does not do
---------------------------------------

It says nothing about the *size* of a shift in metric units.  A
perturbation that moves every run by a negligible +0.0001 in the same
direction still separates the two samples completely and still scores 0.
That is why the score is never reported alone: every axis also carries
the raw mean difference, its confidence interval, the shift in units of
seed noise, and -- when the metric declares a tolerance -- an equivalence
test, which is what actually answers "is this reproduction acceptable?".

Read stability as *detectability* and equivalence as *acceptability*.
They answer different questions, they can disagree, and when they do the
report says so.

The overall fingerprint is the **minimum** across the measured
non-seed axes, because a result is only as reproducible as its most
fragile dependency, with the geometric mean beside it.  Neither number
should be quoted without the per-axis table.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Sequence

from fabric.analysis.determinism import DeterminismReport
from fabric.analysis.results import ConditionResults, ResultSet
from fabric.core.entities import MetricSpec
from fabric.statistics.confidence import Interval, paired_interval, t_interval, welch_interval
from fabric.statistics.descriptive import mean, stdev
from fabric.statistics.effect_size import cliffs_delta
from fabric.statistics.hypothesis import TestResult, tost_equivalence
from fabric.statistics.selection import (
    minimum_attainable_p,
    run_selected,
    select_test,
)

AXIS_ORDER = ("implementation", "environment", "hardware", "configuration",
              "dataset", "model")

NULL_RESAMPLES = 400
"""Relabellings used to calibrate the null distribution of |Cliff's delta|."""

DECISION_RESAMPLES = 300
"""Seed subsamples used to estimate decision stability."""


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class NullCalibration:
    """The null distribution of ``|Cliff's delta|`` for two samples."""

    expectation: float
    p_value: float
    resamples: int

    def to_dict(self) -> dict[str, Any]:
        return {"expectation": self.expectation, "p_value": self.p_value,
                "resamples": self.resamples}


def null_calibration(a: Sequence[float], b: Sequence[float], *,
                     observed: float | None = None,
                     resamples: int = NULL_RESAMPLES,
                     seed: int = 20260101) -> NullCalibration:
    """Calibrate ``|Cliff's delta|`` against random relabelling.

    Pools both samples and relabels them at random into groups of the
    original sizes.  Returns the mean ``|delta|`` under relabelling --
    the value an observed delta must beat before any of it can be
    attributed to the perturbation -- and the permutation p-value for the
    observed delta, which says whether it beat the null at all.
    """
    if len(a) < 2 or len(b) < 2:
        return NullCalibration(0.0, float("nan"), 0)
    if observed is None:
        observed = cliffs_delta(a, b).value
    pool = list(a) + list(b)
    rng = random.Random(seed)
    n_a = len(a)
    total = 0.0
    at_least_as_extreme = 0
    for _ in range(resamples):
        rng.shuffle(pool)
        sampled = abs(cliffs_delta(pool[:n_a], pool[n_a:]).value)
        total += sampled
        if sampled >= abs(observed) - 1e-12:
            at_least_as_extreme += 1
    return NullCalibration(
        expectation=total / resamples,
        p_value=(at_least_as_extreme + 1) / (resamples + 1),
        resamples=resamples,
    )


def calibrated_stability(observed_delta: float, null_delta: float) -> float:
    """Turn an observed delta into a calibrated stability score."""
    magnitude = abs(observed_delta)
    if null_delta >= 1.0:
        return 1.0
    excess = (magnitude - null_delta) / (1.0 - null_delta)
    return 1.0 - min(1.0, max(0.0, excess))


# --------------------------------------------------------------------------
# axes
# --------------------------------------------------------------------------


@dataclass
class AxisStability:
    """How much one perturbation axis moves one metric."""

    axis: str
    metric: str
    label: str = ""
    n_reference: int = 0
    n_perturbed: int = 0
    reference_mean: float = math.nan
    perturbed_mean: float = math.nan
    delta_mean: float = math.nan
    cliffs_delta: float = math.nan
    null_delta: float = math.nan
    null_p_value: float = math.nan
    stability: float = math.nan
    stability_low: float = math.nan
    stability_high: float = math.nan
    interval: Interval | None = None
    seed_noise: float = math.nan
    noise_multiples: float = math.nan
    equivalence: TestResult | None = None
    tolerance: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def equivalent(self) -> bool | None:
        """Whether equivalence was *established* (never "assumed")."""
        if self.equivalence is None:
            return None
        return bool(self.equivalence.extra.get("equivalent"))

    @property
    def tolerance_verdict(self) -> str:
        """Three-way reading of the tolerance test.

        A non-significant TOST does not mean "outside the tolerance"; it
        means equivalence was not established, which is a different and
        weaker statement. Conflating them is the equivalence-testing
        equivalent of reading p > 0.05 as proof of no effect, so Fabric
        keeps the third answer:

        ``within``
            equivalence established: the difference is smaller than the
            declared tolerance.
        ``beyond``
            the confidence interval lies entirely outside the tolerance
            band: the difference is real and larger than acceptable.
        ``inconclusive``
            the interval straddles the tolerance boundary. More runs
            would settle it; the data as it stands does not.
        """
        if self.equivalence is None or self.tolerance is None:
            return "no_tolerance"
        if self.equivalent:
            return "within"
        if self.interval is not None:
            tolerance = abs(self.tolerance)
            if self.interval.low > tolerance or self.interval.high < -tolerance:
                return "beyond"
        return "inconclusive"

    @property
    def verdict(self) -> str:
        """Combined reading of detectability and acceptability."""
        if self.stability != self.stability:
            return "not_measured"
        tolerance_verdict = self.tolerance_verdict
        if tolerance_verdict == "within":
            return "reproducible"
        if tolerance_verdict == "beyond":
            return "deviates_beyond_tolerance"
        if tolerance_verdict == "inconclusive":
            return "tolerance_inconclusive"
        if self.stability >= 0.8:
            return "indistinguishable"
        if self.stability >= 0.5:
            return "detectable_shift"
        return "large_shift"

    def describe(self) -> str:
        if self.stability != self.stability:
            return f"{self.axis}: not measured."
        parts = [f"{self.axis} ({self.label})" if self.label else self.axis]
        text = (
            f"{' '.join(parts)}: moved {self.metric} by {self.delta_mean:+.4f} "
            f"{self.interval.format() if self.interval else ''}"
        ).strip()
        if self.noise_multiples == self.noise_multiples and self.seed_noise > 0:
            text += f", {self.noise_multiples:.1f}x the seed noise"
        text += f"; stability {self.stability:.2f}"
        if self.stability_low == self.stability_low:
            text += f" [{self.stability_low:.2f}, {self.stability_high:.2f}]"
        if self.null_p_value == self.null_p_value and self.null_p_value > 0.05:
            text += (" (not distinguishable from relabelling noise, permutation "
                     f"p={self.null_p_value:.3f})")
        tolerance_verdict = self.tolerance_verdict
        if tolerance_verdict == "within":
            text += f", within the declared tolerance of {self.tolerance:g}"
        elif tolerance_verdict == "beyond":
            text += f", beyond the declared tolerance of {self.tolerance:g}"
        elif tolerance_verdict == "inconclusive":
            text += (
                f", and whether it stays within the declared tolerance of "
                f"{self.tolerance:g} is undecided: the interval straddles that boundary"
            )
        return text + "."

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "metric": self.metric,
            "label": self.label,
            "n_reference": self.n_reference,
            "n_perturbed": self.n_perturbed,
            "reference_mean": self.reference_mean,
            "perturbed_mean": self.perturbed_mean,
            "delta_mean": self.delta_mean,
            "cliffs_delta": self.cliffs_delta,
            "null_delta": self.null_delta,
            "null_p_value": self.null_p_value,
            "stability": self.stability,
            "stability_interval": [self.stability_low, self.stability_high],
            "interval": self.interval.to_dict() if self.interval else None,
            "seed_noise": self.seed_noise,
            "noise_multiples": self.noise_multiples,
            "equivalence": self.equivalence.to_dict() if self.equivalence else None,
            "tolerance": self.tolerance,
            "equivalent": self.equivalent,
            "tolerance_verdict": self.tolerance_verdict,
            "verdict": self.verdict,
            "description": self.describe(),
            "detail": self.detail,
            "warnings": self.warnings,
        }


@dataclass
class SeedReference:
    """The seed axis: the noise floor everything else is measured against."""

    metric: str
    condition: str = ""
    n_seeds: int = 0
    mean: float = math.nan
    std: float = math.nan
    cv: float = math.nan
    interval: Interval | None = None
    decision_stability: float = math.nan
    decision_detail: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> str:
        if self.n_seeds < 2:
            return "seed: too few seeds to estimate the noise floor."
        text = (
            f"seed: {self.n_seeds} seeds of '{self.condition}' give "
            f"{self.metric} = {self.mean:.4f} +/- {self.std:.4f} "
            f"(CV {self.cv * 100:.2f}%)"
        )
        if self.interval:
            text += f", mean {self.interval.format()}"
        if self.decision_stability == self.decision_stability:
            k = self.decision_detail.get("subsample_size", 0)
            text += (
                f". A replication using {k} of these seeds would reach the same "
                f"conclusion {self.decision_stability * 100:.0f}% of the time"
            )
            if self.decision_detail.get("floor_limited"):
                floor = self.decision_detail.get("minimum_attainable_p")
                text += (
                    f", but that is a limit of the test rather than of the effect: with "
                    f"{k} runs the smallest attainable p-value is {floor:.4f}, which is "
                    f"above alpha, so no {k}-seed replication could reach significance "
                    "whatever it measured"
                )
        return text + "."

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": "seed",
            "metric": self.metric,
            "condition": self.condition,
            "n_seeds": self.n_seeds,
            "mean": self.mean,
            "std": self.std,
            "cv": self.cv,
            "interval": self.interval.to_dict() if self.interval else None,
            "decision_stability": self.decision_stability,
            "decision_detail": self.decision_detail,
            "description": self.describe(),
            "role": "reference noise floor; not scored as a perturbation",
            "warnings": self.warnings,
        }


@dataclass
class ReproducibilityProfile:
    """The full reproducibility fingerprint of an experiment."""

    metric: str = ""
    seed: SeedReference | None = None
    axes: list[AxisStability] = field(default_factory=list)
    determinism: DeterminismReport | None = None
    notes: list[str] = field(default_factory=list)
    method: str = (
        "stability(axis) = 1 - max(0, (|Cliff's delta| - E|delta|_null) / (1 - E|delta|_null)), "
        "where E|delta|_null is the mean |Cliff's delta| under random relabelling of the "
        "same samples. The seed axis is the reference noise floor and is not scored."
    )

    def axis(self, name: str) -> AxisStability | None:
        for item in self.axes:
            if item.axis == name:
                return item
        return None

    @property
    def measured(self) -> list[AxisStability]:
        return [a for a in self.axes if a.stability == a.stability]

    @property
    def worst_axis(self) -> AxisStability | None:
        measured = self.measured
        return min(measured, key=lambda a: a.stability) if measured else None

    @property
    def overall(self) -> float:
        """Worst-case stability across the measured non-seed axes."""
        worst = self.worst_axis
        return worst.stability if worst else math.nan

    @property
    def geometric_mean(self) -> float:
        measured = self.measured
        if not measured:
            return math.nan
        product = 1.0
        for item in measured:
            product *= max(item.stability, 1e-9)
        return product ** (1.0 / len(measured))

    @property
    def beyond_tolerance(self) -> list["AxisStability"]:
        return [a for a in self.measured if a.tolerance_verdict == "beyond"]

    @property
    def inconclusive(self) -> list["AxisStability"]:
        return [a for a in self.measured if a.tolerance_verdict == "inconclusive"]

    @property
    def verdict(self) -> str:
        if not self.measured:
            return "not_measured"
        if self.beyond_tolerance:
            return "fails_tolerance"
        if self.overall >= 0.8:
            return "robust"
        if self.overall >= 0.5:
            return "sensitive"
        return "fragile"

    def describe(self) -> str:
        if not self.measured:
            return (
                "No perturbation axes were measured beyond the seed. Declare a "
                "'reproducibility.perturbations' block to quantify how far results move "
                "when the environment, hardware, dataset, model or implementation changes."
            )
        worst = self.worst_axis
        assert worst is not None
        head = (
            f"Across {len(self.measured)} perturbation axes, the most fragile is "
            f"'{worst.axis}' (stability {worst.stability:.2f}); the geometric mean is "
            f"{self.geometric_mean:.2f}."
        )
        failing = [a.axis for a in self.beyond_tolerance]
        undecided = [a.axis for a in self.inconclusive]
        if failing:
            head += (
                f" Changing {', '.join(sorted(set(failing)))} moves {self.metric} beyond "
                f"the tolerance declared for it, so a reproduction differing on "
                f"{'those axes' if len(set(failing)) > 1 else 'that axis'} should not be "
                "expected to match."
            )
        if undecided:
            head += (
                f" On {', '.join(sorted(set(undecided)))} the data cannot decide whether "
                "the shift stays within tolerance; more runs would be needed to say."
            )
        return head

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "method": self.method,
            "seed": self.seed.to_dict() if self.seed else None,
            "axes": [a.to_dict() for a in self.axes],
            "overall_worst_case": self.overall,
            "geometric_mean": self.geometric_mean,
            "worst_axis": self.worst_axis.axis if self.worst_axis else "",
            "verdict": self.verdict,
            "determinism": self.determinism.to_dict() if self.determinism else None,
            "description": self.describe(),
            "notes": self.notes,
        }


# --------------------------------------------------------------------------
# decision stability
# --------------------------------------------------------------------------


def decision_stability(
    treatment_values: Sequence[float],
    baseline_values: Sequence[float],
    *,
    paired: bool,
    alpha: float = 0.05,
    subsample: int = 0,
    resamples: int = DECISION_RESAMPLES,
    seed: int = 20260101,
) -> tuple[float, dict[str, Any]]:
    """How often a smaller replication would reach the same conclusion.

    Takes the verdict from the full seed set as the reference, then draws
    ``subsample`` seeds at random (default: half) and re-runs the whole
    test-selection procedure on the smaller sample.  The fraction of
    subsamples that agree is the estimate.

    This is the number that answers the question researchers actually
    have: *if I had run five seeds instead of ten, would I have concluded
    the same thing?*
    """
    n = min(len(treatment_values), len(baseline_values))
    if n < 4:
        return math.nan, {"reason": "fewer than four paired seeds"}
    k = subsample or max(2, n // 2)

    def verdict(left: Sequence[float], right: Sequence[float]) -> str:
        decision = select_test(left, right, paired=paired, alpha=alpha)
        result = run_selected(decision, left, right, alpha=alpha)
        if result.p_value != result.p_value:
            return "untestable"
        if result.p_value >= alpha:
            return "no_difference"
        return "better" if mean(left) > mean(right) else "worse"

    reference = verdict(treatment_values[:n], baseline_values[:n])
    floor = minimum_attainable_p(
        select_test(treatment_values[:k], baseline_values[:k], paired=paired,
                    alpha=alpha).test,
        paired, k, k,
    )
    rng = random.Random(seed)
    indices = list(range(n))
    agree = 0
    outcomes: dict[str, int] = {}
    for _ in range(resamples):
        chosen = rng.sample(indices, k)
        if paired:
            left = [treatment_values[i] for i in chosen]
            right = [baseline_values[i] for i in chosen]
        else:
            left = [treatment_values[i] for i in rng.sample(indices, k)]
            right = [baseline_values[i] for i in chosen]
        outcome = verdict(left, right)
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        if outcome == reference:
            agree += 1
    return agree / resamples, {
        "full_verdict": reference,
        "subsample_size": k,
        "n_seeds": n,
        "resamples": resamples,
        "outcomes": dict(sorted(outcomes.items(), key=lambda kv: -kv[1])),
        "minimum_attainable_p": floor,
        "floor_limited": bool(floor is not None and floor > alpha),
        "alpha": alpha,
    }


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------


def analyse_reproducibility(
    results: ResultSet,
    *,
    metric: str,
    metrics: Sequence[MetricSpec] = (),
    determinism: DeterminismReport | None = None,
    alpha: float = 0.05,
    reference_name: str = "",
    decision_baseline: str = "",
) -> ReproducibilityProfile:
    """Measure stability along every perturbation axis that was run."""
    profile = ReproducibilityProfile(metric=metric, determinism=determinism)
    spec = results.metric_spec(metric)
    tolerance = spec.tolerance

    reference = (
        results.conditions.get(reference_name)
        if reference_name else results.primary_treatment(metric)
    )
    if reference is None:
        profile.notes.append("No reference condition produced results.")
        return profile

    reference_values = reference.values(metric)
    seed_noise = stdev(reference_values) if len(reference_values) > 1 else 0.0
    profile.seed = _seed_reference(
        results, reference, metric, reference_values, seed_noise,
        decision_baseline=decision_baseline, alpha=alpha,
    )

    grouped: dict[str, list[ConditionResults]] = {}
    for condition in results.perturbations:
        axis = str(condition.perturbation.get("axis", ""))
        if axis and axis != "seed":
            grouped.setdefault(axis, []).append(condition)

    for axis in AXIS_ORDER:
        for condition in sorted(grouped.get(axis, []), key=lambda c: c.name):
            row = _compare_axis(
                axis=axis, metric=metric, reference=reference, perturbed=condition,
                results=results, seed_noise=seed_noise, tolerance=tolerance, alpha=alpha,
            )
            if row is not None:
                profile.axes.append(row)

    if determinism and determinism.tested and not determinism.deterministic:
        floor = determinism.noise_floor(metric)
        profile.notes.append(
            f"The pipeline is not deterministic: identical runs differ by up to "
            f"{floor:.3g} on {metric}. Perturbation effects smaller than that cannot be "
            "distinguished from the pipeline's own jitter."
        )
        for row in profile.axes:
            if row.delta_mean == row.delta_mean and abs(row.delta_mean) < floor:
                row.warnings.append(
                    f"the shift ({abs(row.delta_mean):.3g}) is smaller than the "
                    f"pipeline's own non-determinism ({floor:.3g})"
                )

    _add_notes(profile)
    return profile


def _seed_reference(results: ResultSet, reference: ConditionResults, metric: str,
                    values: Sequence[float], seed_noise: float, *,
                    decision_baseline: str, alpha: float) -> SeedReference:
    row = SeedReference(
        metric=metric,
        condition=reference.name,
        n_seeds=len(values),
        mean=mean(values) if values else math.nan,
        std=seed_noise,
        cv=(seed_noise / abs(mean(values))) if values and mean(values) else math.nan,
        interval=t_interval(values, 1 - alpha) if len(values) > 1 else None,
    )
    if len(values) < 3:
        row.warnings.append(
            f"only {len(values)} seed(s): the noise floor cannot be estimated"
        )
    elif len(values) < 5:
        row.warnings.append(
            f"{len(values)} seeds gives an unstable estimate of the noise floor"
        )

    if decision_baseline and decision_baseline in results.conditions:
        left, right, seeds = results.aligned(reference.name, decision_baseline, metric)
        if len(seeds) >= 4:
            paired = results.can_pair(reference.name, decision_baseline, metric)
            stability, detail = decision_stability(
                left, right, paired=paired, alpha=alpha,
            )
            detail["baseline"] = decision_baseline
            row.decision_stability = stability
            row.decision_detail = detail
    return row


def _compare_axis(
    *,
    axis: str,
    metric: str,
    reference: ConditionResults,
    perturbed: ConditionResults,
    results: ResultSet,
    seed_noise: float,
    tolerance: float | None,
    alpha: float,
) -> AxisStability | None:
    reference_values = reference.values(metric)
    perturbed_values = perturbed.values(metric)
    if len(reference_values) < 2 or len(perturbed_values) < 2:
        return None

    paired = results.can_pair(perturbed.name, reference.name, metric)
    if paired:
        left, right, _seeds = results.aligned(perturbed.name, reference.name, metric)
    else:
        left, right = perturbed_values, reference_values

    delta = cliffs_delta(left, right)
    calibration = null_calibration(left, right, observed=delta.value)
    difference = mean(left) - mean(right)
    interval = (paired_interval(left, right, 1 - alpha) if paired
                else welch_interval(left, right, 1 - alpha))
    equivalence = (
        tost_equivalence(left, right, tolerance, paired=paired, alpha=alpha)
        if tolerance else None
    )
    row = AxisStability(
        axis=axis,
        metric=metric,
        label=str(perturbed.perturbation.get("label", perturbed.name)),
        n_reference=len(right),
        n_perturbed=len(left),
        reference_mean=mean(right),
        perturbed_mean=mean(left),
        delta_mean=difference,
        cliffs_delta=delta.value,
        null_delta=calibration.expectation,
        null_p_value=calibration.p_value,
        stability=calibrated_stability(delta.value, calibration.expectation),
        stability_low=(calibrated_stability(_furthest(delta), calibration.expectation)
                       if delta.interval else math.nan),
        stability_high=(calibrated_stability(_closest(delta), calibration.expectation)
                        if delta.interval else math.nan),
        interval=interval,
        seed_noise=seed_noise,
        noise_multiples=abs(difference) / seed_noise if seed_noise > 0 else math.inf,
        equivalence=equivalence,
        tolerance=tolerance,
        detail={
            "condition": perturbed.name,
            "reference": reference.name,
            "paired": paired,
            "perturbation": perturbed.perturbation,
            "cliffs_delta_interval": delta.interval.to_dict() if delta.interval else None,
            "null_calibration": calibration.to_dict(),
        },
    )
    if not paired:
        row.warnings.append(
            "the perturbed and reference runs do not share a seed set, so the "
            "comparison is unpaired and less sensitive"
        )
    if min(len(left), len(right)) < 5:
        row.warnings.append(
            f"only {min(len(left), len(right))} runs per side: the calibrated stability "
            "estimate is coarse"
        )
    return row


def _furthest(delta: Any) -> float:
    """The end of the delta interval furthest from zero (pessimistic)."""
    return max(abs(delta.interval.low), abs(delta.interval.high))


def _closest(delta: Any) -> float:
    """The end closest to zero, or zero when the interval spans it."""
    if delta.interval.low <= 0 <= delta.interval.high:
        return 0.0
    return min(abs(delta.interval.low), abs(delta.interval.high))


def _add_notes(profile: ReproducibilityProfile) -> None:
    measured = profile.measured
    if not measured:
        profile.notes.append(
            "Only the seed noise floor was measured. The effect of a different "
            "environment, hardware, dataset version, model version or implementation on "
            "these conclusions is unknown, and unknown is not the same as small."
        )
        return

    ranked = sorted(measured, key=lambda a: a.stability)
    profile.notes.append(
        "Axes from most to least fragile: "
        + ", ".join(f"{a.axis} ({a.stability:.2f})" for a in ranked)
    )
    if profile.seed and profile.seed.std > 0:
        louder = [a for a in ranked if a.noise_multiples == a.noise_multiples
                  and a.noise_multiples > 1.0]
        if louder:
            profile.notes.append(
                "These axes move the result further than one seed standard deviation: "
                + ", ".join(f"{a.axis} ({a.noise_multiples:.1f}x)" for a in louder)
                + ". A paper reporting seed variance alone would understate its own "
                "uncertainty on them."
            )
        else:
            profile.notes.append(
                "No perturbation moved the result by more than one seed standard "
                "deviation, which is the good case: seed variance dominates, and it is "
                "already being reported."
            )
