"""Comparison against published reference values.

A specification may declare what a paper reported::

    baselines:
      - name: published-baseline
        source: "Smith et al. 2023, Table 2"
        reference_value: {accuracy: 0.918}

That number is *evidence of a different kind* from a run, and Fabric
refuses to blur the two.  An empirical baseline is a distribution Fabric
executed and can characterise; a published value is a single scalar
someone else reported, with an uncertainty that is almost never stated
and a protocol that is almost never fully recoverable.

So no hypothesis test is fabricated here.  A one-sample t-test against
the published scalar would be arithmetically fine and scientifically
misleading: it would answer "is our mean equal to that constant", while
implying it had answered "did we reproduce their experiment".  It treats
the published number as exact, which it is not, and it converts every
difference in protocol, data version and hardware into a p-value that
looks like a statement about the method.

What Fabric reports instead is descriptive and complete enough to judge:
the observed distribution with its interval, the published scalar, the
gap between them in absolute terms, in relative terms, in units of the
seed noise actually measured, and against the metric's declared
tolerance if it has one.  The reader draws the conclusion.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from fabric.analysis.results import ResultSet
from fabric.core.entities import MetricSpec
from fabric.statistics.confidence import Interval, t_interval
from fabric.statistics.descriptive import Summary


@dataclass
class ReferenceComparison:
    """One observed arm measured against one published number."""

    condition: str
    metric: str
    published: float
    observed: Summary
    interval: Interval | None = None
    source: str = ""
    tolerance: float | None = None
    seed_noise: float = 0.0
    higher_is_better: bool = True

    # -- derived ------------------------------------------------------------

    @property
    def difference(self) -> float:
        """Observed minus published, in the metric's own units."""
        if self.observed.n == 0:
            return math.nan
        return self.observed.mean - self.published

    @property
    def signed_difference(self) -> float:
        """The difference signed so that positive always means better."""
        return self.difference if self.higher_is_better else -self.difference

    @property
    def relative_pct(self) -> float:
        """Difference as a percentage of the published value."""
        if self.published == 0 or self.difference != self.difference:
            return math.nan
        return 100.0 * self.difference / abs(self.published)

    @property
    def noise_multiple(self) -> float:
        """|difference| expressed in units of the measured seed noise.

        The most informative single number here: a gap of half a standard
        deviation between seeds is unremarkable, and the same gap when the
        seeds agree to four decimal places is not.
        """
        if self.seed_noise <= 0 or self.difference != self.difference:
            return math.nan
        return abs(self.difference) / self.seed_noise

    @property
    def brackets(self) -> bool | None:
        """Whether the observed interval contains the published value."""
        if self.interval is None:
            return None
        return self.interval.contains(self.published)

    @property
    def within_tolerance(self) -> bool | None:
        """Whether the gap is inside the metric's declared tolerance."""
        if self.tolerance is None or self.difference != self.difference:
            return None
        return abs(self.difference) <= self.tolerance

    @property
    def verdict(self) -> str:
        """``consistent`` / ``diverges`` / ``undetermined``.

        Deliberately not "pass" and "fail".  Fabric is comparing a
        distribution it measured against a scalar someone else reported;
        a divergence is a prompt to investigate, not a finding of error.
        """
        if self.observed.n == 0 or self.difference != self.difference:
            return "undetermined"
        within = self.within_tolerance
        if within is not None:
            return "consistent" if within else "diverges"
        brackets = self.brackets
        if brackets is None:
            return "undetermined"
        return "consistent" if brackets else "diverges"

    def describe(self, digits: int = 4) -> str:
        """A sentence that states what was compared, and to what."""
        if self.observed.n == 0:
            return (
                f"{self.condition} reports a published {self.metric} of "
                f"{self.published:.{digits}f}, but produced no runs, so nothing "
                "was reproduced."
            )
        head = (
            f"{self.condition} scored {self.observed.mean:.{digits}f} on "
            f"{self.metric} over {self.observed.n} runs"
            + (f" {self.interval.format(digits)}" if self.interval else "")
            + f" against a published {self.published:.{digits}f}"
            + (f" ({self.source})" if self.source else "")
            + f": a gap of {self.difference:+.{digits}f}"
        )
        if self.relative_pct == self.relative_pct:
            head += f" ({self.relative_pct:+.2f}%)"
        if self.noise_multiple == self.noise_multiple:
            head += f", {self.noise_multiple:.2f}x the seed-to-seed spread"
        if self.within_tolerance is not None:
            head += (
                f", {'inside' if self.within_tolerance else 'outside'} the declared "
                f"tolerance of {self.tolerance:g}"
            )
        elif self.brackets is not None:
            head += (
                f"; the published value lies "
                f"{'inside' if self.brackets else 'outside'} the observed interval"
            )
        return head + "."

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "metric": self.metric,
            "kind": "published_reference",
            "published": self.published,
            "source": self.source,
            "observed": self.observed.to_dict(include_values=False),
            "interval": self.interval.to_dict() if self.interval else None,
            "difference": self.difference,
            "signed_difference": self.signed_difference,
            "relative_pct": self.relative_pct,
            "noise_multiple": self.noise_multiple,
            "seed_noise": self.seed_noise,
            "tolerance": self.tolerance,
            "within_tolerance": self.within_tolerance,
            "brackets": self.brackets,
            "verdict": self.verdict,
            "narrative": self.describe(),
        }


@dataclass
class ReferenceAnalysis:
    """Every published number the specification declared, and how it fared."""

    comparisons: list[ReferenceComparison] = field(default_factory=list)
    missing: list[dict[str, Any]] = field(default_factory=list)
    """Reference values whose arm produced no runs at all."""
    notes: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.comparisons or self.missing)

    def for_metric(self, metric: str) -> list[ReferenceComparison]:
        return [c for c in self.comparisons if c.metric == metric]

    @property
    def diverging(self) -> list[ReferenceComparison]:
        return [c for c in self.comparisons if c.verdict == "diverges"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparisons": [c.to_dict() for c in self.comparisons],
            "missing": self.missing,
            "notes": self.notes,
        }


# The caveat that must travel with every one of these comparisons.
CAVEAT = (
    "A published value is a scalar reported by someone else. Its own "
    "uncertainty is unknown unless the source stated it, and the protocol "
    "behind it is rarely recoverable in full, so the gap below mixes any "
    "genuine difference in method with differences in data version, "
    "implementation, hardware and evaluation detail. Fabric therefore "
    "describes the gap and does not test it."
)


def analyse_references(
    results: ResultSet,
    baselines: Sequence[Any],
    *,
    metrics: Sequence[MetricSpec],
    confidence_level: float = 0.95,
) -> ReferenceAnalysis:
    """Compare each declared published value against what was observed.

    ``baselines`` is the sequence of :class:`~fabric.core.spec.BaselineSpec`
    from the specification; only those carrying ``reference_value`` produce
    a comparison.
    """
    analysis = ReferenceAnalysis()
    by_name = {spec.name: spec for spec in metrics}

    for baseline in baselines:
        published_values = dict(getattr(baseline, "reference_value", {}) or {})
        if not published_values:
            continue
        condition = results.conditions.get(baseline.name)
        source = str(getattr(baseline, "source", "") or "")

        for metric_name, published in sorted(published_values.items()):
            spec = by_name.get(metric_name)
            if spec is None:
                analysis.notes.append(
                    f"Baseline '{baseline.name}' declares a reference value for "
                    f"'{metric_name}', which is not one of the experiment's metrics "
                    f"({', '.join(m.name for m in metrics)}). It was not compared."
                )
                continue
            if condition is None or not condition.series(metric_name):
                analysis.missing.append({
                    "condition": baseline.name,
                    "metric": metric_name,
                    "published": float(published),
                    "source": source,
                    "reason": (
                        "the arm produced no runs" if condition is None or not condition.runs
                        else f"no run reported '{metric_name}'"
                    ),
                })
                continue

            summary = condition.summary(metric_name)
            values = condition.values(metric_name)
            analysis.comparisons.append(ReferenceComparison(
                condition=baseline.name,
                metric=metric_name,
                published=float(published),
                observed=summary,
                interval=t_interval(values, confidence_level) if len(values) > 1 else None,
                source=source,
                tolerance=spec.tolerance,
                seed_noise=summary.std,
                higher_is_better=spec.direction.sign >= 0,
            ))

    if analysis.comparisons or analysis.missing:
        analysis.notes.insert(0, CAVEAT)
    for entry in analysis.missing:
        analysis.notes.append(
            f"A published {entry['metric']} of {entry['published']:g} was declared for "
            f"'{entry['condition']}' but {entry['reason']}, so it was not independently "
            "reproduced. The number is carried through the report as a citation only."
        )
    return analysis
