"""Multiple-comparison control.

Twenty baselines at alpha = 0.05 buys you one significant result for
free.  Fabric knows how many comparisons an experiment implies, because
it generated them, so correction is applied automatically rather than
left to the author's conscience.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

METHODS = ("none", "bonferroni", "holm", "benjamini-hochberg", "sidak")


@dataclass(frozen=True)
class Adjusted:
    """One p-value before and after correction."""

    label: str
    p_value: float
    adjusted_p: float
    rejected: bool
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "p_value": self.p_value,
            "adjusted_p": self.adjusted_p,
            "rejected": self.rejected,
            "rank": self.rank,
        }


@dataclass(frozen=True)
class Correction:
    """The full outcome of a correction over a family of tests."""

    method: str
    alpha: float
    family_size: int
    results: tuple[Adjusted, ...]
    rationale: str = ""

    def by_label(self, label: str) -> Adjusted | None:
        for item in self.results:
            if item.label == label:
                return item
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "alpha": self.alpha,
            "family_size": self.family_size,
            "rationale": self.rationale,
            "results": [r.to_dict() for r in self.results],
            "n_rejected": sum(1 for r in self.results if r.rejected),
        }


_RATIONALE = {
    "none": "No correction applied; every p-value is reported raw. Only defensible "
            "for a single pre-registered comparison.",
    "bonferroni": "Bonferroni controls the family-wise error rate. Simple and always "
                  "valid, but conservative when comparisons are correlated.",
    "holm": "Holm-Bonferroni controls the family-wise error rate with uniformly more "
            "power than Bonferroni and no extra assumptions.",
    "benjamini-hochberg": "Benjamini-Hochberg controls the false discovery rate, which "
                          "is the right target for exploratory sweeps where some false "
                          "positives are acceptable.",
    "sidak": "Sidak controls the family-wise error rate assuming independent tests; "
             "slightly less conservative than Bonferroni when that holds.",
}


def adjust(pairs: Sequence[tuple[str, float]], *, method: str = "holm",
           alpha: float = 0.05) -> Correction:
    """Correct a family of p-values.

    ``pairs`` is a sequence of ``(label, p_value)``.  Entries whose
    p-value is not a number (a test that could not run) are carried
    through unrejected so the family size stays honest.

    The family size is the number of hypotheses *declared*, not the number
    that produced a usable p-value.  A comparison that could not be
    evaluated is still a comparison somebody planned, and shrinking the
    family because a test happened to fail is the same manoeuvre as
    deleting a comparison after seeing the results -- the selection bias
    these corrections exist to prevent.  It is anti-conservative in
    exactly the cases that matter: it turns ``not significant`` into
    ``significant``.

    Only the *ranking* uses the usable p-values, because an untested
    hypothesis has no p-value to rank.  This matches the treatment in
    R's ``p.adjust``, which likewise takes the family size from the
    declared ``n`` and filters only the working vector.
    """
    if method not in METHODS:
        raise ValueError(f"unknown correction {method!r}; expected one of {METHODS}")
    usable = [(label, p) for label, p in pairs if p == p]  # drops NaN
    unusable = tuple(Adjusted(label, p, p, False, 0) for label, p in pairs if p != p)
    family = len(pairs)
    adjusted: dict[str, float] = {}
    rejected: dict[str, bool] = {}

    rationale = _RATIONALE[method]
    if unusable:
        rationale += (
            f" The family is {family} declared hypotheses but only "
            f"{len(usable)} produced a p-value; the {len(unusable)} that could not be "
            "tested still count towards multiplicity."
        )

    if not usable:
        return Correction(method, alpha, family, unusable, rationale)

    ordered = sorted(usable, key=lambda item: item[1])

    if method == "none":
        for label, p in ordered:
            adjusted[label] = p
            rejected[label] = p < alpha
    elif method == "bonferroni":
        for label, p in ordered:
            adjusted[label] = min(1.0, p * family)
            rejected[label] = adjusted[label] < alpha
    elif method == "sidak":
        for label, p in ordered:
            adjusted[label] = min(1.0, 1.0 - (1.0 - p) ** family)
            rejected[label] = adjusted[label] < alpha
    elif method == "holm":
        running = 0.0
        for i, (label, p) in enumerate(ordered):
            value = min(1.0, (family - i) * p)
            running = max(running, value)  # enforce monotonicity
            adjusted[label] = running
            rejected[label] = running < alpha
        # Holm stops at the first failure: everything after stays accepted.
        stop = False
        for label, _ in ordered:
            if stop:
                rejected[label] = False
            elif not rejected[label]:
                stop = True
    else:  # benjamini-hochberg
        running = 1.0
        for i in range(len(ordered) - 1, -1, -1):
            label, p = ordered[i]
            value = min(1.0, p * family / (i + 1))
            running = min(running, value)  # enforce monotonicity
            adjusted[label] = running
            rejected[label] = running < alpha

    results = tuple(
        Adjusted(label, p, adjusted[label], rejected[label], rank + 1)
        for rank, (label, p) in enumerate(ordered)
    )
    return Correction(method, alpha, family, results + unusable, rationale)
