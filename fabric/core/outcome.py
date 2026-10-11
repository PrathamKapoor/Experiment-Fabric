"""Did this experiment produce evidence?

One question, asked by five different layers -- the command line, the
certificate issuer, the integrity checks, the job queue and the
reproduction engine -- and previously answered differently by each. They
disagreed in the one direction that matters: a run in which every attempt
failed was reported as a success by all of them, because the tests guarding
them were written as `all(...)` over a set that was empty.

This module is the single place that answers it. The states are chosen so
that a caller can be conservative without inventing a new one: a
*partially* failed experiment is not a failure, because some evidence is
still evidence and the failures stay in the record.

``no_runs`` is deliberately distinct from ``total_failure``. "You have not
run this yet" and "you ran it and nothing worked" are different facts and a
user's next action differs, but neither of them is success.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum

from fabric.core.entities import Run, RunStatus


class EvidenceOutcome(str, Enum):
    """What a set of recorded runs actually establishes."""

    NO_RUNS = "no_runs"
    """Nothing was executed. Not yet run, or nothing was pending."""

    TOTAL_FAILURE = "total_failure"
    """Runs were attempted and none succeeded. There is no evidence."""

    PARTIAL = "partial"
    """Some runs succeeded and some failed. The failures stay on the record."""

    COMPLETE = "complete"
    """Every recorded run succeeded."""

    @property
    def has_evidence(self) -> bool:
        """Whether there is at least one successful run behind this outcome.

        This is the property the certificate and the exit codes care about.
        It is deliberately narrower than ``is_success``: a partially failed
        experiment *has* evidence and is still evidence a reader can judge.
        """
        return self in (EvidenceOutcome.PARTIAL, EvidenceOutcome.COMPLETE)


def outcome_of(runs: Iterable[Run]) -> EvidenceOutcome:
    """Classify a set of recorded runs.

    Uses ``Run.succeeded`` rather than a status comparison so that cached
    runs count: a run served from the cache is real evidence and must not be
    read as a failure.
    """
    total = failed = succeeded = 0
    for run in runs:
        total += 1
        if run.status == RunStatus.FAILED:
            failed += 1
        if run.succeeded:
            succeeded += 1

    if total == 0:
        return EvidenceOutcome.NO_RUNS
    if succeeded == 0:
        return EvidenceOutcome.TOTAL_FAILURE
    if failed:
        return EvidenceOutcome.PARTIAL
    return EvidenceOutcome.COMPLETE


def describe(runs: Iterable[Run], outcome: EvidenceOutcome | None = None) -> str:
    """One sentence a report or a certificate can print verbatim."""
    runs = list(runs)
    outcome = outcome or outcome_of(runs)
    total = len(runs)
    if outcome is EvidenceOutcome.NO_RUNS:
        return "no run was executed, so there is no evidence to assess"
    failed = sum(1 for r in runs if r.status == RunStatus.FAILED)
    if outcome is EvidenceOutcome.TOTAL_FAILURE:
        return (f"no run succeeded: {failed} of {total} attempted runs failed, "
                "so this experiment produced no evidence")
    if outcome is EvidenceOutcome.PARTIAL:
        return (f"{total - failed} of {total} runs succeeded and {failed} failed; "
                "the failures remain in the record and bound what can be claimed")
    return f"all {total} runs succeeded"
