"""Runs must attach to the condition they were recorded under.

`ResultSet.build` used to key runs by `condition_id` alone. That id is
content-addressed, so it changes whenever anything about how a condition is
identified changes -- including a change to the source fingerprint, which
D-015 widened and D-017 re-scoped. Replaying a bundle recorded before such a
change therefore attached **zero** runs to every condition, and every
downstream statistic read as "no data" rather than as an error.

Observed on the shipped `demo-bundle`: 93 runs across 9 conditions, not one
id matching, and `fabric reproduce` reporting

    reproducibility   INCONCLUSIVE
    note: The original experiment has no condition to reproduce.

one line after reporting that all 93 runs had been re-executed successfully.
"""

from __future__ import annotations

from fabric.analysis.results import ResultSet
from fabric.core.entities import (
    Condition,
    Direction,
    MetricSpec,
    Role,
    Run,
    RunStatus,
)

METRICS = (MetricSpec(name="accuracy", direction=Direction.MAXIMIZE),)


def _condition(name: str, *, role: Role = Role.TREATMENT, config=None) -> Condition:
    """A condition whose content-addressed id varies with its config."""
    return Condition(
        name=name,
        role=role,
        algorithm_id="alg-1",
        config=dict(config or {}),
    )


def _run(condition: Condition, seed: int, *, value: float = 0.5,
         recorded_id: str = "") -> Run:
    return Run(
        id=f"RUN-{condition.name}-{seed}",
        experiment_id="EXP-1",
        condition_id=recorded_id or condition.id,
        condition_name=condition.name,
        seed=seed,
        status=RunStatus.COMPLETED,
        metrics={"accuracy": value},
    )


def test_runs_attach_when_the_id_matches():
    condition = _condition("proposed")
    runs = [_run(condition, seed) for seed in (1, 2, 3)]
    result = ResultSet.build(runs, [condition], METRICS)
    assert len(result["proposed"].runs) == 3


def test_runs_attach_by_name_when_the_id_was_recorded_under_another_one():
    """The bundle case: identical arm, different content-addressed id."""
    condition = _condition("proposed")
    runs = [_run(condition, seed, recorded_id="cond-OLDID00000001") for seed in (1, 2)]
    result = ResultSet.build(runs, [condition], METRICS)

    attached = result["proposed"].runs
    assert len(attached) == 2, "a re-identified condition must still find its runs"
    assert sorted(r.seed for r in attached) == [1, 2]
    assert len(result["proposed"].series("accuracy")) == 2


def test_the_default_treatment_is_findable_after_re_identification():
    """The failure this caused: `primary_treatment` returned None."""
    condition = _condition("proposed")
    runs = [_run(condition, seed, recorded_id="cond-OLDID00000001") for seed in (1, 2, 3)]
    result = ResultSet.build(runs, [condition], METRICS)

    reference = result.primary_treatment("accuracy")
    assert reference is not None, (
        "an experiment with results must not report that it has no condition "
        "to compare"
    )
    assert reference.name == "proposed"


def test_two_conditions_with_different_names_keep_their_own_runs():
    """The guard against making the name match too permissive.

    Two arms whose ids were both re-identified must each recover only their
    own runs, by name -- never the other's.
    """
    proposed = _condition("proposed")
    baseline = _condition("baseline", role=Role.BASELINE)
    runs = [
        _run(proposed, 1, value=0.7, recorded_id="cond-OLDPROPOSED0001"),
        _run(baseline, 1, value=0.3, recorded_id="cond-OLDBASELINE0001"),
    ]

    result = ResultSet.build(runs, [proposed, baseline], METRICS)
    assert list(result["proposed"].series("accuracy").values()) == [0.7]
    assert list(result["baseline"].series("accuracy").values()) == [0.3]


def test_a_condition_with_no_runs_reports_empty_rather_than_inheriting():
    missing = _condition("never-run")
    present = _condition("proposed")
    runs = [_run(present, 1)]
    result = ResultSet.build(runs, [present, missing], METRICS)
    assert result["never-run"].runs == ()
    assert result["proposed"].runs != ()


def test_a_condition_with_no_runs_reports_empty_rather_than_inheriting():
    missing = _condition("never-run")
    present = _condition("proposed")
    runs = [_run(present, 1)]
    result = ResultSet.build(runs, [present, missing], METRICS)
    assert result["never-run"].runs == ()
    assert result["proposed"].runs != ()