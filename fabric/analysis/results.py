"""Turning runs into aligned series.

Everything downstream depends on one operation being done correctly:
lining up two conditions *by seed*.  Comparing the sorted accuracies of
two arms instead of their seed-matched pairs throws away the pairing and
usually flatters the result, so the alignment lives here, once, and every
analysis uses it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from fabric.core.entities import Condition, MetricSpec, Role, Run, RunStatus
from fabric.execution.resources import aggregate
from fabric.statistics.descriptive import Summary, summarize


@dataclass(frozen=True)
class ConditionResults:
    """Every run of one condition, indexed by seed."""

    name: str
    role: Role
    condition_id: str
    group: str = "main"
    config: dict[str, Any] = field(default_factory=dict)
    ablated: tuple[str, ...] = ()
    perturbation: dict[str, Any] = field(default_factory=dict)
    runs: tuple[Run, ...] = ()

    @property
    def successful(self) -> tuple[Run, ...]:
        return tuple(r for r in self.runs if r.succeeded and not r.replicate)

    @property
    def failed(self) -> tuple[Run, ...]:
        return tuple(r for r in self.runs if r.status == RunStatus.FAILED)

    @property
    def replicates(self) -> tuple[Run, ...]:
        """Repeat executions of an identical run, used for determinism checks."""
        return tuple(r for r in self.runs if r.replicate and r.succeeded)

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(sorted({r.seed for r in self.successful}))

    @property
    def failure_rate(self) -> float:
        total = len(self.runs)
        return len(self.failed) / total if total else 0.0

    def series(self, metric: str) -> dict[int, float]:
        """Metric value per seed."""
        return {r.seed: r.metrics[metric] for r in self.successful if metric in r.metrics}

    def values(self, metric: str) -> list[float]:
        series = self.series(metric)
        return [series[s] for s in sorted(series)]

    def summary(self, metric: str) -> Summary:
        return summarize(self.values(metric))

    def resources(self) -> dict[str, Any]:
        return aggregate([r.resources for r in self.successful])

    def to_dict(self, metrics: Sequence[MetricSpec]) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role.value,
            "condition_id": self.condition_id,
            "group": self.group,
            "config": self.config,
            "ablated": list(self.ablated),
            "perturbation": self.perturbation,
            "n_runs": len(self.runs),
            "n_successful": len(self.successful),
            "n_failed": len(self.failed),
            "failure_rate": self.failure_rate,
            "seeds": list(self.seeds),
            "metrics": {
                spec.name: self.summary(spec.name).to_dict()
                for spec in metrics if self.series(spec.name)
            },
            "resources": self.resources(),
        }


@dataclass
class ResultSet:
    """All results of one experiment, organised for analysis."""

    conditions: dict[str, ConditionResults] = field(default_factory=dict)
    metrics: tuple[MetricSpec, ...] = ()

    # -- construction ------------------------------------------------------

    @staticmethod
    def build(runs: Iterable[Run], conditions: Iterable[Condition],
              metrics: Sequence[MetricSpec]) -> ResultSet:
        """Group runs by the condition they belong to.

        A run records both a condition id and a condition *name*. The id is
        content-addressed, so it is not stable across versions: any change to
        how a condition is identified gives the same logical condition a new
        id. Matching only on the id therefore attached **zero** runs to every
        condition when replaying an older bundle, and every downstream
        statistic read as "no data" rather than as an error.

        Matching only on the name is not enough either: two different
        experiments can name a condition the same way, and a name alone
        cannot tell a re-identification from a different arm. So the name is
        used as the key and the id is used as corroboration -- a run whose id
        matches a condition's id is attached directly, and one whose id does
        not is still attached by name, because the name is what the author
        wrote and what every report shows.

        This was observed on a real bundle: 93 recorded runs, nine conditions,
        and not one id matched, so `fabric reproduce` reported "the original
        experiment has no condition to reproduce" while reporting that all 93
        runs had been re-executed.
        """
        by_name: dict[str, list[Run]] = {}
        runs = list(runs)
        for run in runs:
            by_name.setdefault(run.condition_name, []).append(run)
        result = ResultSet(metrics=tuple(metrics))
        # Two passes. First give every condition the runs whose id matches it
        # exactly -- the normal path, and the only one where an id is
        # authoritative. Then, for whatever is left, attach by name: an older
        # bundle recorded under different ids, and a name that no condition
        # has claimed can only belong to the condition that declares it.
        exact: dict[str, list[Run]] = {}
        for run in runs:
            exact.setdefault(run.condition_id, []).append(run)

        taken: set[str] = set()
        attachment: dict[str, list[Run]] = {}
        for condition in conditions:
            matched = exact.get(condition.id, [])
            taken.update(r.id for r in matched)
            attachment[condition.id] = matched
        for run in runs:
            if run.id in taken:
                continue
            for condition in conditions:
                if run.condition_name == condition.name:
                    attachment[condition.id].append(run)
                    taken.add(run.id)
                    break

        for condition in conditions:
            condition_runs = sorted(
                attachment[condition.id],
                key=lambda r: (r.seed, r.replicate),
            )
            result.conditions[condition.name] = ConditionResults(
                name=condition.name,
                role=condition.role,
                condition_id=condition.id,
                group=condition.group,
                config=dict(condition.config),
                ablated=tuple(condition.ablated),
                perturbation=dict(condition.perturbation),
                runs=tuple(condition_runs),
            )
        return result

    # -- access ------------------------------------------------------------

    def __getitem__(self, name: str) -> ConditionResults:
        return self.conditions[name]

    def __contains__(self, name: str) -> bool:
        return name in self.conditions

    def names(self, role: Role | None = None, group: str | None = None) -> list[str]:
        return [
            name for name, condition in self.conditions.items()
            if (role is None or condition.role == role)
            and (group is None or condition.group == group)
        ]

    @property
    def treatments(self) -> list[ConditionResults]:
        return [c for c in self.conditions.values()
                if c.role == Role.TREATMENT and not c.perturbation]

    @property
    def baselines(self) -> list[ConditionResults]:
        return [c for c in self.conditions.values() if c.role == Role.BASELINE]

    @property
    def ablations(self) -> list[ConditionResults]:
        return [c for c in self.conditions.values() if c.role == Role.ABLATION]

    @property
    def perturbations(self) -> list[ConditionResults]:
        return [c for c in self.conditions.values() if c.perturbation]

    @property
    def all_runs(self) -> list[Run]:
        return [r for c in self.conditions.values() for r in c.runs]

    def primary_treatment(self, metric: str) -> ConditionResults | None:
        """The treatment arm that a comparison should default to.

        When a sweep produced several treatment configurations, the best
        one on the primary metric is chosen -- and the analysis says so,
        because silently reporting the best of eight configurations
        against a single baseline configuration is a real form of
        selection bias.
        """
        treatments = [t for t in self.treatments if t.series(metric)]
        if not treatments:
            return None
        spec = self.metric_spec(metric)
        best = max(treatments, key=lambda c: c.summary(metric).mean * (spec.direction.sign or 1))
        return best

    def metric_spec(self, name: str) -> MetricSpec:
        for spec in self.metrics:
            if spec.name == name:
                return spec
        return MetricSpec(name=name)

    # -- alignment ---------------------------------------------------------

    def aligned(self, left: str, right: str, metric: str) -> tuple[list[float], list[float], list[int]]:
        """Seed-matched values for two conditions.

        Returns ``(left_values, right_values, seeds)`` over the seeds both
        conditions completed.  A seed where either arm failed is dropped
        from both, so the pairing stays honest.
        """
        left_series = self.conditions[left].series(metric)
        right_series = self.conditions[right].series(metric)
        seeds = sorted(set(left_series) & set(right_series))
        return ([left_series[s] for s in seeds],
                [right_series[s] for s in seeds],
                seeds)

    def can_pair(self, left: str, right: str, metric: str) -> bool:
        """Whether two conditions share enough seeds to be paired."""
        _, _, seeds = self.aligned(left, right, metric)
        return len(seeds) >= 2 and len(seeds) == len(self.conditions[left].series(metric))

    def to_dict(self) -> dict[str, Any]:
        return {
            "conditions": [c.to_dict(self.metrics) for c in self.conditions.values()],
            "metrics": [m.to_dict() for m in self.metrics],
        }
