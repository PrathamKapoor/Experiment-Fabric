"""The experiment specification.

A specification is a *scientific claim in structured form*: a research
question, the conditions that bear on it, the replications that make it
measurable, and the analysis that will be applied -- declared before
anything runs, so the analysis cannot be chosen after seeing the
results.

The full schema is documented in ``docs/specification.md``; this module
is its parser, its validator, and the single source of truth for what a
valid experiment looks like.
"""

from __future__ import annotations

import math
import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from fabric.core.benchmark import (
    BenchmarkSpec,
    ConstraintResult,
    check_constraints,
)
from fabric.core.benchmark import apply as apply_benchmark
from fabric.core.benchmark import resolve as resolve_benchmark
from fabric.core.entities import MetricSpec, metric_specs
from fabric.util.hashing import full_digest
from fabric.util.yamlio import load_yaml, loads_yaml


class SpecError(ValueError):
    """Raised when a specification is malformed or self-contradictory.

    Errors name the offending key path so the message points at a line in
    the researcher's YAML rather than at Fabric's internals.
    """

    def __init__(self, message: str, path: str = "") -> None:
        self.path = path
        super().__init__(f"{path}: {message}" if path else message)


def _require(data: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in data or data[key] is None:
        raise SpecError(f"missing required key '{key}'", path)
    return data[key]


def _as_mapping(value: Any, path: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SpecError(f"expected a mapping, got {type(value).__name__}", path)
    return dict(value)


def _as_sequence(value: Any, path: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, Mapping)):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


# --------------------------------------------------------------------------
# references to registered entities
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EntityRef:
    """A reference to a dataset, model, or algorithm.

    Either names a registered entity (``name`` plus optional ``version``)
    or carries an inline definition that Fabric registers on the fly.
    Inline definitions keep small experiments to a single file; the
    registry keeps large ones consistent.
    """

    name: str
    version: str = ""
    id: str = ""
    inline: dict[str, Any] = field(default_factory=dict)

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}" if self.version else self.name

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "id": self.id, "inline": self.inline}

    @staticmethod
    def parse(value: Any, path: str) -> EntityRef:
        if value is None:
            raise SpecError("reference is empty", path)
        if isinstance(value, str):
            if "@" in value:
                name, _, version = value.partition("@")
                return EntityRef(name=name.strip(), version=version.strip())
            return EntityRef(name=value.strip())
        data = _as_mapping(value, path)
        name = str(_require(data, "name", path))
        known = {"name", "version", "id"}
        inline = {k: v for k, v in data.items() if k not in known}
        return EntityRef(
            name=name,
            version=str(data.get("version", "") or ""),
            id=str(data.get("id", "") or ""),
            inline=inline,
        )


# --------------------------------------------------------------------------
# configuration sweeps
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Parameter:
    """One configuration knob: either fixed or swept."""

    name: str
    values: tuple[Any, ...]
    swept: bool
    generator: str = "fixed"
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "values": list(self.values),
            "swept": self.swept,
            "generator": self.generator,
            "detail": self.detail,
        }


def _parse_parameter(name: str, value: Any, path: str) -> Parameter:
    """Interpret one entry of the ``configuration`` block.

    A bare list means "sweep these values".  A value that must stay a
    list (a layer-size tuple, say) is written ``{fixed: [64, 64]}``.
    """
    if isinstance(value, Mapping):
        data = dict(value)
        if "fixed" in data:
            return Parameter(name, (data["fixed"],), False, "fixed")
        if "values" in data:
            values = _as_sequence(data["values"], path)
            if not values:
                raise SpecError("'values' is empty", path)
            return Parameter(name, tuple(values), len(values) > 1, "explicit")
        sweep = str(data.get("sweep", "")).lower()
        if sweep in ("linear", "log", "logspace", "geometric"):
            low = float(_require(data, "low", path))
            high = float(_require(data, "high", path))
            count = int(data.get("count", 5))
            if count < 1:
                raise SpecError("'count' must be at least 1", path)
            if sweep in ("log", "logspace", "geometric"):
                if low <= 0 or high <= 0:
                    raise SpecError("a log sweep needs strictly positive bounds", path)
                if count == 1:
                    values = [low]
                else:
                    step = (math.log10(high) - math.log10(low)) / (count - 1)
                    values = [10 ** (math.log10(low) + i * step) for i in range(count)]
            else:
                if count == 1:
                    values = [low]
                else:
                    step = (high - low) / (count - 1)
                    values = [low + i * step for i in range(count)]
            return Parameter(name, tuple(values), count > 1, sweep,
                             {"low": low, "high": high, "count": count})
        if sweep in ("range", "arange"):
            start = float(_require(data, "start", path))
            stop = float(_require(data, "stop", path))
            step = float(data.get("step", 1))
            if step == 0:
                raise SpecError("'step' cannot be zero", path)
            values = []
            current = start
            while (step > 0 and current < stop) or (step < 0 and current > stop):
                values.append(int(current) if float(current).is_integer() else current)
                current += step
            if not values:
                raise SpecError("range produced no values", path)
            return Parameter(name, tuple(values), len(values) > 1, "range",
                             {"start": start, "stop": stop, "step": step})
        raise SpecError(
            "a mapping parameter must use 'fixed', 'values', or 'sweep'", path
        )
    if isinstance(value, (list, tuple)):
        values = list(value)
        if not values:
            raise SpecError("an empty list is not a valid parameter", path)
        return Parameter(name, tuple(values), len(values) > 1, "explicit")
    return Parameter(name, (value,), False, "fixed")


@dataclass(frozen=True)
class SweepPlan:
    """How to traverse the configuration space."""

    strategy: str = "grid"
    samples: int = 0
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def parse(value: Any, path: str = "sweep") -> SweepPlan:
        data = _as_mapping(value, path)
        strategy = str(data.get("strategy", "grid")).lower()
        if strategy not in ("grid", "random"):
            raise SpecError(
                f"unknown sweep strategy {strategy!r}; Fabric v{__spec_version__} supports "
                "'grid' and 'random' (adaptive search is deliberately out of scope "
                "until deterministic execution is proven)",
                path,
            )
        samples = int(data.get("samples", 0) or 0)
        if strategy == "random" and samples <= 0:
            raise SpecError("a random sweep needs 'samples' greater than zero", path)
        return SweepPlan(strategy, samples, int(data.get("seed", 0) or 0))


__spec_version__ = "1"


def expand_configurations(parameters: Sequence[Parameter], plan: SweepPlan) -> list[dict[str, Any]]:
    """Materialise the configuration grid.

    Grid order is deterministic (parameters sorted by name, values in
    declaration order) so the same specification produces the same run
    identifiers on every machine.
    """
    swept = [p for p in sorted(parameters, key=lambda p: p.name) if p.swept]
    fixed = {p.name: p.values[0] for p in parameters if not p.swept}

    combinations: list[dict[str, Any]] = [dict(fixed)]
    for parameter in swept:
        expanded: list[dict[str, Any]] = []
        for base in combinations:
            for value in parameter.values:
                item = dict(base)
                item[parameter.name] = value
                expanded.append(item)
        combinations = expanded

    if plan.strategy == "random" and plan.samples and plan.samples < len(combinations):
        rng = random.Random(plan.seed)
        indices = sorted(rng.sample(range(len(combinations)), plan.samples))
        combinations = [combinations[i] for i in indices]
    return combinations


# --------------------------------------------------------------------------
# seeds
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedPlan:
    """How replications are seeded.

    Seeds are the cheapest source of scientific honesty available: a
    single-seed result is an anecdote.  Fabric therefore requires the
    seed plan to be explicit and records the exact values used.
    """

    strategy: str = "fixed"
    values: tuple[int, ...] = (0,)
    count: int = 0
    start: int = 1
    generator_seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "values": list(self.values),
            "count": self.count,
            "start": self.start,
            "generator_seed": self.generator_seed,
        }

    @staticmethod
    def parse(value: Any, path: str = "seeds") -> SeedPlan:
        if value is None:
            return SeedPlan("fixed", (0,), 1)
        if isinstance(value, int):
            return SeedPlan("fixed", (int(value),), 1)
        if isinstance(value, (list, tuple)):
            values = tuple(int(v) for v in value)
            if not values:
                raise SpecError("seed list is empty", path)
            return SeedPlan("fixed", values, len(values))
        data = _as_mapping(value, path)
        strategy = str(data.get("strategy", "fixed")).lower()
        if strategy == "fixed":
            raw = data.get("values")
            if raw is None:
                count = int(data.get("count", 1))
                start = int(data.get("start", 1))
                values = tuple(range(start, start + count))
            else:
                values = tuple(int(v) for v in _as_sequence(raw, path))
            if not values:
                raise SpecError("no seed values were produced", path)
            return SeedPlan("fixed", values, len(values), int(data.get("start", 1)))
        if strategy == "range":
            count = int(_require(data, "count", path))
            start = int(data.get("start", 1))
            if count < 1:
                raise SpecError("'count' must be at least 1", path)
            return SeedPlan("range", tuple(range(start, start + count)), count, start)
        if strategy == "random":
            count = int(_require(data, "count", path))
            generator_seed = int(data.get("seed", 1234))
            rng = random.Random(generator_seed)
            # Drawn once, then recorded: a "random" seed set still has to be
            # reproducible or the experiment is not.
            values = tuple(sorted(rng.sample(range(1, 2 ** 31 - 1), count)))
            return SeedPlan("random", values, count, 1, generator_seed)
        raise SpecError(f"unknown seed strategy {strategy!r}", path)


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineSpec:
    """A comparison arm.

    Baselines are first-class: Fabric refuses to describe a result as an
    improvement unless there is something to improve *over*.
    """

    name: str
    algorithm: EntityRef | None = None
    model: EntityRef | None = None
    dataset: EntityRef | None = None
    configuration: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    reference_value: dict[str, float] = field(default_factory=dict)
    """Published numbers to reproduce, keyed by metric name."""
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "algorithm": self.algorithm.to_dict() if self.algorithm else None,
            "model": self.model.to_dict() if self.model else None,
            "dataset": self.dataset.to_dict() if self.dataset else None,
            "configuration": self.configuration,
            "description": self.description,
            "reference_value": self.reference_value,
            "source": self.source,
        }

    @staticmethod
    def parse(value: Any, index: int) -> BaselineSpec:
        path = f"baselines[{index}]"
        if isinstance(value, str):
            return BaselineSpec(name=value)
        data = _as_mapping(value, path)
        name = str(_require(data, "name", path))
        return BaselineSpec(
            name=name,
            algorithm=EntityRef.parse(data["algorithm"], f"{path}.algorithm") if data.get("algorithm") else None,
            model=EntityRef.parse(data["model"], f"{path}.model") if data.get("model") else None,
            dataset=EntityRef.parse(data["dataset"], f"{path}.dataset") if data.get("dataset") else None,
            configuration=_as_mapping(data.get("configuration"), f"{path}.configuration"),
            description=str(data.get("description", "")),
            reference_value={str(k): float(v) for k, v in
                             _as_mapping(data.get("reference_value"), path).items()},
            source=str(data.get("source", "")),
        )


@dataclass(frozen=True)
class AblationSpec:
    """A component removal.

    An ablation is a *causal* statement -- "the system without X" -- so
    Fabric models it as a set of removed components rather than as an
    arbitrary config override.  That is what makes the ablation matrix
    automatic.
    """

    name: str
    remove: tuple[str, ...]
    description: str = ""
    configuration: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "remove": list(self.remove),
            "description": self.description,
            "configuration": self.configuration,
        }

    @staticmethod
    def parse(value: Any, index: int, components: Sequence[str]) -> AblationSpec:
        path = f"ablations[{index}]"
        if isinstance(value, str):
            name = value
            removed = _components_from_name(value, components, path)
            return AblationSpec(name=name, remove=removed)
        data = _as_mapping(value, path)
        name = str(_require(data, "name", path))
        raw_remove = data.get("remove")
        if raw_remove is None:
            removed = _components_from_name(name, components, path)
        else:
            removed = tuple(str(c) for c in _as_sequence(raw_remove, path))
        if components:
            unknown = [c for c in removed if c not in components]
            if unknown:
                raise SpecError(
                    f"ablation removes unknown component(s) {unknown}; declared "
                    f"components are {list(components)}",
                    path,
                )
        return AblationSpec(
            name=name,
            remove=removed,
            description=str(data.get("description", "")),
            configuration=_as_mapping(data.get("configuration"), f"{path}.configuration"),
        )


def _components_from_name(name: str, components: Sequence[str], path: str) -> tuple[str, ...]:
    """Read ``remove_retrieval+memory`` or ``-retrieval`` as a component set."""
    text = name.strip()
    for prefix in ("remove_", "remove-", "no_", "no-", "-", "minus_"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    parts = [p.strip() for p in text.replace("+", ",").replace(" and ", ",").split(",") if p.strip()]
    parts = [p[len("remove_"):] if p.startswith("remove_") else p for p in parts]
    if components:
        unknown = [p for p in parts if p not in components]
        if unknown:
            raise SpecError(
                f"cannot infer which components {name!r} removes: {unknown} are not in "
                f"{list(components)}. Declare them explicitly with 'remove: [...]'.",
                path,
            )
    return tuple(parts)


# --------------------------------------------------------------------------
# analysis, execution, reproducibility, integrity
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalysisSpec:
    """How results will be analysed -- declared before they exist."""

    primary_metric: str = ""
    alpha: float = 0.05
    correction: str = "holm"
    paired: str = "auto"
    bootstrap_resamples: int = 5000
    bootstrap_seed: int = 12345
    permutation_resamples: int = 10000
    permutation_seed: int = 12345
    confidence_level: float = 0.95
    """Width of every reported interval. Defaults to ``1 - alpha``.

    Kept separate from alpha because they answer different questions: alpha
    is the error rate the decision accepts, the confidence level is how
    wide an interval the reader is shown. Declaring 0.99 here reports 99%
    intervals while still testing at alpha.
    """
    omnibus: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def parse(value: Any, metrics: Sequence[MetricSpec], path: str = "analysis") -> AnalysisSpec:
        data = _as_mapping(value, path)
        alpha = float(data.get("alpha", 0.05))
        if not 0 < alpha < 1:
            raise SpecError("alpha must lie strictly between 0 and 1", path)
        correction = str(data.get("correction", "holm")).lower()
        from fabric.statistics.multiple import METHODS

        if correction not in METHODS:
            raise SpecError(f"unknown correction {correction!r}; expected one of {list(METHODS)}", path)
        paired = str(data.get("paired", "auto")).lower()
        if paired not in ("auto", "always", "never"):
            raise SpecError("'paired' must be auto, always, or never", path)
        confidence_level = float(data.get("confidence_level", 1 - alpha))
        if not 0 < confidence_level < 1:
            raise SpecError(
                f"confidence_level must lie strictly between 0 and 1, "
                f"received {confidence_level!r}",
                f"{path}.confidence_level",
            )
        permutation_resamples = int(data.get("permutation_resamples", 10000))
        if permutation_resamples < 1:
            raise SpecError(
                f"permutation_resamples must be at least 1, "
                f"received {permutation_resamples!r}",
                f"{path}.permutation_resamples",
            )
        bootstrap_resamples = int(data.get("bootstrap_resamples", 5000))
        if bootstrap_resamples < 1:
            raise SpecError(
                f"bootstrap_resamples must be at least 1, "
                f"received {bootstrap_resamples!r}",
                f"{path}.bootstrap_resamples",
            )
        primary = str(data.get("primary_metric", "") or "")
        if primary and metrics and primary not in {m.name for m in metrics}:
            raise SpecError(
                f"primary_metric {primary!r} is not in the declared metrics "
                f"{[m.name for m in metrics]}",
                path,
            )
        if not primary and metrics:
            primary = next((m.name for m in metrics if m.primary), metrics[0].name)
        return AnalysisSpec(
            primary_metric=primary,
            alpha=alpha,
            correction=correction,
            paired=paired,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=int(data.get("bootstrap_seed", 12345)),
            permutation_resamples=permutation_resamples,
            permutation_seed=int(data.get("permutation_seed", 12345)),
            confidence_level=confidence_level,
            omnibus=bool(data.get("omnibus", True)),
        )


@dataclass(frozen=True)
class ExecutionSpec:
    """How runs are executed."""

    max_workers: int = 1
    timeout_seconds: float = 0.0
    cache: bool = True
    retries: int = 0
    isolation: str = "process"
    continue_on_failure: bool = True
    determinism_replicates: int = 0
    energy_watts: float | None = None
    """Average power draw of the machine, in watts, if the user knows it.

    Fabric has no way to measure power, so it will not report energy unless
    this is declared -- and when it is, every figure derived from it is
    labelled an estimate, never a measurement. The honest reading of
    ``energy_joules`` is "this wattage multiplied by the wall time".
    """

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def parse(value: Any, path: str = "execution") -> ExecutionSpec:
        data = _as_mapping(value, path)
        isolation = str(data.get("isolation", "process")).lower()
        if isolation not in ("process", "inline"):
            raise SpecError(
                "'isolation' must be 'process' (default, one subprocess per run) or "
                "'inline' (same interpreter, faster but not crash-isolated)",
                path,
            )
        workers = int(data.get("max_workers", 1))
        if workers < 1:
            raise SpecError("'max_workers' must be at least 1", path)
        watts = data.get("energy_watts")
        if watts is not None:
            watts = float(watts)
            if watts <= 0:
                raise SpecError(
                    f"'energy_watts' must be a positive average power draw in watts, "
                    f"received {watts!r}; omit it entirely if you do not know it",
                    f"{path}.energy_watts",
                )
        return ExecutionSpec(
            max_workers=workers,
            timeout_seconds=float(data.get("timeout_seconds", 0) or 0),
            cache=bool(data.get("cache", True)),
            retries=int(data.get("retries", 0)),
            isolation=isolation,
            continue_on_failure=bool(data.get("continue_on_failure", True)),
            determinism_replicates=int(data.get("determinism_replicates", 0)),
            energy_watts=watts,
        )


@dataclass(frozen=True)
class PerturbationSpec:
    """One axis of a controlled reproducibility perturbation."""

    axis: str
    values: tuple[Any, ...] = ()
    label: str = ""
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"axis": self.axis, "values": list(self.values),
                "label": self.label, "description": self.description}


AXES = ("seed", "environment", "hardware", "dataset", "model", "implementation", "configuration")


@dataclass(frozen=True)
class ReproducibilitySpec:
    """The reproducibility study attached to an experiment."""

    enabled: bool = False
    perturbations: tuple[PerturbationSpec, ...] = ()
    determinism_replicates: int = 3
    reference_condition: str = ""
    tolerance: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "perturbations": [p.to_dict() for p in self.perturbations],
            "determinism_replicates": self.determinism_replicates,
            "reference_condition": self.reference_condition,
            "tolerance": self.tolerance,
        }

    @staticmethod
    def parse(value: Any, path: str = "reproducibility") -> ReproducibilitySpec:
        data = _as_mapping(value, path)
        if not data:
            return ReproducibilitySpec()
        raw = data.get("perturbations") or {}
        perturbations: list[PerturbationSpec] = []
        if isinstance(raw, Mapping):
            items = raw.items()
        else:
            items = [(str(_require(_as_mapping(item, path), "axis", path)), item)
                     for item in _as_sequence(raw, path)]
        for axis, config in items:
            axis = str(axis).lower()
            if axis not in AXES:
                raise SpecError(f"unknown perturbation axis {axis!r}; expected one of {list(AXES)}", path)
            config_map = _as_mapping(config, f"{path}.{axis}") if not isinstance(config, (list, tuple)) else {"values": list(config)}
            values = tuple(_as_sequence(config_map.get("values"), f"{path}.{axis}.values"))
            perturbations.append(PerturbationSpec(
                axis=axis,
                values=values,
                label=str(config_map.get("label", "")),
                description=str(config_map.get("description", "")),
            ))
        return ReproducibilitySpec(
            enabled=bool(data.get("enabled", True)),
            perturbations=tuple(perturbations),
            determinism_replicates=int(data.get("determinism_replicates", 3)),
            reference_condition=str(data.get("reference_condition", "")),
            tolerance={str(k): float(v) for k, v in _as_mapping(data.get("tolerance"), path).items()},
        )


@dataclass(frozen=True)
class IntegritySpec:
    """Evaluation-integrity rules enforced before and after execution."""

    forbid_test_in_train: bool = True
    require_dataset_digest: bool = True
    freeze_baselines: bool = True
    detect_duplicate_examples: bool = True
    max_train_test_overlap: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def parse(value: Any, path: str = "integrity") -> IntegritySpec:
        data = _as_mapping(value, path)
        return IntegritySpec(
            forbid_test_in_train=bool(data.get("forbid_test_in_train", True)),
            require_dataset_digest=bool(data.get("require_dataset_digest", True)),
            freeze_baselines=bool(data.get("freeze_baselines", True)),
            detect_duplicate_examples=bool(data.get("detect_duplicate_examples", True)),
            max_train_test_overlap=float(data.get("max_train_test_overlap", 0.0)),
        )


# --------------------------------------------------------------------------
# the specification itself
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentSpec:
    """A complete, validated experiment specification."""

    name: str
    research_question: str = ""
    hypothesis: str = ""
    description: str = ""
    tags: tuple[str, ...] = ()
    benchmark: BenchmarkSpec | None = None
    """The benchmark this experiment claims to evaluate on, if any.

    Its digest is part of this specification's identity, so an experiment
    cannot claim a benchmark and then be compared against one that has
    since changed underneath it.
    """
    benchmark_notes: tuple[str, ...] = ()
    """What the benchmark contributed, for the report to state plainly."""
    dataset: EntityRef | None = None
    model: EntityRef | None = None
    algorithm: EntityRef | None = None
    treatment_name: str = "proposed"
    parameters: tuple[Parameter, ...] = ()
    sweep: SweepPlan = field(default_factory=SweepPlan)
    seeds: SeedPlan = field(default_factory=SeedPlan)
    baselines: tuple[BaselineSpec, ...] = ()
    components: tuple[str, ...] = ()
    ablations: tuple[AblationSpec, ...] = ()
    ablation_scope: str = "reference"
    """``reference`` runs ablations at the first configuration only; ``all``
    crosses every ablation with every swept configuration."""
    metrics: tuple[MetricSpec, ...] = ()
    analysis: AnalysisSpec = field(default_factory=AnalysisSpec)
    execution: ExecutionSpec = field(default_factory=ExecutionSpec)
    reproducibility: ReproducibilitySpec = field(default_factory=ReproducibilitySpec)
    integrity: IntegritySpec = field(default_factory=IntegritySpec)
    source_path: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    # -- identity ----------------------------------------------------------

    @property
    def digest(self) -> str:
        """Content digest of the specification's scientific content."""
        return full_digest(self.identity_payload())

    def identity_payload(self) -> dict[str, Any]:
        data = self.to_dict()
        data.pop("source_path", None)
        data.pop("raw", None)
        return data

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "research_question": self.research_question,
            "hypothesis": self.hypothesis,
            "description": self.description,
            "tags": list(self.tags),
            "benchmark": (
                {"ref": self.benchmark.ref, "digest": self.benchmark.digest,
                 "schema_version": self.benchmark.schema_version}
                if self.benchmark else None
            ),
            "dataset": self.dataset.to_dict() if self.dataset else None,
            "model": self.model.to_dict() if self.model else None,
            "algorithm": self.algorithm.to_dict() if self.algorithm else None,
            "treatment_name": self.treatment_name,
            "parameters": [p.to_dict() for p in self.parameters],
            "sweep": self.sweep.to_dict(),
            "seeds": self.seeds.to_dict(),
            "baselines": [b.to_dict() for b in self.baselines],
            "components": list(self.components),
            "ablations": [a.to_dict() for a in self.ablations],
            "ablation_scope": self.ablation_scope,
            "metrics": [m.to_dict() for m in self.metrics],
            "analysis": self.analysis.to_dict(),
            "execution": self.execution.to_dict(),
            "reproducibility": self.reproducibility.to_dict(),
            "integrity": self.integrity.to_dict(),
            "source_path": self.source_path,
        }

    # -- derived quantities ------------------------------------------------

    @property
    def configurations(self) -> list[dict[str, Any]]:
        """Every configuration the sweep produces."""
        return expand_configurations(self.parameters, self.sweep)

    @property
    def n_conditions(self) -> int:
        """Treatment configurations, plus baselines, plus ablations."""
        configs = len(self.configurations)
        ablation_configs = configs if self.ablation_scope == "all" else 1
        return configs + ablation_configs * len(self.ablations) + len(self.baselines)

    @property
    def n_runs(self) -> int:
        """Total executions the plan implies, before caching."""
        return self.n_conditions * len(self.seeds.values)

    def metric(self, name: str) -> MetricSpec:
        for spec in self.metrics:
            if spec.name == name:
                return spec
        return MetricSpec(name=name)

    @property
    def primary_metric(self) -> str:
        if self.analysis.primary_metric:
            return self.analysis.primary_metric
        if self.metrics:
            return self.metrics[0].name
        return ""

    # -- construction ------------------------------------------------------

    @staticmethod
    def from_dict(data: Mapping[str, Any], *, source_path: str = "") -> ExperimentSpec:
        """Parse and validate a specification mapping."""
        if not isinstance(data, Mapping):
            raise SpecError("the specification must be a mapping at the top level")
        raw = dict(data)

        # A benchmark contributes its metrics, dataset, baselines and
        # constraints before any of them are parsed, so every section below
        # stays the single authority on its own shape.
        benchmark: BenchmarkSpec | None = None
        benchmark_notes: tuple[str, ...] = ()
        if raw.get("benchmark") is not None:
            benchmark = resolve_benchmark(raw["benchmark"], spec_path=source_path)
            application = apply_benchmark(benchmark, raw)
            raw = application.data
            benchmark_notes = tuple(application.contributed)

        experiment = raw.get("experiment")
        if isinstance(experiment, str):
            header: dict[str, Any] = {"name": experiment}
        else:
            header = _as_mapping(experiment, "experiment")
        name = str(header.get("name") or raw.get("name") or "")
        if not name:
            raise SpecError("the experiment needs a name", "experiment.name")

        research_question = str(
            header.get("research_question") or raw.get("research_question") or ""
        ).strip()
        if not research_question:
            raise SpecError(
                "an experiment must state its research question. Fabric treats an "
                "experiment as a claim, and a claim needs a question it answers.",
                "research_question",
            )

        metrics = metric_specs(_as_sequence(raw.get("metrics"), "metrics"))
        if not metrics:
            raise SpecError("declare at least one metric", "metrics")

        parameters = tuple(
            _parse_parameter(str(key), value, f"configuration.{key}")
            for key, value in _as_mapping(raw.get("configuration"), "configuration").items()
        )
        components = tuple(str(c) for c in _as_sequence(raw.get("components"), "components"))
        ablations = tuple(
            AblationSpec.parse(value, i, components)
            for i, value in enumerate(_as_sequence(raw.get("ablations"), "ablations"))
        )
        baselines = tuple(
            BaselineSpec.parse(value, i)
            for i, value in enumerate(_as_sequence(raw.get("baselines"), "baselines"))
        )

        ablation_scope = str(
            header.get("ablation_scope") or raw.get("ablation_scope") or "reference"
        ).lower()
        if ablation_scope not in ("reference", "all"):
            raise SpecError(
                "'ablation_scope' must be 'reference' (ablate at the first configuration) "
                "or 'all' (cross ablations with every configuration)",
                "ablation_scope",
            )

        seen: set[str] = set()
        for arm in [*(b.name for b in baselines), *(a.name for a in ablations)]:
            if arm in seen:
                raise SpecError(f"duplicate arm name {arm!r}: arm names must be unique")
            seen.add(arm)

        spec = ExperimentSpec(
            name=name,
            benchmark=benchmark,
            benchmark_notes=benchmark_notes,
            research_question=research_question,
            hypothesis=str(header.get("hypothesis") or raw.get("hypothesis") or ""),
            description=str(header.get("description") or raw.get("description") or ""),
            tags=tuple(str(t) for t in _as_sequence(header.get("tags") or raw.get("tags"), "tags")),
            dataset=EntityRef.parse(raw["dataset"], "dataset") if raw.get("dataset") else None,
            model=EntityRef.parse(raw["model"], "model") if raw.get("model") else None,
            algorithm=EntityRef.parse(raw["algorithm"], "algorithm") if raw.get("algorithm") else None,
            treatment_name=str(header.get("treatment_name") or raw.get("treatment_name") or "proposed"),
            parameters=parameters,
            sweep=SweepPlan.parse(raw.get("sweep")),
            seeds=SeedPlan.parse(raw.get("seeds")),
            baselines=baselines,
            components=components,
            ablations=ablations,
            ablation_scope=ablation_scope,
            metrics=metrics,
            analysis=AnalysisSpec.parse(raw.get("analysis"), metrics),
            execution=ExecutionSpec.parse(raw.get("execution")),
            reproducibility=ReproducibilitySpec.parse(raw.get("reproducibility")),
            integrity=IntegritySpec.parse(raw.get("integrity")),
            source_path=source_path,
            raw=raw,
        )
        spec.validate()
        return spec

    @staticmethod
    def from_yaml(text: str, *, source_path: str = "") -> ExperimentSpec:
        return ExperimentSpec.from_dict(loads_yaml(text) or {}, source_path=source_path)

    @staticmethod
    def from_file(path: str | os.PathLike[str]) -> ExperimentSpec:
        data = load_yaml(path)
        if data is None:
            raise SpecError(f"{os.fspath(path)} is empty")
        return ExperimentSpec.from_dict(data, source_path=os.path.abspath(os.fspath(path)))

    # -- validation --------------------------------------------------------

    def validate(self) -> list[str]:
        """Raise on fatal problems; return non-fatal advisories.

        The advisories are deliberately opinionated.  They are the things
        a good reviewer would say before the compute is spent rather than
        after.
        """
        if self.algorithm is None:
            raise SpecError("an experiment needs an algorithm to run", "algorithm")
        if not self.algorithm.id and "entrypoint" not in self.algorithm.inline:
            raise SpecError(
                "the algorithm needs an 'entrypoint' (for example "
                "'python:mypkg.train:run') unless it is already registered",
                "algorithm.entrypoint",
            )
        if not self.seeds.values:
            raise SpecError("no seeds were produced", "seeds")
        if len(set(self.seeds.values)) != len(self.seeds.values):
            raise SpecError("seed values must be distinct", "seeds.values")

        advisories: list[str] = []
        n_seeds = len(self.seeds.values)
        if n_seeds < 3:
            advisories.append(
                f"only {n_seeds} seed(s): no meaningful variance estimate is possible. "
                "Five is a working minimum; ten is the point at which effect sizes "
                "start to be trustworthy."
            )
        elif n_seeds < 5:
            advisories.append(
                f"{n_seeds} seeds gives low statistical power; consider at least five."
            )
        if not self.baselines:
            advisories.append(
                "no baselines declared: Fabric can measure this method but cannot say "
                "it is better than anything."
            )
        if self.ablations and not self.components:
            advisories.append(
                "ablations were declared without a 'components' list, so Fabric cannot "
                "check that they remove real components."
            )
        if self.n_runs > 5000:
            advisories.append(
                f"this specification expands to {self.n_runs} runs; consider a random "
                "sweep or fewer seeds before committing the compute."
            )
        for metric in self.metrics:
            if metric.primary and metric.tolerance is None:
                advisories.append(
                    f"metric '{metric.name}' has no 'tolerance', so Fabric can test for a "
                    "difference but not for practical equivalence. Reproduction checks "
                    "need a tolerance."
                )

        # Benchmark constraints are advisory, deliberately. Section 20 of
        # the design: Fabric's job is to make a weak design visible, not
        # to forbid research someone has a reason to do. The report and
        # the integrity section repeat these, so an unmet constraint is
        # hard to miss and still possible to override.
        for result in self.benchmark_constraints:
            if not result.satisfied:
                advisories.append(
                    f"benchmark {self.benchmark.ref if self.benchmark else ''} "
                    f"constraint '{result.kind}' is not met: {result.detail}"
                    + (f" ({result.reason})" if result.reason else "")
                )
        return advisories

    @property
    def benchmark_constraints(self) -> list[ConstraintResult]:
        """Every rule the claimed benchmark imposes, and whether it holds."""
        if self.benchmark is None:
            return []
        return check_constraints(self.benchmark, self)

    def summary_lines(self) -> list[str]:
        """Human-readable one-line-per-fact summary used by the CLI."""
        configs = len(self.configurations)
        return [
            f"experiment      {self.name}",
            *([f"benchmark       {self.benchmark.ref} "
               f"(digest {self.benchmark.digest[:19]})"] if self.benchmark else []),
            f"question        {self.research_question}",
            f"metrics         {', '.join(m.name for m in self.metrics)} (primary: {self.primary_metric})",
            f"configurations  {configs}",
            f"arms            1 treatment + {len(self.baselines)} baselines + {len(self.ablations)} ablations",
            f"seeds           {len(self.seeds.values)} ({self.seeds.strategy})",
            f"conditions      {self.n_conditions}",
            f"runs            {self.n_runs}",
        ]
