"""The nouns of an experiment.

Every entity here is immutable and content-addressed: its identifier is
derived from its own contents, so two researchers who register the same
dataset on different continents produce the same identifier, and a
metric can name the exact dataset that produced it forever.

The rule that keeps the system honest: *nothing carries a mutable
pointer*.  A run stores ``dataset_id``, not "the current dataset".
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, ClassVar

from fabric.util.hashing import content_id

# --------------------------------------------------------------------------
# enumerations
# --------------------------------------------------------------------------


class Role(str, Enum):
    """What scientific job an experimental arm performs."""

    TREATMENT = "treatment"
    BASELINE = "baseline"
    ABLATION = "ablation"
    REFERENCE = "reference"


class RunStatus(str, Enum):
    """Lifecycle of a single run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"
    SKIPPED = "skipped"


class Direction(str, Enum):
    """Whether a metric is better when larger or smaller."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    NEUTRAL = "neutral"

    @property
    def sign(self) -> int:
        """+1 when larger is better, -1 when smaller is better, 0 otherwise."""
        return {"maximize": 1, "minimize": -1, "neutral": 0}[self.value]


class FailureClass(str, Enum):
    """Taxonomy used by the failure analyser.

    Failures are data.  A run that dies of OOM at ``batch_size=128`` tells
    you something about the method, so Fabric classifies rather than
    discards them.
    """

    OOM = "oom"
    TIMEOUT = "timeout"
    NAN = "nan"
    DIVERGENCE = "divergence"
    SEGFAULT = "segfault"
    DEPENDENCY = "dependency"
    NETWORK = "network"
    INVALID_OUTPUT = "invalid_output"
    METRIC_FAILURE = "metric_failure"
    USER_ERROR = "user_error"
    UNKNOWN = "unknown"


# --------------------------------------------------------------------------
# base
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Entity:
    """Base class providing content-addressed identity."""

    #: Subclasses override with a short prefix such as ``ds-`` or ``mdl-``.
    #: A ClassVar rather than a field, so subclasses stay free to declare
    #: required fields without tripping dataclass default ordering.
    id_prefix: ClassVar[str] = ""

    def identity_payload(self) -> dict[str, Any]:
        """The fields that define this entity's identity.

        Bookkeeping fields (registration timestamps, local file paths)
        are excluded so that identity survives being moved between
        machines.
        """
        payload = {k: v for k, v in asdict(self).items() if k not in self._excluded_fields()}
        payload["__kind__"] = type(self).__name__
        return payload

    @staticmethod
    def _excluded_fields() -> frozenset[str]:
        """Fields that never participate in identity.

        Bookkeeping (when it was registered, where the bytes happen to
        live on this machine) plus free-text annotation. Annotation is
        excluded so that describing the same dataset more fully in a
        second experiment does not create a second, incompatible
        dataset -- a trap that turns the registry's conflict guard from a
        safety feature into an obstacle.
        """
        return frozenset({
            "registered_at", "local_path", "id", "dirty",
            "description", "notes", "source", "preprocessing", "license",
        })

    @property
    def id(self) -> str:
        """Stable content identifier."""
        return content_id(self.identity_payload(), prefix=self.id_prefix)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping, with ``id`` materialised for readers."""
        data = dict(asdict(self))
        data["id"] = self.id
        data["kind"] = type(self).__name__
        return _plain(data)

    def with_(self, **changes: Any) -> Entity:
        """Return a copy with fields replaced (identity changes with it)."""
        return replace(self, **changes)


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


# --------------------------------------------------------------------------
# scientific inputs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Dataset(Entity):
    """An immutable reference to data.

    ``digest`` is the content hash of the underlying bytes when Fabric can
    see them, or a declared digest when the data lives behind an API.  A
    dataset whose bytes change is a *different dataset*, which is exactly
    the property that makes "has the benchmark changed since the paper?"
    an answerable question.
    """

    name: str
    version: str = "v1"
    digest: str = ""
    source: str = ""
    preprocessing: str = ""
    schema: dict[str, Any] = field(default_factory=dict)
    splits: dict[str, dict[str, Any]] = field(default_factory=dict)
    n_examples: int | None = None
    license: str = ""
    notes: str = ""
    local_path: str = ""
    registered_at: float = 0.0
    id_prefix: ClassVar[str] = "ds-"

    @property
    def ref(self) -> str:
        """Human-facing reference such as ``benchmark-x@v3``."""
        return f"{self.name}@{self.version}"

    def identity_payload(self) -> dict[str, Any]:
        """Identity is the content, when the content is known.

        A dataset with a digest *is* those bytes under that name and
        version; schema and split annotations describe it but do not
        change which data it is. Without a digest there is nothing to
        address it by, so every declared field counts instead.
        """
        if self.digest:
            return {
                "__kind__": "Dataset",
                "name": self.name,
                "version": self.version,
                "digest": self.digest,
            }
        return super().identity_payload()


@dataclass(frozen=True)
class Model(Entity):
    """A model version, down to precision and tokenizer.

    "GPT-style model" is not a scientific reference.  ``foo-7b`` at
    ``bf16`` with tokenizer ``v2`` and weights ``sha256:...`` is.
    """

    name: str
    version: str = "v1"
    weights_digest: str = ""
    architecture: str = ""
    commit: str = ""
    precision: str = ""
    quantization: str = "none"
    tokenizer: str = ""
    parameters: int | None = None
    provider: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    local_path: str = ""
    registered_at: float = 0.0
    id_prefix: ClassVar[str] = "mdl-"

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"

    def identity_payload(self) -> dict[str, Any]:
        """Identity is the weights plus everything that changes inference."""
        if self.weights_digest:
            return {
                "__kind__": "Model",
                "name": self.name,
                "version": self.version,
                "weights_digest": self.weights_digest,
                "precision": self.precision,
                "quantization": self.quantization,
                "tokenizer": self.tokenizer,
                "config": self.config,
            }
        return super().identity_payload()


@dataclass(frozen=True)
class Algorithm(Entity):
    """The method under study, pinned to source code.

    ``entrypoint`` is deliberately framework-agnostic; see
    :mod:`fabric.execution.adapters` for the supported forms.
    """

    name: str
    entrypoint: str = ""
    commit: str = ""
    source_digest: str = ""
    version: str = "v1"
    description: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    dirty: bool = False
    """The working tree had uncommitted changes when this was captured.

    Recorded but *excluded from identity*. What pins the method is
    ``source_digest``, the hash of the entrypoint's own source file; the
    commit is provenance around it. If dirtiness were part of identity,
    editing any unrelated file in the repository would mint a new
    algorithm version and invalidate every cached run.

    The honest caveat, which the certificate states rather than hides: a
    dirty tree means the commit does not fully describe the code, and a
    change in a *helper* module the entrypoint imports is not covered by
    ``source_digest``. Commit before the runs that matter.
    """
    registered_at: float = 0.0
    id_prefix: ClassVar[str] = "alg-"

    @property
    def ref(self) -> str:
        short = self.commit[:7] if self.commit else self.version
        return f"{self.name}@{short}"


@dataclass(frozen=True)
class Environment(Entity):
    """Captured software environment."""

    python_version: str = ""
    platform: str = ""
    os_name: str = ""
    os_release: str = ""
    packages: dict[str, str] = field(default_factory=dict)
    interpreter: str = ""
    git_commit: str = ""
    git_dirty: bool = False
    container: str = ""
    lockfiles: dict[str, str] = field(default_factory=dict)
    env_vars: dict[str, str] = field(default_factory=dict)
    accelerator_runtime: dict[str, str] = field(default_factory=dict)
    captured_at: float = 0.0
    id_prefix: ClassVar[str] = "env-"

    @staticmethod
    def _excluded_fields() -> frozenset[str]:
        return Entity._excluded_fields() | {"captured_at"}


@dataclass(frozen=True)
class Hardware(Entity):
    """Captured hardware context.

    Recorded, never assumed: moving from an A100 to an H100 is an
    experimental perturbation, and Fabric can only measure it if both
    runs said what they ran on.
    """

    machine_id: str = ""
    cpu_model: str = ""
    cpu_count: int = 0
    ram_bytes: int = 0
    architecture: str = ""
    gpus: list[dict[str, Any]] = field(default_factory=list)
    accelerator: str = "cpu"
    driver: str = ""
    distributed: dict[str, Any] = field(default_factory=dict)
    captured_at: float = 0.0
    id_prefix: ClassVar[str] = "hw-"

    @staticmethod
    def _excluded_fields() -> frozenset[str]:
        return Entity._excluded_fields() | {"captured_at"}

    @property
    def label(self) -> str:
        if self.gpus:
            names = sorted({str(g.get("name", "gpu")) for g in self.gpus})
            return f"{len(self.gpus)}x {', '.join(names)}"
        return self.cpu_model or self.architecture or "unknown"


# --------------------------------------------------------------------------
# metrics and artifacts
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricSpec:
    """Declaration of a measurable quantity."""

    name: str
    direction: Direction = Direction.MAXIMIZE
    unit: str = ""
    primary: bool = False
    description: str = ""
    tolerance: float | None = None
    """Practical-equivalence margin, in metric units.

    When set, Fabric can run equivalence tests (TOST) instead of only
    difference tests, which is what a reproduction attempt actually wants
    to know: not "is it different?" but "is it the same within X?".
    """

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "direction": self.direction.value,
            "unit": self.unit,
            "primary": self.primary,
            "description": self.description,
            "tolerance": self.tolerance,
        }

    @staticmethod
    def from_any(value: Any) -> MetricSpec:
        """Accept ``"accuracy"``, ``{"accuracy": "maximize"}`` or a full mapping."""
        if isinstance(value, str):
            return MetricSpec(name=value, direction=_infer_direction(value))
        if isinstance(value, Mapping):
            data = dict(value)
            if len(data) == 1 and "name" not in data:
                (name, spec), = data.items()
                if isinstance(spec, str):
                    return MetricSpec(name=str(name), direction=Direction(spec))
                data = {"name": name, **(spec or {})}
            name = str(data["name"])
            direction = data.get("direction")
            return MetricSpec(
                name=name,
                direction=Direction(direction) if direction else _infer_direction(name),
                unit=str(data.get("unit", "")),
                primary=bool(data.get("primary", False)),
                description=str(data.get("description", "")),
                tolerance=_opt_float(data.get("tolerance")),
            )
        raise TypeError(f"cannot read a metric specification from {value!r}")


_LOWER_IS_BETTER = ("loss", "error", "latency", "time", "cost", "memory", "perplexity", "regret", "rate_of_failure")


def _infer_direction(name: str) -> Direction:
    lowered = name.lower()
    if any(token in lowered for token in _LOWER_IS_BETTER):
        return Direction.MINIMIZE
    return Direction.MAXIMIZE


def _opt_float(value: Any) -> float | None:
    return None if value is None else float(value)


@dataclass(frozen=True)
class Artifact:
    """An immutable file produced by a run."""

    name: str
    path: str
    digest: str
    bytes: int = 0
    media_type: str = "application/octet-stream"
    role: str = "output"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Failure:
    """A classified run failure."""

    failure_class: FailureClass
    message: str
    traceback: str = ""
    exit_code: int | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["failure_class"] = self.failure_class.value
        return data


@dataclass(frozen=True)
class Resources:
    """Resource accounting for a run.

    Accuracy without cost is half a result.  These fields let a report ask
    whether +0.2 points was worth 6.5x the compute.
    """

    wall_seconds: float = 0.0
    cpu_seconds: float = 0.0
    gpu_seconds: float = 0.0
    peak_rss_bytes: int = 0
    gpu_memory_bytes: int = 0
    energy_joules: float | None = None
    cost_usd: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# experimental structure
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Condition(Entity):
    """One experimental arm: everything except the seed.

    A condition is the unit that gets *compared*; a run is the unit that
    gets *executed*.  Keeping them distinct is what lets Fabric say "these
    ten runs are replications of the same condition" rather than "here are
    ten unrelated numbers".
    """

    name: str
    role: Role = Role.TREATMENT
    algorithm_id: str = ""
    model_id: str = ""
    dataset_id: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    ablated: tuple[str, ...] = ()
    group: str = "main"
    """Comparison group: only conditions in the same group are compared."""
    axis: dict[str, Any] = field(default_factory=dict)
    """Which sweep coordinates produced this condition (for slicing later)."""
    perturbation: dict[str, Any] = field(default_factory=dict)
    """Non-empty when this condition is a reproducibility counterfactual."""
    id_prefix: ClassVar[str] = "cond-"

    @property
    def is_perturbed(self) -> bool:
        return bool(self.perturbation)


@dataclass(frozen=True)
class Run:
    """A single execution: one condition at one seed.

    Mutable by design during execution, then frozen into ``run.json``.
    """

    id: str
    experiment_id: str
    condition_id: str
    condition_name: str
    seed: int
    role: Role = Role.TREATMENT
    status: RunStatus = RunStatus.PENDING
    config: dict[str, Any] = field(default_factory=dict)
    dataset_id: str = ""
    model_id: str = ""
    algorithm_id: str = ""
    environment_id: str = ""
    hardware_id: str = ""
    ablated: tuple[str, ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: list[Artifact] = field(default_factory=list)
    resources: Resources = field(default_factory=Resources)
    failure: Failure | None = None
    cache_key: str = ""
    cached_from: str = ""
    replicate: int = 0
    """Index of a repeated execution of an identical run (determinism checks)."""
    started_at: float = 0.0
    finished_at: float = 0.0
    logs: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        if not self.finished_at:
            return 0.0
        return max(0.0, self.finished_at - self.started_at)

    @property
    def succeeded(self) -> bool:
        return self.status in (RunStatus.COMPLETED, RunStatus.CACHED)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        data["role"] = self.role.value
        data["ablated"] = list(self.ablated)
        data["artifacts"] = [a.to_dict() for a in self.artifacts]
        data["resources"] = self.resources.to_dict()
        data["failure"] = self.failure.to_dict() if self.failure else None
        data["duration_seconds"] = self.duration
        return data

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> Run:
        failure = data.get("failure")
        resources = data.get("resources") or {}
        return Run(
            id=str(data["id"]),
            experiment_id=str(data.get("experiment_id", "")),
            condition_id=str(data.get("condition_id", "")),
            condition_name=str(data.get("condition_name", "")),
            seed=int(data.get("seed", 0)),
            role=Role(data.get("role", "treatment")),
            status=RunStatus(data.get("status", "pending")),
            config=dict(data.get("config") or {}),
            dataset_id=str(data.get("dataset_id", "")),
            model_id=str(data.get("model_id", "")),
            algorithm_id=str(data.get("algorithm_id", "")),
            environment_id=str(data.get("environment_id", "")),
            hardware_id=str(data.get("hardware_id", "")),
            ablated=tuple(data.get("ablated") or ()),
            metrics={str(k): float(v) for k, v in (data.get("metrics") or {}).items()},
            artifacts=[Artifact(**a) for a in (data.get("artifacts") or [])],
            resources=Resources(**{k: v for k, v in resources.items() if k in Resources.__annotations__}),
            failure=(
                Failure(
                    failure_class=FailureClass(failure.get("failure_class", "unknown")),
                    message=str(failure.get("message", "")),
                    traceback=str(failure.get("traceback", "")),
                    exit_code=failure.get("exit_code"),
                )
                if failure
                else None
            ),
            cache_key=str(data.get("cache_key", "")),
            cached_from=str(data.get("cached_from", "")),
            replicate=int(data.get("replicate", 0)),
            started_at=float(data.get("started_at", 0.0)),
            finished_at=float(data.get("finished_at", 0.0)),
            logs=dict(data.get("logs") or {}),
            extra=dict(data.get("extra") or {}),
        )


@dataclass(frozen=True)
class Experiment:
    """The top-level scientific object."""

    id: str
    name: str
    research_question: str = ""
    hypothesis: str = ""
    spec_digest: str = ""
    created_at: float = field(default_factory=time.time)
    tags: tuple[str, ...] = ()
    metrics: tuple[MetricSpec, ...] = ()
    primary_metric: str = ""
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "research_question": self.research_question,
            "hypothesis": self.hypothesis,
            "spec_digest": self.spec_digest,
            "created_at": self.created_at,
            "tags": list(self.tags),
            "metrics": [m.to_dict() for m in self.metrics],
            "primary_metric": self.primary_metric,
            "description": self.description,
        }

    def metric(self, name: str) -> MetricSpec:
        for spec in self.metrics:
            if spec.name == name:
                return spec
        return MetricSpec(name=name, direction=_infer_direction(name))


def metric_specs(values: Iterable[Any]) -> tuple[MetricSpec, ...]:
    """Normalise a heterogeneous metric list from a specification."""
    specs = tuple(MetricSpec.from_any(v) for v in values)
    if specs and not any(s.primary for s in specs):
        first = specs[0]
        specs = (replace(first, primary=True),) + specs[1:]
    return specs
