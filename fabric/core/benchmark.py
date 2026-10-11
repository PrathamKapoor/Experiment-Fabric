"""Benchmarks as reusable, versioned scientific contracts.

A benchmark is the part of an experiment that is *not* the researcher's
contribution: the task, the data, the metrics, the arms everyone compares
against, and the rules a result has to satisfy before it means anything.
Today that part gets retyped into every specification, which is how two
papers end up reporting "accuracy on benchmark-x" having measured
different things.

So Fabric makes it a first-class object with its own identity::

    benchmark: agent-qa@v1

The definition behind that reference contributes its metrics, datasets,
baselines and constraints to the experiment, and its digest travels into
the experiment's identity.  Change the benchmark and every experiment
that used it is provably a different experiment.

Two rules keep this from becoming a configuration framework:

*   A benchmark contributes, the experiment refines.  The experiment may
    add metrics and baselines; it may not silently redefine one the
    benchmark pinned.  A contradiction is an error with a location, not a
    last-writer-wins merge.
*   Constraints are checked, not decorative.  ``min_seeds: 5`` means an
    experiment with three seeds is told so before the compute is spent.

What a benchmark is deliberately *not*: a place to put scripts.  It
declares what must be measured and against what; the code that measures
it stays in the experiment's algorithm, which is the part that varies.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from fabric.util.hashing import full_digest
from fabric.util.yamlio import load_yaml

BENCHMARK_SCHEMA_VERSION = "1.0"

#: Environment variable holding extra ``:``-separated search directories.
SEARCH_PATH_VARIABLE = "FABRIC_BENCHMARK_PATH"


class BenchmarkError(Exception):
    """A benchmark definition, or its use, is invalid."""

    def __init__(self, message: str, path: str = "") -> None:
        self.path = path
        super().__init__(f"{path}: {message}" if path else message)


# --------------------------------------------------------------------------
# pieces
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskSpec:
    """One task within a benchmark.

    Most benchmarks have exactly one.  Those that do not (a suite with a
    retrieval task and a generation task) need the metrics attributed to
    the right one, or the aggregate means nothing.
    """

    name: str
    description: str = ""
    dataset: str = ""
    """Name of the benchmark dataset this task evaluates on."""
    metrics: tuple[str, ...] = ()
    """Metric names this task reports; empty means all of the benchmark's."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def parse(value: Any, index: int) -> TaskSpec:
        path = f"tasks[{index}]"
        if isinstance(value, str):
            return TaskSpec(name=value)
        if not isinstance(value, Mapping):
            raise BenchmarkError("a task must be a name or a mapping", path)
        name = str(value.get("name") or "")
        if not name:
            raise BenchmarkError("a task needs a name", path)
        return TaskSpec(
            name=name,
            description=str(value.get("description", "")),
            dataset=str(value.get("dataset", "")),
            metrics=tuple(str(m) for m in value.get("metrics", ()) or ()),
        )


@dataclass(frozen=True)
class BenchmarkDataset:
    """A dataset the benchmark is defined on, by identity rather than bytes.

    Fabric tracks *which* data, not the data itself.  A benchmark may name
    a local path, a URL, or an object-store key; what it must always carry
    is a version and, where the bytes are reachable, a digest.
    """

    name: str
    version: str = "v1"
    digest: str = ""
    path: str = ""
    uri: str = ""
    """External location for data too large to sit beside the definition."""
    split: str = ""
    n_examples: int | None = None
    license: str = ""
    source: str = ""

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def identity(self) -> dict[str, Any]:
        """Only the fields that decide which data this is."""
        return {"name": self.name, "version": self.version, "digest": self.digest}

    @staticmethod
    def parse(value: Any, index: int) -> BenchmarkDataset:
        path = f"datasets[{index}]"
        if isinstance(value, str):
            name, _, version = value.partition("@")
            return BenchmarkDataset(name=name, version=version or "v1")
        if not isinstance(value, Mapping):
            raise BenchmarkError("a dataset must be a reference or a mapping", path)
        name = str(value.get("name") or "")
        if not name:
            raise BenchmarkError("a dataset needs a name", path)
        n = value.get("n_examples")
        return BenchmarkDataset(
            name=name,
            version=str(value.get("version", "v1")),
            digest=str(value.get("digest", "")),
            path=str(value.get("path", "")),
            uri=str(value.get("uri", "")),
            split=str(value.get("split", "")),
            n_examples=int(n) if n is not None else None,
            license=str(value.get("license", "")),
            source=str(value.get("source", "")),
        )


@dataclass(frozen=True)
class BenchmarkMetric:
    """A metric the benchmark defines, with the rules for reading it.

    ``pinned`` is the important flag.  A pinned metric is part of the
    benchmark's contract: an experiment that redefines its direction or
    tolerance is no longer reporting the same quantity, and Fabric says so
    rather than letting the two silently diverge.
    """

    name: str
    direction: str = "maximize"
    aggregation: str = "mean"
    unit: str = ""
    tolerance: float | None = None
    primary: bool = False
    required: bool = True
    """Whether an experiment claiming this benchmark must report it."""
    pinned: bool = True
    """Whether the experiment may redefine direction, tolerance or unit."""
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def as_metric_spec(self) -> dict[str, Any]:
        """The mapping the experiment's own metric parser understands."""
        data: dict[str, Any] = {
            "name": self.name,
            "direction": self.direction,
            "aggregation": self.aggregation,
            "primary": self.primary,
        }
        if self.unit:
            data["unit"] = self.unit
        if self.tolerance is not None:
            data["tolerance"] = self.tolerance
        if self.description:
            data["description"] = self.description
        return data

    @staticmethod
    def parse(value: Any, index: int) -> BenchmarkMetric:
        path = f"metrics[{index}]"
        if isinstance(value, str):
            return BenchmarkMetric(name=value)
        if not isinstance(value, Mapping):
            raise BenchmarkError("a metric must be a name or a mapping", path)
        name = str(value.get("name") or "")
        if not name:
            raise BenchmarkError("a metric needs a name", path)
        direction = str(value.get("direction", "maximize")).lower()
        if direction not in ("maximize", "minimize", "neutral"):
            raise BenchmarkError(
                f"direction must be maximize, minimize or neutral, not {direction!r}",
                f"{path}.direction",
            )
        tolerance = value.get("tolerance")
        return BenchmarkMetric(
            name=name,
            direction=direction,
            aggregation=str(value.get("aggregation", "mean")).lower(),
            unit=str(value.get("unit", "")),
            tolerance=float(tolerance) if tolerance is not None else None,
            primary=bool(value.get("primary", False)),
            required=bool(value.get("required", True)),
            pinned=bool(value.get("pinned", True)),
            description=str(value.get("description", "")),
        )


@dataclass(frozen=True)
class BenchmarkConstraint:
    """One rule a conforming experiment must satisfy.

    Constraints exist so that "we evaluated on benchmark-x" carries
    information.  A benchmark whose authors think five seeds is the
    minimum for a meaningful claim can say so once, here, instead of in a
    paragraph of a README that nobody reads.
    """

    kind: str
    value: Any = None
    reason: str = ""

    KINDS = (
        "min_seeds",
        "max_alpha",
        "require_baselines",
        "require_dataset_digest",
        "require_determinism_check",
        "required_metrics",
    )

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "value": self.value, "reason": self.reason}

    @staticmethod
    def parse(value: Any, index: int) -> BenchmarkConstraint:
        path = f"constraints[{index}]"
        if isinstance(value, str):
            kind, _, literal = value.partition(":")
            parsed: Any = literal.strip() or True
            if isinstance(parsed, str) and parsed.isdigit():
                parsed = int(parsed)
            return BenchmarkConstraint(kind=kind.strip(), value=parsed)
        if not isinstance(value, Mapping):
            raise BenchmarkError("a constraint must be a string or a mapping", path)
        kind = str(value.get("kind") or "")
        if kind not in BenchmarkConstraint.KINDS:
            raise BenchmarkError(
                f"unknown constraint {kind!r}; expected one of "
                f"{list(BenchmarkConstraint.KINDS)}",
                path,
            )
        return BenchmarkConstraint(
            kind=kind,
            value=value.get("value", True),
            reason=str(value.get("reason", "")),
        )


# --------------------------------------------------------------------------
# the benchmark itself
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkSpec:
    """A versioned benchmark definition with a content-addressed identity."""

    name: str
    version: str = "v1"
    description: str = ""
    schema_version: str = BENCHMARK_SCHEMA_VERSION
    citation: str = ""
    license: str = ""
    tasks: tuple[TaskSpec, ...] = ()
    datasets: tuple[BenchmarkDataset, ...] = ()
    metrics: tuple[BenchmarkMetric, ...] = ()
    baselines: tuple[dict[str, Any], ...] = ()
    """Baseline arms in the experiment specification's own shape.

    Kept as raw mappings on purpose: the experiment's ``BaselineSpec``
    parser is the single authority on what a baseline may contain, and
    duplicating it here is how the two would drift apart.
    """
    constraints: tuple[BenchmarkConstraint, ...] = ()
    output_schema: dict[str, Any] = field(default_factory=dict)
    """What a conforming run must return, as ``{metric: type}``."""
    source_path: str = ""

    # -- identity ----------------------------------------------------------

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"

    @property
    def digest(self) -> str:
        """Content digest over everything that defines the benchmark."""
        return full_digest(self.identity_payload())

    def identity_payload(self) -> dict[str, Any]:
        return {
            "__kind__": "Benchmark",
            "name": self.name,
            "version": self.version,
            "schema_version": self.schema_version,
            "tasks": [t.to_dict() for t in self.tasks],
            "datasets": [d.identity() for d in self.datasets],
            "metrics": [m.to_dict() for m in self.metrics],
            "baselines": list(self.baselines),
            "constraints": [c.to_dict() for c in self.constraints],
            "output_schema": self.output_schema,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "ref": self.ref,
            "digest": self.digest,
            "description": self.description,
            "schema_version": self.schema_version,
            "citation": self.citation,
            "license": self.license,
            "tasks": [t.to_dict() for t in self.tasks],
            "datasets": [d.to_dict() for d in self.datasets],
            "metrics": [m.to_dict() for m in self.metrics],
            "baselines": list(self.baselines),
            "constraints": [c.to_dict() for c in self.constraints],
            "output_schema": self.output_schema,
            "source_path": self.source_path,
        }

    # -- convenience -------------------------------------------------------

    @property
    def primary_metric(self) -> str:
        for metric in self.metrics:
            if metric.primary:
                return metric.name
        return self.metrics[0].name if self.metrics else ""

    @property
    def required_metrics(self) -> tuple[str, ...]:
        return tuple(m.name for m in self.metrics if m.required)

    def metric(self, name: str) -> BenchmarkMetric | None:
        for metric in self.metrics:
            if metric.name == name:
                return metric
        return None

    def constraint(self, kind: str) -> BenchmarkConstraint | None:
        for item in self.constraints:
            if item.kind == kind:
                return item
        return None

    def describe(self) -> str:
        parts = [f"{self.ref}"]
        if self.tasks:
            parts.append(f"{len(self.tasks)} task(s)")
        if self.datasets:
            parts.append(", ".join(d.ref for d in self.datasets))
        if self.metrics:
            parts.append(f"metrics: {', '.join(m.name for m in self.metrics)}")
        if self.constraints:
            parts.append(f"{len(self.constraints)} constraint(s)")
        return " -- ".join(parts)

    # -- construction ------------------------------------------------------

    @staticmethod
    def from_dict(data: Mapping[str, Any], *, source_path: str = "") -> BenchmarkSpec:
        if not isinstance(data, Mapping):
            raise BenchmarkError("a benchmark definition must be a mapping")
        header = data.get("benchmark")
        if isinstance(header, str):
            header = {"name": header}
        elif header is None:
            header = {}
        elif not isinstance(header, Mapping):
            raise BenchmarkError("'benchmark' must be a name or a mapping", "benchmark")

        name = str(header.get("name") or data.get("name") or "")
        if not name:
            raise BenchmarkError("a benchmark needs a name", "benchmark.name")

        schema_version = str(header.get("schema_version")
                             or data.get("schema_version")
                             or BENCHMARK_SCHEMA_VERSION)
        major = schema_version.split(".")[0]
        if major != BENCHMARK_SCHEMA_VERSION.split(".")[0]:
            raise BenchmarkError(
                f"benchmark '{name}' declares schema version {schema_version}, and this "
                f"Fabric understands {BENCHMARK_SCHEMA_VERSION}. Refusing to guess what "
                "the fields mean rather than reinterpreting them under a newer schema.",
                "benchmark.schema_version",
            )

        metrics = tuple(BenchmarkMetric.parse(v, i)
                        for i, v in enumerate(data.get("metrics") or ()))
        if not metrics:
            raise BenchmarkError(
                "a benchmark must declare at least one metric; a benchmark that does "
                "not say what to measure is a folder, not a benchmark",
                "metrics",
            )
        if sum(1 for m in metrics if m.primary) > 1:
            raise BenchmarkError("only one metric may be primary", "metrics")

        seen: set[str] = set()
        for metric in metrics:
            if metric.name in seen:
                raise BenchmarkError(f"duplicate metric {metric.name!r}", "metrics")
            seen.add(metric.name)

        datasets = tuple(BenchmarkDataset.parse(v, i)
                         for i, v in enumerate(data.get("datasets") or ()))
        tasks = tuple(TaskSpec.parse(v, i)
                      for i, v in enumerate(data.get("tasks") or ()))

        dataset_names = {d.name for d in datasets}
        for task in tasks:
            if task.dataset and task.dataset not in dataset_names:
                raise BenchmarkError(
                    f"task {task.name!r} evaluates on dataset {task.dataset!r}, which "
                    f"the benchmark does not define (it defines "
                    f"{sorted(dataset_names) or 'none'})",
                    f"tasks.{task.name}.dataset",
                )
            for metric_name in task.metrics:
                if metric_name not in seen:
                    raise BenchmarkError(
                        f"task {task.name!r} reports metric {metric_name!r}, which the "
                        f"benchmark does not define",
                        f"tasks.{task.name}.metrics",
                    )

        baselines = tuple(
            dict(b) if isinstance(b, Mapping) else {"name": str(b)}
            for b in data.get("baselines") or ()
        )
        constraints = tuple(BenchmarkConstraint.parse(v, i)
                            for i, v in enumerate(data.get("constraints") or ()))

        required = data.get("required_metrics")
        if required:
            constraints += (BenchmarkConstraint(
                kind="required_metrics",
                value=[str(m) for m in required],
                reason="declared by the benchmark's required_metrics list",
            ),)

        return BenchmarkSpec(
            name=name,
            version=str(header.get("version") or data.get("version") or "v1"),
            description=str(header.get("description") or data.get("description") or ""),
            schema_version=schema_version,
            citation=str(header.get("citation") or data.get("citation") or ""),
            license=str(header.get("license") or data.get("license") or ""),
            tasks=tasks,
            datasets=datasets,
            metrics=metrics,
            baselines=baselines,
            constraints=constraints,
            output_schema=dict(data.get("output_schema") or {}),
            source_path=source_path,
        )

    @staticmethod
    def from_file(path: str | os.PathLike[str]) -> BenchmarkSpec:
        data = load_yaml(path)
        if data is None:
            raise BenchmarkError(f"{os.fspath(path)} is empty")
        return BenchmarkSpec.from_dict(data, source_path=os.path.abspath(os.fspath(path)))


# --------------------------------------------------------------------------
# the registry: finding a benchmark by reference
# --------------------------------------------------------------------------


def search_paths(*, root: str = "", spec_path: str = "") -> list[str]:
    """Directories searched for benchmark definitions, in priority order.

    Nearest first: the experiment's own directory, then the working tree,
    then anything the environment adds, then the definitions that ship
    with Fabric.  A project can therefore pin its own copy of a benchmark
    without touching the installation.
    """
    candidates: list[str] = []
    if spec_path:
        directory = os.path.dirname(os.path.abspath(spec_path))
        candidates += [os.path.join(directory, "benchmarks"), directory]
    candidates.append(os.path.join(os.path.abspath(root or os.getcwd()), "benchmarks"))
    for entry in os.environ.get(SEARCH_PATH_VARIABLE, "").split(os.pathsep):
        if entry.strip():
            candidates.append(os.path.abspath(entry.strip()))
    # The definitions that ship inside the package, at fabric/benchmarks/.
    # Two dirnames from fabric/core/benchmark.py is the fabric/ directory,
    # so this resolves inside the installed distribution and works for a
    # wheel as well as a checkout.
    candidates.append(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "benchmarks"))

    ordered: list[str] = []
    for path in candidates:
        if path not in ordered and os.path.isdir(path):
            ordered.append(path)
    return ordered


def available(*, root: str = "", spec_path: str = "") -> list[BenchmarkSpec]:
    """Every benchmark definition on the search path, nearest first.

    A name found in an earlier directory shadows a later one, so the list
    holds at most one definition per ``name@version``.
    """
    found: dict[str, BenchmarkSpec] = {}
    for directory in search_paths(root=root, spec_path=spec_path):
        for entry in sorted(os.listdir(directory)):
            if not entry.endswith((".yaml", ".yml")):
                continue
            path = os.path.join(directory, entry)
            try:
                benchmark = BenchmarkSpec.from_file(path)
            except (BenchmarkError, OSError, ValueError):
                continue
            found.setdefault(benchmark.ref, benchmark)
    return sorted(found.values(), key=_version_order)


def _version_order(benchmark: BenchmarkSpec) -> tuple[Any, tuple[int, ...], Any]:
    """Sort by name, then by version compared as a number where possible.

    ``v10`` is a later version than ``v2``; sorting the strings makes it
    look earlier, so an unversioned reference would resolve to the older
    contract. A version that is not numeric at all falls back to comparing
    the string, which keeps an unparseable version orderable rather than
    dropping it from the catalogue.
    """
    parts: list[tuple[int, int, str]] = []
    for piece in re.findall(r"\d+|\D+", benchmark.version):
        if piece.isdigit():
            parts.append((1, int(piece), ""))
        else:
            parts.append((0, 0, piece))
    return (benchmark.name, tuple(parts), benchmark.version)


def resolve(reference: Any, *, root: str = "", spec_path: str = "") -> BenchmarkSpec:
    """Turn ``agent-qa@v1``, or a mapping, into a loaded benchmark.

    Accepts three shapes, in the order a user is likely to reach for them:
    a bare reference, a mapping with an explicit ``path``, and a mapping
    with ``name``/``version``.
    """
    if isinstance(reference, BenchmarkSpec):
        return reference

    explicit_path = ""
    name = version = ""
    expected_digest = ""
    if isinstance(reference, str):
        text = reference.strip()
        if text.endswith((".yaml", ".yml")) or os.sep in text or "/" in text:
            explicit_path = text
        else:
            name, _, version = text.partition("@")
    elif isinstance(reference, Mapping):
        explicit_path = str(reference.get("path", "") or "")
        name = str(reference.get("name", "") or "")
        version = str(reference.get("version", "") or "")
        expected_digest = str(reference.get("digest", "") or "")
        if not explicit_path and not name:
            raise BenchmarkError(
                "a benchmark reference needs a 'name' or a 'path'", "benchmark")
    else:
        raise BenchmarkError(
            "'benchmark' must be a reference like 'agent-qa@v1', a path to a "
            "definition, or a mapping", "benchmark")

    if explicit_path:
        candidates = [explicit_path]
        if spec_path:
            candidates.append(os.path.join(
                os.path.dirname(os.path.abspath(spec_path)), explicit_path))
        candidates.append(os.path.join(os.path.abspath(root or os.getcwd()),
                                       explicit_path))
        for candidate in candidates:
            if os.path.isfile(candidate):
                benchmark = BenchmarkSpec.from_file(candidate)
                _check_digest(benchmark, expected_digest)
                return benchmark
        raise BenchmarkError(
            f"no benchmark definition at {explicit_path!r} (looked in "
            f"{', '.join(candidates)})", "benchmark")

    catalogue = available(root=root, spec_path=spec_path)
    # `catalogue` is already ordered by _version_order, so the last match is
    # the newest -- taking the first would silently serve v2 over v10.
    matches = [b for b in catalogue
               if b.name == name and (not version or b.version == version)]
    if matches:
        _check_digest(matches[-1], expected_digest)
        return matches[-1]

    known = ", ".join(b.ref for b in catalogue) or "none"
    directories = ", ".join(search_paths(root=root, spec_path=spec_path)) or "none"
    wanted = f"{name}@{version}" if version else name
    raise BenchmarkError(
        f"no benchmark named {wanted!r} was found. Searched: {directories}. "
        f"Available: {known}. Add a definition to a 'benchmarks' directory, or "
        f"set {SEARCH_PATH_VARIABLE} to where yours live.",
        "benchmark",
    )


def _check_digest(benchmark: BenchmarkSpec, expected: str) -> None:
    """A pinned digest that no longer matches is a changed benchmark."""
    if expected and benchmark.digest != expected:
        raise BenchmarkError(
            f"benchmark {benchmark.ref} has digest {benchmark.digest[:19]}..., but the "
            f"experiment pinned {expected[:19]}.... The definition has changed since "
            "this experiment was written, so results are not comparable to the ones "
            "that pinned the old digest. Update the pin deliberately, or restore the "
            "definition.",
            "benchmark.digest",
        )


# --------------------------------------------------------------------------
# applying a benchmark to a specification
# --------------------------------------------------------------------------


@dataclass
class Application:
    """The result of merging a benchmark into a raw specification."""

    benchmark: BenchmarkSpec
    data: dict[str, Any]
    contributed: list[str] = field(default_factory=list)
    """Human-readable notes on what the benchmark supplied."""


def apply(benchmark: BenchmarkSpec, raw: Mapping[str, Any]) -> Application:
    """Merge a benchmark's contract into a raw specification mapping.

    Works on the raw mapping rather than the parsed objects so that every
    existing parser remains the single authority on its own section. The
    benchmark contributes; the experiment refines; a contradiction on a
    pinned field raises rather than resolving silently.
    """
    data = dict(raw)
    notes: list[str] = []

    # -- metrics -----------------------------------------------------------
    declared = list(data.get("metrics") or ())
    by_name: dict[str, Any] = {}
    order: list[str] = []
    for entry in declared:
        entry_name = entry if isinstance(entry, str) else str(
            (entry or {}).get("name", ""))
        if entry_name:
            by_name[entry_name] = entry
            order.append(entry_name)

    merged: list[Any] = []
    for metric in benchmark.metrics:
        existing = by_name.pop(metric.name, None)
        if existing is None:
            merged.append(metric.as_metric_spec())
            continue
        if isinstance(existing, Mapping) and metric.pinned:
            _refuse_metric_override(benchmark, metric, existing)
        combined = dict(metric.as_metric_spec())
        if isinstance(existing, Mapping):
            combined.update({k: v for k, v in existing.items() if k != "name"})
        merged.append(combined)
    for name in order:
        if name in by_name:
            merged.append(by_name[name])
    data["metrics"] = merged
    notes.append(
        f"metrics {', '.join(m.name for m in benchmark.metrics)} defined by "
        f"{benchmark.ref}"
    )

    # -- primary metric ----------------------------------------------------
    analysis = dict(data.get("analysis") or {})
    if benchmark.primary_metric and not analysis.get("primary_metric"):
        analysis["primary_metric"] = benchmark.primary_metric
        notes.append(f"primary metric '{benchmark.primary_metric}' from {benchmark.ref}")
    if analysis:
        data["analysis"] = analysis

    # -- dataset -----------------------------------------------------------
    if benchmark.datasets and not data.get("dataset"):
        first = benchmark.datasets[0]
        dataset: dict[str, Any] = {"name": first.name, "version": first.version}
        if first.digest:
            dataset["digest"] = first.digest
        if first.path:
            dataset["path"] = first.path
        if first.uri:
            dataset["uri"] = first.uri
        if first.source:
            dataset["source"] = first.source
        if first.license:
            dataset["license"] = first.license
        if first.n_examples is not None:
            dataset["n_examples"] = first.n_examples
        data["dataset"] = dataset
        notes.append(f"dataset {first.ref} from {benchmark.ref}")

    # -- baselines ---------------------------------------------------------
    if benchmark.baselines:
        own = list(data.get("baselines") or ())
        own_names = {
            (b if isinstance(b, str) else str((b or {}).get("name", "")))
            for b in own
        }
        inherited = [dict(b) for b in benchmark.baselines
                     if str(b.get("name", "")) not in own_names]
        if inherited:
            data["baselines"] = inherited + own
            notes.append(
                f"baselines {', '.join(str(b['name']) for b in inherited)} "
                f"from {benchmark.ref}"
            )

    return Application(benchmark=benchmark, data=data, contributed=notes)


def _refuse_metric_override(benchmark: BenchmarkSpec, metric: BenchmarkMetric,
                            declared: Mapping[str, Any]) -> None:
    """A pinned metric redefined by the experiment is a silent divergence."""
    for field_name, benchmark_value in (("direction", metric.direction),
                                        ("tolerance", metric.tolerance),
                                        ("unit", metric.unit)):
        if field_name not in declared:
            continue
        given = declared[field_name]
        if field_name == "tolerance" and given is not None:
            given = float(given)
        if field_name == "direction":
            given = str(given).lower()
        if benchmark_value in (None, "") and given in (None, ""):
            continue
        if given != benchmark_value:
            raise BenchmarkError(
                f"benchmark {benchmark.ref} pins {metric.name}.{field_name} to "
                f"{benchmark_value!r}, and this experiment declares {given!r}. Two "
                f"experiments that disagree here are not measuring the same thing, "
                f"so 'evaluated on {benchmark.name}' would mean nothing. Either match "
                f"the benchmark, mark the metric 'pinned: false' in the definition, or "
                f"rename the metric to make the difference explicit.",
                f"metrics.{metric.name}.{field_name}",
            )


# --------------------------------------------------------------------------
# constraints
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ConstraintResult:
    """Whether one benchmark constraint was satisfied."""

    kind: str
    satisfied: bool
    detail: str
    reason: str = ""
    fatal: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "satisfied": self.satisfied,
            "detail": self.detail,
            "reason": self.reason,
            "fatal": self.fatal,
        }


def check_constraints(benchmark: BenchmarkSpec, spec: Any) -> list[ConstraintResult]:
    """Evaluate every constraint against a parsed :class:`ExperimentSpec`.

    Returns results rather than raising: whether an unsatisfied constraint
    blocks the run is the caller's decision, and section 20 of the design
    is explicit that Fabric should make bad designs *visible* rather than
    forbidden.
    """
    results: list[ConstraintResult] = []
    metric_names = {m.name for m in spec.metrics}

    for constraint in benchmark.constraints:
        kind = constraint.kind
        if kind == "min_seeds":
            wanted = int(constraint.value)
            have = len(spec.seeds.values)
            results.append(ConstraintResult(
                kind, have >= wanted,
                f"{have} seed(s) declared, {benchmark.ref} requires at least {wanted}",
                constraint.reason,
            ))
        elif kind == "max_alpha":
            wanted = float(constraint.value)
            results.append(ConstraintResult(
                kind, spec.analysis.alpha <= wanted,
                f"alpha={spec.analysis.alpha}, {benchmark.ref} requires "
                f"alpha <= {wanted}",
                constraint.reason,
            ))
        elif kind == "require_baselines":
            wanted = constraint.value
            if isinstance(wanted, bool):
                have = len(spec.baselines)
                results.append(ConstraintResult(
                    kind, have > 0 if wanted else True,
                    f"{have} baseline(s) declared, {benchmark.ref} requires at least one",
                    constraint.reason,
                ))
            else:
                names = [str(n) for n in (wanted if isinstance(wanted, list) else [wanted])]
                declared = {b.name for b in spec.baselines}
                missing = [n for n in names if n not in declared]
                results.append(ConstraintResult(
                    kind, not missing,
                    f"required baselines missing: {', '.join(missing)}" if missing
                    else f"all required baselines present: {', '.join(names)}",
                    constraint.reason,
                ))
        elif kind == "require_dataset_digest":
            required = bool(constraint.value)
            has = bool(spec.dataset and (
                spec.dataset.inline.get("digest") or spec.dataset.inline.get("path")))
            results.append(ConstraintResult(
                kind, has or not required,
                "the dataset carries a digest or a path Fabric can hash" if has
                else f"{benchmark.ref} requires the dataset to be identified by content",
                constraint.reason,
            ))
        elif kind == "require_determinism_check":
            required = bool(constraint.value)
            # The reproducibility block carries a default replicate count
            # that only means anything once it is switched on.
            replicates = max(
                spec.reproducibility.determinism_replicates
                if spec.reproducibility.enabled else 0,
                spec.execution.determinism_replicates,
            )
            results.append(ConstraintResult(
                kind, replicates >= 2 or not required,
                f"{replicates} determinism replicate(s) declared, {benchmark.ref} "
                f"requires at least 2",
                constraint.reason,
            ))
        elif kind == "required_metrics":
            names = [str(n) for n in (constraint.value
                                      if isinstance(constraint.value, list)
                                      else [constraint.value])]
            missing = [n for n in names if n not in metric_names]
            results.append(ConstraintResult(
                kind, not missing,
                f"metrics not declared: {', '.join(missing)}" if missing
                else f"all required metrics declared: {', '.join(names)}",
                constraint.reason,
            ))

    # The benchmark's own required metrics are a constraint even when the
    # definition does not spell one out.
    missing_required = [name for name in benchmark.required_metrics
                        if name not in metric_names]
    if benchmark.required_metrics:
        results.append(ConstraintResult(
            "benchmark_metrics",
            not missing_required,
            f"required metrics not declared: {', '.join(missing_required)}"
            if missing_required else
            f"every required metric is declared: "
            f"{', '.join(benchmark.required_metrics)}",
            f"{benchmark.ref} defines these as the metrics it is read on",
        ))
    return results


def unsatisfied(results: Sequence[ConstraintResult]) -> list[ConstraintResult]:
    return [r for r in results if not r.satisfied]
