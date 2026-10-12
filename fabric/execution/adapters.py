"""How Fabric talks to your code.

The core understands experiments, not neural networks.  An algorithm is
anything reachable through one of these entrypoint forms:

``python:package.module:function``
    Import and call.  The function receives a :class:`RunContext` and
    returns a mapping of metric names to numbers (or records them on the
    context).  This is the fast path and the one the examples use.

``script:relative/path.py:function``
    The same, for a file that is not on the import path.

``command:./train.sh --config {config_path}``
    Any executable at all.  Fabric writes the run description to a JSON
    file, passes its path in ``FABRIC_RUN``, and reads metrics back from
    ``metrics.json`` in the run's output directory.  This is how Fabric
    stays framework-agnostic: PyTorch, JAX, scikit-learn, R, a compiled
    binary, or a shell script are all the same to it.

Nothing in this module imports a machine-learning framework, and nothing
in the core ever will.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import random
import shlex
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from fabric.execution.process import run_bounded
from fabric.util.hashing import file_digest, tree_digest
from fabric.util.jsonio import write_json


class EntrypointError(RuntimeError):
    """Raised when an entrypoint cannot be resolved or does not behave."""


# --------------------------------------------------------------------------
# the contract handed to user code
# --------------------------------------------------------------------------


@dataclass
class RunContext:
    """Everything a run is allowed to know about itself.

    Deliberately narrow.  A run receives its configuration, its seed, and
    descriptions of its inputs; it cannot see other runs, the experiment's
    results so far, or which arm is the "proposed" one.  Code that cannot
    see whether it is the treatment cannot accidentally be tuned to win.
    """

    run_id: str
    experiment_id: str
    condition: str
    role: str
    seed: int
    config: dict[str, Any] = field(default_factory=dict)
    dataset: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    ablated: tuple[str, ...] = ()
    components: tuple[str, ...] = ()
    output_dir: str = ""
    workdir: str = ""
    replicate: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    _metrics: dict[str, float] = field(default_factory=dict, repr=False)
    _logs: list[str] = field(default_factory=list, repr=False)
    _artifacts: list[dict[str, Any]] = field(default_factory=list, repr=False)

    # -- inputs ------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Read a configuration value."""
        return self.config.get(key, default)

    def enabled(self, component: str) -> bool:
        """Whether a named component is active in this condition.

        This is the ablation contract: write the system once, guard each
        component with ``if ctx.enabled("retrieval")``, and every ablation
        in the specification becomes runnable without touching the code.
        """
        return component not in self.ablated

    def rng(self, stream: str = "") -> random.Random:
        """A deterministic RNG derived from the run's seed.

        Named streams let different parts of a pipeline draw independently
        without interfering, while staying reproducible: ``ctx.rng("data")``
        and ``ctx.rng("init")`` are stable and unrelated.
        """
        if not stream:
            return random.Random(self.seed)
        return random.Random(f"{self.seed}:{stream}")

    # -- outputs -----------------------------------------------------------

    def record(self, name: str, value: float) -> None:
        """Record one metric."""
        try:
            self._metrics[str(name)] = float(value)
        except (TypeError, ValueError) as exc:
            raise EntrypointError(
                f"metric {name!r} must be a number, got {value!r}"
            ) from exc

    def record_all(self, metrics: Mapping[str, Any]) -> None:
        """Record several metrics at once."""
        for name, value in metrics.items():
            self.record(name, value)

    def log(self, message: str) -> None:
        """Append a line to the run's log."""
        self._logs.append(str(message))

    def artifact_path(self, name: str) -> str:
        """A path inside the run's artifact directory.

        Anything written here is hashed and recorded in the manifest, so
        predictions written this way remain addressable evidence rather
        than a file someone has to find later.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        return os.path.join(self.output_dir, name)

    def write_artifact(self, name: str, content: str | bytes, *, role: str = "output") -> str:
        """Write an artifact and register it."""
        path = self.artifact_path(name)
        mode = "wb" if isinstance(content, bytes) else "w"
        kwargs: dict[str, Any] = {} if isinstance(content, bytes) else {"encoding": "utf-8", "newline": "\n"}
        with open(path, mode, **kwargs) as fh:
            fh.write(content)
        self._artifacts.append({"name": name, "path": path, "role": role})
        return path

    def write_json_artifact(self, name: str, value: Any, *, role: str = "output") -> str:
        path = self.artifact_path(name)
        write_json(path, value)
        self._artifacts.append({"name": name, "path": path, "role": role})
        return path

    # -- serialisation for out-of-process runs -----------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "condition": self.condition,
            "role": self.role,
            "seed": self.seed,
            "config": self.config,
            "dataset": self.dataset,
            "model": self.model,
            "ablated": list(self.ablated),
            "components": list(self.components),
            "output_dir": self.output_dir,
            "workdir": self.workdir,
            "replicate": self.replicate,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> RunContext:
        return RunContext(
            run_id=str(data.get("run_id", "")),
            experiment_id=str(data.get("experiment_id", "")),
            condition=str(data.get("condition", "")),
            role=str(data.get("role", "treatment")),
            seed=int(data.get("seed", 0)),
            config=dict(data.get("config") or {}),
            dataset=dict(data.get("dataset") or {}),
            model=dict(data.get("model") or {}),
            ablated=tuple(data.get("ablated") or ()),
            components=tuple(data.get("components") or ()),
            output_dir=str(data.get("output_dir", "")),
            workdir=str(data.get("workdir", "")),
            replicate=int(data.get("replicate", 0)),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class AdapterResult:
    """What an adapter produced."""

    metrics: dict[str, float]
    logs: list[str] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0


# --------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------


def parse_entrypoint(entrypoint: str) -> tuple[str, str]:
    """Split an entrypoint into ``(scheme, target)``."""
    text = (entrypoint or "").strip()
    if not text:
        raise EntrypointError("the algorithm has no entrypoint")
    for scheme in ("python:", "script:", "command:", "shell:"):
        if text.startswith(scheme):
            return scheme[:-1], text[len(scheme):].strip()
    if ":" in text and not text.startswith((".", "/", "\\")) and " " not in text:
        return "python", text
    return "command", text


def load_callable(target: str, *, root: str = "") -> Callable[[RunContext], Any]:
    """Import ``module:function`` and return the callable."""
    if ":" not in target:
        raise EntrypointError(
            f"python entrypoint {target!r} must be 'module.path:function'"
        )
    module_name, _, function_name = target.rpartition(":")
    if root and root not in sys.path:
        sys.path.insert(0, root)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise EntrypointError(
            f"cannot import {module_name!r}: {exc}. "
            f"Is it importable from {root or os.getcwd()!r}?"
        ) from exc
    try:
        function = getattr(module, function_name)
    except AttributeError as exc:
        available = [n for n in dir(module) if not n.startswith("_")][:10]
        raise EntrypointError(
            f"{module_name} has no attribute {function_name!r}; found {available}"
        ) from exc
    if not callable(function):
        raise EntrypointError(f"{target} is not callable")
    return function


def load_script(target: str, *, root: str = "") -> Callable[[RunContext], Any]:
    """Load ``path.py:function`` from a file that is not on the import path."""
    path_part, _, function_name = target.rpartition(":")
    if not path_part:
        raise EntrypointError("script entrypoint must be 'path/to/file.py:function'")
    path = path_part if os.path.isabs(path_part) else os.path.join(root or os.getcwd(), path_part)
    if not os.path.isfile(path):
        raise EntrypointError(f"script {path} does not exist")
    spec = importlib.util.spec_from_file_location(f"fabric_script_{abs(hash(path))}", path)
    if spec is None or spec.loader is None:
        raise EntrypointError(f"cannot load {path} as a Python module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, function_name, None)
    if not callable(function):
        raise EntrypointError(f"{path} has no callable {function_name!r}")
    return function


def resolve(entrypoint: str, *, root: str = "") -> Callable[[RunContext], Any] | str:
    """Resolve an entrypoint to a callable, or to a command string."""
    scheme, target = parse_entrypoint(entrypoint)
    if scheme == "python":
        return load_callable(target, root=root)
    if scheme == "script":
        return load_script(target, root=root)
    return target


def entrypoint_digest(entrypoint: str, *, root: str = "",
                      exclude: Iterable[str] = ()) -> str:
    """Digest of the source file behind an entrypoint, when reachable.

    Included in the cache key so that editing the method invalidates
    cached results instead of silently reusing them -- the failure mode
    that makes experiment caches untrustworthy.

    The fingerprint also covers the package's source so that a change to
    any imported module or dependency is visible to the cache.

    ``exclude`` names files that live inside the snapshot but are not
    source. The specification belongs there: it is hashed independently as
    ``spec_digest``, and counting it again re-identifies the algorithm every
    time an experiment is re-tuned, which made the registry refuse to re-plan
    a specification the user had merely amended. Duplicate coverage must not
    become contradictory coverage.

    Exclusions match on **content as well as path**, because the same
    specification exists at more than one path. ``fabric run`` reads the
    user's ``experiment.yaml``; ``fabric export`` writes a copy into the
    bundle; ``fabric reproduce`` then reads *that copy*. Excluding by path
    alone let the original back into the snapshot during reproduction, so
    the fingerprint changed between planning an experiment and re-running it
    -- and the registry rejected its own result.
    """
    extra_excludes = {os.path.abspath(p) for p in exclude if p}
    # Exclude by *semantic* identity, not by file bytes. The same
    # specification is written out more than once -- the user's YAML, the
    # copy stored in the workspace, the copy inside an exported bundle -- and
    # those copies are not byte-identical (re-serialisation reorders keys and
    # normalises quoting) while meaning exactly the same thing. A byte
    # comparison therefore stops excluding the specification the moment it is
    # read from a second path, which is precisely when the fingerprint starts
    # disagreeing with itself.
    spec_identities = set()
    for path in extra_excludes:
        identity = _spec_identity(path)
        if identity:
            spec_identities.add(identity)

    def is_excluded(path: str) -> bool:
        return _spec_identity(path) in spec_identities if spec_identities else False
    scheme, target = parse_entrypoint(entrypoint)
    file_digest_value = ""
    try:
        if scheme == "python":
            module_name, _, _ = target.rpartition(":")
            if root and root not in sys.path:
                sys.path.insert(0, root)
            spec = importlib.util.find_spec(module_name)
            if spec and spec.origin and os.path.isfile(spec.origin):
                file_digest_value = file_digest(spec.origin)
        elif scheme == "script":
            path_part, _, _ = target.rpartition(":")
            path = path_part if os.path.isabs(path_part) else os.path.join(root or os.getcwd(), path_part)
            if os.path.isfile(path):
                file_digest_value = file_digest(path)
        else:
            first = shlex.split(target, posix=os.name != "nt")[0] if target else ""
            candidate = first if os.path.isabs(first) else os.path.join(root or os.getcwd(), first)
            if os.path.isfile(candidate):
                file_digest_value = file_digest(candidate)
    except (OSError, ValueError, ImportError):
        pass

    snapshot = ""
    source_root = _source_root_for(entrypoint, root)
    if source_root and os.path.isdir(source_root):
        try:
            snapshot = tree_digest(
                source_root,
                exclude=_SNAPSHOT_EXCLUDES,
                suffixes=_SNAPSHOT_SUFFIXES,
                exclude_paths=extra_excludes,
                exclude_predicate=is_excluded if spec_identities else None,
            )
        except (OSError, ValueError):
            pass
    if file_digest_value and snapshot:
        return f"{file_digest_value}:{snapshot}"
    if file_digest_value:
        return file_digest_value
    return snapshot or ""


#: Directories that never hold the user's source. ``.fabric`` is Fabric's own
#: output; the rest are caches, build products and virtual environments.
_SNAPSHOT_EXCLUDES = frozenset({
    ".fabric", ".git", "__pycache__", ".pytest_cache", ".ruff_cache",
    ".mypy_cache", ".venv", "venv", ".tox", ".nox", "build", "dist",
    ".eggs", "node_modules", ".idea", ".vscode", ".DS_Store",
})

#: Files that can change what an experiment computes.
#:
#: This allowlist is the mechanism that keeps generated output out of the
#: fingerprint, and it is the reason the previous blocklist approach had to
#: grow an entry for every new kind of artifact: ``bundle/``,
#: ``variant.yaml``, ``graph.html`` and finally the data files
#: ``scripts/smoke.sh`` generates were all unknown-in-advance names that a
#: deny-list cannot anticipate. Selecting source by what it *is* means the
#: next artifact is excluded without anyone remembering it.
#:
#: Reports (``.html``/``.md``/``.tex``), run records, datasets, SQLite files
#: and coverage data are excluded by construction, because none of them can
#: change what the code does when re-executed.
_SNAPSHOT_SUFFIXES = frozenset({
    ".py", ".pyi", ".pyx",
    ".yaml", ".yml", ".toml", ".cfg", ".ini",
    ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".rs", ".go", ".java", ".rb", ".pl",
    ".R", ".r", ".jl", ".m", ".lua", ".sql",
})


def _spec_identity(path: str) -> str:
    """The canonical digest of a specification file, or ``""``.

    Used to recognise the same specification at a different path. Parsing is
    best-effort: a file that is not a specification is simply not a
    specification, and returns nothing to exclude.
    """
    try:
        from fabric.core.spec import ExperimentSpec

        return ExperimentSpec.from_file(path).digest
    except Exception:  # noqa: BLE001 - any parse problem means "not this spec"
        return ""


def _source_root_for(entrypoint: str, root: str) -> str:
    """The directory whose source changes should invalidate the cache.

    Derived from the entrypoint's own location rather than from the
    workspace. The workspace is where experiments *write*, so a snapshot
    taken over it changes every time Fabric or the user's scripts produce
    output — which silently re-identified the algorithm mid-experiment and
    made the registry refuse to re-plan a specification it had just run.
    The entrypoint's directory is where the code that computes the result
    actually lives, so it is both stable and the correct unit.

    Walks up to the enclosing package (the directory containing the
    entrypoint's top-level module) so that sibling helpers it imports are
    covered, and stops at a project marker so it does not climb into an
    unrelated parent. Falls back to ``root`` when the entrypoint resolves to
    no file at all.
    """
    scheme, target = parse_entrypoint(entrypoint)
    origin = ""
    try:
        if scheme == "python":
            top_level = target.rpartition(":")[0].split(".")[0]
            if top_level:
                spec = importlib.util.find_spec(top_level)
                if spec and spec.origin and os.path.isfile(spec.origin):
                    origin = spec.origin
        elif scheme == "script":
            path_part = target.rpartition(":")[0]
            if os.path.isfile(path_part):
                origin = path_part
        else:
            first = shlex.split(target, posix=os.name != "nt")[0] if target else ""
            if first and os.path.isfile(first):
                origin = first
    except (ImportError, OSError, ValueError):
        origin = ""

    if not origin:
        return root or os.getcwd()
    # `find_spec` was given the *top-level* module, so `origin` is already
    # that package's own file: its directory is the scope. Climbing further --
    # to a project marker, or even out of the package -- makes the fingerprint
    # depend on where that directory happens to sit, and `sys.path` decides
    # that: `examples.agent_pipeline` can resolve to a copy inside a
    # workspace or to the source checkout. One run, one digest; the next
    # command over the same code, a different one, and the registry refused to
    # re-plan an experiment it had just produced. The package directory is
    # identical under both resolutions, so the fingerprint is too.
    return os.path.dirname(os.path.abspath(origin))


# --------------------------------------------------------------------------
# invocation
# --------------------------------------------------------------------------


def call_python(function: Callable[[RunContext], Any], context: RunContext) -> AdapterResult:
    """Invoke a Python entrypoint and normalise whatever it returns."""
    returned = function(context)
    metrics = dict(context._metrics)
    if isinstance(returned, Mapping):
        for key, value in returned.items():
            try:
                metrics[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
    elif returned is not None and not metrics:
        raise EntrypointError(
            f"the entrypoint returned {type(returned).__name__}; it must return a mapping "
            "of metric names to numbers, or record metrics on the context"
        )
    if not metrics:
        raise EntrypointError(
            "the run produced no metrics. Return a mapping like {'accuracy': 0.9} "
            "or call ctx.record('accuracy', 0.9)."
        )
    return AdapterResult(metrics=metrics, logs=list(context._logs),
                         artifacts=list(context._artifacts))


def call_command(command: str, context: RunContext, *, root: str = "",
                 timeout: float = 0.0, env_overlay: Mapping[str, str] | None = None) -> AdapterResult:
    """Invoke an external command and read metrics back from ``metrics.json``."""
    os.makedirs(context.output_dir, exist_ok=True)
    run_json = os.path.join(context.output_dir, "run_context.json")
    write_json(run_json, context.to_dict())
    metrics_path = os.path.join(context.output_dir, "metrics.json")
    if os.path.exists(metrics_path):
        os.unlink(metrics_path)

    formatted = command.format(
        config_path=run_json,
        output_dir=context.output_dir,
        seed=context.seed,
        run_id=context.run_id,
    )
    environment = dict(os.environ)
    environment.update({
        "FABRIC_RUN": run_json,
        "FABRIC_OUTPUT_DIR": context.output_dir,
        "FABRIC_SEED": str(context.seed),
        "FABRIC_RUN_ID": context.run_id,
        "FABRIC_CONDITION": context.condition,
        "PYTHONHASHSEED": str(context.seed),
    })
    if env_overlay:
        environment.update({str(k): str(v) for k, v in env_overlay.items()})

    started = time.time()
    completed = run_bounded(
        formatted, shell=True, cwd=root or os.getcwd(), env=environment,
        timeout=timeout,
    )
    if completed.timed_out:
        raise TimeoutError(
            f"command exceeded the {timeout:g}s timeout after {time.time() - started:.1f}s"
        )

    metrics: dict[str, float] = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, encoding="utf-8") as fh:
                raw = json.load(fh)
            for key, value in (raw.get("metrics", raw) or {}).items():
                try:
                    metrics[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
        except (OSError, json.JSONDecodeError) as exc:
            raise EntrypointError(f"could not read {metrics_path}: {exc}") from exc

    if completed.returncode != 0:
        raise RuntimeError(
            f"command exited with code {completed.returncode}\n"
            f"{(completed.stderr or completed.stdout or '').strip()[-2000:]}"
        )
    if not metrics:
        raise EntrypointError(
            f"the command completed but wrote no metrics. Write a JSON object of "
            f"metric names to numbers at {metrics_path} (the path is also in "
            "$FABRIC_OUTPUT_DIR)."
        )
    return AdapterResult(metrics=metrics, stdout=completed.stdout or "",
                         stderr=completed.stderr or "", exit_code=completed.returncode)
