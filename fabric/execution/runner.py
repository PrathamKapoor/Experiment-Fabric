"""Executing a plan.

The runner turns pending runs into evidence: it isolates each execution,
seeds it, measures it, captures whatever it produced, classifies whatever
went wrong, hashes every artifact, and writes an immutable record.

Two properties matter more than speed:

* **Nothing is lost.**  A crashed run, a timeout, an OOM -- all become
  records with a classification, not gaps.
* **Nothing is invented.**  If a metric was not produced, the run failed;
  the runner never substitutes a default.
"""

from __future__ import annotations

import concurrent.futures
import os
import shutil
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from fabric.core.entities import (
    Artifact,
    Condition,
    Environment,
    Failure,
    FailureClass,
    Hardware,
    Resources,
    Run,
    RunStatus,
)
from fabric.core.registry import Registry
from fabric.core.store import Workspace
from fabric.environments.capture import capture_environment
from fabric.environments.hardware import capture_hardware
from fabric.execution import failures as failure_module
from fabric.execution.adapters import (
    AdapterResult,
    EntrypointError,
    RunContext,
    call_command,
    call_python,
    parse_entrypoint,
    resolve,
)
from fabric.execution.cache import CachePolicy, cache_key, is_cacheable
from fabric.execution.planner import ExperimentPlan
from fabric.execution.process import run_bounded
from fabric.execution.resources import ResourceMonitor
from fabric.util.hashing import file_digest
from fabric.util.jsonio import atomic_write_text, read_json, write_json

ProgressCallback = Callable[["RunEvent"], None]


@dataclass
class RunEvent:
    """Progress notification emitted as runs complete."""

    run: Run
    index: int
    total: int
    cached: bool = False

    @property
    def label(self) -> str:
        return f"{self.run.condition_name} seed={self.run.seed}" + (
            f" rep={self.run.replicate}" if self.run.replicate else ""
        )


@dataclass
class ExecutionOptions:
    """Knobs for one execution pass."""

    max_workers: int = 1
    timeout_seconds: float = 0.0
    isolation: str = "process"
    retries: int = 0
    continue_on_failure: bool = True
    cache: CachePolicy = field(default_factory=CachePolicy)
    track_gpu: bool = False
    cost_per_gpu_hour: float | None = None
    cost_per_cpu_hour: float | None = None
    energy_watts: float | None = None
    only: tuple[str, ...] = ()
    """Restrict execution to these condition names."""
    rerun: bool = False
    """Re-execute runs that already have results."""


@dataclass
class ExecutionReport:
    """What one execution pass did."""

    experiment_id: str
    executed: list[Run] = field(default_factory=list)
    skipped: list[Run] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0
    environment: Environment | None = None
    hardware: Hardware | None = None

    @property
    def completed(self) -> list[Run]:
        return [r for r in self.executed if r.status == RunStatus.COMPLETED]

    @property
    def cached(self) -> list[Run]:
        return [r for r in self.executed if r.status == RunStatus.CACHED]

    @property
    def failed(self) -> list[Run]:
        return [r for r in self.executed if r.status == RunStatus.FAILED]

    @property
    def wall_seconds(self) -> float:
        return max(0.0, self.finished_at - self.started_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "executed": len(self.executed),
            "completed": len(self.completed),
            "cached": len(self.cached),
            "failed": len(self.failed),
            "skipped": len(self.skipped),
            "wall_seconds": self.wall_seconds,
            "environment_id": self.environment.id if self.environment else "",
            "hardware_id": self.hardware.id if self.hardware else "",
            "failure_classes": failure_module.summarise(
                [r.failure for r in self.failed if r.failure]
            ),
        }

    def summary_line(self) -> str:
        parts = [f"{len(self.completed)} completed"]
        if self.cached:
            parts.append(f"{len(self.cached)} from cache")
        if self.failed:
            parts.append(f"{len(self.failed)} failed")
        if self.skipped:
            parts.append(f"{len(self.skipped)} already done")
        return ", ".join(parts) + f" in {self.wall_seconds:.1f}s"


class Runner:
    """Executes the runs of a plan."""

    def __init__(self, workspace: Workspace, plan: ExperimentPlan,
                 options: ExecutionOptions | None = None, *, root: str = "") -> None:
        self.workspace = workspace
        self.plan = plan
        self.options = options or ExecutionOptions()
        self.root = root or workspace.root
        self.registry = Registry(workspace)
        self.environment = capture_environment(self.root)
        self.hardware = capture_hardware()
        self.registry.register_environment(self.environment)
        self.registry.register_hardware(self.hardware)

    # -- selection ---------------------------------------------------------

    def pending_runs(self) -> list[Run]:
        """Runs that still need executing."""
        selected = []
        existing = {r.id: r for r in self.workspace.iter_runs(self.plan.experiment.id)}
        for run in self.plan.runs:
            if self.options.only and run.condition_name not in self.options.only:
                continue
            previous = existing.get(run.id)
            if previous and previous.succeeded and not self.options.rerun:
                continue
            selected.append(run)
        return selected

    # -- execution ---------------------------------------------------------

    def execute(self, runs: Sequence[Run] | None = None, *,
                on_progress: ProgressCallback | None = None) -> ExecutionReport:
        """Execute runs, honouring the cache and the worker limit."""
        queue = list(runs if runs is not None else self.pending_runs())
        report = ExecutionReport(
            experiment_id=self.plan.experiment.id,
            started_at=time.time(),
            environment=self.environment,
            hardware=self.hardware,
        )
        already = {r.id for r in self.workspace.iter_runs(self.plan.experiment.id)
                   if r.succeeded}
        report.skipped = [r for r in self.plan.runs
                          if r.id in already and r.id not in {q.id for q in queue}]

        total = len(queue)
        if total == 0:
            report.finished_at = time.time()
            return report

        workers = max(1, min(self.options.max_workers, total))
        if workers == 1:
            for index, run in enumerate(queue, start=1):
                finished = self._execute_one(run)
                report.executed.append(finished)
                if on_progress:
                    on_progress(RunEvent(finished, index, total,
                                         finished.status == RunStatus.CACHED))
                if finished.status == RunStatus.FAILED and not self.options.continue_on_failure:
                    break
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(self._execute_one, run): run for run in queue}
                for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                    finished = future.result()
                    report.executed.append(finished)
                    if on_progress:
                        on_progress(RunEvent(finished, index, total,
                                             finished.status == RunStatus.CACHED))
            report.executed.sort(key=lambda r: (r.condition_name, r.seed, r.replicate))

        report.finished_at = time.time()
        return report

    # -- one run -----------------------------------------------------------

    def _execute_one(self, run: Run) -> Run:
        condition = self.plan.condition(run.condition_id)
        algorithm = self.plan.algorithm_for(condition)
        if algorithm is None:
            return self._record_failure(
                run, Failure(FailureClass.USER_ERROR, "no algorithm is attached to this condition")
            )

        env_overlay = dict(condition.perturbation.get("env", {})) if condition.is_perturbed else {}
        key = cache_key(
            run,
            algorithm_id=algorithm.id,
            source_digest=algorithm.source_digest,
            environment=self.environment,
            policy=self.options.cache,
            hardware_id=self.hardware.id,
            env_overlay=env_overlay,
        )

        if is_cacheable(run, self.options.cache) and not self.options.rerun:
            hit = self.workspace.cache_lookup(key)
            if hit:
                return self._record_cache_hit(run, key, hit)

        attempt = 0
        last: Run | None = None
        while attempt <= max(0, self.options.retries):
            last = self._run_once(run, condition, algorithm, env_overlay, key,
                                  attempt=attempt)
            if last.status != RunStatus.FAILED:
                return last
            if last.failure and not failure_module.is_infrastructure(last.failure.failure_class):
                # Only infrastructure failures are worth retrying; retrying a
                # KeyError just produces the same KeyError more slowly.
                break
            attempt += 1
        return last if last else run

    def _run_once(self, run: Run, condition: Condition, algorithm: Any,
                  env_overlay: Mapping[str, str], key: str, *, attempt: int) -> Run:
        output_dir = self.workspace.artifacts_dir(self.plan.experiment.id, run.id)
        os.makedirs(output_dir, exist_ok=True)
        dataset = self.plan.dataset_for(condition)
        model = self.plan.model_for(condition)

        context = RunContext(
            run_id=run.id,
            experiment_id=self.plan.experiment.id,
            condition=condition.name,
            role=condition.role.value,
            seed=run.seed,
            config=dict(run.config),
            dataset=dataset.to_dict() if dataset else {},
            model=model.to_dict() if model else {},
            ablated=tuple(run.ablated),
            components=tuple(self.plan.spec.components),
            output_dir=output_dir,
            workdir=self.root,
            replicate=run.replicate,
            metadata={"attempt": attempt, "perturbation": condition.perturbation},
        )

        started = time.time()
        monitor = ResourceMonitor(
            track_gpu=self.options.track_gpu,
            cost_per_gpu_hour=self.options.cost_per_gpu_hour,
            cost_per_cpu_hour=self.options.cost_per_cpu_hour,
            energy_watts=self.options.energy_watts,
        )
        failure: Failure | None = None
        result: AdapterResult | None = None

        try:
            with monitor:
                if self.options.isolation == "inline":
                    result = self._run_inline(algorithm.entrypoint, context, env_overlay)
                else:
                    result = self._run_subprocess(algorithm.entrypoint, context, env_overlay)
        except WorkerCrash as exc:
            # The subprocess already classified this; keep its verdict.
            failure = exc.failure
        except TimeoutError as exc:
            failure = Failure(FailureClass.TIMEOUT, str(exc))
        except EntrypointError as exc:
            failure = Failure(FailureClass.INVALID_OUTPUT, str(exc),
                              traceback.format_exc()[-4000:])
        except (KeyboardInterrupt, SystemExit):
            # An interrupt is the user asking to stop, not a failed run.
            # Recording it as data would let the sweep continue past a
            # Ctrl-C and report a tidy summary for an aborted experiment.
            # The run is left unrecorded rather than marked failed, so a
            # resumed experiment re-attempts exactly this run.
            raise
        except Exception as exc:
            failure = failure_module.from_exception(exc, traceback_text=traceback.format_exc())

        finished = time.time()
        artifacts = self._collect_artifacts(output_dir)

        if failure is not None:
            return self._record_failure(run, failure, started=started, finished=finished,
                                        resources=monitor.result, artifacts=artifacts,
                                        cache_key=key)

        assert result is not None
        nan_metrics = [k for k, v in result.metrics.items() if v != v]
        if nan_metrics:
            return self._record_failure(
                run,
                Failure(FailureClass.NAN,
                        f"metric(s) {nan_metrics} came back as NaN"),
                started=started, finished=finished, resources=monitor.result,
                artifacts=artifacts, cache_key=key,
            )

        logs = self._write_logs(run, result)
        completed = Run(
            id=run.id,
            experiment_id=run.experiment_id,
            condition_id=run.condition_id,
            condition_name=run.condition_name,
            seed=run.seed,
            role=run.role,
            status=RunStatus.COMPLETED,
            config=run.config,
            dataset_id=run.dataset_id,
            model_id=run.model_id,
            algorithm_id=run.algorithm_id,
            environment_id=self.environment.id,
            hardware_id=self.hardware.id,
            ablated=run.ablated,
            metrics=result.metrics,
            artifacts=artifacts,
            resources=monitor.result,
            cache_key=key,
            replicate=run.replicate,
            started_at=started,
            finished_at=finished,
            logs=logs,
            extra={**run.extra, "measured": monitor.measured,
                   "env_overlay": dict(env_overlay)},
        )
        self.workspace.save_run(self.plan.experiment.id, completed)
        if is_cacheable(completed, self.options.cache):
            self.workspace.cache_store(key, self.plan.experiment.id, completed.id,
                                       completed.metrics)
        return completed

    # -- dispatch ----------------------------------------------------------

    def _run_inline(self, entrypoint: str, context: RunContext,
                    env_overlay: Mapping[str, str]) -> AdapterResult:
        """Run in this interpreter: fast, but shares our fate on a crash."""
        scheme, _ = parse_entrypoint(entrypoint)
        previous = {k: os.environ.get(k) for k in env_overlay}
        os.environ.update({str(k): str(v) for k, v in env_overlay.items()})
        os.environ["PYTHONHASHSEED"] = str(context.seed)
        try:
            target = resolve(entrypoint, root=self.root)
            if scheme in ("python", "script"):
                import random as _random

                _random.seed(context.seed)
                return call_python(target, context)  # type: ignore[arg-type]
            return call_command(str(target), context, root=self.root,
                                timeout=self.options.timeout_seconds,
                                env_overlay=env_overlay)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def _run_subprocess(self, entrypoint: str, context: RunContext,
                        env_overlay: Mapping[str, str]) -> AdapterResult:
        """Run in a fresh interpreter, so a segfault costs one run."""
        with tempfile.TemporaryDirectory(prefix="fabric-run-") as tmp:
            request_path = os.path.join(tmp, "request.json")
            result_path = os.path.join(tmp, "result.json")
            write_json(request_path, {
                "entrypoint": entrypoint,
                "context": context.to_dict(),
                "root": self.root,
                "timeout": self.options.timeout_seconds,
                "env_overlay": dict(env_overlay),
                "result_path": result_path,
            })

            environment = dict(os.environ)
            environment.update({str(k): str(v) for k, v in env_overlay.items()})
            environment["PYTHONHASHSEED"] = str(context.seed)
            fabric_root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
            existing_path = environment.get("PYTHONPATH", "")
            parts = [p for p in (fabric_root, self.root, existing_path) if p]
            environment["PYTHONPATH"] = os.pathsep.join(parts)
            timeout = self.options.timeout_seconds

            try:
                completed = run_bounded(
                    [sys.executable, "-m", "fabric.execution.worker", request_path],
                    cwd=self.root, env=environment, timeout=timeout,
                )
            except OSError:
                raise
            if completed.timed_out:
                raise TimeoutError(f"run exceeded the {timeout:g}s timeout")

            payload = read_json(result_path)
            if payload is None:
                # The worker died before it could report: classify from the
                # process output rather than losing the failure entirely.
                failure = failure_module.from_process(
                    completed.stderr, completed.stdout, completed.returncode
                )
                raise WorkerCrash(failure)
            if not payload.get("ok"):
                raise WorkerCrash(Failure(
                    failure_class=failure_module.classify(
                        payload.get("traceback", "") or payload.get("error", "")
                    ),
                    message=str(payload.get("error", ""))[:2000],
                    traceback=str(payload.get("traceback", ""))[-8000:],
                    exit_code=completed.returncode,
                ))
            return AdapterResult(
                metrics={str(k): float(v) for k, v in (payload.get("metrics") or {}).items()},
                logs=list(payload.get("logs") or []),
                artifacts=list(payload.get("artifacts") or []),
                stdout=(payload.get("stdout") or "") + (completed.stdout or ""),
                stderr=(payload.get("stderr") or "") + (completed.stderr or ""),
            )

    # -- bookkeeping -------------------------------------------------------

    def _collect_artifacts(self, output_dir: str) -> list[Artifact]:
        """Hash everything the run left behind."""
        artifacts: list[Artifact] = []
        if not os.path.isdir(output_dir):
            return artifacts
        for dirpath, _dirnames, filenames in os.walk(output_dir):
            for name in sorted(filenames):
                path = os.path.join(dirpath, name)
                relative = os.path.relpath(path, output_dir).replace(os.sep, "/")
                try:
                    size = os.path.getsize(path)
                    digest = file_digest(path)
                except OSError:
                    continue
                artifacts.append(Artifact(
                    name=relative,
                    path=relative,
                    digest=digest,
                    bytes=size,
                    media_type=_media_type(name),
                    role="context" if name == "run_context.json" else "output",
                ))
        return artifacts

    def _write_logs(self, run: Run, result: AdapterResult) -> dict[str, str]:
        directory = self.workspace.run_dir(self.plan.experiment.id, run.id)
        os.makedirs(directory, exist_ok=True)
        written: dict[str, str] = {}
        for name, content in (("stdout.log", result.stdout), ("stderr.log", result.stderr),
                              ("run.log", "\n".join(result.logs))):
            if content and content.strip():
                atomic_write_text(os.path.join(directory, name), content)
                written[name.split(".")[0]] = name
        return written

    def _record_failure(self, run: Run, failure: Failure, *, started: float = 0.0,
                        finished: float = 0.0, resources: Any = None,
                        artifacts: Sequence[Artifact] = (), cache_key: str = "") -> Run:
        failed = Run(
            id=run.id,
            experiment_id=run.experiment_id,
            condition_id=run.condition_id,
            condition_name=run.condition_name,
            seed=run.seed,
            role=run.role,
            status=RunStatus.FAILED,
            config=run.config,
            dataset_id=run.dataset_id,
            model_id=run.model_id,
            algorithm_id=run.algorithm_id,
            environment_id=self.environment.id,
            hardware_id=self.hardware.id,
            ablated=run.ablated,
            metrics={},
            artifacts=list(artifacts),
            resources=resources or Resources(),
            failure=failure,
            cache_key=cache_key,
            replicate=run.replicate,
            started_at=started or time.time(),
            finished_at=finished or time.time(),
            extra=dict(run.extra),
        )
        directory = self.workspace.run_dir(self.plan.experiment.id, run.id)
        os.makedirs(directory, exist_ok=True)
        if failure.traceback:
            atomic_write_text(os.path.join(directory, "traceback.log"), failure.traceback)
        self.workspace.save_run(self.plan.experiment.id, failed)
        return failed

    def _record_cache_hit(self, run: Run, key: str, hit: Mapping[str, Any]) -> Run:
        source = self.workspace.load_run(str(hit["experiment_id"]), str(hit["run_id"]))
        metrics = dict(hit.get("metrics") or {})
        artifacts = self._copy_cached_artifacts(run, source, hit) if source else []
        cached = Run(
            id=run.id,
            experiment_id=run.experiment_id,
            condition_id=run.condition_id,
            condition_name=run.condition_name,
            seed=run.seed,
            role=run.role,
            status=RunStatus.CACHED,
            config=run.config,
            dataset_id=run.dataset_id,
            model_id=run.model_id,
            algorithm_id=run.algorithm_id,
            environment_id=source.environment_id if source else self.environment.id,
            hardware_id=source.hardware_id if source else self.hardware.id,
            ablated=run.ablated,
            metrics=metrics or (source.metrics if source else {}),
            artifacts=artifacts,
            resources=source.resources if source else Resources(),
            cache_key=key,
            cached_from=f"{hit['experiment_id']}/{hit['run_id']}",
            replicate=run.replicate,
            started_at=time.time(),
            finished_at=time.time(),
            extra={**run.extra, "cache_hit": True},
        )
        self.workspace.save_run(self.plan.experiment.id, cached)
        return cached

    def _copy_cached_artifacts(self, run: Run, source: Run,
                               hit: Mapping[str, Any]) -> list[Artifact]:
        """Materialise a cached run's artifacts in its own directory.

        A cache hit reuses a *computation*, but the run it produces still has
        to stand on its own: its artifacts must be present and hashable where
        the run says they are, or verification reports them missing and a
        bundle exports a dangling reference. Compute is what the cache saves;
        a few kilobytes of output is not worth the broken evidence chain.
        """
        source_dir = self.workspace.artifacts_dir(str(hit["experiment_id"]), source.id)
        target_dir = self.workspace.artifacts_dir(self.plan.experiment.id, run.id)
        copied: list[Artifact] = []
        for artifact in source.artifacts:
            source_path = os.path.join(source_dir, artifact.path.replace("/", os.sep))
            if not os.path.isfile(source_path):
                continue
            target_path = os.path.join(target_dir, artifact.path.replace("/", os.sep))
            if os.path.abspath(source_path) == os.path.abspath(target_path):
                # The cache hit is this run's own earlier output; the file is
                # already exactly where the record says it is.
                copied.append(artifact)
                continue
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            try:
                shutil.copyfile(source_path, target_path)
            except (OSError, shutil.Error):
                continue
            copied.append(artifact)
        return copied


class WorkerCrash(Exception):
    """Raised inside the runner to carry a classified worker failure."""

    def __init__(self, failure: Failure) -> None:
        self.failure = failure
        super().__init__(failure.message)


_MEDIA_TYPES = {
    ".json": "application/json",
    ".csv": "text/csv",
    ".txt": "text/plain",
    ".log": "text/plain",
    ".md": "text/markdown",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".jsonl": "application/x-ndjson",
    ".pt": "application/octet-stream",
    ".npy": "application/octet-stream",
    ".html": "text/html",
}


def _media_type(name: str) -> str:
    _, extension = os.path.splitext(name.lower())
    return _MEDIA_TYPES.get(extension, "application/octet-stream")
