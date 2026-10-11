"""Workers: the part of the server that runs other people's code.

This is the security boundary of the whole system, so it is worth being
exact about where it sits (sections 36 and 72).

A Fabric experiment names an entrypoint, and executing one means
importing and calling arbitrary Python.  That is fine on a laptop, where
the code and the user are the same person.  It is **not** fine on a
server that accepts submissions over HTTP, because it turns "post a YAML
file" into "run this on your machine".

So execution is off unless it is switched on:

*   ``FabricServer`` starts with ``execute=False``.  Submitted
    specifications are parsed, expanded, hashed and stored; the run
    endpoint returns 403 with an explanation.
*   ``fabric serve --execute`` turns it on, and the server says at
    startup, in plain words, what that means.
*   Even switched on, each run goes through the same subprocess isolation
    the command line uses.  A submitted specification may declare
    ``isolation: inline``; the server overrides it, because that would run
    the submitted code in this process.  The request is recorded so the run
    still says what was asked for.

What that subprocess isolation is, and is not:

*   It **is** a crash boundary.  A segfault, an OOM kill or a
    ``sys.exit()`` takes down one run, is classified, and leaves the rest
    of the experiment intact.
*   It **is** a timeout boundary.  A run that hangs is killed.
*   It is **not** a sandbox.  The subprocess is the same user, the same
    filesystem, the same network.  Nothing here stops submitted code
    reading the workspace, opening a socket, or deleting files it can
    reach.

Fabric will not claim otherwise.  If you need to accept submissions from
people you do not trust, the isolation has to come from underneath --
a container per worker with no credentials and no outbound network, a
VM, or a separate machine.  ``docker-compose.yml`` in this repository
runs the worker as a separate unprivileged container for exactly that
reason, and that is a starting point, not a finished sandbox.
"""

from __future__ import annotations

import os
import threading
import time
import traceback
from collections.abc import Sequence
from typing import Any

from fabric.core.entities import Run
from fabric.core.outcome import describe, outcome_of
from fabric.core.store import Workspace
from fabric.execution.cache import CachePolicy
from fabric.execution.runner import ExecutionOptions, Runner
from fabric.server import service
from fabric.server.jobs import FAILED, SUCCEEDED, Job, JobStore

#: How long a worker sleeps when the queue is empty. Short enough that a
#: submitted job starts promptly, long enough that an idle server is not
#: spinning on SQLite.
POLL_SECONDS = 0.25


class Worker(threading.Thread):
    """Claims jobs and executes them, one at a time.

    A thread rather than a process, because the work it does is itself
    already spawning subprocesses: the expensive, dangerous part runs in
    ``fabric.execution.worker``, and this thread only orchestrates.
    """

    def __init__(self, name: str, store: JobStore, workspace: Workspace, *,
                 execute: bool = False, poll_seconds: float = POLL_SECONDS) -> None:
        super().__init__(name=name, daemon=True)
        self.name = name
        self.store = store
        self.workspace = workspace
        self.execute = execute
        self.poll_seconds = poll_seconds
        # Deliberately not `_stop`: Thread already has one, and shadowing it
        # breaks join() in a way that only shows up at shutdown.
        self._stopping = threading.Event()
        self.jobs_handled = 0

    def stop(self) -> None:
        self._stopping.set()

    def run(self) -> None:  # pragma: no cover - exercised through the server
        while not self._stopping.is_set():
            job = self.store.claim(self.name)
            if job is None:
                self._stopping.wait(self.poll_seconds)
                continue
            self.handle(job)

    # -- one job -----------------------------------------------------------

    def handle(self, job: Job) -> None:
        """Execute one claimed job, recording whatever happens."""
        self.jobs_handled += 1
        try:
            if job.kind == "run":
                result = self._run_experiment(job)
            elif job.kind == "reproduce":
                result = self._reproduce(job)
            else:
                raise service.ServiceError(f"unknown job kind {job.kind!r}", 400)
        except service.ServiceError as exc:
            self.store.finish(job.id, FAILED, message=str(exc),
                              result={"error": exc.to_dict()})
        except Exception as exc:
            # The traceback goes into the job record rather than to stderr,
            # so a failed job is diagnosable from the API without shell
            # access to the machine.
            self.store.finish(
                job.id, FAILED, message=f"{type(exc).__name__}: {exc}",
                result={"error": {"type": type(exc).__name__, "message": str(exc),
                                  "traceback": traceback.format_exc()}},
            )
        else:
            # The execution itself returned; whether that counts as a
            # successful *job* depends on whether anything ran successfully.
            # A job in which every run timed out, crashed or was invalid must
            # not be reported as succeeded -- that is the same false success
            # the CLI used to exit zero on, in a place nobody was watching.
            state, note = self._job_state(job, result)
            if state == SUCCEEDED:
                self.store.finish(job.id, SUCCEEDED,
                                  message=result.get("message", ""), result=result)
            else:
                self.store.finish(job.id, FAILED,
                                  message=note or "the job produced no usable evidence",
                                  result={**result, "outcome": note})

    def _job_state(self, job: Job, result: dict[str, Any]) -> tuple[str, str]:
        """The job's terminal state, judged by what it actually produced.

        A reproduction is judged by its own comparison verdict, because the
        original experiment's records say nothing about whether the rerun
        matched. An execution is judged by the persisted run records rather
        than by the counters in the result, so the queue cannot disagree with
        the evidence it is reporting on.
        """
        if job.kind == "reproduce":
            if result.get("passed"):
                return SUCCEEDED, str(result.get("verdict", "reproduced"))
            return FAILED, f"reproduction verdict: {result.get('verdict', 'unknown')}"
        return job_outcome(self.workspace.load_runs(job.experiment_id))

    def _run_experiment(self, job: Job) -> dict[str, Any]:
        if not self.execute:
            raise service.ServiceError(
                "this server was started without --execute, so it will not run "
                "submitted code", 403,
                remedy="Restart with 'fabric serve --execute' if the submissions "
                       "are trusted, or run the experiment from the command line.",
            )
        plan = service.load_plan(self.workspace, job.experiment_id)
        options = _options_from(plan.spec, job.payload)
        runner = Runner(self.workspace, plan, options, root=self.workspace.root)
        pending = runner.pending_runs()
        total = len(pending)
        self.store.progress(job.id, 0, total, "starting")

        done = 0

        def on_progress(event: Any) -> None:
            nonlocal done
            done += 1
            self.store.progress(job.id, done, total, event.label)

        report = runner.execute(on_progress=on_progress)
        completed = len(report.completed) + len(report.cached)
        failed = len(report.failed)
        return {
            "experiment_id": job.experiment_id,
            "executed": len(report.executed),
            "completed": completed,
            "cached": len(report.cached),
            "failed": failed,
            "message": f"{completed} completed, {failed} failed of {total} pending",
        }

    def _reproduce(self, job: Job) -> dict[str, Any]:
        """Re-execute an experiment and compare, without touching the original.

        Reproduction is a run like any other as far as the security
        boundary is concerned, so it needs the same permission.
        """
        if not self.execute:
            raise service.ServiceError(
                "this server was started without --execute, so it will not "
                "re-run experiments", 403,
            )
        from fabric.reproduce import reproduce

        total = len(self.workspace.load_runs(job.experiment_id))
        done = 0

        def on_progress(event: Any) -> None:
            nonlocal done
            done += 1
            self.store.progress(job.id, done, total, event.label)

        result = reproduce(
            self.workspace, job.experiment_id,
            conditions=tuple(str(c) for c in job.payload.get("conditions") or ()),
            on_progress=on_progress,
        )
        return {
            "experiment_id": job.experiment_id,
            "reproduction_id": result.plan.experiment.id,
            "verdict": result.verdict,
            "passed": result.succeeded,
            "message": f"reproduction verdict: {result.verdict}",
            "reproduction": result.report.to_dict(),
        }


MAX_SERVER_WORKERS = 8
"""Hard ceiling on concurrent runs inside one server process.

The specification chooses how many workers it wants, and the request may
only narrow that. Neither of those is a bound the *server* agreed to: a
specification declaring ``max_workers: 100000`` would have Fabric's runner
build a ThreadPoolExecutor with that many threads inside a process that is
also serving HTTP. The cap is the server's, and it applies to the
specification as well as to the request. The command line has no such cap,
because there the author is the person running the experiment.
"""


def job_outcome(runs: Sequence[Run]) -> tuple[str, str]:
    """Whether a finished job should be recorded as succeeded or failed.

    Returns ``(state, detail)`` where ``state`` is ``"succeeded"`` or
    ``"failed"``. Delegating to :mod:`fabric.core.outcome` keeps the queue,
    the command line and the certificate saying the same thing about the
    same runs -- they used to disagree, and the queue was the one nobody
    looked at.

    A job whose runs were all *skipped* as already complete still has the
    earlier records to judge, so it is not special-cased here.
    """
    outcome = outcome_of(runs)
    return (SUCCEEDED if outcome.has_evidence else FAILED), describe(runs, outcome)


def _options_from(spec: Any, payload: dict[str, Any]) -> ExecutionOptions:
    """Execution options for a queued run.

    The specification decides; the request may only narrow. A submitted
    job cannot ask for more workers than the spec allows, or turn off the
    cache for everyone -- those are properties of the experiment, not of
    the person asking for it.

    Two properties are additionally the *server's*, not the specification's:

    * isolation is forced to ``process``. ``inline`` runs the submitted code
      in the server's own interpreter, which throws away the crash boundary
      and the timeout that every other execution mode has, and lets one
      submitted specification take the service down permanently.
    * the worker count is capped at :data:`MAX_SERVER_WORKERS`.
    """
    execution = spec.execution
    workers = int(payload.get("max_workers") or execution.max_workers)
    return ExecutionOptions(
        max_workers=max(1, min(workers, execution.max_workers, MAX_SERVER_WORKERS)),
        timeout_seconds=execution.timeout_seconds,
        # The specification may ask for inline execution, but a specification
        # arrives over the network, so it does not get to choose where the
        # code runs. Recorded so the run record still says what was asked for.
        isolation="process" if execution.isolation == "inline" else execution.isolation,
        retries=execution.retries,
        continue_on_failure=execution.continue_on_failure,
        cache=CachePolicy(enabled=execution.cache),
        energy_watts=execution.energy_watts,
        only=tuple(str(name) for name in payload.get("only") or ()),
        rerun=bool(payload.get("rerun", False)),
    )


class WorkerPool:
    """A fixed set of workers sharing one queue."""

    def __init__(self, store: JobStore, workspace: Workspace, *,
                 size: int = 1, execute: bool = False,
                 poll_seconds: float = POLL_SECONDS) -> None:
        self.store = store
        self.workspace = workspace
        self.execute = execute
        self.workers = [
            Worker(f"worker-{index + 1}@{os.getpid()}", store, workspace,
                   execute=execute, poll_seconds=poll_seconds)
            for index in range(max(1, size))
        ]

    def start(self) -> None:
        # Anything left RUNNING belongs to a previous process that is no
        # longer here to finish it. Completed runs are already on disk and
        # the runner skips them, so requeueing resumes rather than repeats.
        #
        # Scoped to this pool's own worker identities. docs/deployment.md
        # recommends several worker processes sharing one SQLite file, and an
        # unscoped sweep would have the second process requeue the first
        # process's live jobs: two workers, one run directory.
        # Scoped by worker liveness: recover jobs whose process is gone,
        # leave alone what a live sibling process is running.
        # docs/deployment.md recommends several worker processes sharing one
        # SQLite file, and an unscoped sweep would make this process requeue
        # a sibling's live jobs -- two workers, one run directory.
        self.store.requeue_running(
            message=f"requeued after the server restarted (pid {os.getpid()})")
        for worker in self.workers:
            worker.start()

    def stop(self, timeout: float = 5.0) -> None:
        for worker in self.workers:
            worker.stop()
        deadline = time.time() + timeout
        for worker in self.workers:
            worker.join(max(0.0, deadline - time.time()))

    def status(self) -> list[dict[str, Any]]:
        return [
            {"name": w.name, "alive": w.is_alive(), "jobs_handled": w.jobs_handled,
             "may_execute": w.execute}
            for w in self.workers
        ]
