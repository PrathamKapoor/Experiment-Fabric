"""The metadata store behind the server: jobs, and their lifecycle.

Two storage systems, deliberately, because they hold different kinds of
thing (section 73):

*   **Metadata goes in SQLite.**  A job is a small mutable row that
    several threads read and write concurrently and that has to survive a
    restart.  That is a database's job.
*   **Evidence stays on the filesystem**, in the same ``.fabric``
    workspace the command line writes.  Runs, artifacts, analyses and
    bundles are immutable content-addressed files, and putting them in a
    database would make them harder to verify, harder to copy and
    impossible to read without the server running.

The consequence worth stating plainly: **the database is disposable.**
Delete it and every experiment, run, artifact and certificate is still
there and still verifiable; what is lost is the queue. That is the right
way round for a system whose whole claim is that the evidence outlives
the tool.

``sqlite3`` is in the standard library, so this adds no dependency. The
schema is small enough to port to PostgreSQL without changing callers:
every query goes through this module, and none of them use anything
outside the SQL-92 that both understand.
"""

from __future__ import annotations

import ctypes
import json
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

JOB_SCHEMA_VERSION = 1

#: The job lifecycle. Section 78 asks for explicit states rather than one
#: overloaded field, so a job's *state* and its experiment's *status* are
#: separate things: a job can fail while the experiment it was running is
#: perfectly good, and an experiment can be partial with no job at all.
QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
CANCELLED = "cancelled"

TERMINAL = frozenset({SUCCEEDED, FAILED, CANCELLED})


_STILL_ACTIVE = 259


def _process_exists(pid: int) -> bool:
    """Whether ``pid`` names a live process, without signalling it.

    ``os.kill(pid, 0)`` is the POSIX idiom but is *not* safe on Windows:
    there, any signal other than ``CTRL_C_EVENT``/``CTRL_BREAK_EVENT`` is
    delivered by unconditionally calling ``TerminateProcess``, so probing
    existence that way would kill the very sibling process this check
    exists to protect. Windows is asked instead through ``OpenProcess``
    followed by ``GetExitCodeProcess``, which only read state.

    A handle alone is not enough on Windows: ``OpenProcess`` with an access
    mask the caller does not hold can succeed for a process that has
    already exited but not yet been reaped, and the handle value is a small
    integer that looks perfectly valid. The exit code is the reliable
    signal -- ``STILL_ACTIVE`` means running, anything else means gone.
    """
    if os.name == "nt":  # pragma: no cover - exercised on Windows only
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL,
                                         wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE,
                                                ctypes.POINTER(wintypes.DWORD)]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED
        if not handle:
            return False
        code = wintypes.DWORD()
        try:
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == _STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # It exists; it simply is not ours to signal.
        return True
    except OSError:
        return True
    return True


def _worker_is_alive(identity: str) -> bool:
    """Whether the process that claimed a job under ``identity`` still exists.

    Worker identities are written as ``worker-<n>@<pid>``. An identity with
    no pid cannot be judged and is treated as alive, so an unfamiliar format
    causes a job to be left alone rather than duplicated. This is the
    signal that lets several server processes share one queue file: a
    starting server recovers what its predecessors dropped without
    touching what a live sibling is running.
    """
    if "@" not in identity:
        return True
    raw = identity.rsplit("@", 1)[1].strip()
    if not raw.isdigit():
        return True
    pid = int(raw)
    if pid == os.getpid():
        return True
    return _process_exists(pid)

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    state         TEXT NOT NULL,
    created_at    REAL NOT NULL,
    started_at    REAL,
    finished_at   REAL,
    worker        TEXT,
    progress      INTEGER NOT NULL DEFAULT 0,
    total         INTEGER NOT NULL DEFAULT 0,
    message       TEXT NOT NULL DEFAULT '',
    payload       TEXT NOT NULL DEFAULT '{}',
    result        TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS jobs_state ON jobs (state, created_at);
CREATE INDEX IF NOT EXISTS jobs_experiment ON jobs (experiment_id, created_at);
"""


class JobError(Exception):
    """A job could not be created, found, or transitioned."""


@dataclass
class Job:
    """One unit of work the server was asked to do."""

    id: str
    kind: str
    experiment_id: str
    state: str = QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    worker: str = ""
    progress: int = 0
    total: int = 0
    message: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.state in TERMINAL

    @property
    def duration(self) -> float | None:
        if self.started_at is None:
            return None
        return (self.finished_at or time.time()) - self.started_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "experiment_id": self.experiment_id,
            "state": self.state,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration,
            "worker": self.worker,
            "progress": self.progress,
            "total": self.total,
            "message": self.message,
            "payload": self.payload,
            "result": self.result,
        }

    @staticmethod
    def from_row(row: sqlite3.Row) -> Job:
        return Job(
            id=row["id"],
            kind=row["kind"],
            experiment_id=row["experiment_id"],
            state=row["state"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            worker=row["worker"] or "",
            progress=row["progress"],
            total=row["total"],
            message=row["message"] or "",
            payload=json.loads(row["payload"] or "{}"),
            result=json.loads(row["result"] or "{}"),
        )


class JobStore:
    """Durable job queue over SQLite.

    Safe for use from several threads: SQLite serialises writers itself,
    and the one operation that needs to be atomic across workers --
    claiming the next queued job -- is done as a conditional UPDATE inside
    an IMMEDIATE transaction rather than as a read followed by a write.
    """

    def __init__(self, path: str) -> None:
        self.path = os.path.abspath(path)
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._local = threading.local()
        connection = self._connection
        # executescript() commits and starts its own transaction, so this
        # deliberately does not go through _connect(): wrapping it would
        # leave a COMMIT with no transaction to commit.
        connection.executescript(SCHEMA)
        row = connection.execute("SELECT version FROM schema_version").fetchone()
        if row is None:
            connection.execute("INSERT INTO schema_version (version) VALUES (?)",
                               (JOB_SCHEMA_VERSION,))
        elif row["version"] != JOB_SCHEMA_VERSION:
            raise JobError(
                f"{self.path} was written by job schema version {row['version']}, "
                f"and this Fabric speaks version {JOB_SCHEMA_VERSION}. The queue is "
                "disposable -- every experiment, run and artifact lives in the "
                "workspace, not here -- so the fix is to delete this file and let "
                "it be recreated."
            )

    # -- connection --------------------------------------------------------

    @property
    def _connection(self) -> sqlite3.Connection:
        existing = getattr(self._local, "connection", None)
        if existing is None:
            existing = sqlite3.connect(self.path, timeout=30.0,
                                       isolation_level=None)
            existing.row_factory = sqlite3.Row
            existing.execute("PRAGMA journal_mode=WAL")
            existing.execute("PRAGMA busy_timeout=30000")
            self._local.connection = existing
        return existing

    @contextmanager
    def _connect(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        connection = self._connection
        connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        try:
            yield connection
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        else:
            connection.execute("COMMIT")

    def close(self) -> None:
        existing = getattr(self._local, "connection", None)
        if existing is not None:
            existing.close()
            self._local.connection = None

    # -- writing -----------------------------------------------------------

    def enqueue(self, kind: str, experiment_id: str, *,
                payload: dict[str, Any] | None = None, total: int = 0) -> Job:
        """Add a job to the queue and return it."""
        job = Job(
            id="job-" + uuid.uuid4().hex[:12],
            kind=kind,
            experiment_id=experiment_id,
            total=total,
            payload=dict(payload or {}),
        )
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO jobs (id, kind, experiment_id, state, created_at, "
                "progress, total, message, payload, result) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (job.id, job.kind, job.experiment_id, job.state, job.created_at,
                 0, job.total, "", json.dumps(job.payload), "{}"),
            )
        return job

    def claim(self, worker: str) -> Job | None:
        """Atomically take the oldest queued job, or return ``None``.

        The conditional UPDATE is what makes this safe: two workers racing
        for the same row means one of them updates zero rows and tries
        again, rather than both believing they own it.
        """
        while True:
            with self._connect(immediate=True) as connection:
                row = connection.execute(
                    "SELECT * FROM jobs WHERE state = ? ORDER BY created_at LIMIT 1",
                    (QUEUED,),
                ).fetchone()
                if row is None:
                    return None
                now = time.time()
                changed = connection.execute(
                    "UPDATE jobs SET state = ?, started_at = ?, worker = ? "
                    "WHERE id = ? AND state = ?",
                    (RUNNING, now, worker, row["id"], QUEUED),
                ).rowcount
            if changed:
                job = Job.from_row(row)
                job.state = RUNNING
                job.started_at = now
                job.worker = worker
                return job

    def progress(self, job_id: str, done: int, total: int, message: str = "") -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE jobs SET progress = ?, total = ?, message = ? WHERE id = ?",
                (done, total, message, job_id),
            )

    def finish(self, job_id: str, state: str, *, message: str = "",
               result: dict[str, Any] | None = None) -> None:
        if state not in TERMINAL:
            raise JobError(f"{state!r} is not a terminal job state")
        with self._connect() as connection:
            connection.execute(
                "UPDATE jobs SET state = ?, finished_at = ?, message = ?, result = ? "
                "WHERE id = ?",
                (state, time.time(), message, json.dumps(result or {}), job_id),
            )

    def cancel(self, job_id: str) -> bool:
        """Cancel a job that has not started. Returns whether it worked.

        A running job is deliberately not killed: it owns a subprocess and
        a partly written run directory, and tearing that down from another
        thread is how workspaces get corrupted. Section 77 -- preserve
        completed evidence -- outranks responsiveness here.
        """
        with self._connect() as connection:
            changed = connection.execute(
                "UPDATE jobs SET state = ?, finished_at = ?, message = ? "
                "WHERE id = ? AND state = ?",
                (CANCELLED, time.time(), "cancelled before it started",
                 job_id, QUEUED),
            ).rowcount
        return bool(changed)

    def requeue_running(self, *, message: str = "",
                        stale_after: float | None = None,
                        rows: Sequence[Mapping[str, Any]] | None = None) -> int:
        """Return jobs left RUNNING by a crash to the queue.

        Called at startup. A job whose worker died mid-run left no result,
        and the runner skips work that already completed, so re-running it
        resumes rather than repeats.

        Which jobs are swept matters. An unscoped sweep would have a *second*
        server sharing this database requeue the first server's live jobs,
        and two workers would then execute the same runs into the same run
        directories -- the workspace corruption the rest of this module
        works to prevent. ``docs/deployment.md`` recommends exactly that
        deployment, so the sweep must not be blind.

        ``rows`` is the caller already-known set of RUNNING jobs. For each
        one whose worker identity names a process that is no longer alive,
        the job is requeued. Worker identities are written as
        ``worker-<n>@<pid>``, so liveness is decided by ``os.kill(pid, 0)``
        rather than by a timeout. A job with no parseable identity, or whose
        process is alive, is left alone.

        ``stale_after`` adds a second rule for the case liveness cannot see:
        a worker that is alive but wedged holds its job forever. When it is
        a number, jobs that started longer than that many seconds ago are
        requeued regardless. Left unset, liveness alone decides -- which is
        the right default for a process that is the only writer.
        """
        with self._connect() as connection:
            if rows is None:
                rows = [dict(r) for r in connection.execute(
                    "SELECT id, worker, started_at FROM jobs WHERE state = ?",
                    (RUNNING,)).fetchall()]
            dead = [str(r.get("id")) for r in rows
                    if not _worker_is_alive(str(r.get("worker") or ""))]
            if stale_after is not None:
                cutoff = time.time() - max(0.0, stale_after)
                for row in rows:
                    if row["id"] in dead:
                        continue
                    started = row.get("started_at")
                    if started is None or float(started) <= cutoff:
                        dead.append(str(row["id"]))
            if not dead:
                return 0
            recovered = 0
            for job_id in dead:
                recovered += connection.execute(
                    "UPDATE jobs SET state = ?, started_at = NULL, worker = '', "
                    "message = ? WHERE state = ? AND id = ?",
                    (QUEUED, message or "requeued after the server restarted",
                     RUNNING, job_id),
                ).rowcount
        return recovered

    def running_rows(self) -> list[dict[str, Any]]:
        """Every job currently RUNNING, for a liveness-scoped recovery sweep."""
        with self._connect() as connection:
            return [dict(r) for r in connection.execute(
                "SELECT id, worker, started_at FROM jobs WHERE state = ?",
                (RUNNING,)).fetchall()]

    # -- reading -----------------------------------------------------------

    def get(self, job_id: str) -> Job | None:
        row = self._connection.execute(
            "SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return Job.from_row(row) if row else None

    def list(self, *, experiment_id: str = "", state: str = "",
             limit: int = 100) -> list[Job]:
        query = "SELECT * FROM jobs"
        clauses: list[str] = []
        parameters: list[Any] = []
        if experiment_id:
            clauses.append("experiment_id = ?")
            parameters.append(experiment_id)
        if state:
            clauses.append("state = ?")
            parameters.append(state)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC LIMIT ?"
        parameters.append(int(limit))
        rows = self._connection.execute(query, parameters).fetchall()
        return [Job.from_row(row) for row in rows]

    def counts(self) -> dict[str, int]:
        rows = self._connection.execute(
            "SELECT state, COUNT(*) AS n FROM jobs GROUP BY state").fetchall()
        counts = dict.fromkeys((QUEUED, RUNNING, SUCCEEDED, FAILED, CANCELLED), 0)
        for row in rows:
            counts[row["state"]] = row["n"]
        return counts
