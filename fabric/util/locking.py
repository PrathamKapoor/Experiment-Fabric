"""A small cross-process file lock.

Several shared files -- the cache index, the registry index, the
experiment index -- are read-modify-write.  With four workers writing
them, unlocked updates lose entries; on Windows they also fail outright,
because a rename onto a file another handle has open raises
``PermissionError``.

This is a lock file created with ``O_CREAT | O_EXCL``, which is atomic on
every platform Fabric targets.  It carries the owning process id so a
lock left behind by a killed run can be identified and broken rather than
deadlocking the workspace forever.
"""

from __future__ import annotations

import errno
import os
import threading
import time
from types import TracebackType
from typing import Any

_THREAD_LOCKS: dict[str, threading.RLock] = {}
_REGISTRY_GUARD = threading.Lock()

STALE_SECONDS = 120.0
"""A lock older than this is assumed to belong to a dead process."""


def _thread_lock(path: str) -> threading.RLock:
    key = os.path.normcase(os.path.abspath(path))
    with _REGISTRY_GUARD:
        lock = _THREAD_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _THREAD_LOCKS[key] = lock
        return lock


class FileLock:
    """Guard a path against concurrent writers, in-process and out."""

    def __init__(self, path: str, *, timeout: float = 30.0, poll: float = 0.01) -> None:
        self.target = os.path.abspath(path)
        self.lock_path = self.target + ".lock"
        self.timeout = timeout
        self.poll = poll
        self._thread_lock = _thread_lock(self.target)
        self._held = False

    def acquire(self) -> None:
        self._thread_lock.acquire()
        deadline = time.monotonic() + self.timeout
        os.makedirs(os.path.dirname(self.lock_path), exist_ok=True)
        while True:
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(f"{os.getpid()} {time.time()}")
                self._held = True
                return
            except FileExistsError:
                if self._break_if_stale():
                    continue
                if time.monotonic() > deadline:
                    self._thread_lock.release()
                    raise TimeoutError(
                        f"could not acquire the lock on {self.target} within "
                        f"{self.timeout:g}s. If no other Fabric process is running, "
                        f"delete {self.lock_path}."
                    )
                time.sleep(self.poll)
            except OSError as exc:  # pragma: no cover - unusual filesystems
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    self._thread_lock.release()
                    raise
                time.sleep(self.poll)

    def _break_if_stale(self) -> bool:
        try:
            age = time.time() - os.path.getmtime(self.lock_path)
        except OSError:
            return True
        if age <= STALE_SECONDS:
            return False
        try:
            os.unlink(self.lock_path)
            return True
        except OSError:
            return False

    def release(self) -> None:
        if self._held:
            try:
                os.unlink(self.lock_path)
            except OSError:
                pass
            self._held = False
        self._thread_lock.release()

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc: BaseException | None, tb: TracebackType | None) -> None:
        self.release()
