"""Running a child process under a deadline that actually holds.

Fabric's contract with a user is that ``execution.timeout_seconds`` bounds
how long one run may take. It did not. ``subprocess.run(..., timeout=T,
capture_output=True)`` kills the direct child when T expires and then calls
``communicate()`` *with no remaining timeout* to drain the pipes. A
descendant that inherited the pipe write handles keeps them open, so the
drain blocks for as long as that descendant lives.

Measured before this module existed, with ``timeout_seconds: 5``:

    grandchild sleeps 10s  ->  the run took 11.1s
    grandchild sleeps 30s  ->  the run took 31.0s
    grandchild sleeps 60s  ->  the run took 61.0s

The run tracked the grandchild exactly and ignored the timeout completely.
This affects precisely the workloads Fabric exists for: any PyTorch run
with ``DataLoader(num_workers > 0)``, any ``multiprocessing`` pool, and any
``command:`` entrypoint, which additionally spawns through a shell.

Three things are needed together, and the module does all three:

1. **Ownership.** The child is started in its own process group (POSIX) or
   assigned to its own Job Object (Windows), so its descendants can be
   reached without touching anything unrelated.
2. **Termination.** On expiry the whole group is killed, not just the
   process Fabric happened to spawn.
3. **A bounded drain.** Even after termination, the pipes are drained with
   their own deadline. If a descendant is unreachable, the run still ends on
   time with whatever output was captured. This is the belt to the
   platform-specific braces: if Job Objects are unavailable, the deadline
   still holds.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

#: How long to wait for a killed process tree to close its pipes before
#: giving up on the output and returning anyway.
DEFAULT_GRACE_SECONDS = 2.0

_POSIX = os.name == "posix"
_WINDOWS = os.name == "nt"


@dataclass
class BoundedResult:
    """The outcome of a bounded child process.

    Mirrors the fields of ``subprocess.CompletedProcess`` plus what a
    caller needs in order to report an honest timeout.
    """

    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    """True when the deadline expired. The process was terminated."""

    truncated: bool = False
    """True when output was abandoned because the tree would not close its
    pipes. ``stdout``/``stderr`` then hold whatever arrived in time."""

    killed_tree: bool = False
    """True when the whole process group or job was terminated, rather than
    only the direct child."""

    def __post_init__(self) -> None:
        self.check_returncode = self._check

    def _check(self) -> None:
        if self.returncode:
            raise subprocess.CalledProcessError(self.returncode, self.args)  # type: ignore[attr-defined]


def _child_flags() -> dict[str, Any]:
    """Spawn flags that give Fabric ownership of the child's process tree."""
    if _POSIX:
        # A new session makes the child a process-group leader, so
        # os.killpg reaches everything it started.
        return {"start_new_session": True}
    flags: dict[str, Any] = {}
    if _WINDOWS and hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        flags["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    return flags


# --------------------------------------------------------------------------
# Windows Job Objects
# --------------------------------------------------------------------------
#
# CREATE_NEW_PROCESS_GROUP alone does not give a termination handle for
# descendants on Windows; a Job Object does. Assigning can fail when the
# process is already inside a job (some CI runners do this), so every entry
# point degrades to "kill the direct child and bound the drain" rather than
# failing the run.


def _job_object_available() -> bool:
    return _WINDOWS and sys.version_info >= (3, 2)  # pragma: no cover


def _assign_job(process: subprocess.Popen) -> Any:
    """Create a Job Object with KILL_ON_JOB_CLOSE and assign ``process``.

    Returns the job handle, or ``None`` if the platform or the environment
    will not allow it. Closing the handle kills anything still inside.
    """
    if not _WINDOWS:  # pragma: no cover - POSIX uses process groups
        return None
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

    class IO_COUNTERS(ctypes.Structure):  # pragma: no cover - layout only
        _fields_ = [(n, ctypes.c_ulonglong) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):  # pragma: no cover
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):  # pragma: no cover
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.TerminateJobObject.restype = wintypes.BOOL
    kernel32.QueryInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD)]
    kernel32.QueryInformationJobObject.restype = wintypes.BOOL
    kernel32.IsProcessInJob.argtypes = [
        wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
    kernel32.IsProcessInJob.restype = wintypes.BOOL
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return None
    limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(
            job, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(limits), ctypes.sizeof(limits)):
        kernel32.CloseHandle(job)
        return None

    handle = wintypes.HANDLE(int(process._handle))  # type: ignore[attr-defined]
    if not kernel32.AssignProcessToJobObject(job, handle):
        kernel32.CloseHandle(job)
        return None

    return _WindowsJob(kernel32, job)


class _WindowsJob:  # pragma: no cover - exercised only on Windows
    """A Job Object handle that kills its members when closed."""

    def __init__(self, kernel32: Any, handle: Any) -> None:
        self._kernel32 = kernel32
        self._handle = handle
        self.assigned = False
        self.terminated = False

    def terminate(self) -> bool:
        if self._handle:
            self.terminated = True
            return bool(self._kernel32.TerminateJobObject(self._handle, 1))
        return False

    def close(self) -> None:
        # KILL_ON_JOB_CLOSE means closing terminates any survivor.
        if self._handle:
            self._kernel32.CloseHandle(self._handle)
            self._handle = None

    def __enter__(self) -> _WindowsJob:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _terminate_tree(process: subprocess.Popen, job: Any) -> bool:
    """Kill the process and everything it started.

    Returns whether the *whole tree* was targeted, which the caller records
    so that a degraded kill is visible rather than silently partial.
    """
    if job is not None:
        try:
            if job.terminate():
                return True
        except Exception:  # pragma: no cover - defensive
            pass

    if _POSIX:
        try:
            pgid = os.getpgid(process.pid)
        except (ProcessLookupError, PermissionError, OSError):
            pgid = None
        if pgid is not None and pgid != os.getpgid(0):
            try:
                os.killpg(pgid, signal.SIGKILL)
                return True
            except (ProcessLookupError, PermissionError, OSError):
                pass

    try:
        process.kill()
    except (ProcessLookupError, OSError):  # pragma: no cover - already gone
        pass
    return False


def run_bounded(
    argv: str | list[str],
    *,
    cwd: str | os.PathLike[str] | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
    grace: float = DEFAULT_GRACE_SECONDS,
    shell: bool = False,
    text: bool = True,
) -> BoundedResult:
    """Run a child, and return within ``timeout`` whatever happens.

    ``timeout=None`` (or a non-positive value) means no deadline, which is
    the documented "unlimited" behaviour and is the caller's choice to make.

    The returned object is never an exception path: a process that outlives
    its deadline comes back with ``timed_out=True`` and the output captured
    before the deadline, so the caller can classify the failure instead of
    handling an exception that says only "it took too long".
    """
    options: dict[str, Any] = dict(
        cwd=str(cwd) if cwd else None,
        env=env,
        shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if text:
        options["encoding"] = "utf-8"
        options["errors"] = "replace"
    else:
        options["text"] = False
    options.update(_child_flags())

    process = subprocess.Popen(argv, **options)  # noqa: S603 - argv is built by Fabric
    job = _assign_job(process) if _WINDOWS else None

    try:
        stdout, stderr = process.communicate(
            timeout=timeout if timeout and timeout > 0 else None)
        return BoundedResult(returncode=process.returncode,
                             stdout=_text(stdout), stderr=_text(stderr))
    except subprocess.TimeoutExpired:
        pass

    killed_tree = _terminate_tree(process, job)
    timed_out = True
    truncated = False
    try:
        # The drain gets its own deadline. Without this the whole point is
        # lost: an unreachable descendant holding the pipe would block here
        # for as long as it lives, which is the behaviour being fixed.
        stdout, stderr = process.communicate(timeout=grace)
    except subprocess.TimeoutExpired:  # pragma: no cover - needs a wedged pipe
        truncated = True
        stdout, stderr = b"", b""
        try:
            process.kill()
        except (ProcessLookupError, OSError):
            pass
    finally:
        if job is not None:
            job.close()

    return BoundedResult(returncode=process.returncode or -signal.SIGKILL,
                         stdout=_text(stdout), stderr=_text(stderr),
                         timed_out=timed_out, truncated=truncated,
                         killed_tree=killed_tree)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)