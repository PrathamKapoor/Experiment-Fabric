"""The timeout must bound wall time, including a run's descendants.

Fabric's contract is that ``execution.timeout_seconds`` bounds how long one
run takes. Before :mod:`fabric.execution.process`, it did not: a descendant
that inherited the output pipes kept them open, so collecting output blocked
for as long as that descendant lived.

These tests drive real child processes, so they are slower than the rest of
the unit suite and worth it -- this is the guarantee that makes a run
terminable at all. Descendants that sleep are always given a generous
upper bound in ``wait_for_death`` so a genuine leak cannot wedge the suite.
"""

from __future__ import annotations

import os
import sys
import time

import pytest

from fabric.execution.process import DEFAULT_GRACE_SECONDS, run_bounded

PYTHON = sys.executable

#: Sleeps well beyond any test's own budget, so a *surviving* descendant is
#: unambiguously a leak rather than one that happened to finish.
LONG_SLEEP = 120
SHORT_SLEEP = 2


def _spawn_grandchild(code: str) -> "subprocess.Popen":
    import subprocess

    return subprocess.Popen([PYTHON, "-c", code])


def wait_for_death(pid: int, *, timeout: float = 20.0) -> bool:
    """Whether ``pid`` stops existing within ``timeout``."""
    from fabric.core.entities import Run  # noqa: F401  (import sanity)

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _exists(pid):
            return True
        time.sleep(0.1)
    return not _exists(pid)


def _exists(pid: int) -> bool:
    if os.name == "nt":  # pragma: no cover - Windows branch
        import ctypes
        from ctypes import wintypes

        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k.OpenProcess.restype = wintypes.HANDLE
        k.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        k.GetExitCodeProcess.restype = wintypes.BOOL
        h = k.OpenProcess(0x1000, False, pid)
        if not h:
            return False
        code = wintypes.DWORD()
        try:
            if not k.GetExitCodeProcess(h, ctypes.byref(code)):
                return False
            return code.value == 259          # STILL_ACTIVE
        finally:
            k.CloseHandle(h)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _spawner_code(child_body: str) -> str:
    """A parent that spawns a descendant running ``child_body``, then sleeps."""
    return (
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {child_body!r}])\n"
        f"time.sleep({LONG_SLEEP})\n"
    )


# --------------------------------------------------------------------------
# the no-descendant case must keep working
# --------------------------------------------------------------------------


def test_a_process_that_finishes_in_time_is_untouched():
    result = run_bounded([PYTHON, "-c", "print('hello')"], timeout=30)
    assert result.returncode == 0
    assert "hello" in result.stdout
    assert not result.timed_out
    assert not result.truncated


def test_a_process_that_exits_nonzero_is_reported_not_raised():
    result = run_bounded([PYTHON, "-c", "raise SystemExit(3)"], timeout=30)
    assert result.returncode == 3
    assert not result.timed_out


def test_no_deadline_means_no_deadline():
    """`timeout=None` is the documented unlimited case, not an accident."""
    started = time.time()
    result = run_bounded([PYTHON, "-c", "import time; time.sleep(0.4)"])
    assert not result.timed_out
    assert time.time() - started >= 0.3


# --------------------------------------------------------------------------
# the defect: a descendant outliving its parent
# --------------------------------------------------------------------------


@pytest.mark.parametrize("descendant_sleep,timeout", [(LONG_SLEEP, 3)])
def test_a_run_is_bounded_even_when_a_descendant_outlives_it(descendant_sleep, timeout):
    """The measured defect, restated as an assertion.

    Before the fix this run took as long as the descendant lived. The
    assertion is on wall time, which is what a user experiences, with a
    generous allowance for interpreter startup and the kill grace period.
    """
    body = f"import time; time.sleep({descendant_sleep})"
    started = time.time()
    result = run_bounded([PYTHON, "-c", _spawner_code(body)], timeout=timeout)
    elapsed = time.time() - started

    assert result.timed_out, "the deadline should have expired"
    assert elapsed < timeout + DEFAULT_GRACE_SECONDS + 5, (
        f"a {timeout}s timeout took {elapsed:.1f}s; the deadline did not hold"
    )


def test_several_descendants_still_do_not_extend_the_run():
    body = f"import time; time.sleep({LONG_SLEEP})"
    parent = (
        "import subprocess, sys, time\n"
        "for _ in range(4):\n"
        f"    subprocess.Popen([sys.executable, '-c', {body!r}])\n"
        f"time.sleep({LONG_SLEEP})\n"
    )
    started = time.time()
    result = run_bounded([PYTHON, "-c", parent], timeout=3)
    elapsed = time.time() - started
    assert result.timed_out
    assert elapsed < 3 + DEFAULT_GRACE_SECONDS + 5, f"{elapsed:.1f}s"


def test_a_descendant_is_actually_terminated_not_merely_abandoned():
    """The point of killing the tree is that it stops consuming resources."""
    marker = os.path.join(os.path.dirname(__file__), "..", ".descendant.pid")
    marker = os.path.abspath(marker)
    body = (
        "import os, sys, time\n"
        f"open({marker!r}, 'w').write(str(os.getpid()))\n"
        f"time.sleep({LONG_SLEEP})\n"
    )
    try:
        result = run_bounded([PYTHON, "-c", _spawner_code(body)], timeout=4)
        assert result.timed_out
        # Give the descendant a moment to have written its pid at all.
        deadline = time.time() + 5
        while not os.path.isfile(marker) and time.time() < deadline:
            time.sleep(0.1)
        if not os.path.isfile(marker):
            pytest.skip("the descendant never started")
        pid = int(open(marker, encoding="utf-8").read().strip())
        assert wait_for_death(pid), (
            f"descendant {pid} survived the timeout; the run was abandoned, "
            "not terminated"
        )
        assert result.killed_tree, "the whole tree should have been targeted"
    finally:
        if os.path.isfile(marker):
            os.unlink(marker)


def test_a_descendant_that_exits_during_cleanup_is_handled():
    """A race that must not raise: the descendant dies while we are killing."""
    body = "import time; time.sleep(0.2)"
    for _ in range(3):
        result = run_bounded([PYTHON, "-c", _spawner_code(body)], timeout=1)
        assert not result.truncated, "a short-lived descendant must not wedge the drain"


def test_a_run_still_works_after_a_timeout():
    """A timed-out run must not poison the process that supervises it."""
    run_bounded([PYTHON, "-c", _spawner_code("import time; time.sleep(60)")],
                timeout=2)
    result = run_bounded([PYTHON, "-c", "print('still alive')"], timeout=30)
    assert result.returncode == 0
    assert "still alive" in result.stdout


def test_a_shell_command_is_bounded_too():
    """`command:` entrypoints spawn through a shell, which spawns again."""
    started = time.time()
    result = run_bounded(
        f'{PYTHON} -c "import time; time.sleep({LONG_SLEEP})"',
        shell=True, timeout=3)
    elapsed = time.time() - started
    assert result.timed_out
    assert elapsed < 3 + DEFAULT_GRACE_SECONDS + 5, f"{elapsed:.1f}s"


def test_output_captured_before_the_deadline_is_kept():
    result = run_bounded(
        [PYTHON, "-c",
         "import sys; print('progress line'); sys.stdout.flush();"
         f" import time; time.sleep({LONG_SLEEP})"],
        timeout=3)
    assert result.timed_out
    assert "progress line" in result.stdout