"""Failure classification.

A failed run is not a gap in the results; it is a result.  "This method
OOMs above batch size 64" is a finding, and it is only visible if
failures are recorded and classified rather than retried into silence.

Classification is pattern-based and deliberately conservative: anything
it cannot place confidently becomes ``UNKNOWN`` with the original message
preserved, because a wrong label is worse than no label.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from fabric.core.entities import Failure, FailureClass

_PATTERNS: tuple[tuple[FailureClass, tuple[str, ...]], ...] = (
    (FailureClass.OOM, (
        r"\bout of memory\b", r"\bOOM\b", r"CUDA out of memory",
        r"MemoryError", r"Cannot allocate memory", r"std::bad_alloc",
        r"Killed process .* \(python\)", r"DefaultCPUAllocator: can't allocate",
    )),
    (FailureClass.TIMEOUT, (
        r"\btimed? ?out\b", r"TimeoutError", r"exceeded the .* timeout",
        r"DeadlineExceeded", r"SIGKILL.*time limit",
    )),
    (FailureClass.NAN, (
        r"\bnan\b", r"not a number", r"invalid value encountered",
        r"Loss is NaN", r"inf(inity)? encountered",
    )),
    (FailureClass.DIVERGENCE, (
        r"\bdiverge", r"loss (exploded|blew up)", r"gradient overflow",
        r"failed to converge", r"ConvergenceWarning.*maximum",
    )),
    (FailureClass.SEGFAULT, (
        r"Segmentation fault", r"SIGSEGV", r"core dumped", r"Fatal Python error",
        r"access violation", r"exit(ed)? with code -11",
    )),
    (FailureClass.DEPENDENCY, (
        r"ModuleNotFoundError", r"ImportError", r"No module named",
        r"undefined symbol", r"DLL load failed", r"GLIBC.*not found",
        r"version `GLIBCXX", r"command not found",
    )),
    (FailureClass.NETWORK, (
        r"ConnectionError", r"ConnectionResetError", r"Temporary failure in name resolution",
        r"Max retries exceeded", r"SSLError", r"Read timed out", r"HTTPError",
        r"socket\.gaierror", r"Connection refused",
    )),
    (FailureClass.INVALID_OUTPUT, (
        r"produced no metrics", r"must return a mapping", r"wrote no metrics",
        r"JSONDecodeError", r"could not read .*metrics",
    )),
    (FailureClass.METRIC_FAILURE, (
        r"metric .* must be a number", r"ZeroDivisionError",
        r"cannot compute .*metric",
    )),
    (FailureClass.USER_ERROR, (
        r"KeyError", r"AttributeError", r"TypeError", r"ValueError",
        r"IndexError", r"AssertionError", r"NotImplementedError",
        r"FileNotFoundError",
    )),
)


def classify(text: str, *, exit_code: int | None = None) -> FailureClass:
    """Classify a failure from its message and traceback."""
    if exit_code is not None and exit_code in (-11, 139):
        return FailureClass.SEGFAULT
    if exit_code is not None and exit_code in (-9, 137):
        # 137 is the canonical "the OOM killer took it" exit code.
        return FailureClass.OOM
    haystack = text or ""
    for failure_class, patterns in _PATTERNS:
        for pattern in patterns:
            if re.search(pattern, haystack, re.IGNORECASE):
                return failure_class
    return FailureClass.UNKNOWN


def from_exception(exc: BaseException, *, traceback_text: str = "") -> Failure:
    """Build a :class:`Failure` from an exception."""
    message = f"{type(exc).__name__}: {exc}"
    return Failure(
        failure_class=classify(traceback_text or message),
        message=message[:2000],
        traceback=traceback_text[-8000:],
    )


def from_process(stderr: str, stdout: str, exit_code: int) -> Failure:
    """Build a :class:`Failure` from a subprocess result."""
    combined = (stderr or "") + "\n" + (stdout or "")
    message = (stderr or stdout or f"process exited with code {exit_code}").strip()
    return Failure(
        failure_class=classify(combined, exit_code=exit_code),
        message=message[-2000:],
        traceback=combined[-8000:],
        exit_code=exit_code,
    )


def describe(failure_class: FailureClass) -> str:
    """A short explanation suitable for a report."""
    return {
        FailureClass.OOM: "ran out of memory",
        FailureClass.TIMEOUT: "exceeded its time limit",
        FailureClass.NAN: "produced a NaN or infinite value",
        FailureClass.DIVERGENCE: "failed to converge",
        FailureClass.SEGFAULT: "crashed at the native level",
        FailureClass.DEPENDENCY: "could not load a dependency",
        FailureClass.NETWORK: "hit a network failure",
        FailureClass.INVALID_OUTPUT: "did not produce usable metrics",
        FailureClass.METRIC_FAILURE: "failed while computing a metric",
        FailureClass.USER_ERROR: "raised an error in the experiment code",
        FailureClass.UNKNOWN: "failed for an unclassified reason",
    }[failure_class]


def is_infrastructure(failure_class: FailureClass) -> bool:
    """Whether a failure says something about the environment, not the method.

    The distinction matters for reporting: a network blip is noise to be
    retried, while an OOM at a particular batch size is a property of the
    method worth putting in a table.
    """
    return failure_class in (FailureClass.NETWORK, FailureClass.DEPENDENCY)


def summarise(failures: Iterable[Failure]) -> dict[str, int]:
    """Count failures by class."""
    counts: dict[str, int] = {}
    for failure in failures:
        key = failure.failure_class.value
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
