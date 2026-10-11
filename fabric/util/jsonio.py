"""Crash-safe JSON and text persistence.

Experiments run for hours; a half-written ``results.json`` produced by a
Ctrl-C at the wrong moment destroys the evidence chain.  Every write here
goes to a temporary file in the destination directory and is then
atomically renamed into place.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from typing import Any

_INFINITY = float("inf")


def sanitise(value: Any) -> Any:
    """Turn an object graph into data JSON can actually represent.

    Two jobs, done in one walk.

    The first is the usual one: unwrap entities via ``to_dict``, sets into
    sorted lists, datetimes into ISO strings.

    The second is the one that matters. Python's ``json`` module happily
    writes bare ``NaN``, ``Infinity`` and ``-Infinity``. Python reads them
    back; almost nothing else does. They are not in the JSON grammar, so
    ``JSON.parse``, ``jq``, Go, Rust and every strict parser reject the
    *entire document* over one of them. A results file only Python can
    read is not machine-readable, and for a project whose whole claim is
    that the evidence outlives the tool that wrote it, that is a defect
    rather than a quirk.

    So a non-finite float becomes ``null`` -- the honest JSON for "no
    number here". What ``null`` cannot express (NaN meaning *not
    computable*, an infinity meaning *unbounded*) is never carried by the
    number alone in Fabric anyway: every such field travels beside a
    ``verdict``, a ``reason`` or a narrative that says which it was. A
    zero-variance comparison reports an unbounded effect size in its prose
    and ``null`` in its number.
    """
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        # NaN is the only value unequal to itself; the infinities are
        # equal to themselves but not finite.
        if value != value or value in (_INFINITY, -_INFINITY):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): sanitise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitise(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return [sanitise(v) for v in sorted(value, key=repr)]
    if hasattr(value, "to_dict"):
        return sanitise(value.to_dict())
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def dumps(value: Any, *, indent: int | None = 2) -> str:
    """Serialise to strictly valid JSON.

    The one place Fabric turns an object into JSON text. ``allow_nan`` is
    off so that anything :func:`sanitise` somehow missed raises here rather
    than producing a file no other language can read.
    """
    return json.dumps(sanitise(value), indent=indent, ensure_ascii=False,
                      allow_nan=False)


class _FabricEncoder(json.JSONEncoder):
    """Kept for callers that pass ``cls=`` directly; prefer :func:`dumps`."""

    def default(self, o: Any) -> Any:
        return sanitise(o)


def atomic_write_text(path: str | os.PathLike[str], text: str) -> None:
    """Write ``text`` to ``path`` atomically, creating parent directories."""
    path = os.fspath(path)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=parent, prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        _replace_with_retry(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _retry_on_sharing_violation(action, *, what: str, attempts: int = 20):
    """Run ``action``, retrying the ``PermissionError`` Windows raises on a
    file another handle currently holds open.

    Windows refuses to rename over a file that any handle has open, and a
    rename briefly locks the destination.  A reader can therefore catch a
    sharing violation it did nothing wrong about, purely because a writer
    was mid-swap.  Retrying briefly is the documented remedy, and both
    halves of this module need it: a writer that gives up loses the update,
    a reader that gives up fails a whole experiment for no reason.
    """
    delay = 0.005
    for attempt in range(attempts):
        try:
            return action()
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 1.6, 0.2)


def _replace_with_retry(source: str, destination: str, *, attempts: int = 20) -> None:
    """``os.replace`` with a short retry; see :func:`_retry_on_sharing_violation`."""
    _retry_on_sharing_violation(
        lambda: os.replace(source, destination), what="rename", attempts=attempts
    )


def write_json(path: str | os.PathLike[str], value: Any, *, indent: int = 2) -> None:
    """Serialise ``value`` as strictly valid JSON to ``path``, atomically."""
    atomic_write_text(path, dumps(value, indent=indent) + "\n")


class CorruptRecordError(ValueError):
    """Raised when a stored record exists but cannot be parsed.

    A missing file is a normal condition and returns the default. A file
    that is present but unreadable is a damaged scientific record, and
    Fabric says so by name rather than letting a JSONDecodeError surface
    from four frames down.
    """


def read_json(path: str | os.PathLike[str], default: Any = None) -> Any:
    """Read JSON from ``path``, returning ``default`` when absent.

    A transient Windows sharing violation is retried rather than raised:
    the cache index and the experiment index are read while the runner's
    workers are writing them, and a reader that aborted on that would
    fail a run that was working perfectly well.
    """
    def _read() -> Any:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    try:
        return _retry_on_sharing_violation(_read, what="read")
    except FileNotFoundError:
        return default
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CorruptRecordError(
            f"{os.fspath(path)} is not readable JSON: {exc}. "
            "The file has been truncated or edited since Fabric wrote it; "
            "run 'fabric verify' to see what else changed."
        ) from exc
