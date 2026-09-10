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


class _FabricEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:  # noqa: D102 - json protocol
        if hasattr(o, "to_dict"):
            return o.to_dict()
        if isinstance(o, (set, frozenset)):
            return sorted(o, key=repr)
        if hasattr(o, "isoformat"):
            return o.isoformat()
        return str(o)


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


def _replace_with_retry(source: str, destination: str, *, attempts: int = 20) -> None:
    """``os.replace`` with a short retry.

    On Windows a rename fails while any other handle has the destination
    open, which an antivirus scanner or a concurrent reader can do at any
    moment. Retrying briefly is the documented remedy.
    """
    delay = 0.005
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 1.6, 0.2)


def write_json(path: str | os.PathLike[str], value: Any, *, indent: int = 2) -> None:
    """Serialise ``value`` as JSON to ``path`` atomically."""
    atomic_write_text(path, json.dumps(value, indent=indent, cls=_FabricEncoder, ensure_ascii=False) + "\n")


def read_json(path: str | os.PathLike[str], default: Any = None) -> Any:
    """Read JSON from ``path``, returning ``default`` when absent."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return default
