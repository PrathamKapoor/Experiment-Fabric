"""Content addressing.

Every entity in Fabric is identified by the SHA-256 of its *canonical*
JSON serialisation.  Canonical means: sorted keys, no insignificant
whitespace, UTF-8, and a stable representation for floats.  Two entities
with the same identity therefore hash identically on any machine, which
is what makes cache hits and provenance verification meaningful.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterable
from typing import Any

_FLOAT_PRECISION = 12
"""Floats are rounded to this many significant digits before hashing.

Without it, ``0.1 + 0.2`` and ``0.30000000000000004`` would be distinct
identities on some platforms.  Twelve digits keeps every value a human
would write in a config file exact while absorbing last-bit noise.
"""


def _canonicalise(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return "__nan__"
        if math.isinf(value):
            return "__inf__" if value > 0 else "__-inf__"
        if value == int(value) and abs(value) < 1e15:
            return int(value)
        return float(f"%.{_FLOAT_PRECISION}g" % value)
    if isinstance(value, dict):
        return {str(k): _canonicalise(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonicalise(v) for v in value]
    if isinstance(value, set):
        return [_canonicalise(v) for v in sorted(value, key=repr)]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if hasattr(value, "to_dict"):
        return _canonicalise(value.to_dict())
    return str(value)


def canonical_json(value: Any) -> str:
    """Serialise ``value`` to a byte-stable JSON string."""
    return json.dumps(
        _canonicalise(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def content_id(value: Any, *, prefix: str = "", length: int = 16) -> str:
    """Return a short content-addressed identifier for ``value``.

    ``length`` is in hex characters.  16 hex chars = 64 bits, which for the
    number of entities a research group produces keeps collision odds far
    below the odds of the disk failing.
    """
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    body = digest[:length]
    return f"{prefix}{body}" if prefix else body


def full_digest(value: Any) -> str:
    """Return the full ``sha256:...`` digest of a canonicalised value."""
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_digest(path: str | os.PathLike[str], *, chunk_size: int = 1 << 20) -> str:
    """Return ``sha256:<hex>`` for a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def tree_digest(
    root: str | os.PathLike[str],
    *,
    exclude: Iterable[str] = (),
) -> str:
    """Return a digest over a directory tree.

    The digest covers relative paths *and* content, so a renamed file
    changes the fingerprint.  Paths are normalised to forward slashes so a
    dataset registered on Windows matches the same dataset on Linux.
    """
    excluded = tuple(exclude)
    entries: list[tuple[str, str]] = []
    root = os.fspath(root)
    if os.path.isfile(root):
        return file_digest(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in excluded)
        for name in sorted(filenames):
            if name in excluded:
                continue
            abs_path = os.path.join(dirpath, name)
            rel = os.path.relpath(abs_path, root).replace(os.sep, "/")
            entries.append((rel, file_digest(abs_path)))
    return full_digest(entries)


def short_id(value: str, length: int = 8) -> str:
    """Trim a digest or identifier for display."""
    if value.startswith("sha256:"):
        value = value[len("sha256:"):]
    return value[:length]
