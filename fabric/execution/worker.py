"""The out-of-process run worker.

Invoked as ``python -m fabric.execution.worker <request.json>``.  It
resolves the entrypoint, runs it, and writes a result file.  Fabric uses
a subprocess by default because scientific code segfaults, leaks memory,
and calls ``sys.exit`` -- none of which should be able to take down a
480-run sweep.

The worker deliberately imports as little as possible before handing
control to user code, so that a heavy framework import is attributed to
the run that asked for it.
"""

from __future__ import annotations

import json
import os
import random
import sys
import traceback
from typing import Any


def _seed_global_rngs(seed: int) -> list[str]:
    """Seed every RNG we can reach without importing frameworks.

    NumPy and PyTorch are seeded only if they are *already* imported by
    the user's code path, which is checked through ``sys.modules`` rather
    than by importing them here.  Returns the list of what was seeded so
    the run record can state it.
    """
    seeded = ["python:random"]
    random.seed(seed)
    numpy = sys.modules.get("numpy")
    if numpy is not None:  # pragma: no cover - depends on user code
        try:
            numpy.random.seed(seed % (2 ** 32))
            seeded.append("numpy")
        except Exception:
            pass
    torch = sys.modules.get("torch")
    if torch is not None:  # pragma: no cover - depends on user code
        try:
            torch.manual_seed(seed)
            if getattr(torch, "cuda", None) and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            seeded.append("torch")
        except Exception:
            pass
    return seeded


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        sys.stderr.write("usage: python -m fabric.execution.worker <request.json>\n")
        return 2
    request_path = argv[1]
    with open(request_path, "r", encoding="utf-8") as fh:
        request: dict[str, Any] = json.load(fh)

    result_path = request["result_path"]
    root = request.get("root") or os.getcwd()
    if root not in sys.path:
        sys.path.insert(0, root)

    payload: dict[str, Any] = {"ok": False}
    try:
        from fabric.execution.adapters import (
            RunContext,
            call_command,
            call_python,
            parse_entrypoint,
            resolve,
        )

        context = RunContext.from_dict(request["context"])
        os.environ["PYTHONHASHSEED"] = str(context.seed)
        seeded = _seed_global_rngs(context.seed)

        entrypoint = request["entrypoint"]
        scheme, _ = parse_entrypoint(entrypoint)
        target = resolve(entrypoint, root=root)
        if scheme in ("python", "script"):
            result = call_python(target, context)  # type: ignore[arg-type]
        else:
            result = call_command(
                str(target), context, root=root,
                timeout=float(request.get("timeout") or 0.0),
                env_overlay=request.get("env_overlay") or {},
            )
        payload = {
            "ok": True,
            "metrics": result.metrics,
            "logs": result.logs,
            "artifacts": result.artifacts,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "seeded": seeded,
        }
    except BaseException as exc:  # noqa: BLE001 - the worker must report everything
        payload = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    with open(result_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":  # pragma: no cover - process entrypoint
    raise SystemExit(main(sys.argv))
