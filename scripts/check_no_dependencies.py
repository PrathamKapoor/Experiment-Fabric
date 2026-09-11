"""Fail if anything third-party leaked into the core.

Zero dependencies is a load-bearing claim, not an aspiration: a tool that
measures reproducibility must not perturb the environment it measures. It
is also the kind of claim that decays silently the first time someone adds
``import numpy`` to a helper, so it is enforced rather than documented.

The one permitted exception is the optional, guarded ``import yaml`` in
the YAML module, which falls back to the bundled parser.

Run directly, or as the ``no-dependency-guard`` CI job.
"""

from __future__ import annotations

import ast
import pathlib
import sys

ALLOWED_THIRD_PARTY = {
    # fabric/util/yamlio.py imports PyYAML inside a try/except and works
    # without it. Every other third-party import is a regression.
    ("yaml", "fabric/util/yamlio.py"),
}

ROOT = pathlib.Path(__file__).resolve().parent.parent


def guarded_by_try(tree: ast.Module, node: ast.stmt) -> bool:
    """Whether an import sits inside a try/except ImportError block."""
    for candidate in ast.walk(tree):
        if not isinstance(candidate, ast.Try):
            continue
        if not any(child is node for child in ast.walk(candidate)):
            continue
        for handler in candidate.handlers:
            names = []
            if isinstance(handler.type, ast.Name):
                names = [handler.type.id]
            elif isinstance(handler.type, ast.Tuple):
                names = [n.id for n in handler.type.elts if isinstance(n, ast.Name)]
            if "ImportError" in names or "ModuleNotFoundError" in names:
                return True
    return False


def main() -> int:
    standard = set(sys.stdlib_module_names)
    violations: list[str] = []
    checked = 0

    for path in sorted((ROOT / "fabric").rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        checked += 1
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                modules = [node.module]

            for module in modules:
                root = module.split(".")[0]
                if root in standard or root == "fabric":
                    continue
                if (root, relative) in ALLOWED_THIRD_PARTY and guarded_by_try(tree, node):
                    continue
                violations.append(
                    f"{relative}:{node.lineno}: imports third-party module {root!r}"
                )

    if violations:
        print("Third-party imports found in the core:\n")
        for violation in violations:
            print(f"  {violation}")
        print(
            "\nThe core must run on the standard library alone. If a dependency is "
            "genuinely required, put it behind a try/except ImportError with a "
            "working fallback and add it to ALLOWED_THIRD_PARTY here, with a reason."
        )
        return 1

    print(f"No third-party imports in {checked} core modules (1 guarded exception).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
