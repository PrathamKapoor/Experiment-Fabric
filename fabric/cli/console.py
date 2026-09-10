"""Terminal output.

Colour is applied only when the output is an interactive terminal that
has not opted out, so piping ``fabric analyze`` into a file produces
clean text.  ``NO_COLOR`` and ``FABRIC_NO_COLOR`` are both honoured.
"""

from __future__ import annotations

import os
import shutil
import sys
from typing import Any, Iterable, Sequence


def _supports_colour(stream: Any) -> bool:
    if os.environ.get("NO_COLOR") or os.environ.get("FABRIC_NO_COLOR"):
        return False
    if os.environ.get("FABRIC_FORCE_COLOR"):
        return True
    if not hasattr(stream, "isatty") or not stream.isatty():
        return False
    if sys.platform == "win32":
        # Windows 10+ terminals understand ANSI once virtual terminal
        # processing is on, which modern consoles enable by default.
        return os.environ.get("WT_SESSION") is not None or \
            os.environ.get("TERM_PROGRAM") is not None or \
            os.environ.get("ANSICON") is not None or \
            int(os.environ.get("FABRIC_ANSI", "1")) == 1
    return True


COLOUR = _supports_colour(sys.stdout)

_CODES = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "grey": "\033[90m",
}


def paint(text: str, *styles: str) -> str:
    """Wrap text in ANSI styles when colour is available."""
    if not COLOUR or not styles:
        return text
    prefix = "".join(_CODES.get(s, "") for s in styles)
    return f"{prefix}{text}{_CODES['reset']}"


def width(default: int = 100) -> int:
    try:
        return min(shutil.get_terminal_size((default, 24)).columns, 160)
    except OSError:  # pragma: no cover - unusual terminals
        return default


def echo(text: str = "") -> None:
    print(text)


def heading(text: str) -> None:
    echo("")
    echo(paint(text, "bold"))
    echo(paint("-" * min(len(text), width()), "grey"))


def field(label: str, value: Any, label_width: int = 16) -> None:
    echo(f"  {paint(label.ljust(label_width), 'grey')} {value}")


def bullet(text: str, marker: str = "-") -> None:
    echo(f"  {marker} {text}")


def note(text: str) -> None:
    echo(paint(f"  {text}", "grey"))


def success(text: str) -> None:
    echo(f"  {paint('ok', 'green')} {text}")


def warn(text: str) -> None:
    echo(f"  {paint('warning', 'yellow')} {text}")


def error(text: str) -> None:
    print(f"  {paint('error', 'red')} {text}", file=sys.stderr)


def status_mark(status: str) -> str:
    colour = {
        "completed": "green", "cached": "cyan", "failed": "red",
        "pending": "grey", "running": "yellow", "skipped": "grey",
        "pass": "green", "verified": "green", "complete": "green", "recorded": "cyan",
        "warn": "yellow", "incomplete": "yellow", "unverified": "yellow",
        "fail": "red", "info": "blue",
    }.get(status.lower(), "")
    return paint(status, colour) if colour else status


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]], *,
          aligns: Sequence[str] = (), indent: int = 2,
          max_width: int | None = None) -> None:
    """Print an aligned table that fits the terminal."""
    if not rows:
        return
    columns = len(headers)
    cells = [[str(c) for c in row] + [""] * (columns - len(row)) for row in rows]
    widths = [len(str(h)) for h in headers]
    for row in cells:
        for index in range(columns):
            widths[index] = max(widths[index], len(row[index]))

    limit = max_width or (width() - indent - 2)
    total = sum(widths) + 2 * (columns - 1)
    if total > limit and columns:
        # Squeeze the widest column first; never below eight characters.
        excess = total - limit
        order = sorted(range(columns), key=lambda i: -widths[i])
        for index in order:
            if excess <= 0:
                break
            reducible = max(0, widths[index] - 8)
            take = min(reducible, excess)
            widths[index] -= take
            excess -= take

    def render(row: Sequence[str], style: str = "") -> str:
        parts = []
        for index, value in enumerate(row):
            align = aligns[index] if index < len(aligns) else "left"
            text = value if len(value) <= widths[index] else value[: widths[index] - 1] + "…"
            parts.append(text.rjust(widths[index]) if align == "right"
                         else text.ljust(widths[index]))
        line = "  ".join(parts).rstrip()
        return paint(line, style) if style else line

    pad = " " * indent
    echo(pad + render([str(h) for h in headers], "bold"))
    echo(pad + paint("  ".join("-" * w for w in widths), "grey"))
    for row in cells:
        echo(pad + render(row))


class Progress:
    """A single-line progress indicator that degrades to plain lines."""

    def __init__(self, total: int, label: str = "") -> None:
        self.total = max(1, total)
        self.label = label
        self.done = 0
        self.interactive = COLOUR and sys.stdout.isatty()

    def advance(self, text: str = "", failed: bool = False) -> None:
        self.done += 1
        fraction = self.done / self.total
        if self.interactive:
            bar_width = 24
            filled = int(fraction * bar_width)
            bar = "#" * filled + "." * (bar_width - filled)
            mark = paint("x", "red") if failed else " "
            line = (f"  [{bar}] {self.done}/{self.total} {mark} "
                    f"{text[: max(10, width() - bar_width - 26)]}")
            sys.stdout.write("\r" + line.ljust(width() - 1))
            sys.stdout.flush()
        elif failed or self.done == self.total or self.done % 25 == 0:
            print(f"  [{self.done}/{self.total}] {text}")

    def finish(self, summary: str = "") -> None:
        if self.interactive:
            sys.stdout.write("\r" + " " * (width() - 1) + "\r")
            sys.stdout.flush()
        if summary:
            echo(f"  {summary}")


def bar(value: float, maximum: float = 1.0, size: int = 20) -> str:
    """A small inline bar, for stability and rate displays."""
    if maximum <= 0:
        return " " * size
    filled = int(round(max(0.0, min(1.0, value / maximum)) * size))
    return "#" * filled + "." * (size - filled)
