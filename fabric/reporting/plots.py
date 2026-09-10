"""Charts, drawn as inline SVG with no dependencies.

Fabric will not pull in matplotlib to draw six chart types, and a report
that needs a plotting stack installed to be readable is not a portable
scientific artifact.  Everything here emits SVG text.

Design rules followed throughout (they are not stylistic preferences --
each one fixes a specific way charts mislead):

* **One axis, never two.**  Two measures on two scales invite whatever
  comparison the author wants; separate charts do not.
* **Colour carries polarity or identity, never rank.**  Improvement and
  regression use a validated diverging pair (blue/red) with a neutral
  grey for "not distinguishable"; the three condition roles use the first
  three categorical slots, which are the ones that clear colour-vision
  separation for all pairs.
* **Colour is never the only channel.**  Every mark carries a direct
  label or sits in a labelled row, and every chart in the report is
  accompanied by the table of the same numbers.
* **Text wears ink tokens**, never the series colour.
* Marks are thin, data-ends are rounded and anchored to the baseline,
  adjacent fills are separated by a surface gap, and grid lines recede.

Colours are emitted as ``var(--role, #fallback)`` so the report's own
light/dark tokens drive them, and a chart pasted into another document
still renders.
"""

from __future__ import annotations

import html
import math
from typing import Any, Iterable, Sequence

# Role -> light-mode fallback. The dark values live in the report stylesheet.
INK = "var(--viz-ink, #0b0b0b)"
INK_2 = "var(--viz-ink-2, #52514e)"
MUTED = "var(--viz-muted, #898781)"
GRID = "var(--viz-grid, #e1e0d9)"
AXIS = "var(--viz-axis, #c3c2b7)"
SURFACE = "var(--viz-surface, #fcfcfb)"
POSITIVE = "var(--viz-pos, #2a78d6)"
NEGATIVE = "var(--viz-neg, #d03b3b)"
NEUTRAL = "var(--viz-neutral, #a8a69e)"
SERIES = (
    "var(--viz-s1, #2a78d6)",
    "var(--viz-s2, #eb6834)",
    "var(--viz-s3, #1baf7a)",
)

# Single-quoted inside the family list: this string is emitted into a
# double-quoted SVG attribute, and a nested double quote would break the XML.
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"
LABEL_SIZE = 12
TICK_SIZE = 11
ROW_HEIGHT = 26
BAR_HEIGHT = 14
BAR_RADIUS = 4
MARGIN_TOP = 30
MARGIN_BOTTOM = 38
MARGIN_RIGHT = 18


def _esc(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _num(value: float, digits: int = 4) -> str:
    if value != value:
        return "n/a"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if value == 0:
        return "0"
    if abs(value) >= 1000 or abs(value) < 10 ** -(digits + 1):
        return f"{value:.3g}"
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def _label_width(labels: Iterable[str], minimum: int = 90, maximum: int = 260) -> int:
    longest = max((len(l) for l in labels), default=0)
    return max(minimum, min(maximum, int(longest * 6.6) + 14))


def _nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    """Human-friendly tick positions covering ``[low, high]``."""
    if not math.isfinite(low) or not math.isfinite(high):
        return [0.0]
    if high <= low:
        pad = abs(high) * 0.1 or 1.0
        low, high = low - pad, high + pad
    span = high - low
    raw = span / max(1, count - 1)
    magnitude = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    for multiple in (1, 2, 2.5, 5, 10):
        step = magnitude * multiple
        if span / step <= count:
            break
    start = math.floor(low / step) * step
    ticks = []
    value = start
    while value <= high + step * 0.5 and len(ticks) < 20:
        ticks.append(round(value, 12))
        value += step
    return ticks


def _bar_path(x_zero: float, x_value: float, y: float, height: float,
              radius: float = BAR_RADIUS) -> str:
    """A bar rounded only at the data end, square against the baseline."""
    if abs(x_value - x_zero) < 0.5:
        x_value = x_zero + (0.5 if x_value >= x_zero else -0.5)
    r = min(radius, abs(x_value - x_zero), height / 2)
    if x_value >= x_zero:
        return (
            f"M{x_zero:.1f},{y:.1f} H{x_value - r:.1f} "
            f"a{r:.1f},{r:.1f} 0 0 1 {r:.1f},{r:.1f} "
            f"V{y + height - r:.1f} a{r:.1f},{r:.1f} 0 0 1 {-r:.1f},{r:.1f} "
            f"H{x_zero:.1f} Z"
        )
    return (
        f"M{x_zero:.1f},{y:.1f} H{x_value + r:.1f} "
        f"a{r:.1f},{r:.1f} 0 0 0 {-r:.1f},{r:.1f} "
        f"V{y + height - r:.1f} a{r:.1f},{r:.1f} 0 0 0 {r:.1f},{r:.1f} "
        f"H{x_zero:.1f} Z"
    )


def _open(width: int, height: int, title: str, description: str = "") -> list[str]:
    return [
        f'<svg class="viz" viewBox="0 0 {width} {height}" width="100%" '
        f'height="{height}" role="img" xmlns="http://www.w3.org/2000/svg" '
        f'aria-label="{_esc(title)}" style="font-family:{FONT};max-width:100%">',
        f"<title>{_esc(title)}</title>",
        (f"<desc>{_esc(description)}</desc>" if description else ""),
    ]


def _x_axis(parts: list[str], ticks: Sequence[float], scale, top: float,
            bottom: float, digits: int = 4) -> None:
    """Recessive vertical grid plus tick labels along the bottom."""
    for tick in ticks:
        x = scale(tick)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top:.1f}" x2="{x:.1f}" y2="{bottom:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{bottom + 15:.1f}" text-anchor="middle" '
            f'font-size="{TICK_SIZE}" fill="{MUTED}" '
            f'style="font-variant-numeric:tabular-nums">{_esc(_num(tick, digits))}</text>'
        )


# --------------------------------------------------------------------------
# 1. forest plot
# --------------------------------------------------------------------------


def forest(rows: Sequence[dict[str, Any]], *, title: str = "",
           unit: str = "", width: int = 760) -> str:
    """Effect estimates with confidence intervals.

    ``rows`` need ``label``, ``estimate``, ``low``, ``high`` and may carry
    ``verdict`` (``improvement`` / ``regression`` / anything else) and
    ``note``.  The zero line is drawn heavier than the grid because
    whether an interval crosses it is the whole point of the chart.
    """
    rows = [r for r in rows if r.get("estimate") == r.get("estimate")]
    if not rows:
        return ""
    labels = [str(r["label"]) for r in rows]
    left = _label_width(labels)
    plot_width = width - left - MARGIN_RIGHT - 70
    height = MARGIN_TOP + len(rows) * ROW_HEIGHT + MARGIN_BOTTOM

    low = min(min(r["low"], r["estimate"]) for r in rows)
    high = max(max(r["high"], r["estimate"]) for r in rows)
    low, high = min(low, 0.0), max(high, 0.0)
    pad = (high - low) * 0.12 or 0.01
    low, high = low - pad, high + pad

    def scale(value: float) -> float:
        return left + (value - low) / (high - low) * plot_width

    parts = _open(width, height, title or "effect estimates with confidence intervals",
                  "Each row shows the estimated difference and its interval; "
                  "an interval crossing zero means the difference is not resolved.")
    ticks = _nice_ticks(low, high)
    _x_axis(parts, ticks, scale, MARGIN_TOP - 12, height - MARGIN_BOTTOM + 6)

    zero_x = scale(0.0)
    parts.append(
        f'<line x1="{zero_x:.1f}" y1="{MARGIN_TOP - 12:.1f}" x2="{zero_x:.1f}" '
        f'y2="{height - MARGIN_BOTTOM + 6:.1f}" stroke="{AXIS}" stroke-width="2"/>'
    )
    parts.append(
        f'<text x="{zero_x:.1f}" y="{MARGIN_TOP - 18:.1f}" text-anchor="middle" '
        f'font-size="{TICK_SIZE}" fill="{MUTED}">no difference</text>'
    )

    for index, row in enumerate(rows):
        y = MARGIN_TOP + index * ROW_HEIGHT + ROW_HEIGHT / 2
        verdict = str(row.get("verdict", ""))
        colour = (POSITIVE if verdict == "improvement"
                  else NEGATIVE if verdict == "regression" else NEUTRAL)
        x_low, x_high, x_est = scale(row["low"]), scale(row["high"]), scale(row["estimate"])
        tooltip = (
            f"{row['label']}: {_num(row['estimate'])}{unit} "
            f"[{_num(row['low'])}, {_num(row['high'])}]"
            + (f" - {row['note']}" if row.get("note") else "")
        )
        parts.append(f'<g><title>{_esc(tooltip)}</title>')
        parts.append(
            f'<line x1="{x_low:.1f}" y1="{y:.1f}" x2="{x_high:.1f}" y2="{y:.1f}" '
            f'stroke="{colour}" stroke-width="2" stroke-linecap="round"/>'
        )
        for cap in (x_low, x_high):
            parts.append(
                f'<line x1="{cap:.1f}" y1="{y - 4:.1f}" x2="{cap:.1f}" '
                f'y2="{y + 4:.1f}" stroke="{colour}" stroke-width="2"/>'
            )
        parts.append(
            f'<circle cx="{x_est:.1f}" cy="{y:.1f}" r="5" fill="{colour}" '
            f'stroke="{SURFACE}" stroke-width="2"/>'
        )
        parts.append("</g>")
        parts.append(
            f'<text x="{left - 10:.1f}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="{LABEL_SIZE}" fill="{INK}">{_esc(row["label"])}</text>'
        )
        parts.append(
            f'<text x="{width - MARGIN_RIGHT:.1f}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="{TICK_SIZE}" fill="{INK_2}" '
            f'style="font-variant-numeric:tabular-nums">'
            f'{_esc(_num(row["estimate"]))}</text>'
        )
    parts.append("</svg>")
    return "\n".join(p for p in parts if p)


# --------------------------------------------------------------------------
# 2. per-seed strip plot
# --------------------------------------------------------------------------


def strip(rows: Sequence[dict[str, Any]], *, title: str = "", width: int = 760) -> str:
    """Every replication as a dot, with the mean marked.

    The chart a bar-of-means hides: it shows the spread the conclusion is
    resting on, and whether two conditions overlap at all.
    """
    rows = [r for r in rows if r.get("values")]
    if not rows:
        return ""
    labels = [str(r["label"]) for r in rows]
    left = _label_width(labels)
    plot_width = width - left - MARGIN_RIGHT - 60
    height = MARGIN_TOP + len(rows) * ROW_HEIGHT + MARGIN_BOTTOM

    everything = [v for r in rows for v in r["values"]]
    low, high = min(everything), max(everything)
    pad = (high - low) * 0.1 or (abs(high) * 0.05 or 0.01)
    low, high = low - pad, high + pad

    def scale(value: float) -> float:
        return left + (value - low) / (high - low) * plot_width

    parts = _open(width, height, title or "per-replication values",
                  "One dot per seed; the vertical rule marks the mean.")
    _x_axis(parts, _nice_ticks(low, high), scale, MARGIN_TOP - 12,
            height - MARGIN_BOTTOM + 6)

    for index, row in enumerate(rows):
        y = MARGIN_TOP + index * ROW_HEIGHT + ROW_HEIGHT / 2
        values = list(row["values"])
        mean = sum(values) / len(values)
        emphasis = bool(row.get("emphasise"))
        colour = SERIES[0] if emphasis else MUTED
        parts.append(
            f'<line x1="{scale(min(values)):.1f}" y1="{y:.1f}" '
            f'x2="{scale(max(values)):.1f}" y2="{y:.1f}" stroke="{GRID}" '
            f'stroke-width="1"/>'
        )
        for value in values:
            parts.append(
                f'<circle cx="{scale(value):.1f}" cy="{y:.1f}" r="4" fill="{colour}" '
                f'fill-opacity="0.72" stroke="{SURFACE}" stroke-width="1.5">'
                f"<title>{_esc(row['label'])}: {_esc(_num(value))}</title></circle>"
            )
        parts.append(
            f'<line x1="{scale(mean):.1f}" y1="{y - 9:.1f}" x2="{scale(mean):.1f}" '
            f'y2="{y + 9:.1f}" stroke="{INK}" stroke-width="2">'
            f"<title>mean {_esc(_num(mean))} over {len(values)} runs</title></line>"
        )
        weight = ' font-weight="600"' if emphasis else ""
        parts.append(
            f'<text x="{left - 10:.1f}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="{LABEL_SIZE}" fill="{INK}"{weight}>'
            f'{_esc(row["label"])}</text>'
        )
        parts.append(
            f'<text x="{width - MARGIN_RIGHT:.1f}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="{TICK_SIZE}" fill="{INK_2}" '
            f'style="font-variant-numeric:tabular-nums">{_esc(_num(mean))}</text>'
        )
    parts.append("</svg>")
    return "\n".join(p for p in parts if p)


# --------------------------------------------------------------------------
# 3. diverging bars (ablation deltas)
# --------------------------------------------------------------------------


def diverging_bars(rows: Sequence[dict[str, Any]], *, title: str = "",
                   width: int = 760, positive_label: str = "",
                   negative_label: str = "") -> str:
    """Signed magnitudes around a neutral zero.

    Used for ablation effects, where the sign is the finding: a bar to one
    side means the component was carrying the result.
    """
    rows = [r for r in rows if r.get("value") == r.get("value")]
    if not rows:
        return ""
    labels = [str(r["label"]) for r in rows]
    left = _label_width(labels)
    plot_width = width - left - MARGIN_RIGHT - 66
    height = MARGIN_TOP + len(rows) * ROW_HEIGHT + MARGIN_BOTTOM

    values = [r["value"] for r in rows]
    extent = max(abs(min(values)), abs(max(values))) or 1.0
    low, high = -extent * 1.15, extent * 1.15

    def scale(value: float) -> float:
        return left + (value - low) / (high - low) * plot_width

    parts = _open(width, height, title or "signed effect by condition")
    _x_axis(parts, _nice_ticks(low, high), scale, MARGIN_TOP - 12,
            height - MARGIN_BOTTOM + 6)
    zero_x = scale(0.0)

    if positive_label or negative_label:
        if negative_label:
            parts.append(
                f'<text x="{left:.1f}" y="{MARGIN_TOP - 18:.1f}" text-anchor="start" '
                f'font-size="{TICK_SIZE}" fill="{MUTED}">{_esc(negative_label)}</text>'
            )
        if positive_label:
            parts.append(
                f'<text x="{left + plot_width:.1f}" y="{MARGIN_TOP - 18:.1f}" '
                f'text-anchor="end" font-size="{TICK_SIZE}" fill="{MUTED}">'
                f'{_esc(positive_label)}</text>'
            )

    for index, row in enumerate(rows):
        y = MARGIN_TOP + index * ROW_HEIGHT + (ROW_HEIGHT - BAR_HEIGHT) / 2
        value = row["value"]
        significant = bool(row.get("significant", True))
        colour = NEUTRAL if not significant else (POSITIVE if value >= 0 else NEGATIVE)
        tooltip = f"{row['label']}: {_num(value)}" + (
            "" if significant else " (not distinguishable from noise)"
        )
        parts.append(
            f'<path d="{_bar_path(zero_x, scale(value), y, BAR_HEIGHT)}" fill="{colour}" '
            f'fill-opacity="{"1" if significant else "0.55"}">'
            f"<title>{_esc(tooltip)}</title></path>"
        )
        text_y = y + BAR_HEIGHT / 2 + 4
        parts.append(
            f'<text x="{left - 10:.1f}" y="{text_y:.1f}" text-anchor="end" '
            f'font-size="{LABEL_SIZE}" fill="{INK}">{_esc(row["label"])}</text>'
        )
        parts.append(
            f'<text x="{width - MARGIN_RIGHT:.1f}" y="{text_y:.1f}" text-anchor="end" '
            f'font-size="{TICK_SIZE}" fill="{INK_2}" '
            f'style="font-variant-numeric:tabular-nums">{_esc(_num(value))}'
            f'{"" if significant else " ns"}</text>'
        )
    parts.append(
        f'<line x1="{zero_x:.1f}" y1="{MARGIN_TOP - 12:.1f}" x2="{zero_x:.1f}" '
        f'y2="{height - MARGIN_BOTTOM + 6:.1f}" stroke="{AXIS}" stroke-width="2"/>'
    )
    parts.append("</svg>")
    return "\n".join(p for p in parts if p)


# --------------------------------------------------------------------------
# 4. stability bars
# --------------------------------------------------------------------------


_VERDICT_COLOUR = {
    "within": "var(--viz-good, #0ca30c)",
    "beyond": "var(--viz-crit, #d03b3b)",
    "inconclusive": "var(--viz-warn, #fab219)",
}


def stability_bars(rows: Sequence[dict[str, Any]], *, title: str = "",
                   width: int = 760) -> str:
    """Reproducibility stability per axis, on a fixed 0-1 scale.

    The scale is fixed rather than data-driven so two experiments'
    fingerprints can be put side by side.  The verdict is shown as a
    labelled chip rather than by colour alone.
    """
    rows = [r for r in rows if r.get("stability") == r.get("stability")]
    if not rows:
        return ""
    labels = [str(r["label"]) for r in rows]
    left = _label_width(labels, minimum=110)
    right_pad = 130
    plot_width = width - left - MARGIN_RIGHT - right_pad
    height = MARGIN_TOP + len(rows) * ROW_HEIGHT + MARGIN_BOTTOM

    def scale(value: float) -> float:
        return left + max(0.0, min(1.0, value)) * plot_width

    parts = _open(width, height, title or "stability by perturbation axis",
                  "1.0 means the perturbed runs are indistinguishable from a "
                  "relabelling of the reference runs.")
    _x_axis(parts, [0.0, 0.25, 0.5, 0.75, 1.0], scale, MARGIN_TOP - 12,
            height - MARGIN_BOTTOM + 6, digits=2)

    for index, row in enumerate(rows):
        y = MARGIN_TOP + index * ROW_HEIGHT + (ROW_HEIGHT - BAR_HEIGHT) / 2
        value = float(row["stability"])
        verdict = str(row.get("verdict", ""))
        parts.append(
            f'<path d="{_bar_path(left, scale(value), y, BAR_HEIGHT)}" '
            f'fill="{SERIES[0]}">'
            f"<title>{_esc(row['label'])}: stability {_esc(_num(value, 2))}</title></path>"
        )
        low, high = row.get("low"), row.get("high")
        if low is not None and high is not None and low == low and high == high:
            mid = y + BAR_HEIGHT / 2
            parts.append(
                f'<line x1="{scale(low):.1f}" y1="{mid:.1f}" x2="{scale(high):.1f}" '
                f'y2="{mid:.1f}" stroke="{INK}" stroke-width="1.5" stroke-opacity="0.55"/>'
            )
        text_y = y + BAR_HEIGHT / 2 + 4
        parts.append(
            f'<text x="{left - 10:.1f}" y="{text_y:.1f}" text-anchor="end" '
            f'font-size="{LABEL_SIZE}" fill="{INK}">{_esc(row["label"])}</text>'
        )
        parts.append(
            f'<text x="{left + plot_width + 8:.1f}" y="{text_y:.1f}" '
            f'font-size="{TICK_SIZE}" fill="{INK_2}" '
            f'style="font-variant-numeric:tabular-nums">{_esc(_num(value, 2))}</text>'
        )
        if verdict and verdict != "no_tolerance":
            colour = _VERDICT_COLOUR.get(verdict, MUTED)
            parts.append(
                f'<circle cx="{left + plot_width + 48:.1f}" cy="{text_y - 4:.1f}" r="4" '
                f'fill="{colour}"/>'
            )
            parts.append(
                f'<text x="{left + plot_width + 57:.1f}" y="{text_y:.1f}" '
                f'font-size="{TICK_SIZE}" fill="{INK_2}">{_esc(verdict)}</text>'
            )
    parts.append("</svg>")
    return "\n".join(p for p in parts if p)


# --------------------------------------------------------------------------
# 5. sweep line
# --------------------------------------------------------------------------


def sweep_line(points: Sequence[dict[str, Any]], *, title: str = "",
               x_label: str = "", y_label: str = "", width: int = 760,
               height: int = 300) -> str:
    """One parameter against one metric, with a spread band.

    The band is the point of the chart: a difference between settings that
    sits inside it is not a finding.
    """
    points = [p for p in points if p.get("mean") == p.get("mean")]
    if len(points) < 2:
        return ""
    left = 74
    plot_width = width - left - MARGIN_RIGHT - 10
    plot_height = height - MARGIN_TOP - MARGIN_BOTTOM

    xs = list(range(len(points)))
    lows = [p["mean"] - p.get("std", 0.0) for p in points]
    highs = [p["mean"] + p.get("std", 0.0) for p in points]
    low, high = min(lows), max(highs)
    pad = (high - low) * 0.18 or (abs(high) * 0.05 or 0.01)
    low, high = low - pad, high + pad

    def sx(index: int) -> float:
        return left + (index / max(1, len(points) - 1)) * plot_width

    def sy(value: float) -> float:
        return MARGIN_TOP + plot_height - (value - low) / (high - low) * plot_height

    parts = _open(width, height, title or "parameter sweep")
    for tick in _nice_ticks(low, high):
        y = sy(tick)
        parts.append(
            f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left + plot_width:.1f}" '
            f'y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 8:.1f}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="{TICK_SIZE}" fill="{MUTED}" '
            f'style="font-variant-numeric:tabular-nums">{_esc(_num(tick))}</text>'
        )

    band_top = " ".join(f"{sx(i):.1f},{sy(highs[i]):.1f}" for i in xs)
    band_bottom = " ".join(f"{sx(i):.1f},{sy(lows[i]):.1f}" for i in reversed(xs))
    parts.append(
        f'<polygon points="{band_top} {band_bottom}" fill="{SERIES[0]}" '
        f'fill-opacity="0.14"/>'
    )
    line = " ".join(f"{sx(i):.1f},{sy(points[i]['mean']):.1f}" for i in xs)
    parts.append(
        f'<polyline points="{line}" fill="none" stroke="{SERIES[0]}" '
        f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
    )
    for index, point in enumerate(points):
        parts.append(
            f'<circle cx="{sx(index):.1f}" cy="{sy(point["mean"]):.1f}" r="4.5" '
            f'fill="{SERIES[0]}" stroke="{SURFACE}" stroke-width="2">'
            f"<title>{_esc(point['label'])}: {_esc(_num(point['mean']))} "
            f"+/- {_esc(_num(point.get('std', 0.0)))}</title></circle>"
        )
        parts.append(
            f'<text x="{sx(index):.1f}" y="{height - MARGIN_BOTTOM + 20:.1f}" '
            f'text-anchor="middle" font-size="{TICK_SIZE}" fill="{MUTED}">'
            f'{_esc(point["label"])}</text>'
        )
    if x_label:
        parts.append(
            f'<text x="{left + plot_width / 2:.1f}" y="{height - 4:.1f}" '
            f'text-anchor="middle" font-size="{TICK_SIZE}" fill="{INK_2}">'
            f'{_esc(x_label)}</text>'
        )
    if y_label:
        parts.append(
            f'<text x="14" y="{MARGIN_TOP + plot_height / 2:.1f}" '
            f'text-anchor="middle" font-size="{TICK_SIZE}" fill="{INK_2}" '
            f'transform="rotate(-90 14 {MARGIN_TOP + plot_height / 2:.1f})">'
            f'{_esc(y_label)}</text>'
        )
    parts.append("</svg>")
    return "\n".join(p for p in parts if p)


# --------------------------------------------------------------------------
# 6. cost against quality
# --------------------------------------------------------------------------


_ROLE_SLOT = {"treatment": 0, "baseline": 1, "ablation": 2}


def cost_scatter(points: Sequence[dict[str, Any]], *, title: str = "",
                 x_label: str = "compute (seconds)", y_label: str = "",
                 width: int = 760, height: int = 340) -> str:
    """Quality against what it cost.

    Three roles, three categorical slots -- the maximum that keeps every
    pair distinguishable under colour-vision deficiency when marks are
    scattered rather than adjacent.  Every point is directly labelled, so
    colour is reinforcement rather than the only channel.
    """
    points = [p for p in points if p.get("x") == p.get("x") and p.get("y") == p.get("y")]
    if len(points) < 2:
        return ""
    left = 78
    plot_width = width - left - MARGIN_RIGHT - 12
    plot_height = height - MARGIN_TOP - MARGIN_BOTTOM - 16

    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    x_low, x_high = min(xs), max(xs)
    y_low, y_high = min(ys), max(ys)
    x_pad = (x_high - x_low) * 0.16 or (abs(x_high) * 0.1 or 1.0)
    y_pad = (y_high - y_low) * 0.16 or (abs(y_high) * 0.1 or 0.01)
    x_low, x_high = x_low - x_pad, x_high + x_pad
    y_low, y_high = y_low - y_pad, y_high + y_pad

    def sx(value: float) -> float:
        return left + (value - x_low) / (x_high - x_low) * plot_width

    def sy(value: float) -> float:
        return MARGIN_TOP + plot_height - (value - y_low) / (y_high - y_low) * plot_height

    parts = _open(width, height, title or "quality against compute")
    for tick in _nice_ticks(y_low, y_high):
        y = sy(tick)
        parts.append(
            f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left + plot_width:.1f}" '
            f'y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 8:.1f}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="{TICK_SIZE}" fill="{MUTED}" '
            f'style="font-variant-numeric:tabular-nums">{_esc(_num(tick))}</text>'
        )
    for tick in _nice_ticks(x_low, x_high, 4):
        x = sx(tick)
        parts.append(
            f'<text x="{x:.1f}" y="{MARGIN_TOP + plot_height + 18:.1f}" '
            f'text-anchor="middle" font-size="{TICK_SIZE}" fill="{MUTED}" '
            f'style="font-variant-numeric:tabular-nums">{_esc(_num(tick, 3))}</text>'
        )

    for point in points:
        slot = _ROLE_SLOT.get(str(point.get("role", "")), 2)
        colour = SERIES[slot]
        x, y = sx(point["x"]), sy(point["y"])
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{colour}" '
            f'stroke="{SURFACE}" stroke-width="2">'
            f"<title>{_esc(point['label'])}: {_esc(_num(point['y']))} at "
            f"{_esc(_num(point['x'], 3))}</title></circle>"
        )
        anchor = "end" if x > left + plot_width * 0.72 else "start"
        offset = -10 if anchor == "end" else 10
        parts.append(
            f'<text x="{x + offset:.1f}" y="{y + 4:.1f}" text-anchor="{anchor}" '
            f'font-size="{TICK_SIZE}" fill="{INK}">{_esc(point["label"])}</text>'
        )

    legend_y = height - 12
    legend_x = left
    for role, slot in sorted(_ROLE_SLOT.items(), key=lambda kv: kv[1]):
        if not any(str(p.get("role", "")) == role for p in points):
            continue
        parts.append(
            f'<circle cx="{legend_x:.1f}" cy="{legend_y - 4:.1f}" r="5" '
            f'fill="{SERIES[slot]}"/>'
        )
        parts.append(
            f'<text x="{legend_x + 10:.1f}" y="{legend_y:.1f}" '
            f'font-size="{TICK_SIZE}" fill="{INK_2}">{_esc(role)}</text>'
        )
        legend_x += 24 + len(role) * 6.6
    parts.append(
        f'<text x="{left + plot_width:.1f}" y="{MARGIN_TOP + plot_height + 34:.1f}" '
        f'text-anchor="end" font-size="{TICK_SIZE}" fill="{INK_2}">{_esc(x_label)}</text>'
    )
    if y_label:
        parts.append(
            f'<text x="14" y="{MARGIN_TOP + plot_height / 2:.1f}" text-anchor="middle" '
            f'font-size="{TICK_SIZE}" fill="{INK_2}" '
            f'transform="rotate(-90 14 {MARGIN_TOP + plot_height / 2:.1f})">'
            f'{_esc(y_label)}</text>'
        )
    parts.append("</svg>")
    return "\n".join(p for p in parts if p)


# --------------------------------------------------------------------------
# terminal fallback
# --------------------------------------------------------------------------


def ascii_bars(rows: Sequence[tuple[str, float]], *, width: int = 32,
               maximum: float | None = None) -> list[str]:
    """A plain-text bar chart, for terminals and Markdown."""
    if not rows:
        return []
    top = maximum if maximum is not None else max(abs(v) for _, v in rows) or 1.0
    label_width = max(len(label) for label, _ in rows)
    lines = []
    for label, value in rows:
        filled = int(round(abs(value) / top * width)) if top else 0
        bar = "#" * min(width, filled)
        lines.append(f"{label:<{label_width}}  {bar:<{width}}  {_num(value)}")
    return lines


CHART_RENDERERS = {
    "forest": forest,
    "strip": strip,
    "diverging_bars": diverging_bars,
    "stability": stability_bars,
    "sweep_line": sweep_line,
    "cost_scatter": cost_scatter,
}


def render(kind: str, data: dict[str, Any]) -> str:
    """Render a chart block by kind."""
    renderer = CHART_RENDERERS.get(kind)
    if renderer is None:
        return ""
    payload = dict(data)
    rows = payload.pop("rows", None)
    points = payload.pop("points", None)
    if rows is not None:
        return renderer(rows, **payload)
    if points is not None:
        return renderer(points, **payload)
    return ""
