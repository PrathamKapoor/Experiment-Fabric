"""Markdown rendering.

The format that survives everything: readable in a terminal, in a pull
request, in a notebook, and in a repository that will outlive the tool
that wrote it.  Charts degrade to ASCII bars rather than disappearing,
because a report whose evidence is only visible in one format is not a
portable artifact.
"""

from __future__ import annotations

import time

from fabric.reporting import plots
from fabric.reporting.document import (
    Block,
    Bullets,
    Callout,
    Chart,
    Code,
    Document,
    GraphView,
    KeyValues,
    Paragraph,
    Table,
)

_CALLOUT_PREFIX = {
    "good": "**OK**",
    "info": "**Note**",
    "warn": "**Warning**",
    "bad": "**Failure**",
}


def _escape(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_table(table: Table) -> list[str]:
    lines: list[str] = []
    if table.caption:
        lines.append(f"*{table.caption}*")
        lines.append("")
    headers = [_escape(h) for h in table.headers]
    lines.append("| " + " | ".join(headers) + " |")
    separators = []
    for index in range(len(headers)):
        align = table.aligns[index] if index < len(table.aligns) else "left"
        separators.append({"left": ":---", "right": "---:", "center": ":---:"}.get(align, ":---"))
    lines.append("| " + " | ".join(separators) + " |")
    for row_index, row in enumerate(table.rows):
        cells = [_escape(c) for c in row]
        if row_index in table.emphasise:
            cells = [f"**{c}**" if c else c for c in cells]
        lines.append("| " + " | ".join(cells) + " |")
    if table.note:
        lines.append("")
        lines.append(f"> {table.note}")
    return lines


def render_chart(chart: Chart) -> list[str]:
    """Charts become a caption plus, where possible, an ASCII bar chart."""
    lines: list[str] = []
    rows = chart.data.get("rows") or []
    ascii_rows: list[tuple[str, float]] = []
    if chart.kind in ("diverging_bars", "stability"):
        key = "value" if chart.kind == "diverging_bars" else "stability"
        ascii_rows = [(str(r["label"]), float(r[key])) for r in rows
                      if r.get(key) == r.get(key)]
    elif chart.kind == "forest":
        ascii_rows = [(str(r["label"]), float(r["estimate"])) for r in rows
                      if r.get("estimate") == r.get("estimate")]
    if ascii_rows:
        maximum = 1.0 if chart.kind == "stability" else None
        lines.append("```")
        lines.extend(plots.ascii_bars(ascii_rows, maximum=maximum))
        lines.append("```")
    if chart.caption:
        lines.append("")
        lines.append(f"*{chart.caption}*")
    return lines


def render_block(block: Block) -> list[str]:
    if isinstance(block, Paragraph):
        return [block.text]
    if isinstance(block, Bullets):
        marker = "1." if block.ordered else "-"
        if block.ordered:
            return [f"{i}. {item}" for i, item in enumerate(block.items, start=1)]
        return [f"{marker} {item}" for item in block.items]
    if isinstance(block, KeyValues):
        lines = []
        if block.caption:
            lines.append(f"*{block.caption}*")
            lines.append("")
        for key, value in block.pairs:
            lines.append(f"- **{key}**: {_escape(value)}")
        return lines
    if isinstance(block, Table):
        return render_table(block)
    if isinstance(block, Callout):
        prefix = _CALLOUT_PREFIX.get(block.level, "**Note**")
        text = f"> {prefix} — {block.title}"
        if block.text:
            text += f"\n>\n> {block.text}"
        return [text]
    if isinstance(block, Code):
        lines = []
        if block.caption:
            lines.append(f"*{block.caption}*")
            lines.append("")
        lines.append(f"```{block.language}")
        lines.append(block.text)
        lines.append("```")
        return lines
    if isinstance(block, Chart):
        return render_chart(block)
    if isinstance(block, GraphView):
        # Markdown cannot be interactive, but many renderers draw mermaid,
        # and the ones that do not still show a readable definition.
        lines = []
        if block.caption:
            lines.append(f"*{block.caption}*")
            lines.append("")
        if block.mermaid:
            lines.append("```mermaid")
            lines.append(block.mermaid)
            lines.append("```")
        return lines
    return []


def render(document: Document, *, include_toc: bool = True) -> str:
    """Render a document to Markdown."""
    lines: list[str] = [f"# {document.title}", ""]
    if document.subtitle:
        lines.append(f"**Research question.** {document.subtitle}")
        lines.append("")

    metadata = document.metadata
    generated = time.strftime("%Y-%m-%d %H:%M:%S UTC",
                              time.gmtime(metadata.get("generated_at", time.time())))
    lines.append(
        f"`{metadata.get('experiment_id', '')}` · {metadata.get('n_runs', 0)} runs · "
        f"primary metric `{metadata.get('primary_metric', '')}` · "
        f"alpha {metadata.get('alpha', '')} with {metadata.get('correction', '')} "
        f"correction · generated {generated}"
    )
    lines.append("")
    lines.append(f"Specification digest `{metadata.get('spec_digest', '')}`")
    lines.append("")

    if include_toc:
        lines.append("## Contents")
        lines.append("")
        for depth, section in document.walk():
            if depth > 2:
                continue
            indent = "  " * (depth - 1)
            lines.append(f"{indent}- [{section.title}](#{section.anchor})")
        lines.append("")

    for depth, section in document.walk():
        if section.is_empty and not section.summary:
            continue
        lines.append("")
        lines.append("#" * min(6, depth + 1) + f" {section.title}")
        lines.append("")
        if section.summary and depth == 1:
            lines.append(f"*{section.summary}*")
            lines.append("")
        for block in section.blocks:
            rendered = render_block(block)
            if rendered:
                lines.extend(rendered)
                lines.append("")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "Generated by [Experiment Fabric](https://github.com/) — every number above "
        "is addressable back to the run, seed, code, data, environment and hardware "
        "that produced it."
    )
    lines.append("")
    return "\n".join(lines)
