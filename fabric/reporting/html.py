"""Self-contained HTML rendering.

One file, no network, no build step: the report opens on any machine
that has a browser, including an air-gapped one, and keeps its charts.
That matters because the HTML report is the artifact most likely to be
attached to a paper submission or handed to a reviewer.

The stylesheet defines the visualization tokens the SVG charts reference,
in three scopes: the light default, the OS dark preference, and an
explicit theme stamp -- so the charts follow the reader's theme rather
than assuming a white page.
"""

from __future__ import annotations

import html as html_module
import time
from typing import Any

from fabric.reporting import plots
from fabric.reporting.document import (
    Block,
    Bullets,
    Callout,
    Chart,
    Code,
    Document,
    KeyValues,
    Paragraph,
    Table,
)

STYLESHEET = """
:root {
  color-scheme: light;
  --page: #f9f9f7;
  --surface: #ffffff;
  --ink: #0b0b0b;
  --ink-2: #52514e;
  --muted: #898781;
  --line: #e1e0d9;
  --accent: #2a78d6;
  --good: #0ca30c;
  --warn: #fab219;
  --bad: #d03b3b;
  --code-bg: #f2f1ed;
  --viz-surface: #ffffff;
  --viz-ink: #0b0b0b;
  --viz-ink-2: #52514e;
  --viz-muted: #898781;
  --viz-grid: #e1e0d9;
  --viz-axis: #c3c2b7;
  --viz-pos: #2a78d6;
  --viz-neg: #d03b3b;
  --viz-neutral: #a8a69e;
  --viz-s1: #2a78d6;
  --viz-s2: #eb6834;
  --viz-s3: #1baf7a;
  --viz-good: #0ca30c;
  --viz-warn: #fab219;
  --viz-crit: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d;
    --surface: #1a1a19;
    --ink: #ffffff;
    --ink-2: #c3c2b7;
    --muted: #898781;
    --line: #2c2c2a;
    --accent: #3987e5;
    --code-bg: #232321;
    --viz-surface: #1a1a19;
    --viz-ink: #ffffff;
    --viz-ink-2: #c3c2b7;
    --viz-grid: #2c2c2a;
    --viz-axis: #383835;
    --viz-pos: #3987e5;
    --viz-s1: #3987e5;
    --viz-s2: #d95926;
    --viz-s3: #199e70;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d;
  --surface: #1a1a19;
  --ink: #ffffff;
  --ink-2: #c3c2b7;
  --muted: #898781;
  --line: #2c2c2a;
  --accent: #3987e5;
  --code-bg: #232321;
  --viz-surface: #1a1a19;
  --viz-ink: #ffffff;
  --viz-ink-2: #c3c2b7;
  --viz-grid: #2c2c2a;
  --viz-axis: #383835;
  --viz-pos: #3987e5;
  --viz-s1: #3987e5;
  --viz-s2: #d95926;
  --viz-s3: #199e70;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--page);
  color: var(--ink);
  font: 15px/1.62 system-ui, -apple-system, 'Segoe UI', sans-serif;
}
.wrap { max-width: 1080px; margin: 0 auto; padding: 40px 24px 96px; }
header.masthead { border-bottom: 1px solid var(--line); padding-bottom: 22px; margin-bottom: 8px; }
h1 { font-size: 30px; line-height: 1.2; margin: 0 0 10px; letter-spacing: -0.01em; }
h2 { font-size: 21px; margin: 44px 0 10px; padding-top: 14px; border-top: 1px solid var(--line); }
h3 { font-size: 16px; margin: 26px 0 8px; }
h4 { font-size: 14px; margin: 20px 0 6px; color: var(--ink-2); }
p { margin: 0 0 12px; max-width: 74ch; }
.question { font-size: 16px; color: var(--ink-2); max-width: 74ch; margin: 0 0 14px; }
.meta { font-size: 12.5px; color: var(--muted); font-variant-numeric: tabular-nums; }
.meta code { background: none; color: var(--ink-2); padding: 0; }
.section-summary { color: var(--ink-2); font-size: 14px; margin: -2px 0 16px; max-width: 74ch; }
ul, ol { max-width: 74ch; padding-left: 22px; margin: 0 0 14px; }
li { margin-bottom: 5px; }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
code { background: var(--code-bg); padding: 1px 5px; border-radius: 4px; font-size: 12.5px;
       overflow-wrap: anywhere; }
pre { background: var(--code-bg); padding: 12px 14px; border-radius: 8px; overflow-x: auto;
      font-size: 12.5px; line-height: 1.55; margin: 0 0 14px; }
pre code { background: none; padding: 0; }
.table-wrap { overflow-x: auto; margin: 0 0 8px; border: 1px solid var(--line);
              border-radius: 10px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--line);
         white-space: nowrap; font-variant-numeric: tabular-nums; }
th { color: var(--ink-2); font-weight: 600; font-size: 12px; text-transform: uppercase;
     letter-spacing: 0.04em; background: var(--page); position: sticky; top: 0; }
tbody tr:last-child td { border-bottom: none; }
tbody tr:hover td { background: color-mix(in srgb, var(--accent) 7%, transparent); }
tr.emphasis td { font-weight: 600; }
td.right, th.right { text-align: right; }
.caption { font-size: 12.5px; color: var(--ink-2); margin: 10px 0 4px; }
.note { font-size: 12.5px; color: var(--muted); margin: 6px 0 18px; max-width: 78ch; }
.callout { border-left: 3px solid var(--muted); background: var(--surface);
           padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 0 0 14px; max-width: 80ch; }
.callout .callout-title { font-weight: 600; font-size: 13.5px; margin-bottom: 3px; }
.callout p { margin: 0; font-size: 14px; color: var(--ink-2); }
.callout.good { border-left-color: var(--good); }
.callout.info { border-left-color: var(--accent); }
.callout.warn { border-left-color: var(--warn); }
.callout.bad { border-left-color: var(--bad); }
.kv { display: grid; grid-template-columns: minmax(140px, 220px) 1fr; gap: 2px 18px;
      font-size: 13.5px; margin: 0 0 16px; }
.kv dt { color: var(--ink-2); }
.kv dd { margin: 0; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
figure { margin: 6px 0 20px; }
figure .viz { display: block; background: var(--viz-surface); border: 1px solid var(--line);
              border-radius: 10px; padding: 8px 4px; }
figcaption { font-size: 12.5px; color: var(--ink-2); margin-top: 8px; max-width: 78ch; }
nav.toc { background: var(--surface); border: 1px solid var(--line); border-radius: 10px;
          padding: 14px 18px; margin: 22px 0 8px; font-size: 13.5px; }
nav.toc ul { list-style: none; padding-left: 0; margin: 0; columns: 2; column-gap: 32px; }
nav.toc li { margin-bottom: 4px; break-inside: avoid; }
nav.toc a { color: var(--ink-2); text-decoration: none; }
nav.toc a:hover { color: var(--accent); text-decoration: underline; }
footer { margin-top: 56px; padding-top: 18px; border-top: 1px solid var(--line);
         font-size: 12.5px; color: var(--muted); }
@media (max-width: 720px) {
  nav.toc ul { columns: 1; }
  .kv { grid-template-columns: 1fr; }
  .kv dt { margin-top: 8px; }
}
@media print {
  body { background: #fff; }
  h2 { break-before: auto; }
  figure, .table-wrap, .callout { break-inside: avoid; }
}
"""


def _esc(text: Any) -> str:
    return html_module.escape(str(text), quote=True)


def _render_table(table: Table) -> str:
    parts = []
    if table.caption:
        parts.append(f'<p class="caption">{_esc(table.caption)}</p>')
    parts.append('<div class="table-wrap"><table><thead><tr>')
    for index, header in enumerate(table.headers):
        align = table.aligns[index] if index < len(table.aligns) else "left"
        css = ' class="right"' if align == "right" else ""
        parts.append(f"<th{css}>{_esc(header)}</th>")
    parts.append("</tr></thead><tbody>")
    for row_index, row in enumerate(table.rows):
        css = ' class="emphasis"' if row_index in table.emphasise else ""
        parts.append(f"<tr{css}>")
        for index, cell in enumerate(row):
            align = table.aligns[index] if index < len(table.aligns) else "left"
            cell_css = ' class="right"' if align == "right" else ""
            parts.append(f"<td{cell_css}>{_esc(cell)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    if table.note:
        parts.append(f'<p class="note">{_esc(table.note)}</p>')
    return "".join(parts)


def _render_block(block: Block) -> str:
    if isinstance(block, Paragraph):
        return f"<p>{_esc(block.text)}</p>"
    if isinstance(block, Bullets):
        tag = "ol" if block.ordered else "ul"
        items = "".join(f"<li>{_esc(item)}</li>" for item in block.items)
        return f"<{tag}>{items}</{tag}>"
    if isinstance(block, KeyValues):
        parts = []
        if block.caption:
            parts.append(f'<p class="caption">{_esc(block.caption)}</p>')
        parts.append('<dl class="kv">')
        for key, value in block.pairs:
            parts.append(f"<dt>{_esc(key)}</dt><dd>{_esc(value)}</dd>")
        parts.append("</dl>")
        return "".join(parts)
    if isinstance(block, Table):
        return _render_table(block)
    if isinstance(block, Callout):
        text = f"<p>{_esc(block.text)}</p>" if block.text else ""
        return (
            f'<div class="callout {_esc(block.level)}">'
            f'<div class="callout-title">{_esc(block.title)}</div>{text}</div>'
        )
    if isinstance(block, Code):
        caption = f'<p class="caption">{_esc(block.caption)}</p>' if block.caption else ""
        return f"{caption}<pre><code>{_esc(block.text)}</code></pre>"
    if isinstance(block, Chart):
        svg = plots.render(block.kind, block.data)
        if not svg:
            return ""
        caption = f"<figcaption>{_esc(block.caption)}</figcaption>" if block.caption else ""
        return f"<figure>{svg}{caption}</figure>"
    return ""


def render(document: Document, *, standalone: bool = True) -> str:
    """Render a document to a single self-contained HTML file."""
    metadata = document.metadata
    generated = time.strftime("%Y-%m-%d %H:%M:%S UTC",
                              time.gmtime(metadata.get("generated_at", time.time())))

    body: list[str] = ['<div class="wrap">', '<header class="masthead">']
    body.append(f"<h1>{_esc(document.title)}</h1>")
    if document.subtitle:
        body.append(f'<p class="question">{_esc(document.subtitle)}</p>')
    body.append(
        '<p class="meta">'
        f"<code>{_esc(metadata.get('experiment_id', ''))}</code> &middot; "
        f"{_esc(metadata.get('n_runs', 0))} runs &middot; primary metric "
        f"<code>{_esc(metadata.get('primary_metric', ''))}</code> &middot; "
        f"alpha {_esc(metadata.get('alpha', ''))} with "
        f"{_esc(metadata.get('correction', ''))} correction &middot; "
        f"generated {_esc(generated)}</p>"
    )
    body.append(
        f'<p class="meta">specification digest '
        f"<code>{_esc(metadata.get('spec_digest', ''))}</code></p>"
    )
    body.append("</header>")

    body.append('<nav class="toc"><ul>')
    for depth, section in document.walk():
        if depth > 1:
            continue
        body.append(f'<li><a href="#{_esc(section.anchor)}">{_esc(section.title)}</a></li>')
    body.append("</ul></nav>")

    for depth, section in document.walk():
        if section.is_empty and not section.summary:
            continue
        level = min(6, depth + 1)
        body.append(f'<h{level} id="{_esc(section.anchor)}">{_esc(section.title)}</h{level}>')
        if section.summary and depth == 1:
            body.append(f'<p class="section-summary">{_esc(section.summary)}</p>')
        for block in section.blocks:
            rendered = _render_block(block)
            if rendered:
                body.append(rendered)

    body.append(
        "<footer>Generated by Experiment Fabric. Every number in this report is "
        "addressable back to the run, seed, code, data, environment and hardware "
        "that produced it: "
        f"<code>fabric why {_esc(metadata.get('experiment_id', ''))} "
        f"--metric {_esc(metadata.get('primary_metric', ''))}</code></footer>"
    )
    body.append("</div>")
    content = "\n".join(body)

    if not standalone:
        return content
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_esc(document.title)} — Experiment Fabric</title>\n"
        f"<style>{STYLESHEET}</style>\n</head>\n<body>\n{content}\n</body>\n</html>\n"
    )
