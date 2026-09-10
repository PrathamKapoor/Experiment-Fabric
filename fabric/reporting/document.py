"""A format-independent report document.

The report is built once as a tree of typed blocks, then rendered to
Markdown, HTML, LaTeX or JSON.  Building it once is what keeps the four
formats saying the same thing: there is no path by which the HTML report
can claim something the LaTeX one does not.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Block:
    """Base class for report content."""

    def to_dict(self) -> dict[str, Any]:
        return {"type": type(self).__name__.lower()}


@dataclass
class Paragraph(Block):
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"type": "paragraph", "text": self.text}


@dataclass
class Bullets(Block):
    items: list[str] = field(default_factory=list)
    ordered: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"type": "bullets", "items": self.items, "ordered": self.ordered}


@dataclass
class KeyValues(Block):
    """A definition list: fact on the left, value on the right."""

    pairs: list[tuple[str, str]] = field(default_factory=list)
    caption: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"type": "keyvalues", "pairs": [list(p) for p in self.pairs],
                "caption": self.caption}


@dataclass
class Table(Block):
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    caption: str = ""
    aligns: list[str] = field(default_factory=list)
    note: str = ""
    emphasise: list[int] = field(default_factory=list)
    """Row indices to highlight (the treatment row, usually)."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "table", "headers": self.headers, "rows": self.rows,
            "caption": self.caption, "aligns": self.aligns, "note": self.note,
            "emphasise": self.emphasise,
        }


@dataclass
class Callout(Block):
    """A boxed note: ``info``, ``good``, ``warn`` or ``bad``."""

    level: str
    title: str
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"type": "callout", "level": self.level, "title": self.title,
                "text": self.text}


@dataclass
class Code(Block):
    text: str
    language: str = ""
    caption: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"type": "code", "text": self.text, "language": self.language,
                "caption": self.caption}


@dataclass
class Chart(Block):
    """A chart described by data, rendered by whichever format supports it."""

    kind: str
    data: dict[str, Any] = field(default_factory=dict)
    caption: str = ""
    alt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"type": "chart", "kind": self.kind, "data": self.data,
                "caption": self.caption, "alt": self.alt}


@dataclass
class Section:
    """A titled group of blocks, optionally with subsections."""

    title: str
    blocks: list[Block] = field(default_factory=list)
    subsections: list[Section] = field(default_factory=list)
    anchor: str = ""
    summary: str = ""

    def add(self, block: Block) -> Section:
        self.blocks.append(block)
        return self

    def paragraph(self, text: str) -> Section:
        return self.add(Paragraph(text))

    def bullets(self, items: Iterable[str], ordered: bool = False) -> Section:
        items = [i for i in items if i]
        if items:
            self.add(Bullets(list(items), ordered))
        return self

    def table(self, headers: Sequence[str], rows: Sequence[Sequence[Any]], *,
              caption: str = "", aligns: Sequence[str] = (), note: str = "",
              emphasise: Sequence[int] = ()) -> Section:
        self.add(Table(
            headers=[str(h) for h in headers],
            rows=[[str(c) for c in row] for row in rows],
            caption=caption,
            aligns=list(aligns) or ["left"] + ["right"] * (len(headers) - 1),
            note=note,
            emphasise=list(emphasise),
        ))
        return self

    def subsection(self, title: str, anchor: str = "") -> Section:
        section = Section(title, anchor=anchor or _slug(title))
        self.subsections.append(section)
        return section

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "anchor": self.anchor or _slug(self.title),
            "summary": self.summary,
            "blocks": [b.to_dict() for b in self.blocks],
            "subsections": [s.to_dict() for s in self.subsections],
        }

    @property
    def is_empty(self) -> bool:
        return not self.blocks and not any(not s.is_empty for s in self.subsections)


@dataclass
class Document:
    """A complete report."""

    title: str
    subtitle: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    sections: list[Section] = field(default_factory=list)

    def section(self, title: str, anchor: str = "") -> Section:
        section = Section(title, anchor=anchor or _slug(title))
        self.sections.append(section)
        return section

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "subtitle": self.subtitle,
            "metadata": self.metadata,
            "sections": [s.to_dict() for s in self.sections],
        }

    def walk(self) -> Iterable[tuple[int, Section]]:
        """Yield ``(depth, section)`` for every section, depth-first."""

        def visit(section: Section, depth: int) -> Iterable[tuple[int, Section]]:
            yield depth, section
            for child in section.subsections:
                yield from visit(child, depth + 1)

        for section in self.sections:
            yield from visit(section, 1)


def _slug(text: str) -> str:
    out = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in " -_/":
            out.append("-")
    slug = "".join(out)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "section"
