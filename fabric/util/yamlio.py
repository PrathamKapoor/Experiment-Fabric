"""YAML I/O with a bundled fallback parser.

Fabric specifications are YAML because researchers already write YAML.
But Fabric's core has no third-party dependencies, so when PyYAML is not
installed we fall back to the parser below, which implements the subset
of YAML that experiment specifications actually use:

* block mappings and block sequences, nested by indentation
* flow sequences ``[a, b]`` and flow mappings ``{a: 1}``
* single/double quoted scalars, literal ``|`` and folded ``>`` blocks
* ``#`` comments, ``---`` document markers, ``null``/``~``/booleans
* automatic int/float typing

Anchors, aliases, tags, and multiple documents are *not* supported: if a
spec needs them, install PyYAML and Fabric will use it automatically.
"""

from __future__ import annotations

import os
import re
from typing import Any

try:  # pragma: no cover - depends on the host environment
    import yaml as _pyyaml
except ImportError:  # pragma: no cover
    _pyyaml = None


class YamlError(ValueError):
    """Raised when a YAML document cannot be parsed."""

    def __init__(self, message: str, line: int | None = None) -> None:
        self.line = line
        super().__init__(f"line {line}: {message}" if line is not None else message)


# --------------------------------------------------------------------------
# scalar typing
# --------------------------------------------------------------------------

_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?$")
_NULLS = {"", "null", "~", "none"}
_TRUE = {"true", "yes", "on"}
_FALSE = {"false", "no", "off"}


def _scalar(text: str) -> Any:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        body = text[1:-1]
        if text[0] == '"':
            return body.encode("utf-8").decode("unicode_escape")
        return body.replace("''", "'")
    lowered = text.lower()
    if lowered in _NULLS:
        return None
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    if _INT_RE.match(text):
        return int(text)
    if _FLOAT_RE.match(text):
        return float(text)
    return text


def _strip_comment(line: str) -> str:
    """Remove a trailing ``#`` comment that is not inside quotes."""
    quote: str | None = None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1] in " \t"):
            return line[:i]
    return line


def _split_top_level(text: str, sep: str = ",") -> list[str]:
    """Split ``text`` on ``sep`` ignoring separators nested in quotes/brackets."""
    parts: list[str] = []
    depth = 0
    quote: str | None = None
    current: list[str] = []
    for ch in text:
        if quote:
            current.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
            current.append(ch)
        elif ch in "[{":
            depth += 1
            current.append(ch)
        elif ch in "]}":
            depth -= 1
            current.append(ch)
        elif ch == sep and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    tail = "".join(current)
    if tail.strip() or parts:
        parts.append(tail)
    return parts


def _parse_flow(text: str, line_no: int) -> Any:
    """Parse a flow collection. Keys stay raw; _canonical_key normalises them."""
    text = text.strip()
    if text.startswith("["):
        if not text.endswith("]"):
            raise YamlError("unterminated flow sequence", line_no)
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_parse_value(p.strip(), line_no) for p in _split_top_level(inner)]
    if text.startswith("{"):
        if not text.endswith("}"):
            raise YamlError("unterminated flow mapping", line_no)
        inner = text[1:-1].strip()
        if not inner:
            return {}
        out: dict[str, Any] = {}
        for pair in _split_top_level(inner):
            if ":" not in pair:
                raise YamlError("flow mapping entry has no colon: " + repr(pair), line_no)
            key, _, val = pair.partition(":")
            out[_scalar(key)] = _parse_value(val.strip(), line_no)
        return out
    raise YamlError("not a flow collection: " + repr(text), line_no)


def _parse_value(text: str, line_no: int) -> Any:
    text = text.strip()
    if text.startswith("[") or text.startswith("{"):
        return _parse_flow(text, line_no)
    return _scalar(text)


# --------------------------------------------------------------------------
# block parser
# --------------------------------------------------------------------------

class _Reader:
    """A cursor over physical lines with indentation awareness."""

    def __init__(self, text: str) -> None:
        self.raw = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        self.pos = 0

    def _significant(self, index: int) -> bool:
        line = self.raw[index]
        lead = line[: len(line) - len(line.lstrip())]
        if "\t" in lead:
            raise YamlError("tabs are not allowed for indentation", index + 1)
        stripped = _strip_comment(line).strip()
        return bool(stripped) and stripped != "---" and stripped != "..."

    def skip_blank(self) -> None:
        while self.pos < len(self.raw) and not self._significant(self.pos):
            self.pos += 1

    def peek(self) -> tuple[int, str, int] | None:
        """Return ``(indent, content, line_number)`` of the next real line."""
        self.skip_blank()
        if self.pos >= len(self.raw):
            return None
        line = _strip_comment(self.raw[self.pos]).rstrip()
        indent = len(line) - len(line.lstrip(" "))
        return indent, line.strip(), self.pos + 1

    def advance(self) -> None:
        self.pos += 1

    def replace_current(self, new_line: str) -> None:
        self.raw[self.pos] = new_line


def _read_block_scalar(reader: _Reader, parent_indent: int, style: str) -> str:
    lines: list[str] = []
    block_indent: int | None = None
    while reader.pos < len(reader.raw):
        raw = reader.raw[reader.pos]
        if not raw.strip():
            lines.append("")
            reader.advance()
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        if indent <= parent_indent:
            break
        if block_indent is None:
            block_indent = indent
        lines.append(raw[block_indent:].rstrip())
        reader.advance()
    while lines and not lines[-1]:
        lines.pop()
    if style == "|":
        return "\n".join(lines) + "\n"
    return " ".join(line.strip() for line in lines if line.strip()) + "\n"


_KEY_RE = re.compile(r"^(?P<key>(?:\"[^\"]*\"|'[^']*'|[^:#]+?))\s*:(?:\s+(?P<value>.*))?$")


def _match_key(content: str) -> re.Match[str] | None:
    """Match ``key: value``, without mistaking a quoted scalar for a key.

    ``- "quoted: colon"`` is a string, not a mapping. The plain-scalar
    branch of the key pattern would happily match ``"quoted`` as a key, so
    a line that opens with a quote is only a key when the *quoted* branch
    matches all the way to the colon.
    """
    if content[:1] in ("\"", "'"):
        quote = content[0]
        closing = content.find(quote, 1)
        if closing == -1:
            return None
        remainder = content[closing + 1:].lstrip()
        if not remainder.startswith(":"):
            return None
    return _KEY_RE.match(content)


def _parse_node(reader: _Reader, indent: int) -> Any:
    head = reader.peek()
    if head is None:
        return None
    if head[0] < indent:
        return None
    if head[1].startswith("- ") or head[1] == "-":
        return _parse_sequence(reader, head[0])
    return _parse_mapping(reader, head[0])


def _parse_sequence(reader: _Reader, indent: int) -> list[Any]:
    items: list[Any] = []
    while True:
        head = reader.peek()
        if head is None:
            break
        cur_indent, content, line_no = head
        is_item = content.startswith("- ") or content == "-"
        if cur_indent != indent or not is_item:
            if cur_indent > indent:
                raise YamlError("unexpected indentation inside sequence", line_no)
            break
        body = content[1:].strip()
        if not body:
            reader.advance()
            nxt = reader.peek()
            if nxt is not None and nxt[0] > indent:
                items.append(_parse_node(reader, nxt[0]))
            else:
                items.append(None)
            continue
        starts_collection = body.startswith("[") or body.startswith("{")
        if _match_key(body) and not starts_collection:
            # "- key: value" starts a mapping whose columns line up two
            # characters right of the dash; rewrite and re-read as a mapping.
            reader.replace_current(" " * (indent + 2) + body)
            items.append(_parse_mapping(reader, indent + 2))
            continue
        reader.advance()
        items.append(_parse_value(body, line_no))
    return items


def _parse_mapping(reader: _Reader, indent: int) -> dict[Any, Any]:
    # Keys are left as parsed scalars; _canonical_key turns them into strings
    # once, on both parser paths, so the two cannot disagree.
    mapping: dict[Any, Any] = {}
    while True:
        head = reader.peek()
        if head is None:
            break
        cur_indent, content, line_no = head
        if cur_indent < indent:
            break
        if cur_indent > indent:
            raise YamlError("unexpected indentation in mapping", line_no)
        if content.startswith("- "):
            break
        match = _match_key(content)
        if not match:
            raise YamlError("expected a 'key: value' pair, got " + repr(content), line_no)
        key = _scalar(match.group("key"))
        value_text = (match.group("value") or "").strip()
        reader.advance()
        if value_text in ("|", ">", "|-", ">-"):
            mapping[key] = _read_block_scalar(reader, cur_indent, value_text[0])
            continue
        if value_text:
            mapping[key] = _parse_value(value_text, line_no)
            continue
        nxt = reader.peek()
        if nxt is None or nxt[0] < cur_indent:
            mapping[key] = None
            continue
        if nxt[0] == cur_indent:
            if nxt[1].startswith("- ") or nxt[1] == "-":
                mapping[key] = _parse_sequence(reader, cur_indent)
            else:
                mapping[key] = None
            continue
        mapping[key] = _parse_node(reader, nxt[0])
    return mapping


# YAML 1.1, which PyYAML implements, requires a decimal point in the
# mantissa *and* a sign in the exponent before it will call something a
# float. So `1e-4` and `1.0e5` are strings to PyYAML and numbers to almost
# everyone else, including the bundled parser.
#
# Left alone, that means `learning_rate: [1e-4, 5e-5]` parses to strings on
# a machine with PyYAML installed and to floats on one without -- and since
# the specification digest is computed from the parsed document, the same
# file would have two different experiment identities depending on what
# happened to be in the environment. That is precisely the class of bug
# this project exists to catch, so Fabric defines the canonical reading and
# applies it to both parsers rather than inheriting either one.
_SCIENTIFIC = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)[eE][+-]?\d+$")


_KEY_SPELLING = {True: "true", False: "false", None: "null"}


def _canonical_key(key: Any) -> str:
    """Mapping keys are always strings.

    Both parsers type-coerce bare scalars, so a key spelled ``true`` or
    ``null`` arrives as a Python object rather than text. In a
    configuration format a key is a name, so it is converted back to a
    string here on both paths.

    The residual caveat, which is not worth more machinery: a key spelled
    with a YAML boolean or null synonym (``none``, ``yes``, ``off``) is
    normalised to its canonical spelling, so ``none:`` and ``null:`` are
    the same key. Do not name a parameter after a YAML keyword.
    """
    if isinstance(key, str):
        return key
    if isinstance(key, bool) or key is None:
        return _KEY_SPELLING[key]
    return str(key)


def _canonicalise(value: Any) -> Any:
    """Apply Fabric's reading of a parsed document, whichever parser ran."""
    if isinstance(value, str):
        if _SCIENTIFIC.match(value.strip()):
            try:
                return float(value)
            except ValueError:  # pragma: no cover - the regex already guarantees this
                return value
        return value
    if isinstance(value, dict):
        return {_canonical_key(k): _canonicalise(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_canonicalise(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_canonicalise(v) for v in value)
    return value


def loads_yaml(text: str) -> Any:
    """Parse a YAML document from a string.

    Uses PyYAML when it is installed and the bundled parser otherwise, then
    canonicalises the result so the two cannot disagree. Parse failures raise
    :class:`YamlError` whichever parser ran, so callers need one error surface.
    """
    if _pyyaml is not None:
        try:
            return _canonicalise(_pyyaml.safe_load(text))
        except _pyyaml.YAMLError as exc:
            raise YamlError(str(exc)) from exc
    reader = _Reader(text)
    head = reader.peek()
    if head is None:
        return None
    return _canonicalise(_parse_node(reader, head[0]))


def load_yaml(path: str | os.PathLike[str]) -> Any:
    """Parse a YAML document from a file path."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    try:
        return loads_yaml(text)
    except YamlError as exc:
        raise YamlError(os.fspath(path) + ": " + str(exc)) from exc


# --------------------------------------------------------------------------
# emitter
# --------------------------------------------------------------------------

_SPECIAL_CHARS = ":#[]{}\n"


def _emit_float(value: float) -> str:
    """Write a float that every parser reads back as a float.

    ``repr(5e-05)`` is ``5e-05``, which YAML 1.1 treats as a string because
    the mantissa has no decimal point. Emitting ``5.0e-05`` instead keeps
    the round trip intact for PyYAML as well as the bundled parser.
    """
    if value != value or value in (float("inf"), float("-inf")):
        return f'"{value}"'
    text = repr(value)
    if "e" in text or "E" in text:
        mantissa, _, exponent = text.partition("e")
        if "." not in mantissa:
            mantissa += ".0"
        if exponent and exponent[0] not in "+-":
            exponent = "+" + exponent
        return f"{mantissa}e{exponent}"
    return text


def _emit_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return repr(value)
    if isinstance(value, float):
        return _emit_float(value)
    text = str(value)
    reserved = _NULLS | _TRUE | _FALSE
    needs_quote = (
        text == ""
        or text.lower() in reserved
        or bool(_INT_RE.match(text))
        or bool(_FLOAT_RE.match(text))
        or any(ch in text for ch in _SPECIAL_CHARS)
        or "'" in text
        or '"' in text
        or text != text.strip()
        or text.startswith("- ")
    )
    if not needs_quote:
        return text
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return '"' + escaped + '"'


def dump_yaml(value: Any, indent: int = 0) -> str:
    """Emit a YAML document for plain Python data.

    The emitter covers the same subset as the parser, which is enough to
    round-trip every specification Fabric produces.
    """
    pad = " " * indent
    if isinstance(value, dict):
        if not value:
            return pad + "{}\n"
        out: list[str] = []
        for key, val in value.items():
            if isinstance(val, (dict, list)) and val:
                out.append(pad + str(key) + ":\n" + dump_yaml(val, indent + 2))
            elif isinstance(val, dict):
                out.append(pad + str(key) + ": {}\n")
            elif isinstance(val, list):
                out.append(pad + str(key) + ": []\n")
            else:
                out.append(pad + str(key) + ": " + _emit_scalar(val) + "\n")
        return "".join(out)
    if isinstance(value, list):
        if not value:
            return pad + "[]\n"
        out = []
        for item in value:
            if isinstance(item, dict) and item:
                body = dump_yaml(item, indent + 2)
                out.append(pad + "-" + body[indent + 1:])
            elif isinstance(item, list) and item:
                out.append(pad + "-\n" + dump_yaml(item, indent + 2))
            else:
                out.append(pad + "- " + _emit_scalar(item) + "\n")
        return "".join(out)
    return pad + _emit_scalar(value) + "\n"
