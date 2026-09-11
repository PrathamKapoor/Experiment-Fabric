"""The bundled YAML parser must agree with PyYAML.

Fabric ships its own parser so the core needs no dependencies, and uses
PyYAML automatically when it is installed. That means a specification can
be parsed by either one, and the two must not disagree -- a spec that
means different things depending on what happens to be installed would
undermine the identity digest that everything else rests on.

These tests are skipped when PyYAML is absent. CI installs it in one leg
of the matrix precisely so they run.
"""

from __future__ import annotations

import glob
import os

import pytest

import fabric.util.yamlio as yamlio

pyyaml = pytest.importorskip("yaml", reason="PyYAML is not installed")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture()
def bundled(monkeypatch):
    """Force the bundled parser even when PyYAML is importable."""
    monkeypatch.setattr(yamlio, "_pyyaml", None)
    return yamlio.loads_yaml


def specification_files() -> list[str]:
    files = sorted(glob.glob(os.path.join(ROOT, "examples", "**", "*.yaml"),
                             recursive=True))
    study = os.path.join(ROOT, "paper", "study.yaml")
    if os.path.isfile(study):
        files.append(study)
    return files


@pytest.mark.parametrize("path", specification_files(),
                         ids=lambda p: os.path.relpath(p, ROOT).replace(os.sep, "/"))
def test_every_shipped_specification_parses_identically(path, bundled):
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    assert bundled(text) == pyyaml.safe_load(text)


# --------------------------------------------------------------------------
# the YAML 1.1 scientific-notation trap
# --------------------------------------------------------------------------

# PyYAML implements YAML 1.1, which requires a decimal point in the mantissa
# and a sign in the exponent before it will call something a float. So it
# reads these as strings while the bundled parser reads numbers. Fabric
# canonicalises both, so a specification means one thing either way.
CANONICALISED = [
    ("lr: 1e-4\n", {"lr": 1e-4}),
    ("lr: 5E-5\n", {"lr": 5e-5}),
    ("lr: 1.0e5\n", {"lr": 1.0e5}),
    ("lr: 2.5E+3\n", {"lr": 2.5e3}),
    ("values: [1e-4, 5e-5]\n", {"values": [1e-4, 5e-5]}),
    ("nested:\n  a:\n    b: 3e8\n", {"nested": {"a": {"b": 3e8}}}),
    # ...but something that merely looks numeric stays a string.
    ("version: v1\n", {"version": "v1"}),
    ("name: 1e-4-model\n", {"name": "1e-4-model"}),
    ("commit: 1e4f2a\n", {"commit": "1e4f2a"}),
]


@pytest.mark.parametrize("text,expected", CANONICALISED, ids=range(len(CANONICALISED)))
def test_scientific_notation_means_the_same_to_both_parsers(text, expected, bundled):
    assert bundled(text) == expected


@pytest.mark.parametrize("text,expected", CANONICALISED, ids=range(len(CANONICALISED)))
def test_scientific_notation_is_canonicalised_on_the_pyyaml_path(text, expected):
    assert yamlio.loads_yaml(text) == expected


SPEC_WITH_SCIENTIFIC_NOTATION = "\n".join([
    "experiment: {name: d}",
    "research_question: q",
    "algorithm: {name: a, entrypoint: 'python:m:f'}",
    "metrics: [accuracy]",
    "configuration:",
    "  learning_rate: [1e-4, 5e-5]",
    "seeds: {values: [1, 2, 3]}",
    "",
])


def test_the_experiment_identity_survives_installing_pyyaml(monkeypatch):
    """The bug this guards against.

    PyYAML reads ``1e-4`` as a string and the bundled parser as a float.
    The specification digest is computed from the parsed document, so
    without canonicalisation the same file would have two different
    experiment identities depending on whether PyYAML happened to be
    installed -- silently breaking caching, resumption and cross-machine
    comparison, which is the exact failure class this project exists to
    catch.
    """
    from fabric.core.spec import ExperimentSpec

    with_pyyaml = ExperimentSpec.from_yaml(SPEC_WITH_SCIENTIFIC_NOTATION)
    monkeypatch.setattr(yamlio, "_pyyaml", None)
    without_pyyaml = ExperimentSpec.from_yaml(SPEC_WITH_SCIENTIFIC_NOTATION)

    assert with_pyyaml.parameters[0].values == (1e-4, 5e-5)
    assert without_pyyaml.parameters[0].values == (1e-4, 5e-5)
    assert with_pyyaml.digest == without_pyyaml.digest


# --------------------------------------------------------------------------
# the rest of the supported subset
# --------------------------------------------------------------------------


DOCUMENTS = [
    "a: 1\nb: two\n",
    "list: [1, 2.5, true, null]\n",
    "nested:\n  - name: x\n    value: {a: 1, b: 2}\n  - name: y\n",
    "block: |\n  one\n  two\n",
    'quoted: "colon: inside"\n',
    'seq:\n  - "quoted: colon"\n  - plain\n',
    "matrix:\n  - [1, 2]\n  - [3, 4]\n",
    "empty_map: {}\nempty_list: []\n",
    "# only a comment\nvalue: 1  # trailing\n",
    "negatives: [-1, -2.5]\n",
    "bools: [true, false]\n",
    "nulls: [null, ~]\n",
    "deep:\n  a:\n    b:\n      c: 1\n",
    "decimals: [.5, 1.5, 0.25]\n",
]


@pytest.mark.parametrize("text", DOCUMENTS, ids=range(len(DOCUMENTS)))
def test_the_supported_subset_agrees_with_pyyaml(text, bundled):
    assert bundled(text) == pyyaml.safe_load(text)


# --------------------------------------------------------------------------
# the emitter
# --------------------------------------------------------------------------


def test_mapping_keys_are_always_strings(bundled):
    """A key is a name, not a value, so neither parser may coerce it.

    A key spelled with a YAML keyword is normalised to its canonical
    spelling on both paths -- documented, and the reason not to name a
    parameter `none` or `on`.
    """
    text = "\n".join(["true: 1", "null: 2", "3: three", "plain: 4", ""])
    for parsed in (bundled(text), yamlio.loads_yaml(text)):
        assert set(parsed) == {"true", "null", "3", "plain"}
        assert all(isinstance(key, str) for key in parsed)


@pytest.mark.parametrize("value", [5e-05, 1e20, 0.1, 1.0, -2.5e-8, 1234.5, 0.0])
def test_emitted_floats_read_back_as_floats(value, bundled):
    """``repr(5e-05)`` is ``5e-05``, which YAML 1.1 would call a string."""
    emitted = yamlio.dump_yaml({"x": value})
    assert pyyaml.safe_load(emitted)["x"] == pytest.approx(value)
    assert bundled(emitted)["x"] == pytest.approx(value)


def test_the_emitter_round_trips_through_pyyaml(bundled):
    """What Fabric writes, PyYAML must read back unchanged."""
    value = {
        "experiment": {"name": "demo", "tags": ["a", "b"]},
        "configuration": {"lr": [0.1, 0.01], "small": 5e-05, "fixed": 3,
                          "flag": True, "missing": None},
        "nested": [{"name": "x", "params": {"c": 1.5}}, {"name": "y"}],
        "text": "a string: with a colon",
    }
    emitted = yamlio.dump_yaml(value)
    assert pyyaml.safe_load(emitted) == value
    assert bundled(emitted) == value
