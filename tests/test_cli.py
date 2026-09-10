"""The command line, exercised the way a user would.

Every command is run against a real workspace in a temporary directory.
These are slower than the unit tests and worth it: the CLI is the surface
almost everyone touches, and a broken argument name is invisible to every
other test in the suite.
"""

from __future__ import annotations

import json

import pytest

from fabric.cli.main import main

SPEC = """
experiment:
  name: cli-demo
research_question: Does the command line work?
algorithm:
  name: method
  entrypoint: python:tests.helpers:ablatable
components: [alpha, beta]
metrics:
  - name: accuracy
    tolerance: 0.02
baselines:
  - name: control
    algorithm:
      name: control
      entrypoint: python:tests.helpers:deterministic
ablations: [remove_alpha, remove_beta]
seeds:
  values: [1, 2, 3, 4, 5, 6, 7, 8]
reproducibility:
  enabled: true
  determinism_replicates: 2
  perturbations:
    configuration:
      values:
        - label: shifted
          scale: 3.0
execution:
  isolation: inline
  max_workers: 2
"""


@pytest.fixture()
def project(tmp_path, monkeypatch):
    """A workspace with the specification written into it."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FABRIC_NO_COLOR", "1")
    (tmp_path / "experiment.yaml").write_text(SPEC, encoding="utf-8")
    assert main(["init", "--quiet"]) == 0
    return tmp_path


def run_cli(*args: str) -> int:
    return main(list(args))


# --------------------------------------------------------------------------
# lifecycle
# --------------------------------------------------------------------------


def test_init_creates_a_workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert run_cli("init") == 0
    assert (tmp_path / ".fabric" / "config.json").is_file()


def test_commands_outside_a_workspace_explain_themselves(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert run_cli("ls") == 2
    assert "fabric init" in capsys.readouterr().err


def test_plan_reports_the_expansion_without_running(project, capsys):
    assert run_cli("plan", "experiment.yaml") == 0
    output = capsys.readouterr().out
    assert "conditions" in output
    assert "nothing was executed" in output
    assert list((project / ".fabric" / "experiments").glob("EXP-*/runs/*")) == []


def test_run_analyze_report_certify_export_verify(project, capsys):
    assert run_cli("run", "experiment.yaml") == 0
    output = capsys.readouterr().out
    assert "completed" in output

    experiments = json.loads(
        (project / ".fabric" / "experiments" / "index.json").read_text(encoding="utf-8"))
    experiment_id = experiments["experiments"][0]["id"]

    assert run_cli("status", experiment_id) == 0
    assert "complete" in capsys.readouterr().out

    assert run_cli("analyze", experiment_id) == 0
    analysis_output = capsys.readouterr().out
    assert "baseline comparisons" in analysis_output
    assert "ablations" in analysis_output
    assert "reproducibility" in analysis_output
    assert "limitations" in analysis_output

    assert run_cli("report", experiment_id, "--format", "all") == 0
    capsys.readouterr()
    reports = project / ".fabric" / "experiments" / experiment_id / "reports"
    for name in ("report.md", "report.html", "report.tex", "report.json"):
        assert (reports / name).is_file(), name
    assert (reports / "tables").is_dir()

    assert run_cli("certify", experiment_id) == 0
    certificate_output = capsys.readouterr().out
    assert "EXPERIMENT CERTIFICATE" in certificate_output
    assert "VERDICT" in certificate_output

    bundle = project / "bundle"
    assert run_cli("export", experiment_id, "--out", str(bundle)) == 0
    capsys.readouterr()
    assert (bundle / "manifest.json").is_file()
    assert (bundle / "reproduce.sh").is_file()

    assert run_cli("verify", str(bundle)) == 0
    assert "INTACT" in capsys.readouterr().out

    (bundle / "README.md").write_text("edited", encoding="utf-8")
    assert run_cli("verify", str(bundle)) == 1
    assert "MODIFIED" in capsys.readouterr().out


def test_run_is_idempotent(project, capsys):
    assert run_cli("run", "experiment.yaml") == 0
    capsys.readouterr()
    assert run_cli("run", "experiment.yaml") == 0
    assert "already complete" in capsys.readouterr().out


def test_why_traces_a_number(project, capsys):
    assert run_cli("run", "experiment.yaml") == 0
    capsys.readouterr()
    experiments = json.loads(
        (project / ".fabric" / "experiments" / "index.json").read_text(encoding="utf-8"))
    experiment_id = experiments["experiments"][0]["id"]

    assert run_cli("why", experiment_id, "--metric", "accuracy", "--seed", "3") == 0
    output = capsys.readouterr().out
    assert "accuracy" in output
    for expected in ("run", "condition", "algorithm", "environment", "hardware"):
        assert expected in output

    assert run_cli("why", experiment_id, "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["chain"] and payload["metric"] == "accuracy"


def test_graph_exports_every_format(project, capsys):
    assert run_cli("run", "experiment.yaml") == 0
    capsys.readouterr()
    experiments = json.loads(
        (project / ".fabric" / "experiments" / "index.json").read_text(encoding="utf-8"))
    experiment_id = experiments["experiments"][0]["id"]

    for fmt, marker in (("mermaid", "flowchart"), ("dot", "digraph"), ("json", "nodes")):
        assert run_cli("graph", experiment_id, "--format", fmt) == 0
        assert marker in capsys.readouterr().out


def test_reproduce_compares_against_the_original(project, capsys):
    assert run_cli("run", "experiment.yaml") == 0
    capsys.readouterr()
    experiments = json.loads(
        (project / ".fabric" / "experiments" / "index.json").read_text(encoding="utf-8"))
    experiment_id = experiments["experiments"][0]["id"]

    assert run_cli("reproduce", experiment_id) == 0
    output = capsys.readouterr().out
    assert "REPRODUCTION COMPLETE" in output
    assert "PASS" in output


def test_analyze_json_is_machine_readable(project, capsys):
    assert run_cli("run", "experiment.yaml") == 0
    capsys.readouterr()
    experiments = json.loads(
        (project / ".fabric" / "experiments" / "index.json").read_text(encoding="utf-8"))
    experiment_id = experiments["experiments"][0]["id"]

    assert run_cli("analyze", experiment_id, "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["primary_metric"] == "accuracy"
    assert payload["baselines"]["comparisons"]
    assert payload["reproducibility"]["axes"]
    assert payload["claims"]


# --------------------------------------------------------------------------
# registry and diagnostics
# --------------------------------------------------------------------------


def test_register_and_list_a_dataset(project, capsys):
    data = project / "data"
    data.mkdir()
    (data / "train.jsonl").write_text('{"x": 1}\n', encoding="utf-8")

    assert run_cli("register", "dataset", "corpus", "--version", "v1",
                   "--path", str(data)) == 0
    assert "digest" in capsys.readouterr().out

    assert run_cli("registry", "ls") == 0
    assert "corpus@v1" in capsys.readouterr().out

    assert run_cli("registry", "verify") == 0
    assert "hashes to its own identifier" in capsys.readouterr().out


def test_registering_a_changed_version_is_refused(project, capsys):
    data = project / "data"
    data.mkdir()
    (data / "a.txt").write_text("one", encoding="utf-8")
    assert run_cli("register", "dataset", "corpus", "--path", str(data)) == 0
    capsys.readouterr()
    (data / "a.txt").write_text("two", encoding="utf-8")
    assert run_cli("register", "dataset", "corpus", "--path", str(data)) == 2
    assert "already registered" in capsys.readouterr().err


def test_doctor_reports_the_environment(project, capsys):
    assert run_cli("doctor") == 0
    output = capsys.readouterr().out
    assert "environment" in output and "hardware" in output
    assert "fingerprint" in output


def test_cache_stats_and_clear(project, capsys):
    assert run_cli("run", "experiment.yaml") == 0
    capsys.readouterr()
    assert run_cli("cache", "stats") == 0
    assert "entries" in capsys.readouterr().out
    assert run_cli("cache", "clear") == 0
    assert "cleared" in capsys.readouterr().out


# --------------------------------------------------------------------------
# error handling
# --------------------------------------------------------------------------


def test_a_missing_spec_is_a_clean_error(project, capsys):
    assert run_cli("run", "nope.yaml") == 2
    assert "no such specification" in capsys.readouterr().err


def test_an_invalid_spec_names_the_problem(project, capsys):
    (project / "bad.yaml").write_text(
        "experiment:\n  name: x\nmetrics: [accuracy]\n", encoding="utf-8")
    assert run_cli("plan", "bad.yaml") == 2
    assert "research question" in capsys.readouterr().err


def test_an_unknown_experiment_suggests_ls(project, capsys):
    assert run_cli("analyze", "EXP-nothing") == 2
    assert "fabric ls" in capsys.readouterr().err


def test_a_broken_entrypoint_is_caught_before_anything_runs(project, capsys):
    (project / "broken.yaml").write_text(SPEC.replace(
        "python:tests.helpers:ablatable", "python:no.such.module:fn"), encoding="utf-8")
    assert run_cli("plan", "broken.yaml") == 2
    assert "unusable entrypoint" in capsys.readouterr().err
