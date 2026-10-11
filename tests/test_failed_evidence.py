"""A failed experiment must never be reported as a successful one.

Fabric's promise is that it certifies what the evidence supports. That
promise is void if an experiment in which *every* run failed still exits
zero, still carries a positive-looking certificate verdict, and still
reports its integrity checks as passing -- because ``all([])`` is ``True``.

These tests pin the distinction that matters, and they pin the cases that
must NOT be affected by it:

* an experiment that ran and produced inconclusive evidence is **not** an
  execution failure, and must keep succeeding;
* a partially failed experiment is not a total failure;
* diagnostic reports must still be producible, because they are how a
  researcher finds out what went wrong.

Everything runs against a real workspace in a temporary directory.
"""

from __future__ import annotations

import json

import pytest

from fabric.cli.main import main
from tests.evidence import analysis, integrity_findings, load_runs

FAILING = """
experiment:
  name: total-failure
research_question: What does Fabric report when nothing succeeds?
algorithm:
  name: broken
  entrypoint: python:tests.helpers:always_fails
metrics:
  - name: accuracy
    direction: maximize
    primary: true
seeds:
  values: [1, 2, 3]
execution:
  isolation: inline
"""

PARTIAL = """
experiment:
  name: partial-failure
research_question: Does a partially failed experiment still report honestly?
algorithm:
  name: mostly-broken
  entrypoint: python:tests.helpers:fails_above_threshold
metrics:
  - name: accuracy
    direction: maximize
    primary: true
configuration:
  scale: [1.0, 4.0]
seeds:
  values: [1, 2]
execution:
  isolation: inline
"""

INCONCLUSIVE = """
experiment:
  name: inconclusive-not-failed
research_question: Is an inconclusive result distinguishable from a failure?
algorithm:
  name: method
  entrypoint: python:tests.helpers:deterministic
metrics:
  - name: accuracy
    direction: maximize
    primary: true
    tolerance: 0.000001
baselines:
  - name: identical
    algorithm:
      name: identical
      entrypoint: python:tests.helpers:deterministic
seeds:
  values: [1, 2, 3, 4]
execution:
  isolation: inline
"""

WORKING = """
experiment:
  name: all-good
research_question: Does a working experiment still behave?
algorithm:
  name: method
  entrypoint: python:tests.helpers:ablatable
components: [alpha, beta]
metrics:
  - name: accuracy
    direction: maximize
    primary: true
    tolerance: 0.02
seeds:
  values: [1, 2, 3, 4, 5, 6]
execution:
  isolation: inline
"""


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FABRIC_NO_COLOR", "1")
    assert main(["init", "--quiet"]) == 0
    return tmp_path


def spec(workspace, text: str, filename: str = "experiment.yaml") -> str:
    (workspace / filename).write_text(text, encoding="utf-8")
    return filename


def experiment_id(workspace) -> str:
    index = json.loads((workspace / ".fabric" / "experiments" / "index.json")
                      .read_text(encoding="utf-8"))
    return index["experiments"][0]["id"]


def checks_by_name(workspace, experiment: str) -> dict[str, dict]:
    path = (workspace / ".fabric" / "experiments" / experiment
            / "certificate.json")
    return {c["name"]: c for c in json.loads(path.read_text(encoding="utf-8"))["checks"]}


# --------------------------------------------------------------------------
# the defect: a total failure reported as success
# --------------------------------------------------------------------------


def test_running_an_experiment_where_every_run_fails_is_nonzero(workspace):
    """``fabric run`` returning 0 tells a CI pipeline the experiment worked."""
    spec(workspace, FAILING)
    assert main(["run", "experiment.yaml"]) != 0


def test_no_metrics_are_invented_for_failed_runs(workspace):
    spec(workspace, FAILING)
    main(["run", "experiment.yaml"])
    runs = load_runs(workspace, experiment_id(workspace))
    assert runs, "the failed runs must still be recorded"
    assert all(r.status.value == "failed" for r in runs)
    assert all(not r.metrics for r in runs), "a failed run has no measurements"


def test_the_analysis_says_the_evidence_is_missing(workspace):
    spec(workspace, FAILING)
    main(["run", "experiment.yaml"])
    experiment = experiment_id(workspace)
    assert main(["analyze", experiment]) == 0
    report = analysis(workspace, experiment)
    assert report["headline"] == "No treatment condition produced results."
    assert report["claims"] == []


def test_certifying_an_experiment_with_no_successful_runs_fails(workspace):
    """The product's core promise: no evidence, no certificate."""
    spec(workspace, FAILING)
    main(["run", "experiment.yaml"])
    assert main(["certify", experiment_id(workspace)]) != 0


def test_the_certificate_does_not_claim_vacuous_completeness(workspace):
    """``all(...)`` over an empty sequence is True; a check must not be."""
    spec(workspace, FAILING)
    main(["run", "experiment.yaml"])
    experiment = experiment_id(workspace)
    main(["certify", experiment])
    checks = checks_by_name(workspace, experiment)

    provenance = checks["provenance"]
    assert provenance["status"] != "complete", (
        "no run succeeded, so 'every successful run carries its identity' "
        "is vacuously true and must not be reported as complete"
    )
    assert "no run succeeded" in provenance["detail"].lower()

    assert checks["seeds"]["status"] != "complete"
    assert checks["statistics"]["status"] != "verified"


def test_integrity_does_not_pass_metric_completeness_when_nothing_ran(workspace):
    spec(workspace, FAILING)
    main(["run", "experiment.yaml"])
    experiment = experiment_id(workspace)
    main(["analyze", experiment])
    findings = integrity_findings(workspace, experiment)
    assert "metrics.complete" not in findings, (
        "no arm produced any metric, so metric completeness cannot pass"
    )
    absent = findings.get("metrics.missing") or findings.get("metrics.absent")
    assert absent is not None, sorted(findings)
    assert absent["level"] != "pass"


def test_a_diagnostic_report_is_still_producible(workspace):
    """The report is how a researcher finds out what went wrong.

    Refusing to analyse a failed experiment would hide the failure, which is
    the opposite of the intended correction.
    """
    spec(workspace, FAILING)
    main(["run", "experiment.yaml"])
    assert main(["report", experiment_id(workspace), "--format", "markdown",
                 "--out", "reports"]) == 0
    report = (workspace / "reports" / "report.md").read_text(encoding="utf-8")
    assert "Failure analysis" in report


def test_the_exit_code_is_stable_on_a_rerun_of_a_failed_experiment(workspace):
    """Resuming a broken experiment must not turn failure into success."""
    spec(workspace, FAILING)
    assert main(["run", "experiment.yaml"]) != 0
    assert main(["run", "experiment.yaml"]) != 0


# --------------------------------------------------------------------------
# cases that must keep working
# --------------------------------------------------------------------------


def test_a_partially_failed_experiment_still_exits_zero(workspace):
    """Some evidence is still evidence; the run did what was asked."""
    spec(workspace, PARTIAL)
    assert main(["run", "experiment.yaml"]) == 0


def test_a_failed_run_is_never_silently_dropped(workspace):
    spec(workspace, PARTIAL)
    main(["run", "experiment.yaml"])
    runs = load_runs(workspace, experiment_id(workspace))
    statuses = sorted(r.status.value for r in runs)
    assert "failed" in statuses, "the failure must remain in the record"
    assert "completed" in statuses


def test_an_inconclusive_result_is_not_an_execution_failure(workspace):
    """The distinction that matters most.

    Two identical arms cannot separate, so the evidence is inconclusive. The
    runs succeeded, so this is not an execution failure and must never be
    reported as one.
    """
    spec(workspace, INCONCLUSIVE)
    assert main(["run", "experiment.yaml"]) == 0
    experiment = experiment_id(workspace)
    assert main(["analyze", experiment]) == 0

    report = analysis(workspace, experiment)
    assert report["claims"], "an inconclusive comparison is still a claim"
    verdicts = {c["verdict"] for c in report["claims"] if c["kind"] == "comparison"}
    assert verdicts <= {"inconclusive", "equivalent"}, verdicts

    assert main(["report", experiment, "--format", "markdown",
                 "--out", "reports"]) == 0


def test_a_fully_successful_experiment_is_unchanged(workspace):
    spec(workspace, WORKING)
    assert main(["run", "experiment.yaml"]) == 0
    experiment = experiment_id(workspace)
    assert main(["analyze", experiment]) == 0
    assert main(["report", experiment, "--format", "markdown",
                 "--out", "reports"]) == 0
    runs = load_runs(workspace, experiment)
    assert runs and all(r.status.value == "completed" for r in runs)


# --------------------------------------------------------------------------
# the HTTP job must agree with the CLI
# --------------------------------------------------------------------------


def test_the_http_job_does_not_report_success_for_a_total_failure(workspace):
    """A job whose every run failed is a failed job, whatever the CLI says."""
    from fabric.server.worker import job_outcome

    spec(workspace, FAILING)
    main(["run", "experiment.yaml"])
    runs = load_runs(workspace, experiment_id(workspace))
    assert runs and all(r.status.value == "failed" for r in runs)

    verdict, detail = job_outcome(runs)
    assert verdict == "failed", (
        f"a job whose {len(runs)} runs all failed must be a failed job; "
        f"got {verdict!r} ({detail!r})"
    )


def test_a_partially_failed_job_is_not_a_failed_job(workspace):
    from fabric.server.worker import job_outcome

    spec(workspace, PARTIAL)
    main(["run", "experiment.yaml"])
    runs = load_runs(workspace, experiment_id(workspace))
    verdict, _detail = job_outcome(runs)
    assert verdict == "succeeded", "some evidence is still evidence"


def test_a_job_whose_runs_all_succeeded_is_a_successful_job(workspace):
    from fabric.server.worker import job_outcome

    spec(workspace, WORKING)
    main(["run", "experiment.yaml"])
    runs = load_runs(workspace, experiment_id(workspace))
    verdict, _detail = job_outcome(runs)
    assert verdict == "succeeded"
