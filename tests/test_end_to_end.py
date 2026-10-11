"""The whole chain, once, through the command line.

Every other test file exercises a layer. This one exercises the claim:

    YAML -> plan -> run -> analyse -> report -> graph -> certify
         -> export -> verify -> reproduce -> diff

It runs the real ``fabric`` command in a subprocess against a real
workspace, and checks the *files and their contents* rather than exit
codes. An end-to-end test that only asserts ``returncode == 0`` passes
happily while the reports are empty.

It is the same path ``scripts/demo.sh`` takes, so a break here is a break
in the demo.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SPEC = """
# The canonical demonstration. Deliberately lightweight: the point is to
# exercise the infrastructure, not to establish anything about the method.
experiment:
  name: golden-path
  research_question: Does the trick component earn its place?
  hypothesis: The trick improves accuracy; the decoration does nothing.
  treatment_name: proposed

algorithm:
  name: proposed
  entrypoint: python:tests.helpers:ablatable

components: [alpha, beta]

metrics:
  - name: accuracy
    direction: maximize
    primary: true
    tolerance: 0.02

seeds:
  values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

baselines:
  - name: control
    description: The floor.
    algorithm:
      name: control
      entrypoint: python:tests.helpers:deterministic

ablations: [remove_alpha, remove_beta]

analysis:
  primary_metric: accuracy
  alpha: 0.05
  confidence_level: 0.95

reproducibility:
  enabled: true
  determinism_replicates: 2
  perturbations:
    environment:
      values:
        - {label: single-threaded, OMP_NUM_THREADS: "1"}
    configuration:
      values:
        - {label: shifted, scale: 2.0}

execution:
  isolation: inline
  max_workers: 1
"""


class Fabric:
    """Runs the real command line in a real workspace."""

    def __init__(self, root: str) -> None:
        self.root = root
        self.env = {**os.environ, "PYTHONPATH": ROOT, "FABRIC_NO_COLOR": "1"}

    def __call__(self, *args: str, expect: int | None = 0) -> str:
        result = subprocess.run(
            [sys.executable, "-m", "fabric", *args],
            cwd=self.root, capture_output=True, text=True, timeout=900, env=self.env,
        )
        if expect is not None and result.returncode != expect:
            raise AssertionError(
                f"fabric {' '.join(args)} exited {result.returncode}, expected "
                f"{expect}\n--- stdout ---\n{result.stdout}\n--- stderr ---\n"
                f"{result.stderr}"
            )
        self.last = result
        return result.stdout

    def path(self, *parts: str) -> str:
        return os.path.join(self.root, *parts)

    def read(self, *parts: str) -> str:
        with open(self.path(*parts), encoding="utf-8") as handle:
            return handle.read()

    def read_json(self, *parts: str):
        return json.loads(self.read(*parts))


@pytest.fixture(scope="module")
def golden(tmp_path_factory):
    """Walk the entire chain once; every test below reads what it left."""
    root = str(tmp_path_factory.mktemp("golden"))
    fabric = Fabric(root)
    with open(os.path.join(root, "experiment.yaml"), "w", encoding="utf-8") as handle:
        handle.write(SPEC)

    fabric("init", "--name", "golden")
    plan_output = fabric("plan", "experiment.yaml")
    fabric("run", "experiment.yaml")

    experiment_id = ""
    for line in fabric("ls").splitlines():
        if "golden-path" in line:
            experiment_id = line.split()[0]
            break
    assert experiment_id.startswith("EXP-"), "could not find the experiment in ls"

    analyze_output = fabric("analyze", experiment_id)
    fabric("report", experiment_id, "--format", "all")
    fabric("graph", experiment_id, "--format", "html",
           "--out", fabric.path("graph.html"))
    fabric("certify", experiment_id)
    fabric("export", experiment_id, "--out", fabric.path("bundle"))
    verify_output = fabric("verify", fabric.path("bundle"))
    why_output = fabric("why", experiment_id, "--metric", "accuracy", "--seed", "4")

    return {
        "fabric": fabric,
        "id": experiment_id,
        "plan": plan_output,
        "analyze": analyze_output,
        "verify": verify_output,
        "why": why_output,
    }


# --------------------------------------------------------------------------
# expansion
# --------------------------------------------------------------------------


def test_one_yaml_expands_into_the_right_arms(golden):
    """1 treatment + 1 baseline + 2 ablations + 2 perturbations."""
    output = golden["plan"]
    assert "6 (1 treatment, 1 baseline, 2 ablation" in output
    assert "remove_alpha" in output and "remove_beta" in output
    assert "control" in output


def test_the_plan_counts_the_runs_it_will_execute(golden):
    plan = golden["fabric"].read_json(".fabric", "experiments", golden["id"],
                                      "plan.json")
    # 1 treatment + 1 baseline + 2 ablations + 2 perturbation conditions
    assert len(plan["conditions"]) == 6
    # six conditions across ten seeds, plus the determinism replicates
    assert len(plan["runs"]) > 60


def test_every_planned_run_produced_a_record(golden):
    directory = golden["fabric"].path(".fabric", "experiments", golden["id"], "runs")
    records = os.listdir(directory)
    plan = golden["fabric"].read_json(".fabric", "experiments", golden["id"],
                                      "plan.json")
    assert len(records) == len(plan["runs"])


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------


def test_the_headline_states_a_result_with_its_uncertainty(golden):
    output = golden["analyze"]
    assert "proposed scored" in output
    assert "+/-" in output, "a mean without a spread is not a result"
    assert "over 10 seeds" in output


def test_the_baseline_comparison_is_reported_with_an_interval(golden):
    output = golden["analyze"]
    assert "baseline comparisons" in output
    assert "95% CI" in output
    assert "control" in output


def test_the_ablation_matrix_says_which_components_are_load_bearing(golden):
    output = golden["analyze"]
    assert "ablations" in output
    assert "alpha" in output and "beta" in output
    assert "detectable" in output


def test_determinism_was_tested_rather_than_assumed(golden):
    assert "reproducibility" in golden["analyze"]
    analysis = _analysis(golden)
    assert analysis["determinism"]["tested"] is True


def test_both_perturbation_axes_were_measured(golden):
    analysis = _analysis(golden)
    axes = {axis["axis"] for axis in analysis["reproducibility"]["axes"]}
    assert {"environment", "configuration"} <= axes


def test_the_analysis_reports_its_own_limitations(golden):
    analysis = _analysis(golden)
    assert analysis["limitations"], "an analysis with no caveats is not an analysis"
    assert any("detect" in text or "seed" in text for text in analysis["limitations"])


def test_every_claim_carries_whether_the_evidence_supports_it(golden):
    analysis = _analysis(golden)
    assert analysis["claims"]
    for claim in analysis["claims"]:
        assert "statement" in claim and "verdict" in claim
        assert isinstance(claim["supported"], bool)


def _analysis(golden):
    fabric = golden["fabric"]
    return json.loads(fabric("analyze", golden["id"], "--json"))


# --------------------------------------------------------------------------
# reports
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name,fragment", [
    ("report.md", "# "),
    ("report.html", "<!doctype html"),
    ("report.tex", "documentclass"),
    ("report.json", '"sections"'),
])
def test_every_report_format_is_written_and_non_trivial(golden, name, fragment):
    text = golden["fabric"].read(".fabric", "experiments", golden["id"],
                                 "reports", name)
    assert fragment.lower() in text.lower()
    assert len(text) > 3000, f"{name} is suspiciously short"


def test_the_markdown_report_contains_the_sections_that_matter(golden):
    text = golden["fabric"].read(".fabric", "experiments", golden["id"],
                                 "reports", "report.md")
    for heading in ("Summary", "Experimental setup", "Statistical analysis",
                    "Reproducibility", "Limitations", "Provenance"):
        assert heading in text, f"the report has no {heading} section"


def test_the_report_separates_observation_from_interpretation(golden):
    text = golden["fabric"].read(".fabric", "experiments", golden["id"],
                                 "reports", "report.md")
    assert "Limitations" in text
    assert "Research question" in text
    assert "Hypothesis, stated before the runs" in text


def test_the_html_report_embeds_a_clickable_graph(golden):
    text = golden["fabric"].read(".fabric", "experiments", golden["id"],
                                 "reports", "report.html")
    assert "<svg" in text
    assert "lineage" in text.lower()


def test_the_standalone_graph_renders(golden):
    text = golden["fabric"].read("graph.html")
    assert "<svg" in text and "<!doctype html" in text.lower()


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


def test_why_traces_a_number_to_everything_that_produced_it(golden):
    output = golden["why"]
    assert "accuracy" in output
    for link in ("run", "condition", "algorithm", "environment", "hardware"):
        assert link in output, f"the provenance chain has no {link}"
    assert "seed" in output


def test_why_names_the_links_it_could_not_record(golden):
    """A gap must look like a gap, not like an absence of one."""
    output = golden["why"]
    assert "not recorded" in output
    assert "dataset" in output   # this spec declares none, and it says so


def test_impact_answers_the_inverse_question(golden):
    fabric = golden["fabric"]
    graph = json.loads(fabric("graph", golden["id"], "--format", "json"))
    algorithm = next(n for n in graph["nodes"] if n["kind"] == "algorithm")
    output = fabric("impact", golden["id"], "--node", algorithm["id"])
    assert "depends on it" in output


# --------------------------------------------------------------------------
# certificate and bundle
# --------------------------------------------------------------------------


def test_the_certificate_records_what_it_hashed(golden):
    certificate = golden["fabric"].read_json(".fabric", "experiments",
                                             golden["id"], "certificate.json")
    assert certificate["checks"]
    assert certificate["spec_digest"].startswith("sha256:")
    assert certificate["fabric_version"]


def test_the_bundle_contains_everything_needed_to_understand_it(golden):
    for name in ("experiment.yaml", "results.json", "analysis.json",
                 "provenance.json", "certificate.json", "report.md",
                 "README.md", "manifest.json", "reproduce.sh"):
        assert os.path.isfile(golden["fabric"].path("bundle", name)), \
            f"the bundle has no {name}"


def test_the_manifest_carries_its_schema_and_version(golden):
    manifest = golden["fabric"].read_json("bundle", "manifest.json")
    assert manifest["fabric_version"]
    assert manifest["created_at"]
    assert manifest["schema_version"] == manifest["bundle_version"]
    assert manifest["python_version"]
    assert manifest["files"]
    for name, digest in manifest["files"].items():
        assert digest.startswith("sha256:"), f"{name} has no digest"


def test_an_untouched_bundle_verifies(golden):
    assert "INTACT" in golden["verify"]


def test_a_modified_result_fails_verification(golden, tmp_path):
    tampered = str(tmp_path / "tampered")
    shutil.copytree(golden["fabric"].path("bundle"), tampered)
    with open(os.path.join(tampered, "results.json"), "a", encoding="utf-8") as handle:
        handle.write("\n")

    fabric = golden["fabric"]
    output = fabric("verify", tampered, expect=1)
    assert "INTACT" not in output
    assert "results.json" in output


def test_modified_provenance_fails_verification(golden, tmp_path):
    tampered = str(tmp_path / "tampered-provenance")
    shutil.copytree(golden["fabric"].path("bundle"), tampered)
    path = os.path.join(tampered, "provenance.json")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["nodes"] = payload["nodes"][:-1]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    output = golden["fabric"]("verify", tampered, expect=1)
    assert "INTACT" not in output


def test_a_modified_certificate_fails_verification(golden, tmp_path):
    tampered = str(tmp_path / "tampered-certificate")
    shutil.copytree(golden["fabric"].path("bundle"), tampered)
    path = os.path.join(tampered, "certificate.json")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["summary"] = "everything is fine, honestly"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    output = golden["fabric"]("verify", tampered, expect=1)
    assert "INTACT" not in output


# --------------------------------------------------------------------------
# reproduction and diff
# --------------------------------------------------------------------------


def test_the_bundle_reproduces(golden):
    output = golden["fabric"]("reproduce", golden["fabric"].path("bundle"))
    assert "reproducing" in output
    assert "PASS" in output, output


def test_two_experiments_can_be_compared(golden, tmp_path):
    """The same specification with one thing changed, and Fabric finds it."""
    fabric = golden["fabric"]
    variant = SPEC.replace("name: golden-path", "name: golden-path-variant") \
                  .replace("values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
                           "values: [1, 2, 3, 4, 5]")
    with open(fabric.path("variant.yaml"), "w", encoding="utf-8") as handle:
        handle.write(variant)
    fabric("run", "variant.yaml")

    other = ""
    for line in fabric("ls").splitlines():
        if "golden-path-variant" in line:
            other = line.split()[0]
            break
    assert other.startswith("EXP-")

    payload = json.loads(fabric("diff", golden["id"], other, "--json"))
    assert payload["identical"] is False
    seeds = [d for d in payload["differences"] if d["category"] == "seeds"]
    assert seeds and seeds[0]["material"] is True


# --------------------------------------------------------------------------
# machine-readable output
# --------------------------------------------------------------------------


@pytest.mark.parametrize("command", [
    ("ls",), ("analyze",), ("status",), ("certify",),
])
def test_the_json_output_of_each_command_parses_strictly(golden, command):
    """Strictly: a bare NaN would make the document unreadable elsewhere."""
    arguments = list(command)
    if command[0] != "ls":
        arguments.append(golden["id"])
    arguments.append("--json")
    text = golden["fabric"](*arguments)
    parsed = json.loads(text, parse_constant=_reject)
    assert parsed


def _reject(name):
    raise AssertionError(f"the output contains the bare literal {name!r}, "
                         "which is not JSON")


def test_the_experiment_reports_a_completed_status(golden):
    status = json.loads(golden["fabric"]("status", golden["id"], "--json"))
    assert status["status"] == "completed"
    assert status["runs"]["failed"] == 0
    assert status["runs"]["completed"] == status["runs"]["planned"]
