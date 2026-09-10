"""Evaluation integrity.

The checks here look for the ways a well-intentioned experiment quietly
stops being evidence: data that changed under the results, a test set
that overlaps the training set, an analysis edited after the numbers came
in, arms compared on different seeds.

None of them are accusations.  Each finding names what was observed, why
it matters, and what would resolve it -- and the ones that cannot be
checked say so rather than passing silently, because a green tick you did
not earn is worse than no tick at all.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from fabric.analysis.results import ResultSet
from fabric.core.entities import Dataset, Role
from fabric.execution.planner import ExperimentPlan
from fabric.util.hashing import file_digest, tree_digest

FAIL = "fail"
WARN = "warn"
INFO = "info"
PASS = "pass"

_SEVERITY = {PASS: 0, INFO: 1, WARN: 2, FAIL: 3}


@dataclass
class Finding:
    """One integrity observation."""

    code: str
    level: str
    title: str
    detail: str = ""
    remedy: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "level": self.level,
            "title": self.title,
            "detail": self.detail,
            "remedy": self.remedy,
            "evidence": self.evidence,
        }


@dataclass
class IntegrityReport:
    """Everything the integrity checks found."""

    findings: list[Finding] = field(default_factory=list)

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)

    @property
    def failures(self) -> list[Finding]:
        return [f for f in self.findings if f.level == FAIL]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.level == WARN]

    @property
    def passed(self) -> bool:
        return not self.failures

    @property
    def worst_level(self) -> str:
        return max((f.level for f in self.findings), key=lambda l: _SEVERITY[l], default=PASS)

    def summary_line(self) -> str:
        if not self.findings:
            return "no integrity checks ran"
        counts = {
            level: sum(1 for f in self.findings if f.level == level)
            for level in (PASS, INFO, WARN, FAIL)
        }
        return (f"{counts[PASS]} passed, {counts[INFO]} informational, "
                f"{counts[WARN]} warnings, {counts[FAIL]} failures")

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "worst_level": self.worst_level,
            "summary": self.summary_line(),
            "findings": [f.to_dict() for f in self.findings],
        }


# --------------------------------------------------------------------------
# individual checks
# --------------------------------------------------------------------------


def check_dataset_digest(dataset: Dataset | None, report: IntegrityReport) -> None:
    """Confirm the data on disk is still the data that was registered."""
    if dataset is None:
        report.add(Finding(
            "dataset.absent", INFO, "No dataset is attached to this experiment",
            "Nothing to verify. If the method reads data from somewhere, Fabric cannot "
            "see it and cannot vouch for it.",
        ))
        return
    if not dataset.digest:
        report.add(Finding(
            "dataset.no_digest", WARN, f"Dataset {dataset.ref} has no content digest",
            "Without a digest there is no way to prove which data produced these results, "
            "or to detect that it changed afterwards.",
            "Give the dataset a 'path' so Fabric can hash it, or declare a 'digest'.",
        ))
        return
    if not dataset.local_path or not os.path.exists(dataset.local_path):
        report.add(Finding(
            "dataset.not_local", INFO,
            f"Dataset {dataset.ref} is not present locally",
            f"Its digest {dataset.digest[:23]} is recorded but cannot be re-verified here.",
            evidence={"digest": dataset.digest},
        ))
        return
    current = tree_digest(dataset.local_path)
    if current != dataset.digest:
        report.add(Finding(
            "dataset.mutated", FAIL,
            f"Dataset {dataset.ref} has changed since it was registered",
            f"Registered as {dataset.digest[:23]}, now {current[:23]}. Results produced "
            "before the change and after it are not comparable, and the earlier ones can "
            "no longer be reproduced from what is on disk.",
            "Register the new contents as a new version and rerun, or restore the "
            "original data.",
            {"registered": dataset.digest, "current": current, "path": dataset.local_path},
        ))
        return
    report.add(Finding(
        "dataset.verified", PASS, f"Dataset {dataset.ref} matches its registered digest",
        f"{dataset.digest[:23]} verified against the files on disk.",
        evidence={"digest": dataset.digest},
    ))


def check_split_overlap(dataset: Dataset | None, report: IntegrityReport,
                        *, max_overlap: float = 0.0) -> None:
    """Look for examples that appear in more than one split.

    Only runs when the dataset declares split files.  Contamination is
    detected by hashing each example line, which catches exact duplicates
    -- the common and most damaging case -- but not near-duplicates.
    """
    if dataset is None or not dataset.splits:
        return
    paths: dict[str, str] = {}
    for name, meta in dataset.splits.items():
        path = str((meta or {}).get("path", ""))
        if not path:
            continue
        full = path if os.path.isabs(path) else os.path.join(dataset.local_path or "", path)
        if os.path.isfile(full):
            paths[name] = full
    if len(paths) < 2:
        if dataset.splits:
            report.add(Finding(
                "splits.unverifiable", INFO,
                "Train/test overlap could not be checked",
                "The dataset declares splits but not file paths for them, so Fabric "
                "cannot look for contamination.",
                "Add 'path' to each split to enable the check.",
            ))
        return

    fingerprints: dict[str, set[str]] = {}
    for name, path in paths.items():
        seen: set[str] = set()
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    seen.add(_example_fingerprint(line))
        fingerprints[name] = seen

    names = sorted(fingerprints)
    clean = True
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            shared = fingerprints[left] & fingerprints[right]
            if not shared:
                continue
            smaller = min(len(fingerprints[left]), len(fingerprints[right])) or 1
            fraction = len(shared) / smaller
            clean = False
            level = FAIL if fraction > max_overlap else WARN
            report.add(Finding(
                "splits.overlap", level,
                f"{len(shared)} examples appear in both '{left}' and '{right}'",
                f"That is {fraction * 100:.2f}% of the smaller split. Evaluation on "
                "examples the method has seen does not measure generalisation.",
                "Deduplicate the splits and re-register the dataset as a new version.",
                {"splits": [left, right], "shared": len(shared), "fraction": fraction},
            ))
    if clean:
        report.add(Finding(
            "splits.disjoint", PASS, "Splits contain no exactly duplicated examples",
            f"Checked {', '.join(names)} pairwise by exact example hash. "
            "Near-duplicates are not detected by this check.",
        ))


def _example_fingerprint(line: str) -> str:
    """Hash one example, ignoring key order in JSON records."""
    try:
        parsed = json.loads(line)
        return json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    except (json.JSONDecodeError, TypeError):
        return line


def check_seed_alignment(results: ResultSet, report: IntegrityReport) -> None:
    """Confirm arms were evaluated on the same seeds."""
    comparable = [c for c in results.conditions.values() if not c.perturbation]
    seed_sets = {c.name: set(c.seeds) for c in comparable if c.seeds}
    if len(seed_sets) < 2:
        return
    reference = max(seed_sets.values(), key=len)
    mismatched = {name: sorted(reference - seeds)
                  for name, seeds in seed_sets.items() if seeds != reference}
    if not mismatched:
        report.add(Finding(
            "seeds.aligned", PASS,
            f"All {len(seed_sets)} arms completed the same {len(reference)} seeds",
            "Comparisons can be paired, which removes seed variance from every "
            "difference.",
        ))
        return
    report.add(Finding(
        "seeds.misaligned", WARN,
        f"{len(mismatched)} arm(s) are missing seeds the others completed",
        "Fabric drops unmatched seeds before comparing, so those comparisons lose power. "
        "If the missing runs failed for reasons related to the condition, dropping them "
        "also biases the comparison in that condition's favour.",
        "Rerun the missing runs, or check the failure analysis for why they did not finish.",
        {"missing": {k: v[:10] for k, v in mismatched.items()}},
    ))


def check_analysis_frozen(plan: ExperimentPlan, stored_digest: str,
                          report: IntegrityReport) -> None:
    """Detect an analysis plan that changed after results existed."""
    if not stored_digest:
        return
    if stored_digest == plan.spec.digest:
        report.add(Finding(
            "spec.unchanged", PASS,
            "The specification is unchanged since these runs were executed",
            f"Digest {stored_digest[7:19]} matches. The analysis was declared before the "
            "results were seen.",
        ))
        return
    report.add(Finding(
        "spec.changed", WARN,
        "The specification changed after runs were recorded",
        "The stored results were produced under a different specification than the one "
        "being analysed now. If what changed was the analysis (the test, the alpha, the "
        "primary metric), the p-values below were chosen with knowledge of the data.",
        "Record what changed and why, or rerun under the new specification.",
        {"stored": stored_digest, "current": plan.spec.digest},
    ))


def check_baseline_effort(results: ResultSet, report: IntegrityReport) -> None:
    """Flag baselines that were given less tuning than the treatment.

    An untuned baseline is the most common way to manufacture an
    improvement, and it is visible in the plan: if the treatment was swept
    over configurations and the baselines were not, they were not given
    the same chance.
    """
    treatments = results.treatments
    baselines = results.baselines
    if not baselines or len(treatments) < 2:
        return
    report.add(Finding(
        "baselines.effort", WARN,
        f"The treatment was swept over {len(treatments)} configurations; "
        f"each baseline was run at one",
        "Comparing the best of several treatment configurations against a single "
        "baseline configuration credits the method with tuning effort the baseline "
        "never received.",
        "Sweep the baselines over a comparable space, or report the treatment's median "
        "configuration rather than its best.",
        {"treatment_configurations": len(treatments), "baselines": [b.name for b in baselines]},
    ))


def check_metric_completeness(results: ResultSet, report: IntegrityReport) -> None:
    """Confirm every declared metric was actually produced."""
    missing: dict[str, list[str]] = {}
    for spec in results.metrics:
        absent = [c.name for c in results.conditions.values()
                  if c.successful and not c.series(spec.name)]
        if absent:
            missing[spec.name] = absent
    if not missing:
        report.add(Finding(
            "metrics.complete", PASS,
            f"All {len(results.metrics)} declared metrics were produced by every arm",
        ))
        return
    report.add(Finding(
        "metrics.missing", WARN,
        f"{len(missing)} declared metric(s) were not produced by every arm",
        "A metric reported for some conditions and not others cannot be compared across "
        "them, and its absence may not be random.",
        "Have every entrypoint record every declared metric.",
        {"missing": {k: v[:8] for k, v in missing.items()}},
    ))


def check_replication_depth(results: ResultSet, report: IntegrityReport) -> None:
    """Confirm there are enough replications to say anything."""
    counts = {c.name: len(c.successful) for c in results.conditions.values() if c.runs}
    if not counts:
        return
    smallest = min(counts.values())
    if smallest >= 10:
        report.add(Finding(
            "replication.sufficient", PASS,
            f"Every arm completed at least {smallest} replications",
        ))
    elif smallest >= 5:
        report.add(Finding(
            "replication.modest", INFO,
            f"The smallest arm has {smallest} replications",
            "Enough for an effect size, thin for a confident interval. Ten is where "
            "seed-level conclusions start to be stable.",
        ))
    else:
        report.add(Finding(
            "replication.thin", WARN,
            f"The smallest arm has only {smallest} replication(s)",
            "Below five replications, variance estimates are unreliable and small-sample "
            "tests may not be able to reach significance at all regardless of the effect.",
            "Add seeds before drawing conclusions.",
            {"per_condition": counts},
        ))


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def run_checks(plan: ExperimentPlan, results: ResultSet, *,
               stored_spec_digest: str = "") -> IntegrityReport:
    """Run every applicable integrity check."""
    report = IntegrityReport()
    settings = plan.spec.integrity

    if settings.require_dataset_digest:
        check_dataset_digest(plan.dataset, report)
    if settings.forbid_test_in_train or settings.detect_duplicate_examples:
        check_split_overlap(plan.dataset, report,
                            max_overlap=settings.max_train_test_overlap)
    check_seed_alignment(results, report)
    check_metric_completeness(results, report)
    check_replication_depth(results, report)
    check_analysis_frozen(plan, stored_spec_digest, report)
    if settings.freeze_baselines:
        check_baseline_effort(results, report)
    return report
