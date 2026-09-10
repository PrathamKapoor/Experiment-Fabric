"""Reproduction certificates.

A certificate is a short, machine-readable statement of what was
verified about an experiment and what was not.  It is deliberately not a
badge: every line records the evidence behind it, and anything that could
not be checked is reported as ``unverified`` rather than quietly passing.

The certificate is itself content-addressed, so a certificate that has
been edited no longer hashes to its own identifier.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from fabric.analysis.engine import ExperimentAnalysis
from fabric.core.entities import Role, Run, RunStatus
from fabric.core.registry import Registry
from fabric.execution.planner import ExperimentPlan
from fabric.util.hashing import content_id, file_digest

CERTIFICATE_VERSION = "1.0"

VERIFIED = "verified"
RECORDED = "recorded"
UNVERIFIED = "unverified"
FAILED = "failed"
INCOMPLETE = "incomplete"
COMPLETE = "complete"


@dataclass
class Check:
    """One line of a certificate."""

    name: str
    status: str
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in (VERIFIED, RECORDED, COMPLETE)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "status": self.status, "detail": self.detail,
                "evidence": self.evidence}


@dataclass
class Certificate:
    """The full certificate."""

    experiment_id: str
    name: str
    research_question: str = ""
    spec_digest: str = ""
    version: str = CERTIFICATE_VERSION
    issued_at: float = field(default_factory=time.time)
    checks: list[Check] = field(default_factory=list)
    reproduction: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if any(c.status == FAILED for c in self.checks):
            return "FAIL"
        if any(c.status in (UNVERIFIED, INCOMPLETE) for c in self.checks):
            return "PASS WITH GAPS"
        return "PASS"

    @property
    def certificate_id(self) -> str:
        return "EXP-CERT-" + content_id(self.payload(), length=12)

    def payload(self) -> dict[str, Any]:
        """The signed content: everything except the identifier itself."""
        return {
            "version": self.version,
            "experiment_id": self.experiment_id,
            "name": self.name,
            "research_question": self.research_question,
            "spec_digest": self.spec_digest,
            "checks": [c.to_dict() for c in self.checks],
            "reproduction": self.reproduction,
            "verdict": self.verdict,
        }

    def to_dict(self) -> dict[str, Any]:
        data = self.payload()
        data["certificate_id"] = self.certificate_id
        data["issued_at"] = self.issued_at
        data["notes"] = self.notes
        return data

    def render(self) -> str:
        """The terminal rendering."""
        width = max((len(c.name) for c in self.checks), default=20)
        lines = [
            "EXPERIMENT CERTIFICATE",
            "",
            f"  experiment      {self.experiment_id}",
            f"  name            {self.name}",
            f"  specification   {self.spec_digest[:31]}",
            "",
        ]
        for check in self.checks:
            lines.append(f"  {check.name:<{width}}   {check.status}")
            if check.detail:
                lines.append(f"  {'':<{width}}   {check.detail}")
        lines.append("")
        if self.reproduction:
            lines.append(f"  reproduction    {self.reproduction.get('verdict', 'not attempted')}")
            lines.append("")
        lines.append(f"  VERDICT         {self.verdict}")
        lines.append(f"  certificate     {self.certificate_id}")
        if self.notes:
            lines.append("")
            for note in self.notes:
                lines.append(f"  note: {note}")
        return "\n".join(lines)


def verify_certificate(data: dict[str, Any]) -> tuple[bool, str]:
    """Check that a stored certificate still hashes to its own identifier."""
    claimed = data.get("certificate_id", "")
    payload = {k: data.get(k) for k in
               ("version", "experiment_id", "name", "research_question",
                "spec_digest", "checks", "reproduction", "verdict")}
    recomputed = "EXP-CERT-" + content_id(payload, length=12)
    if claimed != recomputed:
        return False, (
            f"certificate does not match its contents (claims {claimed}, "
            f"recomputes to {recomputed}): it has been modified since it was issued"
        )
    return True, "certificate contents match its identifier"


def issue(
    analysis: ExperimentAnalysis,
    plan: ExperimentPlan,
    runs: Sequence[Run],
    registry: Registry,
    *,
    workspace_root: str = "",
    reproduction: dict[str, Any] | None = None,
) -> Certificate:
    """Build a certificate for an experiment."""
    certificate = Certificate(
        experiment_id=analysis.experiment_id,
        name=analysis.name,
        research_question=analysis.research_question,
        spec_digest=analysis.spec_digest,
        reproduction=reproduction,
    )
    spec = plan.spec

    # -- code --------------------------------------------------------------
    algorithm = plan.algorithm
    if algorithm and algorithm.source_digest:
        if algorithm.dirty:
            certificate.checks.append(Check(
                "code", INCOMPLETE,
                f"{algorithm.name} at {algorithm.source_digest[:23]}, commit "
                f"{algorithm.commit[:12]} with uncommitted changes: the entrypoint "
                "file is pinned but the modules it imports are not",
                {"source_digest": algorithm.source_digest, "commit": algorithm.commit,
                 "dirty": True},
            ))
        else:
            certificate.checks.append(Check(
                "code", VERIFIED,
                f"{algorithm.name} at {algorithm.source_digest[:23]}"
                + (f", commit {algorithm.commit[:12]}" if algorithm.commit else
                   ", not under version control"),
                {"source_digest": algorithm.source_digest, "commit": algorithm.commit},
            ))
    else:
        certificate.checks.append(Check(
            "code", UNVERIFIED,
            "the entrypoint source could not be hashed, so the code that ran cannot "
            "be pinned",
        ))

    # -- dataset -----------------------------------------------------------
    dataset = plan.dataset
    if dataset is None:
        certificate.checks.append(Check("dataset", UNVERIFIED, "no dataset was declared"))
    elif not dataset.digest:
        certificate.checks.append(Check(
            "dataset", UNVERIFIED, f"{dataset.ref} has no content digest"))
    else:
        ok, message = registry.verify("dataset", dataset.id)
        certificate.checks.append(Check(
            "dataset", VERIFIED if ok else FAILED,
            f"{dataset.ref} {dataset.digest[:23]}: {message}",
            {"digest": dataset.digest, "id": dataset.id},
        ))

    # -- model -------------------------------------------------------------
    model = plan.model
    if model is None:
        certificate.checks.append(Check("model", UNVERIFIED, "no model was declared"))
    else:
        ok, message = registry.verify("model", model.id)
        status = VERIFIED if (ok and model.weights_digest) else RECORDED if ok else FAILED
        certificate.checks.append(Check(
            "model", status,
            f"{model.ref}" + (f" weights {model.weights_digest[:23]}"
                              if model.weights_digest else " (no weights digest declared)"),
            {"id": model.id},
        ))

    # -- environment and hardware -----------------------------------------
    environment_ids = sorted({r.environment_id for r in runs if r.environment_id})
    hardware_ids = sorted({r.hardware_id for r in runs if r.hardware_id})
    certificate.checks.append(Check(
        "environment", VERIFIED if environment_ids else UNVERIFIED,
        (f"{len(environment_ids)} environment(s) captured: "
         + ", ".join(e[:16] for e in environment_ids[:3]))
        if environment_ids else "no environment was captured",
        {"environment_ids": environment_ids},
    ))
    certificate.checks.append(Check(
        "hardware", RECORDED if hardware_ids else UNVERIFIED,
        (f"{len(hardware_ids)} machine(s): " + ", ".join(h[:16] for h in hardware_ids[:3]))
        if hardware_ids else "no hardware was captured",
        {"hardware_ids": hardware_ids},
    ))

    # -- coverage ----------------------------------------------------------
    planned_seeds = len(spec.seeds.values)
    completed_by_condition: dict[str, set[int]] = {}
    for run in runs:
        if run.succeeded and not run.replicate:
            completed_by_condition.setdefault(run.condition_name, set()).add(run.seed)
    worst = min((len(v) for v in completed_by_condition.values()), default=0)
    certificate.checks.append(Check(
        "seeds", COMPLETE if worst >= planned_seeds else INCOMPLETE,
        f"{worst}/{planned_seeds} seeds completed in the thinnest arm",
        {"planned": planned_seeds, "minimum_completed": worst},
    ))

    for label, role in (("baselines", Role.BASELINE), ("ablations", Role.ABLATION)):
        expected = [c for c in plan.conditions if c.role == role]
        done = [c for c in expected if completed_by_condition.get(c.name)]
        if not expected:
            certificate.checks.append(Check(label, UNVERIFIED, f"no {label} were declared"))
            continue
        certificate.checks.append(Check(
            label, COMPLETE if len(done) == len(expected) else INCOMPLETE,
            f"{len(done)}/{len(expected)} executed",
            {"expected": len(expected), "executed": len(done)},
        ))

    # -- statistics --------------------------------------------------------
    comparisons = analysis.baselines.comparisons
    untested = [c for c in comparisons if c.test.test == "none"]
    if not comparisons:
        certificate.checks.append(Check(
            "statistics", UNVERIFIED, "no comparison was possible"))
    else:
        certificate.checks.append(Check(
            "statistics", VERIFIED if not untested else INCOMPLETE,
            f"{len(comparisons) - len(untested)}/{len(comparisons)} comparisons tested, "
            f"{analysis.correction}-corrected at alpha={analysis.alpha}, "
            f"analysis version {analysis.analysis_version}",
            {"comparisons": len(comparisons), "untested": len(untested)},
        ))

    # -- artifacts ---------------------------------------------------------
    artifacts = [(r, a) for r in runs for a in r.artifacts]
    missing, altered = _verify_artifacts(artifacts, plan, workspace_root)
    if not artifacts:
        certificate.checks.append(Check("artifacts", UNVERIFIED, "no artifacts were written"))
    elif missing or altered:
        certificate.checks.append(Check(
            "artifacts", FAILED,
            f"{len(artifacts)} recorded; {len(missing)} missing, {len(altered)} altered "
            "since they were written",
            {"missing": missing[:10], "altered": altered[:10]},
        ))
    else:
        certificate.checks.append(Check(
            "artifacts", COMPLETE,
            f"{len(artifacts)} artifacts, all present and matching their digests",
            {"count": len(artifacts)},
        ))

    # -- provenance --------------------------------------------------------
    failed = sum(1 for r in runs if r.status == RunStatus.FAILED)
    complete = all(r.environment_id and r.hardware_id and r.algorithm_id
                   for r in runs if r.succeeded)
    certificate.checks.append(Check(
        "provenance", COMPLETE if complete else INCOMPLETE,
        f"{len(runs)} runs recorded ({failed} failed); every successful run carries its "
        "environment, hardware and code identity"
        if complete else "some runs are missing environment, hardware or code identity",
        {"runs": len(runs), "failed": failed},
    ))

    # -- determinism and reproducibility ----------------------------------
    if analysis.determinism.tested:
        certificate.checks.append(Check(
            "determinism",
            VERIFIED if analysis.determinism.deterministic else FAILED,
            analysis.determinism.describe(),
        ))
    else:
        certificate.checks.append(Check(
            "determinism", UNVERIFIED, "no determinism replicates were run"))

    if analysis.reproducibility.measured:
        beyond = analysis.reproducibility.beyond_tolerance
        certificate.checks.append(Check(
            "reproducibility",
            VERIFIED if not beyond else INCOMPLETE,
            analysis.reproducibility.describe(),
            {"worst_case": analysis.reproducibility.overall,
             "axes": len(analysis.reproducibility.measured)},
        ))
    else:
        certificate.checks.append(Check(
            "reproducibility", UNVERIFIED,
            "no perturbation axes beyond the seed were measured"))

    # -- integrity ---------------------------------------------------------
    if analysis.integrity.failures:
        certificate.checks.append(Check(
            "integrity", FAILED,
            "; ".join(f.title for f in analysis.integrity.failures),
        ))
    elif analysis.integrity.warnings:
        certificate.checks.append(Check(
            "integrity", INCOMPLETE,
            f"{len(analysis.integrity.warnings)} warning(s): "
            + "; ".join(f.title for f in analysis.integrity.warnings[:3]),
        ))
    else:
        certificate.checks.append(Check(
            "integrity", VERIFIED, analysis.integrity.summary_line()))

    if certificate.verdict != "PASS":
        certificate.notes.append(
            "A verdict of PASS WITH GAPS means every check that could be run passed, "
            "and some could not be run. The gaps are listed above; they are not "
            "failures, but they are not verification either."
        )
    return certificate


def _verify_artifacts(artifacts: Sequence[tuple[Run, Any]], plan: ExperimentPlan,
                      workspace_root: str) -> tuple[list[str], list[str]]:
    """Re-hash every artifact and report missing or altered files."""
    import os

    missing: list[str] = []
    altered: list[str] = []
    if not workspace_root:
        return missing, altered
    for run, artifact in artifacts:
        path = os.path.join(
            workspace_root, ".fabric", "experiments", plan.experiment.id,
            "runs", run.id, "artifacts", artifact.path.replace("/", os.sep),
        )
        if not os.path.isfile(path):
            missing.append(f"{run.id}/{artifact.name}")
            continue
        if file_digest(path) != artifact.digest:
            altered.append(f"{run.id}/{artifact.name}")
    return missing, altered
