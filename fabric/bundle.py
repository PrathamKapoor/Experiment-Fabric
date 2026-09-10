"""Portable reproduction bundles.

An experiment that only exists inside one person's ``.fabric`` directory
is not a shareable scientific artifact.  A bundle is a self-describing
directory that another researcher can read without Fabric and re-run with
it:

::

    bundle/
      experiment.yaml     the specification, exactly as declared
      manifest.json       identities, digests, counts, verification data
      results.json        every run, every metric, every failure
      analysis.json       the statistics, with the reasoning
      provenance.json     the evidence graph
      certificate.json    what was verified
      report.md           the human-readable report
      reproduce.sh        the commands that regenerate it
      README.md           what this is and how to use it
      artifacts/          the raw outputs, by run (optional)

Everything in it is plain text.  The bundle verifies against its own
manifest, so a file that changed after export is detectable.
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from fabric.core.entities import Run
from fabric.core.store import Workspace
from fabric.execution.planner import ExperimentPlan
from fabric.util.hashing import file_digest, tree_digest
from fabric.util.jsonio import atomic_write_text, read_json, write_json
from fabric.util.yamlio import dump_yaml

BUNDLE_VERSION = "1.0"

README = """# Reproduction bundle: {name}

`{experiment_id}`

## What this is

A complete, self-describing record of one computational experiment: the
specification that defined it, every run it produced, the statistical analysis,
the provenance graph connecting each number to what produced it, and a
certificate of what was verified.

Everything here is plain text. You can read it without installing anything.

## The claim

> {research_question}

{headline}

## How to reproduce it

With [Experiment Fabric](https://github.com/) installed:

```bash
fabric reproduce {bundle_name}
```

That re-executes the specification in your environment and compares the result
against `results.json`, reporting the difference with a confidence interval, an
equivalence test against the tolerance declared in the specification, and a list
of the environment differences that plausibly explain any deviation.

Without Fabric, `reproduce.sh` lists the commands, and `manifest.json` records
the exact dataset digest, model version, source commit and package versions the
original runs used.

## Verifying this bundle has not changed

```bash
fabric verify {bundle_name}
```

Every file is hashed in `manifest.json`. The certificate is content-addressed:
an edited certificate no longer matches its own identifier.

## What was measured

{summary}

## Limitations

{limitations}

Generated {generated} by Experiment Fabric {version}.
"""

REPRODUCE_SH = """#!/usr/bin/env bash
# Regenerate this experiment from its specification.
#
# The specification is the source of truth: it names the dataset version, the
# model version, the source commit and the seeds. Anything this script cannot
# fetch for you is listed in manifest.json.
set -euo pipefail

BUNDLE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

echo "Experiment:    {name}"
echo "Identifier:    {experiment_id}"
echo "Specification: $BUNDLE/experiment.yaml"
echo

# 1. Recreate the workspace (safe to run in an existing one).
fabric init --quiet || true

# 2. Re-execute every run declared by the specification.
fabric run "$BUNDLE/experiment.yaml"

# 3. Compare what you got against what was originally recorded.
fabric reproduce "$BUNDLE"

# 4. Regenerate the report from your own results.
fabric report {experiment_id} --format html
"""


@dataclass
class BundleManifest:
    """What a bundle contains and what it should hash to."""

    version: str = BUNDLE_VERSION
    experiment_id: str = ""
    name: str = ""
    research_question: str = ""
    spec_digest: str = ""
    created_at: float = field(default_factory=time.time)
    files: dict[str, str] = field(default_factory=dict)
    entities: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    certificate_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_version": self.version,
            "experiment_id": self.experiment_id,
            "name": self.name,
            "research_question": self.research_question,
            "spec_digest": self.spec_digest,
            "created_at": self.created_at,
            "files": self.files,
            "entities": self.entities,
            "environment": self.environment,
            "hardware": self.hardware,
            "counts": self.counts,
            "certificate_id": self.certificate_id,
        }


def export(
    workspace: Workspace,
    plan: ExperimentPlan,
    runs: Sequence[Run],
    destination: str,
    *,
    analysis: Any = None,
    provenance: Any = None,
    certificate: Any = None,
    report_markdown: str = "",
    include_artifacts: bool = True,
    environment: Any = None,
    hardware: Any = None,
) -> BundleManifest:
    """Write a portable reproduction bundle."""
    destination = os.path.abspath(destination)
    os.makedirs(destination, exist_ok=True)
    experiment_id = plan.experiment.id

    spec_dict = plan.spec.raw or plan.spec.to_dict()
    atomic_write_text(os.path.join(destination, "experiment.yaml"), dump_yaml(spec_dict))

    write_json(os.path.join(destination, "results.json"), {
        "experiment_id": experiment_id,
        "spec_digest": plan.spec.digest,
        "conditions": [c.to_dict() for c in plan.conditions],
        "runs": [r.to_dict() for r in runs],
    })

    if analysis is not None:
        write_json(os.path.join(destination, "analysis.json"), analysis.to_dict())
    if provenance is not None:
        write_json(os.path.join(destination, "provenance.json"), provenance.to_dict())
    if certificate is not None:
        write_json(os.path.join(destination, "certificate.json"), certificate.to_dict())
    if report_markdown:
        atomic_write_text(os.path.join(destination, "report.md"), report_markdown)

    if include_artifacts:
        _copy_artifacts(workspace, experiment_id, runs, os.path.join(destination, "artifacts"))

    atomic_write_text(
        os.path.join(destination, "reproduce.sh"),
        REPRODUCE_SH.format(name=plan.spec.name, experiment_id=experiment_id),
    )
    try:
        os.chmod(os.path.join(destination, "reproduce.sh"), 0o755)
    except OSError:  # pragma: no cover - filesystem dependent
        pass

    manifest = BundleManifest(
        experiment_id=experiment_id,
        name=plan.spec.name,
        research_question=plan.spec.research_question,
        spec_digest=plan.spec.digest,
        entities={
            "dataset": plan.dataset.to_dict() if plan.dataset else None,
            "model": plan.model.to_dict() if plan.model else None,
            "algorithm": plan.algorithm.to_dict() if plan.algorithm else None,
        },
        environment=environment.to_dict() if environment is not None else {},
        hardware=hardware.to_dict() if hardware is not None else {},
        counts={
            "conditions": len(plan.conditions),
            "runs": len(runs),
            "successful": sum(1 for r in runs if r.succeeded),
            "failed": sum(1 for r in runs if r.status.value == "failed"),
            "seeds": len(plan.spec.seeds.values),
        },
        certificate_id=getattr(certificate, "certificate_id", "") if certificate else "",
    )

    _write_readme(destination, plan, analysis, manifest)

    # Hash every file last, so the manifest covers the finished bundle.
    manifest.files = _hash_tree(destination)
    write_json(os.path.join(destination, "manifest.json"), manifest.to_dict())
    return manifest


def _copy_artifacts(workspace: Workspace, experiment_id: str, runs: Sequence[Run],
                    destination: str) -> None:
    for run in runs:
        source = workspace.artifacts_dir(experiment_id, run.id)
        if not os.path.isdir(source):
            continue
        target = os.path.join(destination, run.id)
        os.makedirs(target, exist_ok=True)
        for name in os.listdir(source):
            source_path = os.path.join(source, name)
            if os.path.isfile(source_path):
                shutil.copy2(source_path, os.path.join(target, name))


def _write_readme(destination: str, plan: ExperimentPlan, analysis: Any,
                  manifest: BundleManifest) -> None:
    headline = getattr(analysis, "headline", "") if analysis else ""
    limitations = getattr(analysis, "limitations", []) if analysis else []
    summary_lines = []
    if analysis is not None and getattr(analysis, "results", None) is not None:
        summary_lines.append(
            f"- {manifest.counts['runs']} runs across "
            f"{manifest.counts['conditions']} conditions and "
            f"{manifest.counts['seeds']} seeds"
        )
        if getattr(analysis, "determinism", None) and analysis.determinism.tested:
            summary_lines.append(f"- {analysis.determinism.describe()}")
        if getattr(analysis, "reproducibility", None) and analysis.reproducibility.measured:
            summary_lines.append(f"- {analysis.reproducibility.describe()}")
    atomic_write_text(os.path.join(destination, "README.md"), README.format(
        name=plan.spec.name,
        experiment_id=plan.experiment.id,
        bundle_name=os.path.basename(destination),
        research_question=plan.spec.research_question.strip(),
        headline=headline or "_No analysis was attached to this bundle._",
        summary="\n".join(summary_lines) or "_Not analysed._",
        limitations="\n".join(f"- {l}" for l in limitations) or "_None recorded._",
        generated=time.strftime("%Y-%m-%d", time.gmtime()),
        version=BUNDLE_VERSION,
    ))


def _hash_tree(root: str) -> dict[str, str]:
    files: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(dirnames)
        for name in sorted(filenames):
            if name == "manifest.json":
                continue
            path = os.path.join(dirpath, name)
            relative = os.path.relpath(path, root).replace(os.sep, "/")
            files[relative] = file_digest(path)
    return files


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------


@dataclass
class VerificationResult:
    """The outcome of verifying a bundle against its manifest."""

    path: str
    ok: bool = True
    checked: int = 0
    missing: list[str] = field(default_factory=list)
    altered: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    certificate: str = ""
    messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path, "ok": self.ok, "checked": self.checked,
            "missing": self.missing, "altered": self.altered, "extra": self.extra,
            "certificate": self.certificate, "messages": self.messages,
        }

    def render(self) -> str:
        lines = [f"BUNDLE VERIFICATION  {self.path}", ""]
        lines.append(f"  files checked   {self.checked}")
        lines.append(f"  missing         {len(self.missing)}")
        lines.append(f"  altered         {len(self.altered)}")
        lines.append(f"  unlisted        {len(self.extra)}")
        if self.certificate:
            lines.append(f"  certificate     {self.certificate}")
        lines.append("")
        for name in self.missing[:10]:
            lines.append(f"  MISSING  {name}")
        for name in self.altered[:10]:
            lines.append(f"  ALTERED  {name}")
        for name in self.extra[:10]:
            lines.append(f"  UNLISTED {name}")
        for message in self.messages:
            lines.append(f"  {message}")
        lines.append("")
        lines.append(f"  RESULT          {'INTACT' if self.ok else 'MODIFIED'}")
        return "\n".join(lines)


def verify(path: str) -> VerificationResult:
    """Verify a bundle against its own manifest."""
    path = os.path.abspath(path)
    result = VerificationResult(path=path)
    manifest = read_json(os.path.join(path, "manifest.json"))
    if manifest is None:
        result.ok = False
        result.messages.append("no manifest.json: this is not a Fabric bundle")
        return result

    listed = dict(manifest.get("files") or {})
    for relative, digest in sorted(listed.items()):
        full = os.path.join(path, relative.replace("/", os.sep))
        if not os.path.isfile(full):
            result.missing.append(relative)
            continue
        result.checked += 1
        if file_digest(full) != digest:
            result.altered.append(relative)

    present = set(_hash_tree(path))
    result.extra = sorted(present - set(listed))

    certificate = read_json(os.path.join(path, "certificate.json"))
    if certificate:
        from fabric.reporting.certificate import verify_certificate

        ok, message = verify_certificate(certificate)
        result.certificate = certificate.get("certificate_id", "")
        result.messages.append(message)
        if not ok:
            result.ok = False

    if result.missing or result.altered:
        result.ok = False
    if result.extra:
        result.messages.append(
            f"{len(result.extra)} file(s) are present but not listed in the manifest; "
            "they were added after the bundle was exported"
        )
    return result


def load(path: str) -> dict[str, Any]:
    """Read a bundle's recorded state."""
    path = os.path.abspath(path)
    manifest = read_json(os.path.join(path, "manifest.json"))
    if manifest is None:
        raise FileNotFoundError(f"{path} has no manifest.json; it is not a Fabric bundle")
    return {
        "path": path,
        "manifest": manifest,
        "results": read_json(os.path.join(path, "results.json"), {}),
        "analysis": read_json(os.path.join(path, "analysis.json"), {}),
        "certificate": read_json(os.path.join(path, "certificate.json"), {}),
        "spec_path": os.path.join(path, "experiment.yaml"),
    }
