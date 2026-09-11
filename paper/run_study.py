"""Run the pilot study and aggregate it into findings.

Executes every specification listed in ``study.yaml``, exports a certified
bundle for each, then reads the analyses back and answers the
pre-registered hypotheses across experiments.

This is a **pilot**, not the study in ``README.md``. Two real pipelines is
enough to demonstrate the methodology end to end and to produce genuine
per-axis numbers; it is nowhere near enough to support a claim about the
field, and the output says so on every page.

Usage::

    python paper/run_study.py                # run everything, write results
    python paper/run_study.py --analyse-only # re-aggregate what already ran

Anything requiring PyTorch is skipped with a note when it is not installed,
rather than failing, so the pilot degrades to whatever the machine can run.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "results")

sys.path.insert(0, ROOT)

from fabric.analysis.engine import analyse  # noqa: E402
from fabric.core.spec import ExperimentSpec  # noqa: E402
from fabric.core.store import Workspace  # noqa: E402
from fabric.execution.planner import plan_experiment  # noqa: E402
from fabric.execution.runner import ExecutionOptions, Runner  # noqa: E402
from fabric.util.jsonio import atomic_write_text, write_json  # noqa: E402
from fabric.util.yamlio import load_yaml  # noqa: E402


def requirement_met(requires: str) -> tuple[bool, str]:
    if not requires:
        return True, ""
    try:
        __import__(requires)
    except ImportError:
        return False, f"{requires} is not installed"
    return True, ""


def _certify(workspace: Workspace, plan, runs, analysis):
    from fabric.core.registry import Registry
    from fabric.reporting.certificate import issue

    return issue(analysis, plan, runs, Registry(workspace),
                 workspace_root=workspace.root)


def _export_bundle(workspace: Workspace, plan, runs, analysis, name: str) -> tuple[str, str]:
    """Write the bundle the findings promise, so the promise is a fact."""
    from fabric import bundle as bundle_module
    from fabric.reporting import builder
    from fabric.reporting import markdown as markdown_renderer

    destination = os.path.join(RESULTS, "bundles", name)
    document = builder.build_report(analysis, plan, runs)
    manifest = bundle_module.export(
        workspace, plan, runs, destination,
        analysis=analysis,
        certificate=_certify(workspace, plan, runs, analysis),
        report_markdown=markdown_renderer.render(document),
        include_artifacts=False,
    )
    return (os.path.relpath(destination, ROOT).replace(os.sep, "/"),
            manifest.certificate_id)


def run_one(entry: dict, workspace: Workspace, *, analyse_only: bool) -> dict | None:
    """Execute one experiment of the pilot and return its analysis."""
    spec_path = os.path.join(ROOT, entry["spec"])
    ok, reason = requirement_met(str(entry.get("requires", "")))
    if not ok:
        print(f"  skipped {entry['name']}: {reason}")
        return {"name": entry["name"], "skipped": reason}

    spec = ExperimentSpec.from_file(spec_path)
    plan = plan_experiment(spec, workspace, strict=True, root=ROOT)
    workspace.create_experiment_dir(plan.experiment.id)
    with open(spec_path, encoding="utf-8") as handle:
        workspace.write_text(plan.experiment.id, "experiment.yaml", handle.read())
    workspace.write(plan.experiment.id, "plan.json", plan.to_dict())
    workspace.register_experiment({
        "id": plan.experiment.id, "name": spec.name,
        "research_question": spec.research_question,
        "spec_digest": spec.digest, "created_at": time.time(),
        "source": spec_path, "runs": len(plan.runs),
    })

    if not analyse_only:
        started = time.time()
        options = ExecutionOptions(
            max_workers=spec.execution.max_workers,
            isolation=spec.execution.isolation,
            timeout_seconds=spec.execution.timeout_seconds,
        )
        report = Runner(workspace, plan, options, root=ROOT).execute()
        print(f"  {entry['name']}: {report.summary_line()} "
              f"({time.time() - started:.0f}s)")

    runs = workspace.load_runs(plan.experiment.id)
    if not runs:
        return {"name": entry["name"], "skipped": "no runs"}
    analysis = analyse(plan, runs, stored_spec_digest=spec.digest)
    bundle, certificate_id = _export_bundle(workspace, plan, runs, analysis,
                                            entry["name"])
    return {
        "name": entry["name"],
        "kind": entry.get("kind", ""),
        "real": bool(entry.get("real", False)),
        "experiment_id": plan.experiment.id,
        "bundle": bundle,
        "certificate_id": certificate_id,
        "analysis": analysis.to_dict(),
    }


# --------------------------------------------------------------------------
# aggregation: the pre-registered hypotheses, answered
# --------------------------------------------------------------------------


def wilson(successes: int, trials: int) -> tuple[float, float]:
    from fabric.statistics.confidence import proportion_interval

    interval = proportion_interval(successes, trials)
    return interval.low, interval.high


def aggregate(entries: list[dict]) -> dict:
    """Answer H1-H4 over the experiments that ran."""
    usable = [e for e in entries if "analysis" in e]
    real = [e for e in usable if e["real"]]

    axes_rows: list[dict] = []
    for entry in usable:
        profile = entry["analysis"]["reproducibility"]
        for axis in profile.get("axes", []):
            if axis.get("stability") is None or axis["stability"] != axis["stability"]:
                continue
            axes_rows.append({
                "experiment": entry["name"],
                "real": entry["real"],
                "axis": axis["axis"],
                "label": axis["label"],
                "metric": axis["metric"],
                "delta": axis["delta_mean"],
                "noise_multiples": axis["noise_multiples"],
                "stability": axis["stability"],
                "tolerance_verdict": axis["tolerance_verdict"],
                "permutation_p": axis.get("null_p_value"),
            })

    # H1: does any non-seed axis exceed one seed standard deviation?
    h1_rows = []
    for entry in real:
        profile = entry["analysis"]["reproducibility"]
        multiples = [a["noise_multiples"] for a in profile.get("axes", [])
                     if a.get("noise_multiples") == a.get("noise_multiples")
                     and math.isfinite(a["noise_multiples"])]
        if not multiples:
            continue
        h1_rows.append({
            "experiment": entry["name"],
            "max_noise_multiples": max(multiples),
            "exceeds_seed": max(multiples) > 1.0,
            "worst_axis": max(profile.get("axes", []),
                              key=lambda a: a.get("noise_multiples", 0) or 0)["axis"],
        })
    h1_hits = sum(1 for r in h1_rows if r["exceeds_seed"])

    # H2: which axes rank highest, by median shift in seed-noise units?
    by_axis: dict[str, list[float]] = {}
    for row in axes_rows:
        if row["real"] and math.isfinite(row["noise_multiples"]):
            by_axis.setdefault(row["axis"], []).append(row["noise_multiples"])
    h2_ranking = sorted(
        ({"axis": axis, "median_noise_multiples": statistics.median(values),
          "n": len(values)} for axis, values in by_axis.items()),
        key=lambda r: -r["median_noise_multiples"])

    # H3: determinism
    h3_rows = []
    for entry in usable:
        determinism = entry["analysis"]["determinism"]
        if not determinism.get("tested"):
            continue
        h3_rows.append({
            "experiment": entry["name"],
            "real": entry["real"],
            "deterministic": determinism["deterministic"],
            "judged_metrics": determinism.get("judged_metrics", []),
            "excluded_metrics": determinism.get("excluded_metrics", []),
        })

    # H4: decision stability
    h4_rows = []
    for entry in usable:
        seed = entry["analysis"]["reproducibility"].get("seed") or {}
        stability = seed.get("decision_stability")
        if stability is None or stability != stability:
            continue
        detail = seed.get("decision_detail", {})
        h4_rows.append({
            "experiment": entry["name"],
            "real": entry["real"],
            "decision_stability": stability,
            "full_verdict": detail.get("full_verdict"),
            "subsample_size": detail.get("subsample_size"),
            "floor_limited": detail.get("floor_limited"),
        })

    return {
        "generated_at": time.time(),
        "experiments": [{"name": e["name"], "real": e["real"],
                         "experiment_id": e.get("experiment_id", ""),
                         "kind": e.get("kind", ""),
                         "bundle": e.get("bundle", ""),
                         "certificate_id": e.get("certificate_id", "")}
                        for e in usable],
        "skipped": [{"name": e["name"], "reason": e["skipped"]}
                    for e in entries if "skipped" in e],
        "axes": axes_rows,
        "h1": {"rows": h1_rows, "hits": h1_hits, "n": len(h1_rows),
               "wilson": wilson(h1_hits, len(h1_rows)) if h1_rows else None},
        "h2": {"ranking": h2_ranking},
        "h3": {"rows": h3_rows,
               "non_deterministic": sum(1 for r in h3_rows
                                        if r["real"] and not r["deterministic"]),
               "n": sum(1 for r in h3_rows if r["real"])},
        "h4": {"rows": h4_rows},
    }


def render_findings(summary: dict) -> str:
    real = [e for e in summary["experiments"] if e["real"]]
    lines = [
        "# Pilot study results",
        "",
        "**This is a pilot, not the study.** It runs the perturbation design",
        f"end to end on {len(real)} real pipeline(s) and reports genuine per-axis",
        "numbers. It is far too small to support any claim about machine",
        "learning as a field, and nothing here should be read as one. The",
        "design it is a pilot for is in [README.md](README.md).",
        "",
        "Regenerate with `python paper/run_study.py`.",
        "",
        "## Experiments",
        "",
        "| Experiment | Kind | Counts toward findings | Bundle |",
        "|---|---|---|---|",
    ]
    for entry in summary["experiments"]:
        lines.append(f"| {entry['name']} | {entry['kind']} | "
                     f"{'yes' if entry['real'] else 'no (simulated)'} | "
                     f"`{entry.get('bundle') or '-'}` |")
    for entry in summary["skipped"]:
        lines.append(f"| {entry['name']} | - | no ({entry['reason']}) |")
    lines.append("")
    lines.append("Simulated pipelines exercise the instrument and are excluded from")
    lines.append("every finding below, as the protocol requires.")
    lines.append("")

    # --- axes ---
    lines += ["## Every measured axis", "",
              "| Experiment | Axis | Perturbation | Metric | Δ | × seed noise | Stability | Tolerance |",
              "|---|---|---|---|---:|---:|---:|---|"]
    for row in sorted(summary["axes"],
                      key=lambda r: (-(r["noise_multiples"]
                                       if math.isfinite(r["noise_multiples"]) else 0))):
        if not row["real"]:
            continue
        multiples = ("inf" if not math.isfinite(row["noise_multiples"])
                     else f"{row['noise_multiples']:.2f}")
        lines.append(
            f"| {row['experiment']} | {row['axis']} | {row['label']} | {row['metric']} "
            f"| {row['delta']:+.4f} | {multiples} | {row['stability']:.2f} "
            f"| {row['tolerance_verdict']} |")
    lines.append("")

    # --- H1 ---
    h1 = summary["h1"]
    lines += ["## H1 — does a non-seed axis exceed the seed noise?", ""]
    if h1["n"]:
        low, high = h1["wilson"]
        lines.append(f"{h1['hits']} of {h1['n']} real pipelines had at least one "
                     f"non-seed axis above one seed standard deviation "
                     f"(Wilson 95% interval [{low:.2f}, {high:.2f}]).")
        lines.append("")
        lines += ["| Experiment | Largest shift (× seed noise) | Axis | Exceeds seed |",
                  "|---|---:|---|---|"]
        for row in h1["rows"]:
            lines.append(f"| {row['experiment']} | {row['max_noise_multiples']:.2f} "
                         f"| {row['worst_axis']} | {'yes' if row['exceeds_seed'] else 'no'} |")
        lines.append("")
        lines.append("With this many pipelines the interval spans most of the unit "
                     "range, which is the honest reading: the pilot demonstrates the "
                     "measurement, not the prevalence.")
    else:
        lines.append("No real pipeline produced a measurable axis.")
    lines.append("")

    # --- H2 ---
    lines += ["## H2 — which axes dominate?", "",
              "| Axis | Median shift (× seed noise) | Measurements |", "|---|---:|---:|"]
    for row in summary["h2"]["ranking"]:
        lines.append(f"| {row['axis']} | {row['median_noise_multiples']:.2f} | {row['n']} |")
    lines.append("")

    # --- H3 ---
    h3 = summary["h3"]
    lines += ["## H3 — is a fixed seed enough for determinism?", ""]
    if h3["n"]:
        lines.append(f"{h3['non_deterministic']} of {h3['n']} real pipelines were "
                     "non-deterministic on their result metrics under a fixed seed.")
        lines.append("")
        lines += ["| Experiment | Counts | Deterministic | Judged on "
                  "| Excluded (measures the machine) |",
                  "|---|---|---|---|---|"]
        for row in h3["rows"]:
            lines.append(
                f"| {row['experiment']} | {'yes' if row['real'] else 'no'} "
                f"| {'yes' if row['deterministic'] else 'NO'} "
                f"| {', '.join(row['judged_metrics']) or '-'} "
                f"| {', '.join(row['excluded_metrics']) or '-'} |")
    else:
        lines.append("No pipeline declared determinism replicates.")
    lines.append("")

    # --- H4 ---
    h4 = summary["h4"]
    real_rows = sum(1 for r in h4["rows"] if r["real"])
    floor_limited = sum(1 for r in h4["rows"] if r["real"] and r["floor_limited"])
    lines += ["## H4 — do conclusions survive fewer seeds?", ""]
    if real_rows:
        lines.append(
            f"In {floor_limited} of {real_rows} real pipelines, halving the seed count "
            "puts significance out of reach *arithmetically*: an exact test on five "
            "paired runs cannot produce a p-value below 0.0625, whatever it measures. "
            "The agreement rate in those rows therefore reports the resolution of the "
            "design rather than the robustness of the effect, which is itself the "
            "finding.")
        lines.append("")
    lines += ["| Experiment | Counts | Full verdict | Subsample | Same conclusion "
              "| Floor-limited |",
              "|---|---|---|---:|---:|---|"]
    for row in h4["rows"]:
        lines.append(
            f"| {row['experiment']} | {'yes' if row['real'] else 'no'} "
            f"| {row['full_verdict']} | {row['subsample_size']} "
            f"| {row['decision_stability'] * 100:.0f}% "
            f"| {'yes' if row['floor_limited'] else 'no'} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("Generated by `paper/run_study.py` from the analyses Fabric produced. "
                 "Every number is reproducible with `fabric reproduce` against the "
                 "corresponding bundle.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the pilot reproducibility study.")
    parser.add_argument("--analyse-only", action="store_true",
                        help="re-aggregate without re-executing")
    args = parser.parse_args()

    manifest = load_yaml(os.path.join(HERE, "study.yaml"))
    workspace = Workspace.discover(ROOT)
    os.makedirs(RESULTS, exist_ok=True)

    print(f"pilot study: {len(manifest['experiments'])} experiment(s)")
    entries = []
    for entry in manifest["experiments"]:
        result = run_one(entry, workspace, analyse_only=args.analyse_only)
        if result:
            entries.append(result)

    summary = aggregate(entries)
    write_json(os.path.join(RESULTS, "findings.json"), summary)
    atomic_write_text(os.path.join(RESULTS, "findings.md"), render_findings(summary))
    print(f"\nwrote {os.path.join(RESULTS, 'findings.md')}")
    print(f"wrote {os.path.join(RESULTS, 'findings.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
