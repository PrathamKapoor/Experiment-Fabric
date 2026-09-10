"""The ``fabric`` command line.

Every command is a verb a researcher already has a word for: plan, run,
analyze, report, why, reproduce, verify, certify, export.  Errors are
written to be actionable -- they say what was wrong and what to do about
it, because a tool used at 2am before a deadline should not require
reading its source.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from typing import Any

from fabric import __version__
from fabric.analysis.engine import ExperimentAnalysis, analyse
from fabric.cli import console
from fabric.core.entities import Algorithm, Dataset, Model, Role, Run, RunStatus
from fabric.core.graph import ExperimentGraph
from fabric.core.provenance import build_provenance, explain, find_metric_node, impact
from fabric.core.registry import Registry, RegistryError
from fabric.core.spec import ExperimentSpec, SpecError
from fabric.core.store import Workspace, WorkspaceError
from fabric.environments.capture import capture_environment
from fabric.environments.hardware import capture_hardware
from fabric.execution.cache import CachePolicy
from fabric.execution.planner import ExperimentPlan, PlanError, plan_experiment
from fabric.execution.runner import ExecutionOptions, Runner
from fabric.util.hashing import tree_digest
from fabric.util.jsonio import CorruptRecordError, atomic_write_text, read_json, write_json


class CommandError(RuntimeError):
    """An error with a message already written for a human."""


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------


def _workspace(args: argparse.Namespace) -> Workspace:
    try:
        return Workspace.discover(getattr(args, "workspace", None) or os.getcwd())
    except WorkspaceError as exc:
        raise CommandError(str(exc)) from exc


def _load_spec(path: str) -> ExperimentSpec:
    if not os.path.isfile(path):
        raise CommandError(f"no such specification: {path}")
    try:
        return ExperimentSpec.from_file(path)
    except SpecError as exc:
        raise CommandError(f"invalid specification: {exc}") from exc


def _plan(workspace: Workspace, spec: ExperimentSpec, *, strict: bool = True) -> ExperimentPlan:
    try:
        return plan_experiment(spec, workspace, strict=strict, root=workspace.root)
    except (PlanError, RegistryError, SpecError) as exc:
        raise CommandError(str(exc)) from exc


def _stored_plan(workspace: Workspace, experiment_id: str) -> ExperimentPlan:
    """Rebuild the plan of an already-executed experiment from its stored spec."""
    spec_path = os.path.join(workspace.experiment_dir(experiment_id), "experiment.yaml")
    if not os.path.isfile(spec_path):
        raise CommandError(
            f"{experiment_id} has no stored specification. It may have been created by "
            "an older version of Fabric; rerun it from its YAML file."
        )
    spec = _load_spec(spec_path)
    return _plan(workspace, spec, strict=False)


def _resolve(workspace: Workspace, reference: str) -> str:
    try:
        return workspace.resolve_experiment(reference)
    except WorkspaceError as exc:
        raise CommandError(str(exc)) from exc


def _analysis_for(workspace: Workspace, experiment_id: str) -> tuple[
        ExperimentPlan, list[Run], ExperimentAnalysis]:
    plan = _stored_plan(workspace, experiment_id)
    runs = workspace.load_runs(experiment_id)
    if not runs:
        raise CommandError(
            f"{experiment_id} has no runs yet. Execute it with 'fabric run' first."
        )
    stored = workspace.read(experiment_id, "plan.json", {}) or {}
    analysis = analyse(plan, runs, stored_spec_digest=str(stored.get("spec_digest", "")))
    return plan, runs, analysis


def _cached_analysis(workspace: Workspace, experiment_id: str) -> tuple[
        ExperimentPlan, list[Run], ExperimentAnalysis]:
    plan, runs, analysis = _analysis_for(workspace, experiment_id)
    workspace.write(experiment_id, "analysis.json", analysis.to_dict())
    return plan, runs, analysis


def _provenance(workspace: Workspace, plan: ExperimentPlan, runs: Sequence[Run],
                analysis: Any = None) -> ExperimentGraph:
    registry = Registry(workspace)
    environments = {}
    hardware = {}
    for run in runs:
        if run.environment_id and run.environment_id not in environments:
            try:
                entity = registry.get("environment", run.environment_id)
                environments[run.environment_id] = (
                    f"python {entity.python_version} on {entity.os_name}")
            except RegistryError:
                pass
        if run.hardware_id and run.hardware_id not in hardware:
            try:
                entity = registry.get("hardware", run.hardware_id)
                hardware[run.hardware_id] = entity.label
            except RegistryError:
                pass
    return build_provenance(plan, list(runs), analysis,
                            environments=environments, hardware=hardware)


# --------------------------------------------------------------------------
# init
# --------------------------------------------------------------------------


def cmd_init(args: argparse.Namespace) -> int:
    root = os.path.abspath(args.path or os.getcwd())
    existing = os.path.isdir(os.path.join(root, ".fabric"))
    if existing and not args.force:
        if args.quiet:
            return 0
        console.warn(f"a workspace already exists at {os.path.join(root, '.fabric')}")
        console.note("pass --force to reinitialise it (this does not delete experiments)")
        return 0
    workspace = Workspace.initialise(root, name=args.name or "", force=True)
    if args.quiet:
        return 0
    console.heading("workspace initialised")
    console.field("location", workspace.path)
    console.field("name", workspace.config.get("name", ""))
    console.echo("")
    console.note("next: write an experiment.yaml, then run")
    console.note("  fabric run experiment.yaml")
    return 0


# --------------------------------------------------------------------------
# register
# --------------------------------------------------------------------------


def cmd_register(args: argparse.Namespace) -> int:
    workspace = _workspace(args)
    registry = Registry(workspace)
    kind = args.kind

    if kind == "dataset":
        digest = args.digest or ""
        local_path = os.path.abspath(args.path) if args.path else ""
        if local_path and not digest:
            if not os.path.exists(local_path):
                raise CommandError(f"no such path: {local_path}")
            digest = tree_digest(local_path)
        entity: Any = Dataset(
            name=args.name, version=args.version, digest=digest,
            source=args.source or "", preprocessing=args.preprocessing or "",
            n_examples=args.examples, license=args.license or "",
            notes=args.notes or "", local_path=local_path,
        )
    elif kind == "model":
        weights = args.weights_digest or ""
        if args.path and not weights:
            if not os.path.exists(args.path):
                raise CommandError(f"no such path: {args.path}")
            weights = tree_digest(args.path)
        entity = Model(
            name=args.name, version=args.version, weights_digest=weights,
            architecture=args.architecture or "", commit=args.commit or "",
            precision=args.precision or "", quantization=args.quantization or "none",
            tokenizer=args.tokenizer or "", parameters=args.parameters,
            provider=args.provider or "",
            local_path=os.path.abspath(args.path) if args.path else "",
        )
    else:
        from fabric.environments.capture import git_state
        from fabric.execution.adapters import entrypoint_digest

        commit = args.commit or ""
        if commit in ("", "auto"):
            detected, dirty, _ = git_state(workspace.root)
            commit = detected + ("-dirty" if dirty and detected else "")
        entity = Algorithm(
            name=args.name, entrypoint=args.entrypoint or "", commit=commit,
            source_digest=entrypoint_digest(args.entrypoint or "", root=workspace.root),
            version=args.version, description=args.description or "",
        )

    try:
        stored = registry.register(kind, entity, force=args.force)
    except RegistryError as exc:
        raise CommandError(str(exc)) from exc

    console.heading(f"{kind} registered")
    console.field("reference", getattr(stored, "ref", stored.id))
    console.field("identifier", stored.id)
    for attribute in ("digest", "weights_digest", "source_digest", "commit"):
        value = getattr(stored, attribute, "")
        if value:
            console.field(attribute, value)
    return 0


def cmd_registry(args: argparse.Namespace) -> int:
    workspace = _workspace(args)
    registry = Registry(workspace)
    if args.action == "verify":
        results = registry.verify_all()
        failures = [r for r in results if not r[2]]
        console.heading("registry verification")
        console.field("entities", str(len(results)))
        console.field("failures", str(len(failures)))
        for kind, entity_id, _ok, message in failures:
            console.error(f"{kind} {entity_id}: {message}")
        if not failures:
            console.success("every stored entity hashes to its own identifier")
        return 1 if failures else 0

    entries = registry.index_entries(args.kind)
    if not entries:
        console.note("nothing registered yet")
        return 0
    console.heading("registry")
    console.table(
        ["kind", "reference", "identifier", "summary"],
        [[e.get("kind", ""),
          f"{e.get('name', '')}@{e.get('version', '')}" if e.get("version") else e.get("name", ""),
          e.get("id", ""), e.get("summary", "")]
         for e in entries],
    )
    return 0


# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------


def _print_plan(plan: ExperimentPlan, *, verbose: bool = False) -> None:
    console.heading("experiment plan")
    for line in plan.summary_lines():
        label, _, value = line.partition("  ")
        console.field(label.strip(), value.strip())

    counts: dict[str, int] = {}
    for condition in plan.conditions:
        counts[condition.role.value] = counts.get(condition.role.value, 0) + 1

    console.heading("conditions")
    console.table(
        ["condition", "role", "removed", "configuration"],
        [[c.name, c.role.value, ", ".join(c.ablated) or "-",
          ", ".join(f"{k}={v}" for k, v in sorted(c.config.items())) or "-"]
         for c in plan.conditions],
    )

    if plan.dataset or plan.model or plan.algorithm:
        console.heading("inputs")
        if plan.dataset:
            console.field("dataset", f"{plan.dataset.ref}  {plan.dataset.digest[:23] or 'no digest'}")
        if plan.model:
            console.field("model", f"{plan.model.ref}  {plan.model.id}")
        if plan.algorithm:
            console.field("algorithm", f"{plan.algorithm.ref}  {plan.algorithm.entrypoint}")

    if plan.advisories:
        console.heading("advisories")
        for advisory in plan.advisories:
            console.warn(advisory)

    if verbose:
        console.heading("runs")
        console.table(
            ["run", "condition", "seed", "replicate"],
            [[r.id, r.condition_name, str(r.seed), str(r.replicate)] for r in plan.runs],
            aligns=["left", "left", "right", "right"],
        )


def cmd_plan(args: argparse.Namespace) -> int:
    workspace = _workspace(args)
    spec = _load_spec(args.spec)
    plan = _plan(workspace, spec, strict=not args.no_check)
    _print_plan(plan, verbose=args.verbose)
    console.echo("")
    console.note(f"nothing was executed. Run it with: fabric run {args.spec}")
    return 0


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    workspace = _workspace(args)
    spec = _load_spec(args.spec)
    plan = _plan(workspace, spec)

    if args.dry_run:
        _print_plan(plan, verbose=args.verbose)
        console.echo("")
        console.note("dry run: nothing was executed")
        return 0

    execution = spec.execution
    options = ExecutionOptions(
        max_workers=args.workers or execution.max_workers,
        timeout_seconds=args.timeout if args.timeout is not None else execution.timeout_seconds,
        isolation=args.isolation or execution.isolation,
        retries=execution.retries,
        continue_on_failure=not args.stop_on_failure and execution.continue_on_failure,
        cache=CachePolicy(
            enabled=execution.cache and not args.no_cache,
            strict_environment=args.strict_environment,
        ),
        track_gpu=args.track_gpu,
        cost_per_gpu_hour=args.gpu_cost,
        cost_per_cpu_hour=args.cpu_cost,
        only=tuple(args.only or ()),
        rerun=args.rerun,
    )

    workspace.create_experiment_dir(plan.experiment.id)
    workspace.write_text(plan.experiment.id, "experiment.yaml",
                         _spec_text(args.spec))
    workspace.write(plan.experiment.id, "plan.json", plan.to_dict())
    workspace.write(plan.experiment.id, "graph.json", plan.graph.to_dict())
    workspace.register_experiment({
        "id": plan.experiment.id,
        "name": plan.experiment.name,
        "research_question": plan.experiment.research_question,
        "spec_digest": plan.spec.digest,
        "created_at": plan.experiment.created_at,
        "source": os.path.abspath(args.spec),
        "runs": len(plan.runs),
    })

    console.heading(f"running {plan.experiment.name}")
    console.field("experiment", plan.experiment.id)
    console.field("conditions", str(len(plan.conditions)))
    console.field("runs", str(len(plan.runs)))
    console.field("workers", str(options.max_workers))
    console.field("isolation", options.isolation)
    console.field("cache", "on" if options.cache.enabled else "off")
    for advisory in plan.advisories:
        console.warn(advisory)
    console.echo("")

    runner = Runner(workspace, plan, options, root=workspace.root)
    pending = runner.pending_runs()
    if not pending:
        console.success("every run is already complete; nothing to do")
        console.note(f"force re-execution with: fabric run {args.spec} --rerun")
        return 0

    progress = console.Progress(len(pending))
    failures: list[Run] = []

    def on_progress(event: Any) -> None:
        failed = event.run.status == RunStatus.FAILED
        if failed:
            failures.append(event.run)
        progress.advance(event.label, failed=failed)

    report = runner.execute(pending, on_progress=on_progress)
    progress.finish()

    console.echo("")
    console.success(report.summary_line())
    workspace.write(plan.experiment.id, "execution.json", report.to_dict())

    if failures:
        console.echo("")
        console.warn(f"{len(failures)} run(s) failed")
        console.table(
            ["condition", "seed", "class", "message"],
            [[r.condition_name, str(r.seed),
              r.failure.failure_class.value if r.failure else "",
              (r.failure.message.splitlines()[0][:80] if r.failure and r.failure.message else "")]
             for r in failures[:10]],
            aligns=["left", "right", "left", "left"],
        )
        console.note("failures are recorded as data; see 'fabric analyze' for the "
                     "failure analysis")

    console.echo("")
    console.note(f"analyse it with:  fabric analyze {plan.experiment.id}")
    console.note(f"report it with:   fabric report {plan.experiment.id} --format html")
    return 0


def _spec_text(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


# --------------------------------------------------------------------------
# ls / status
# --------------------------------------------------------------------------


def cmd_ls(args: argparse.Namespace) -> int:
    workspace = _workspace(args)
    experiments = workspace.list_experiments()
    if not experiments:
        console.note("no experiments yet. Create one with 'fabric run experiment.yaml'")
        return 0
    console.heading("experiments")
    rows = []
    for entry in experiments[: args.limit]:
        runs = workspace.load_runs(str(entry["id"]))
        completed = sum(1 for r in runs if r.succeeded)
        failed = sum(1 for r in runs if r.status == RunStatus.FAILED)
        rows.append([
            entry.get("id", ""),
            entry.get("name", ""),
            f"{completed}/{entry.get('runs', len(runs))}",
            str(failed),
            time.strftime("%Y-%m-%d %H:%M", time.localtime(entry.get("created_at", 0))),
        ])
    console.table(["id", "name", "complete", "failed", "created"], rows,
                  aligns=["left", "left", "right", "right", "left"])
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    workspace = _workspace(args)
    experiment_id = _resolve(workspace, args.experiment)
    plan = _stored_plan(workspace, experiment_id)
    runs = {r.id: r for r in workspace.load_runs(experiment_id)}

    console.heading(plan.experiment.name)
    console.field("experiment", experiment_id)
    console.field("question", plan.experiment.research_question[:88])
    console.field("specification", plan.spec.digest[:31])

    by_condition: dict[str, dict[str, int]] = {}
    for planned in plan.runs:
        bucket = by_condition.setdefault(planned.condition_name,
                                         {"planned": 0, "done": 0, "failed": 0})
        bucket["planned"] += 1
        actual = runs.get(planned.id)
        if actual is None:
            continue
        if actual.succeeded:
            bucket["done"] += 1
        elif actual.status == RunStatus.FAILED:
            bucket["failed"] += 1

    console.heading("progress")
    rows = []
    for name, bucket in by_condition.items():
        fraction = bucket["done"] / bucket["planned"] if bucket["planned"] else 0.0
        rows.append([
            name, console.bar(fraction), f"{bucket['done']}/{bucket['planned']}",
            str(bucket["failed"]) if bucket["failed"] else "",
        ])
    console.table(["condition", "", "complete", "failed"], rows,
                  aligns=["left", "left", "right", "right"])

    total_done = sum(b["done"] for b in by_condition.values())
    total = sum(b["planned"] for b in by_condition.values())
    console.echo("")
    if total_done == total:
        console.success(f"all {total} runs complete")
    else:
        console.note(f"{total_done}/{total} runs complete; "
                     f"resume with 'fabric run <spec>' (finished runs are skipped)")
    return 0


# --------------------------------------------------------------------------
# analyze
# --------------------------------------------------------------------------


def cmd_analyze(args: argparse.Namespace) -> int:
    workspace = _workspace(args)
    experiment_id = _resolve(workspace, args.experiment)
    _plan, _runs, analysis = _cached_analysis(workspace, experiment_id)

    if args.json:
        print(json.dumps(analysis.to_dict(), indent=2, default=str))
        return 0

    console.heading(analysis.name)
    console.echo(f"  {analysis.research_question}")
    console.echo("")
    console.echo(f"  {console.paint(analysis.headline, 'bold')}")

    metric = analysis.primary_metric
    results = analysis.results
    if results is not None:
        console.heading(f"results ({metric})")
        rows = []
        for condition in results.conditions.values():
            if not condition.successful or condition.perturbation:
                continue
            summary = condition.summary(metric)
            rows.append([
                condition.name, condition.role.value, str(summary.n),
                f"{summary.mean:.4f}", f"{summary.std:.4f}",
                f"{summary.minimum:.4f}", f"{summary.maximum:.4f}",
            ])
        console.table(["condition", "role", "n", "mean", "sd", "min", "max"], rows,
                      aligns=["left", "left", "right", "right", "right", "right", "right"])

    comparisons = analysis.baselines.for_metric(metric)
    if comparisons:
        console.heading("baseline comparisons")
        adjusted = any(c.adjusted_p is not None for c in comparisons)
        console.table(
            ["baseline", "difference", f"{round((1 - analysis.alpha) * 100)}% CI",
             "test", "adj. p" if adjusted else "p", "g", "verdict"],
            [[c.baseline, f"{c.difference:+.4f}", c.difference_interval.format(),
              c.test.test, _p(c.effective_p),
              f"{(c.effect('hedges_g').value if c.effect('hedges_g') else float('nan')):.1f}",
              c.verdict]
             for c in comparisons],
            aligns=["left", "right", "left", "left", "right", "right", "left"],
        )

    ranking = analysis.ablations.ranking(metric)
    if ranking:
        console.heading("ablations")
        console.table(
            ["component", "", f"drop in {metric}", "detectable"],
            [[name, console.bar(abs(delta), max(abs(d) for _, d, _ in ranking) or 1.0),
              f"{delta:+.4f}", "yes" if significant else "no"]
             for name, delta, significant in ranking],
            aligns=["left", "left", "right", "left"],
        )

    if analysis.sweep.for_metric(metric):
        console.heading("parameter sensitivity")
        for effect in analysis.sweep.for_metric(metric):
            console.bullet(effect.describe())

    if analysis.failures.failed_runs:
        console.heading("failures")
        console.field("rate", f"{analysis.failures.failure_rate * 100:.1f}%")
        for name, count in analysis.failures.by_class.items():
            console.field(name, str(count))
        for note in analysis.failures.notes:
            console.bullet(note)

    console.heading("reproducibility")
    console.bullet(analysis.determinism.describe())
    if analysis.reproducibility.seed:
        console.bullet(analysis.reproducibility.seed.describe())
    if analysis.reproducibility.measured:
        console.echo("")
        console.table(
            ["axis", "perturbation", "", "stability", "noise", "delta", "tolerance"],
            [[a.axis, a.label[:26], console.bar(a.stability, size=12),
              f"{a.stability:.2f}", f"{a.noise_multiples:.1f}x",
              f"{a.delta_mean:+.4f}", a.tolerance_verdict]
             for a in sorted(analysis.reproducibility.axes, key=lambda x: x.stability)],
            aligns=["left", "left", "left", "right", "right", "right", "left"],
        )
        console.echo("")
        console.bullet(analysis.reproducibility.describe())

    console.heading("integrity")
    for finding in analysis.integrity.findings:
        console.echo(f"  {console.status_mark(finding.level):<18} {finding.title}")

    if analysis.limitations:
        console.heading("limitations")
        for limitation in analysis.limitations:
            console.bullet(limitation)

    console.echo("")
    console.note(f"full report:  fabric report {experiment_id} --format html")
    console.note(f"trace a number:  fabric why {experiment_id} --metric {metric}")
    return 0


def _p(value: float | None) -> str:
    if value is None or value != value:
        return "-"
    return "<0.0001" if value < 0.0001 else f"{value:.4f}"


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def cmd_report(args: argparse.Namespace) -> int:
    from fabric.reporting import builder
    from fabric.reporting import html as html_renderer
    from fabric.reporting import latex as latex_renderer
    from fabric.reporting import markdown as markdown_renderer

    workspace = _workspace(args)
    experiment_id = _resolve(workspace, args.experiment)
    plan, runs, analysis = _cached_analysis(workspace, experiment_id)
    graph = _provenance(workspace, plan, runs, analysis)
    workspace.write(experiment_id, "provenance.json", graph.to_dict())

    registry = Registry(workspace)
    environment = None
    hardware = None
    for run in runs:
        if run.environment_id and environment is None:
            try:
                environment = registry.get("environment", run.environment_id)
            except RegistryError:
                pass
        if run.hardware_id and hardware is None:
            try:
                hardware = registry.get("hardware", run.hardware_id)
            except RegistryError:
                pass

    document = builder.build_report(analysis, plan, runs, graph=graph,
                                    environment=environment, hardware=hardware)

    formats = ["markdown", "html", "latex", "json"] if args.format == "all" else [args.format]
    reports_dir = args.out or workspace.reports_dir(experiment_id)
    os.makedirs(reports_dir, exist_ok=True)
    written: list[str] = []

    for fmt in formats:
        if fmt in ("md", "markdown"):
            path = os.path.join(reports_dir, "report.md")
            atomic_write_text(path, markdown_renderer.render(document))
            written.append(path)
        elif fmt == "html":
            path = os.path.join(reports_dir, "report.html")
            atomic_write_text(path, html_renderer.render(document))
            written.append(path)
        elif fmt in ("tex", "latex"):
            path = os.path.join(reports_dir, "report.tex")
            atomic_write_text(path, latex_renderer.render(document))
            written.append(path)
            tables_dir = os.path.join(reports_dir, "tables")
            os.makedirs(tables_dir, exist_ok=True)
            for label, snippet in latex_renderer.render_tables(document).items():
                table_path = os.path.join(tables_dir, f"{label}.tex")
                atomic_write_text(table_path, snippet + "\n")
                written.append(table_path)
        elif fmt == "json":
            path = os.path.join(reports_dir, "report.json")
            write_json(path, document.to_dict())
            written.append(path)
        else:
            raise CommandError(f"unknown report format: {fmt}")

    console.heading("report written")
    for path in written[:8]:
        console.field("", os.path.relpath(path, workspace.root))
    if len(written) > 8:
        console.note(f"and {len(written) - 8} more (LaTeX table snippets)")

    if args.open:
        html_path = next((p for p in written if p.endswith(".html")), None)
        if html_path:
            import webbrowser

            webbrowser.open("file://" + os.path.abspath(html_path))
    return 0


# --------------------------------------------------------------------------
# why / graph / impact
# --------------------------------------------------------------------------


def cmd_why(args: argparse.Namespace) -> int:
    workspace = _workspace(args)
    experiment_id = _resolve(workspace, args.experiment)
    plan = _stored_plan(workspace, experiment_id)
    runs = workspace.load_runs(experiment_id)
    if not runs:
        raise CommandError(f"{experiment_id} has no runs yet")
    graph = _provenance(workspace, plan, runs)

    target = args.node
    if not target:
        metric = args.metric or plan.spec.primary_metric
        condition = args.condition
        if not condition:
            successful = [r for r in runs if r.succeeded and r.role == Role.TREATMENT]
            condition = successful[0].condition_name if successful else (
                runs[0].condition_name if runs else "")
        node = find_metric_node(graph, condition=condition, metric=metric, seed=args.seed)
        if node is None:
            raise CommandError(
                f"no recorded value of '{metric}' for condition '{condition}'"
                + (f" at seed {args.seed}" if args.seed is not None else "")
                + ". Check 'fabric analyze' for what exists."
            )
        target = node.id

    explanation = explain(graph, target)
    if args.json:
        print(json.dumps(explanation.to_dict(), indent=2, default=str))
        return 0
    console.echo("")
    console.echo(explanation.render())
    console.echo("")
    console.note(f"ancestry tree:  fabric graph {experiment_id} --format ascii --node {target}")
    return 0


def cmd_graph(args: argparse.Namespace) -> int:
    workspace = _workspace(args)
    experiment_id = _resolve(workspace, args.experiment)
    plan = _stored_plan(workspace, experiment_id)
    runs = workspace.load_runs(experiment_id)
    graph = _provenance(workspace, plan, runs)

    if args.node and args.node not in graph:
        raise CommandError(f"no node {args.node} in the provenance graph")

    if args.format == "ascii":
        node = args.node or (graph.nodes_of_kind("claim") or graph.leaves)[0].id
        text = graph.to_ascii(node, depth=args.depth)
    elif args.format == "dot":
        text = (graph.lineage(args.node) if args.node else graph).to_dot()
    elif args.format == "mermaid":
        text = (graph.lineage(args.node) if args.node else graph).to_mermaid()
    else:
        text = json.dumps((graph.lineage(args.node) if args.node else graph).to_dict(),
                          indent=2, default=str)

    if args.out:
        atomic_write_text(args.out, text + "\n")
        console.success(f"written to {args.out}")
    else:
        console.echo(text)
    return 0


def cmd_impact(args: argparse.Namespace) -> int:
    workspace = _workspace(args)
    experiment_id = _resolve(workspace, args.experiment)
    plan = _stored_plan(workspace, experiment_id)
    runs = workspace.load_runs(experiment_id)
    graph = _provenance(workspace, plan, runs, None)

    node = args.node
    if not node:
        if plan.dataset is None:
            raise CommandError("this experiment has no dataset; pass --node explicitly")
        node = plan.dataset.id
    if node not in graph:
        raise CommandError(f"no node {node} in the provenance graph")

    result = impact(graph, node)
    console.heading("impact")
    console.field("node", f"{result['label']} ({node})")
    console.field("depends on it", str(result["affected"]))
    for kind, count in result["by_kind"].items():
        console.field(f"  {kind}", str(count))
    if result["claims"]:
        console.heading("claims that would be invalidated")
        for claim in result["claims"]:
            console.bullet(claim)
    console.echo("")
    console.note("changing this node invalidates everything listed above")
    return 0


# --------------------------------------------------------------------------
# reproduce / verify / certify / export
# --------------------------------------------------------------------------


def cmd_reproduce(args: argparse.Namespace) -> int:
    from fabric import bundle as bundle_module
    from fabric.reproduce import compare_runs

    workspace = _workspace(args)
    target = args.target

    if os.path.isdir(target) and os.path.isfile(os.path.join(target, "manifest.json")):
        loaded = bundle_module.load(target)
        spec = _load_spec(loaded["spec_path"])
        original_runs = [Run.from_dict(r) for r in loaded["results"].get("runs", [])]
        source_label = f"bundle {os.path.basename(target)}"
        original_environment = _entity_or_none(loaded["manifest"].get("environment"), "environment")
        original_hardware = _entity_or_none(loaded["manifest"].get("hardware"), "hardware")
    else:
        experiment_id = _resolve(workspace, target)
        plan_spec_path = os.path.join(workspace.experiment_dir(experiment_id),
                                      "experiment.yaml")
        spec = _load_spec(plan_spec_path)
        original_runs = workspace.load_runs(experiment_id)
        source_label = experiment_id
        registry = Registry(workspace)
        original_environment = _first_entity(registry, "environment",
                                             [r.environment_id for r in original_runs])
        original_hardware = _first_entity(registry, "hardware",
                                          [r.hardware_id for r in original_runs])

    if not original_runs:
        raise CommandError(f"{source_label} has no recorded runs to reproduce")

    plan = _plan(workspace, spec)
    console.heading("reproducing")
    console.field("source", source_label)
    console.field("experiment", plan.experiment.id)
    console.field("runs", str(len(plan.runs)))
    console.echo("")

    options = ExecutionOptions(
        max_workers=args.workers or spec.execution.max_workers,
        timeout_seconds=spec.execution.timeout_seconds,
        isolation=spec.execution.isolation,
        # A reproduction that reads its answer from the cache has
        # reproduced nothing.
        cache=CachePolicy(enabled=False),
        rerun=True,
    )
    workspace.create_experiment_dir(plan.experiment.id)
    runner = Runner(workspace, plan, options, root=workspace.root)
    progress = console.Progress(len(plan.runs))
    execution = runner.execute(list(plan.runs),
                               on_progress=lambda e: progress.advance(
                                   e.label, failed=e.run.status == RunStatus.FAILED))
    progress.finish()
    console.success(execution.summary_line())

    reproduced_runs = workspace.load_runs(plan.experiment.id)
    report = compare_runs(
        plan, original_runs, reproduced_runs,
        original_environment=original_environment,
        reproduced_environment=execution.environment,
        original_hardware=original_hardware,
        reproduced_hardware=execution.hardware,
        alpha=spec.analysis.alpha,
        conditions=tuple(args.condition or ()),
    )
    console.echo("")
    console.echo(report.render())
    workspace.write(plan.experiment.id, "reproduction.json", report.to_dict())
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    return 0 if report.verdict in ("PASS", "PASS WITH DEVIATION") else 1


def _entity_or_none(data: Any, kind: str) -> Any:
    if not data:
        return None
    from fabric.core.registry import _rebuild

    try:
        return _rebuild(kind, data)
    except (TypeError, KeyError):
        return None


def _first_entity(registry: Registry, kind: str, ids: Sequence[str]) -> Any:
    for entity_id in ids:
        if not entity_id:
            continue
        try:
            return registry.get(kind, entity_id)
        except RegistryError:
            continue
    return None


def cmd_verify(args: argparse.Namespace) -> int:
    from fabric import bundle as bundle_module

    target = args.target
    if os.path.isdir(target) and os.path.isfile(os.path.join(target, "manifest.json")):
        result = bundle_module.verify(target)
        console.echo("")
        console.echo(result.render())
        return 0 if result.ok else 1

    workspace = _workspace(args)
    experiment_id = _resolve(workspace, target)
    plan = _stored_plan(workspace, experiment_id)
    runs = workspace.load_runs(experiment_id)
    registry = Registry(workspace)

    console.heading("verification")
    console.field("experiment", experiment_id)
    problems = 0

    if plan.dataset and plan.dataset.local_path and os.path.exists(plan.dataset.local_path):
        current = tree_digest(plan.dataset.local_path)
        if current == plan.dataset.digest:
            console.success(f"dataset {plan.dataset.ref} matches its registered digest")
        else:
            problems += 1
            console.error(
                f"dataset {plan.dataset.ref} has changed: registered "
                f"{plan.dataset.digest[:23]}, now {current[:23]}")

    missing = 0
    altered = 0
    for run in runs:
        for artifact in run.artifacts:
            path = os.path.join(workspace.artifacts_dir(experiment_id, run.id),
                                artifact.path.replace("/", os.sep))
            if not os.path.isfile(path):
                missing += 1
                continue
            from fabric.util.hashing import file_digest

            if file_digest(path) != artifact.digest:
                altered += 1
    total_artifacts = sum(len(r.artifacts) for r in runs)
    if missing or altered:
        problems += 1
        console.error(f"artifacts: {missing} missing, {altered} altered of {total_artifacts}")
    else:
        console.success(f"all {total_artifacts} artifacts match their recorded digests")

    failures = [r for r in registry.verify_all() if not r[2]]
    if failures:
        problems += 1
        for kind, entity_id, _ok, message in failures:
            console.error(f"{kind} {entity_id}: {message}")
    else:
        console.success("every registered entity hashes to its own identifier")

    certificate = workspace.read(experiment_id, "certificate.json")
    if certificate:
        from fabric.reporting.certificate import verify_certificate

        ok, message = verify_certificate(certificate)
        (console.success if ok else console.error)(message)
        if not ok:
            problems += 1

    console.echo("")
    if problems:
        console.error(f"{problems} problem(s) found")
        return 1
    console.success("intact")
    return 0


def cmd_certify(args: argparse.Namespace) -> int:
    from fabric.reporting.certificate import issue

    workspace = _workspace(args)
    experiment_id = _resolve(workspace, args.experiment)
    plan, runs, analysis = _cached_analysis(workspace, experiment_id)
    reproduction = workspace.read(experiment_id, "reproduction.json")
    certificate = issue(analysis, plan, runs, Registry(workspace),
                        workspace_root=workspace.root,
                        reproduction=reproduction)
    workspace.write(experiment_id, "certificate.json", certificate.to_dict())
    if args.json:
        print(json.dumps(certificate.to_dict(), indent=2, default=str))
        return 0
    console.echo("")
    console.echo(certificate.render())
    console.echo("")
    console.note(f"stored at .fabric/experiments/{experiment_id}/certificate.json")
    return 0 if certificate.verdict != "FAIL" else 1


def cmd_export(args: argparse.Namespace) -> int:
    from fabric import bundle as bundle_module
    from fabric.reporting import builder
    from fabric.reporting import markdown as markdown_renderer
    from fabric.reporting.certificate import issue

    workspace = _workspace(args)
    experiment_id = _resolve(workspace, args.experiment)
    plan, runs, analysis = _cached_analysis(workspace, experiment_id)
    graph = _provenance(workspace, plan, runs, analysis)

    registry = Registry(workspace)
    environment = _first_entity(registry, "environment", [r.environment_id for r in runs])
    hardware = _first_entity(registry, "hardware", [r.hardware_id for r in runs])
    certificate = issue(analysis, plan, runs, registry, workspace_root=workspace.root,
                        reproduction=workspace.read(experiment_id, "reproduction.json"))
    document = builder.build_report(analysis, plan, runs, graph=graph,
                                    environment=environment, hardware=hardware)

    destination = args.out or os.path.join(workspace.root, f"{experiment_id}-bundle")
    manifest = bundle_module.export(
        workspace, plan, runs, destination,
        analysis=analysis, provenance=graph, certificate=certificate,
        report_markdown=markdown_renderer.render(document),
        include_artifacts=not args.no_artifacts,
        environment=environment, hardware=hardware,
    )

    console.heading("bundle exported")
    console.field("location", destination)
    console.field("files", str(len(manifest.files)))
    console.field("runs", str(manifest.counts.get("runs", 0)))
    console.field("certificate", manifest.certificate_id)
    console.echo("")
    console.note(f"verify it with:     fabric verify {destination}")
    console.note(f"reproduce it with:  fabric reproduce {destination}")
    return 0


# --------------------------------------------------------------------------
# cache / doctor
# --------------------------------------------------------------------------


def cmd_cache(args: argparse.Namespace) -> int:
    workspace = _workspace(args)
    if args.action == "clear":
        count = workspace.cache_clear()
        console.success(f"cleared {count} cache entries")
        return 0
    index = read_json(workspace.cache_index_path, {}) or {}
    console.heading("cache")
    console.field("entries", str(len(index)))
    if index:
        newest = max(index.values(), key=lambda e: e.get("stored_at", 0))
        console.field("newest", time.strftime(
            "%Y-%m-%d %H:%M", time.localtime(newest.get("stored_at", 0))))
    console.note("a cache hit means the dataset, model, code, configuration, seed and "
                 "environment were all identical")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    from fabric.environments.capture import describe as describe_environment
    from fabric.environments.hardware import describe as describe_hardware

    console.heading("fabric")
    console.field("version", __version__)
    console.field("python", sys.version.split()[0])
    try:
        workspace = Workspace.discover(os.getcwd())
        console.field("workspace", workspace.path)
        console.field("experiments", str(len(workspace.list_experiments())))
    except WorkspaceError:
        console.field("workspace", "none found (run 'fabric init')")

    environment = capture_environment(os.getcwd())
    console.heading("environment")
    for key, value in describe_environment(environment):
        console.field(key, str(value)[:88], label_width=20)
    console.field("fingerprint", environment.id, label_width=20)

    hardware = capture_hardware()
    console.heading("hardware")
    for key, value in describe_hardware(hardware):
        console.field(key, str(value)[:88], label_width=20)
    console.field("fingerprint", hardware.id, label_width=20)

    console.heading("optional dependencies")
    for module, purpose in (("yaml", "faster, fuller YAML parsing"),):
        try:
            __import__(module)
            console.success(f"{module}: available ({purpose})")
        except ImportError:
            console.note(f"{module}: not installed - {purpose}; "
                         "Fabric uses its bundled parser instead")
    return 0


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fabric",
        description="Experiment Fabric - reproducible, provenance-aware experimentation.",
        epilog="Start with: fabric init, then fabric run experiment.yaml",
    )
    parser.add_argument("--version", action="version", version=f"fabric {__version__}")
    parser.add_argument("--workspace", help="workspace root (default: search upwards)")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p = sub.add_parser("init", help="create a workspace")
    p.add_argument("path", nargs="?", help="directory (default: here)")
    p.add_argument("--name", help="workspace name")
    p.add_argument("--force", action="store_true", help="reinitialise an existing workspace")
    p.add_argument("--quiet", action="store_true", help="say nothing on success")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("register", help="register a dataset, model or algorithm")
    p.add_argument("kind", choices=["dataset", "model", "algorithm"])
    p.add_argument("name")
    p.add_argument("--version", default="v1")
    p.add_argument("--path", help="local path to hash")
    p.add_argument("--digest", help="declared content digest")
    p.add_argument("--weights-digest", help="declared weights digest (models)")
    p.add_argument("--source", help="where the data came from")
    p.add_argument("--preprocessing", help="how it was processed")
    p.add_argument("--examples", type=int, help="number of examples")
    p.add_argument("--license")
    p.add_argument("--notes")
    p.add_argument("--architecture")
    p.add_argument("--precision")
    p.add_argument("--quantization")
    p.add_argument("--tokenizer")
    p.add_argument("--parameters", type=int)
    p.add_argument("--provider")
    p.add_argument("--commit", help="source commit, or 'auto'")
    p.add_argument("--entrypoint", help="algorithm entrypoint")
    p.add_argument("--description")
    p.add_argument("--force", action="store_true",
                   help="overwrite a conflicting registration")
    p.set_defaults(func=cmd_register)

    p = sub.add_parser("registry", help="inspect or verify the registry")
    p.add_argument("action", nargs="?", default="ls", choices=["ls", "verify"])
    p.add_argument("--kind", choices=["dataset", "model", "algorithm",
                                      "environment", "hardware"])
    p.set_defaults(func=cmd_registry)

    p = sub.add_parser("plan", help="expand a specification without running it")
    p.add_argument("spec")
    p.add_argument("--verbose", action="store_true", help="list every run")
    p.add_argument("--no-check", action="store_true",
                   help="skip checking that the entrypoint imports")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("run", help="execute an experiment")
    p.add_argument("spec")
    p.add_argument("--workers", type=int, help="parallel runs")
    p.add_argument("--timeout", type=float, help="per-run timeout in seconds")
    p.add_argument("--isolation", choices=["process", "inline"])
    p.add_argument("--only", action="append", help="run only this condition (repeatable)")
    p.add_argument("--rerun", action="store_true", help="re-execute completed runs")
    p.add_argument("--no-cache", action="store_true", help="ignore the cache")
    p.add_argument("--strict-environment", action="store_true",
                   help="treat any environment change as invalidating the cache")
    p.add_argument("--stop-on-failure", action="store_true")
    p.add_argument("--track-gpu", action="store_true", help="measure GPU memory and time")
    p.add_argument("--gpu-cost", type=float, help="USD per GPU hour, for cost accounting")
    p.add_argument("--cpu-cost", type=float, help="USD per CPU hour")
    p.add_argument("--dry-run", action="store_true", help="show the plan and stop")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("ls", help="list experiments")
    p.add_argument("--limit", type=int, default=25)
    p.set_defaults(func=cmd_ls)

    p = sub.add_parser("status", help="show progress of an experiment")
    p.add_argument("experiment")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("analyze", aliases=["analyse"], help="analyse results")
    p.add_argument("experiment")
    p.add_argument("--json", action="store_true", help="emit the full analysis as JSON")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("report", help="generate a scientific report")
    p.add_argument("experiment")
    p.add_argument("--format", default="markdown",
                   choices=["markdown", "md", "html", "latex", "tex", "json", "all"])
    p.add_argument("--out", help="output directory")
    p.add_argument("--open", action="store_true", help="open the HTML report")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("why", help="trace a number back to what produced it")
    p.add_argument("experiment")
    p.add_argument("--metric", help="metric name (default: the primary metric)")
    p.add_argument("--condition", help="condition name (default: the treatment)")
    p.add_argument("--seed", type=int)
    p.add_argument("--node", help="a provenance node id, instead of a metric")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_why)

    p = sub.add_parser("graph", help="export the provenance graph")
    p.add_argument("experiment")
    p.add_argument("--format", default="mermaid",
                   choices=["mermaid", "dot", "json", "ascii"])
    p.add_argument("--node", help="restrict to one node's lineage")
    p.add_argument("--depth", type=int, default=6, help="ascii tree depth")
    p.add_argument("--out", help="write to a file")
    p.set_defaults(func=cmd_graph)

    p = sub.add_parser("impact", help="show what depends on a node")
    p.add_argument("experiment")
    p.add_argument("--node", help="node id (default: the dataset)")
    p.set_defaults(func=cmd_impact)

    p = sub.add_parser("reproduce", help="re-run an experiment and compare")
    p.add_argument("target", help="experiment id or bundle directory")
    p.add_argument("--workers", type=int)
    p.add_argument("--condition", action="append", help="restrict to a condition")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_reproduce)

    p = sub.add_parser("verify", help="check digests and certificates")
    p.add_argument("target", help="experiment id or bundle directory")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("certify", help="issue a reproduction certificate")
    p.add_argument("experiment")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_certify)

    p = sub.add_parser("export", help="write a portable reproduction bundle")
    p.add_argument("experiment")
    p.add_argument("--out", help="destination directory")
    p.add_argument("--no-artifacts", action="store_true",
                   help="omit raw artifacts to keep the bundle small")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("cache", help="inspect or clear the execution cache")
    p.add_argument("action", nargs="?", default="stats", choices=["stats", "clear"])
    p.set_defaults(func=cmd_cache)

    p = sub.add_parser("doctor", help="show environment and hardware capture")
    p.set_defaults(func=cmd_doctor)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    try:
        return int(args.func(args) or 0)
    except CommandError as exc:
        console.error(str(exc))
        return 2
    except KeyboardInterrupt:  # pragma: no cover - interactive
        console.echo("")
        console.warn("interrupted; completed runs are already saved and will be skipped "
                     "on the next 'fabric run'")
        return 130
    except CorruptRecordError as exc:
        console.error(str(exc))
        return 3
    except (SpecError, PlanError, WorkspaceError, RegistryError) as exc:
        console.error(str(exc))
        return 2


if __name__ == "__main__":  # pragma: no cover - process entrypoint
    raise SystemExit(main())
