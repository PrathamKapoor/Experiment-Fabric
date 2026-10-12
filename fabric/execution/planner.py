"""From a specification to an execution plan.

The planner is where "one YAML file" becomes "480 runs that know how they
relate to each other".  It resolves entities, expands the configuration
grid, builds one condition per experimental arm, multiplies conditions by
seeds, adds the reproducibility counterfactuals, and wires the whole
thing into an :class:`~fabric.core.graph.ExperimentGraph`.

Everything here is deterministic.  The same specification produces the
same experiment id, the same condition ids, and the same run ids on any
machine -- which is what makes caching, resumption and cross-machine
comparison possible at all.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from fabric.core import graph as g
from fabric.core.entities import (
    Algorithm,
    Condition,
    Dataset,
    Experiment,
    Model,
    Role,
    Run,
    RunStatus,
)
from fabric.core.graph import ExperimentGraph
from fabric.core.registry import Registry, RegistryError
from fabric.core.spec import EntityRef, ExperimentSpec, SpecError
from fabric.core.store import Workspace
from fabric.environments.capture import git_state
from fabric.execution.adapters import EntrypointError, entrypoint_digest, resolve
from fabric.util.hashing import content_id


class PlanError(RuntimeError):
    """Raised when a specification cannot be turned into a runnable plan."""


@dataclass
class ExperimentPlan:
    """A fully expanded, executable experiment."""

    experiment: Experiment
    spec: ExperimentSpec
    conditions: tuple[Condition, ...]
    runs: tuple[Run, ...]
    graph: ExperimentGraph
    dataset: Dataset | None = None
    model: Model | None = None
    algorithm: Algorithm | None = None
    extra_datasets: dict[str, Dataset] = field(default_factory=dict)
    extra_models: dict[str, Model] = field(default_factory=dict)
    extra_algorithms: dict[str, Algorithm] = field(default_factory=dict)
    advisories: tuple[str, ...] = ()
    created_at: float = field(default_factory=time.time)

    # -- lookup ------------------------------------------------------------

    def condition(self, condition_id: str) -> Condition:
        for item in self.conditions:
            if item.id == condition_id:
                return item
        raise PlanError(f"unknown condition {condition_id}")

    def condition_by_name(self, name: str) -> Condition:
        for item in self.conditions:
            if item.name == name:
                return item
        raise PlanError(f"no condition named {name!r}")

    def runs_for(self, condition_id: str) -> list[Run]:
        return [r for r in self.runs if r.condition_id == condition_id]

    @property
    def reference_condition(self) -> Condition:
        """The condition every perturbation is measured against."""
        for item in self.conditions:
            if item.role == Role.TREATMENT and not item.is_perturbed:
                return item
        return self.conditions[0]

    def dataset_for(self, condition: Condition) -> Dataset | None:
        if condition.dataset_id and condition.dataset_id in self.extra_datasets:
            return self.extra_datasets[condition.dataset_id]
        return self.dataset

    def model_for(self, condition: Condition) -> Model | None:
        if condition.model_id and condition.model_id in self.extra_models:
            return self.extra_models[condition.model_id]
        return self.model

    def algorithm_for(self, condition: Condition) -> Algorithm | None:
        if condition.algorithm_id and condition.algorithm_id in self.extra_algorithms:
            return self.extra_algorithms[condition.algorithm_id]
        return self.algorithm

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment.to_dict(),
            "spec": self.spec.to_dict(),
            "spec_digest": self.spec.digest,
            "dataset": self.dataset.to_dict() if self.dataset else None,
            "model": self.model.to_dict() if self.model else None,
            "algorithm": self.algorithm.to_dict() if self.algorithm else None,
            "extra_datasets": {k: v.to_dict() for k, v in self.extra_datasets.items()},
            "extra_models": {k: v.to_dict() for k, v in self.extra_models.items()},
            "extra_algorithms": {k: v.to_dict() for k, v in self.extra_algorithms.items()},
            "conditions": [c.to_dict() for c in self.conditions],
            "runs": [r.to_dict() for r in self.runs],
            "advisories": list(self.advisories),
            "created_at": self.created_at,
            "counts": self.counts(),
        }

    def counts(self) -> dict[str, int]:
        by_role: dict[str, int] = {}
        for condition in self.conditions:
            by_role[condition.role.value] = by_role.get(condition.role.value, 0) + 1
        return {
            "conditions": len(self.conditions),
            "runs": len(self.runs),
            "seeds": len(self.spec.seeds.values),
            "configurations": len(self.spec.configurations),
            "perturbed_conditions": sum(1 for c in self.conditions if c.is_perturbed),
            **{f"role_{k}": v for k, v in sorted(by_role.items())},
        }

    def summary_lines(self) -> list[str]:
        counts = self.counts()
        lines = [
            f"experiment      {self.experiment.id}  {self.experiment.name}",
            f"question        {self.experiment.research_question}",
            f"conditions      {counts['conditions']} "
            f"({counts.get('role_treatment', 0)} treatment, "
            f"{counts.get('role_baseline', 0)} baseline, "
            f"{counts.get('role_ablation', 0)} ablation)",
            f"seeds           {counts['seeds']}",
            f"runs            {counts['runs']}",
        ]
        if counts["perturbed_conditions"]:
            lines.append(f"perturbations   {counts['perturbed_conditions']} counterfactual conditions")
        return lines


# --------------------------------------------------------------------------
# identity helpers
# --------------------------------------------------------------------------


def experiment_id_for(spec: ExperimentSpec) -> str:
    """Deterministic experiment id.

    Derived from the specification's content, so re-running an unchanged
    specification resumes the same experiment instead of scattering
    near-duplicates across the workspace.
    """
    return "EXP-" + content_id({"spec": spec.digest, "name": spec.name}, length=10)


def run_id_for(experiment_id: str, condition_id: str, seed: int, replicate: int = 0) -> str:
    return "RUN-" + content_id(
        {"experiment": experiment_id, "condition": condition_id,
         "seed": seed, "replicate": replicate},
        length=12,
    )


def _config_label(config: Mapping[str, Any], swept_names: Sequence[str]) -> str:
    """A compact, stable label for a swept configuration."""
    parts = []
    for name in swept_names:
        if name in config:
            value = config[name]
            if isinstance(value, float):
                text = f"{value:g}"
            else:
                text = str(value)
            parts.append(f"{name}={text}")
    return ",".join(parts)


# --------------------------------------------------------------------------
# entity resolution
# --------------------------------------------------------------------------


def _tree_size(path: str) -> int:
    """Total bytes under a path, without reading any of them.

    Size is metadata, never identity -- two datasets can share a byte count
    and share nothing else. It is here so a report can say how much data
    produced a number, and so a bundle can warn before copying a corpus.
    """
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for directory, _subdirs, files in os.walk(path):
        for name in sorted(files):
            try:
                total += os.path.getsize(os.path.join(directory, name))
            except OSError:  # pragma: no cover - a file vanishing mid-walk
                continue
    return total


def _resolve_dataset(ref: EntityRef | None, registry: Registry, *, root: str) -> Dataset | None:
    if ref is None:
        return None
    if ref.id:
        return registry.get("dataset", ref.id)
    if ref.inline:
        inline = dict(ref.inline)
        path = inline.pop("path", "")
        dataset = Dataset(
            name=ref.name,
            version=ref.version or str(inline.pop("version", "v1")),
            digest=str(inline.pop("digest", "")),
            source=str(inline.pop("source", "")),
            preprocessing=str(inline.pop("preprocessing", "")),
            schema=dict(inline.pop("schema", {}) or {}),
            splits=dict(inline.pop("splits", {}) or {}),
            n_examples=inline.pop("n_examples", None),
            size_bytes=inline.pop("size_bytes", None),
            license=str(inline.pop("license", "")),
            notes=str(inline.pop("notes", "")),
            parent=str(inline.pop("parent", "")),
            local_path=os.path.join(root, path) if path and not os.path.isabs(path) else str(path),
        )
        if dataset.local_path and os.path.exists(dataset.local_path):
            from fabric.util.hashing import tree_digest

            if not dataset.digest:
                dataset = dataset.with_(digest=tree_digest(dataset.local_path))
            if dataset.size_bytes is None:
                dataset = dataset.with_(size_bytes=_tree_size(dataset.local_path))
        return registry.register_dataset(dataset)
    try:
        return registry.find("dataset", ref.name, ref.version)
    except RegistryError as exc:
        raise PlanError(str(exc)) from exc


def _resolve_model(ref: EntityRef | None, registry: Registry) -> Model | None:
    if ref is None:
        return None
    if ref.id:
        return registry.get("model", ref.id)
    if ref.inline:
        inline = dict(ref.inline)
        model = Model(
            name=ref.name,
            version=ref.version or str(inline.pop("version", "v1")),
            weights_digest=str(inline.pop("weights_digest", "")),
            architecture=str(inline.pop("architecture", "")),
            commit=str(inline.pop("commit", "")),
            precision=str(inline.pop("precision", "")),
            quantization=str(inline.pop("quantization", "none")),
            tokenizer=str(inline.pop("tokenizer", "")),
            parameters=inline.pop("parameters", None),
            provider=str(inline.pop("provider", "")),
            repository=str(inline.pop("repository", "")),
            checkpoint=str(inline.pop("checkpoint", "")),
            config=dict(inline.pop("config", {}) or {}),
        )
        return registry.register_model(model)
    try:
        return registry.find("model", ref.name, ref.version)
    except RegistryError as exc:
        raise PlanError(str(exc)) from exc


def _resolve_algorithm(ref: EntityRef | None, registry: Registry, *, root: str,
                       strict: bool, spec_paths: Sequence[str] = ()) -> Algorithm | None:
    if ref is None:
        return None
    if ref.id:
        algorithm = registry.get("algorithm", ref.id)
    elif ref.inline:
        inline = dict(ref.inline)
        commit = str(inline.pop("commit", ""))
        dirty = False
        if commit in ("auto", "git", ""):
            commit, dirty, _ = git_state(root)
        entrypoint = str(inline.pop("entrypoint", ""))
        algorithm = Algorithm(
            name=ref.name,
            entrypoint=entrypoint,
            commit=commit,
            source_digest=entrypoint_digest(entrypoint, root=root, exclude=spec_paths),
            version=ref.version or str(inline.pop("version", "v1")),
            description=str(inline.pop("description", "")),
            params=dict(inline.pop("params", {}) or {}),
            dirty=dirty,
        )
        algorithm = registry.register_algorithm(algorithm)
    else:
        try:
            algorithm = registry.find("algorithm", ref.name, ref.version)
        except RegistryError as exc:
            raise PlanError(str(exc)) from exc

    if strict and algorithm.entrypoint:
        try:
            resolve(algorithm.entrypoint, root=root)
        except EntrypointError as exc:
            raise PlanError(
                f"algorithm '{algorithm.name}' has an unusable entrypoint: {exc}"
            ) from exc
    return algorithm


# --------------------------------------------------------------------------
# the planner
# --------------------------------------------------------------------------


def plan_experiment(
    spec: ExperimentSpec,
    workspace: Workspace,
    *,
    strict: bool = True,
    root: str = "",
    exclude_from_digest: Sequence[str] = (),
) -> ExperimentPlan:
    """Expand a specification into a complete execution plan.

    ``exclude_from_digest`` names files that must not contribute to the
    algorithm's source fingerprint. The caller's own specification is always
    among them: it is hashed separately as ``spec_digest``, and counting it
    again means the algorithm's identity depends on which copy of the YAML
    happened to be used to plan, so re-planning a stored experiment
    disagreed with the plan that produced it.
    """
    root = root or workspace.root
    registry = Registry(workspace)
    advisories = list(spec.validate())
    excluded = list(exclude_from_digest)
    if spec.source_path:
        excluded.append(spec.source_path)

    dataset = _resolve_dataset(spec.dataset, registry, root=root)
    model = _resolve_model(spec.model, registry)
    algorithm = _resolve_algorithm(spec.algorithm, registry, root=root, strict=strict,
                                   spec_paths=excluded)
    if algorithm is None:
        raise PlanError("the experiment has no algorithm to run")

    if algorithm.dirty:
        advisories.append(
            f"the working tree has uncommitted changes, so commit "
            f"{algorithm.commit[:7]} does not fully describe the code that will run. "
            "The entrypoint file itself is hashed, but changes in modules it imports "
            "are not. Commit before the runs you intend to publish."
        )
    if not algorithm.commit:
        advisories.append(
            "this is not a git repository, so the code that ran cannot be pinned to a "
            "commit. Only the entrypoint file's own hash is recorded."
        )

    if spec.integrity.require_dataset_digest and dataset is not None and not dataset.digest:
        advisories.append(
            f"dataset '{dataset.ref}' has no digest, so Fabric cannot prove which data "
            "produced these results. Give it a 'path' to hash, or a declared 'digest'."
        )

    experiment_id = experiment_id_for(spec)
    experiment = Experiment(
        id=experiment_id,
        name=spec.name,
        research_question=spec.research_question,
        hypothesis=spec.hypothesis,
        spec_digest=spec.digest,
        tags=spec.tags,
        metrics=spec.metrics,
        primary_metric=spec.primary_metric,
        description=spec.description,
    )

    configurations = spec.configurations
    swept_names = [p.name for p in sorted(spec.parameters, key=lambda p: p.name) if p.swept]

    conditions: list[Condition] = []
    extra_datasets: dict[str, Dataset] = {}
    extra_models: dict[str, Model] = {}
    extra_algorithms: dict[str, Algorithm] = {}

    base_ids = {
        "dataset_id": dataset.id if dataset else "",
        "model_id": model.id if model else "",
        "algorithm_id": algorithm.id,
    }

    # -- treatment arms ----------------------------------------------------
    for index, config in enumerate(configurations):
        label = _config_label(config, swept_names)
        name = spec.treatment_name if not label else f"{spec.treatment_name}[{label}]"
        conditions.append(Condition(
            name=name,
            role=Role.TREATMENT,
            config=dict(config),
            group="main",
            axis={"configuration_index": index, **{k: config[k] for k in swept_names if k in config}},
            **base_ids,
        ))

    reference_config = dict(configurations[0]) if configurations else {}

    # -- ablation arms -----------------------------------------------------
    # Ablations hold the configuration fixed at the reference point by
    # default: crossing every ablation with every configuration multiplies
    # compute without answering the causal question an ablation asks.
    ablation_configs = configurations if spec.ablation_scope == "all" else [reference_config]
    for config in ablation_configs:
        label = _config_label(config, swept_names) if spec.ablation_scope == "all" else ""
        for ablation in spec.ablations:
            merged = dict(config)
            merged.update(ablation.configuration)
            name = ablation.name if not label else f"{ablation.name}[{label}]"
            conditions.append(Condition(
                name=name,
                role=Role.ABLATION,
                config=merged,
                ablated=tuple(ablation.remove),
                group="ablation",
                axis={"ablation": ablation.name, "removed": list(ablation.remove)},
                **base_ids,
            ))

    # -- baseline arms -----------------------------------------------------
    for baseline in spec.baselines:
        b_dataset = dataset
        b_model = model
        b_algorithm = algorithm
        if baseline.dataset:
            b_dataset = _resolve_dataset(baseline.dataset, registry, root=root)
            if b_dataset:
                extra_datasets[b_dataset.id] = b_dataset
        if baseline.model:
            b_model = _resolve_model(baseline.model, registry)
            if b_model:
                extra_models[b_model.id] = b_model
        if baseline.algorithm:
            b_algorithm = _resolve_algorithm(baseline.algorithm, registry, root=root,
                                             strict=strict, spec_paths=excluded)
            if b_algorithm:
                extra_algorithms[b_algorithm.id] = b_algorithm

        merged = dict(reference_config)
        overridden_sweeps = [k for k in baseline.configuration if k in swept_names]
        merged.update(baseline.configuration)
        if overridden_sweeps and spec.integrity.freeze_baselines:
            advisories.append(
                f"baseline '{baseline.name}' overrides swept parameter(s) "
                f"{overridden_sweeps}: it is not being compared at the same operating "
                "point as the treatment. That may be correct, but state it in the paper."
            )
        conditions.append(Condition(
            name=baseline.name,
            role=Role.BASELINE,
            config=merged,
            group="main",
            axis={"baseline": baseline.name},
            dataset_id=b_dataset.id if b_dataset else "",
            model_id=b_model.id if b_model else "",
            algorithm_id=b_algorithm.id if b_algorithm else algorithm.id,
        ))

    # -- reproducibility counterfactuals -----------------------------------
    reference = next((c for c in conditions if c.role == Role.TREATMENT), None)
    perturbation_seeds: list[int] = []
    if spec.reproducibility.enabled and reference is not None:
        if spec.reproducibility.reference_condition:
            named = [c for c in conditions if c.name == spec.reproducibility.reference_condition]
            if not named:
                raise PlanError(
                    f"reproducibility.reference_condition "
                    f"{spec.reproducibility.reference_condition!r} is not an arm of this experiment"
                )
            reference = named[0]
        new_conditions, seeds_added = _build_perturbations(
            spec, reference, registry, root=root, strict=strict,
            extra_datasets=extra_datasets, extra_models=extra_models,
            extra_algorithms=extra_algorithms,
            exclude_from_digest=excluded,
        )
        conditions.extend(new_conditions)
        perturbation_seeds = seeds_added

    # -- uniqueness --------------------------------------------------------
    seen: dict[str, Condition] = {}
    unique: list[Condition] = []
    for condition in conditions:
        if condition.id in seen:
            advisories.append(
                f"conditions '{seen[condition.id].name}' and '{condition.name}' are "
                "identical in every respect; the duplicate was dropped."
            )
            continue
        if condition.name in {c.name for c in unique}:
            raise PlanError(
                f"two different conditions are both named '{condition.name}'. "
                "Arm names must be unique."
            )
        seen[condition.id] = condition
        unique.append(condition)
    conditions = unique

    # -- runs --------------------------------------------------------------
    runs = _build_runs(spec, experiment_id, conditions, perturbation_seeds, dataset, model)

    graph = build_graph(experiment, spec, conditions, runs, dataset, model, algorithm,
                        extra_datasets, extra_models, extra_algorithms)

    return ExperimentPlan(
        experiment=experiment,
        spec=spec,
        conditions=tuple(conditions),
        runs=tuple(runs),
        graph=graph,
        dataset=dataset,
        model=model,
        algorithm=algorithm,
        extra_datasets=extra_datasets,
        extra_models=extra_models,
        extra_algorithms=extra_algorithms,
        advisories=tuple(dict.fromkeys(advisories)),
    )


def _build_perturbations(
    spec: ExperimentSpec,
    reference: Condition,
    registry: Registry,
    *,
    root: str,
    strict: bool,
    extra_datasets: dict[str, Dataset],
    extra_models: dict[str, Model],
    extra_algorithms: dict[str, Algorithm],
    exclude_from_digest: Sequence[str] = (),
) -> tuple[list[Condition], list[int]]:
    """Build one counterfactual condition per perturbation value.

    Exactly one thing changes per condition.  That is the entire design:
    a controlled perturbation experiment, where the effect of "moved to a
    different CUDA version" is measured the same way as the effect of a
    method.
    """
    conditions: list[Condition] = []
    extra_seeds: list[int] = []
    for perturbation in spec.reproducibility.perturbations:
        axis = perturbation.axis
        for index, value in enumerate(perturbation.values):
            # The condition name carries the axis so arm names stay unique;
            # the label does not, so tables can print "axis | label" without
            # repeating the axis twice.
            short = _perturbation_label(value, index)
            name = f"{reference.name}~{axis}:{short}"
            marker = {"axis": axis, "value": value, "label": short,
                      "reference": reference.name}
            if axis == "seed":
                extra_seeds.append(int(value))
                continue
            if axis in ("environment", "hardware"):
                overlay = value if isinstance(value, Mapping) else {"FABRIC_PROFILE": str(value)}
                conditions.append(Condition(
                    name=name, role=Role.REFERENCE, config=dict(reference.config),
                    ablated=reference.ablated, group=f"perturbation:{axis}",
                    dataset_id=reference.dataset_id, model_id=reference.model_id,
                    algorithm_id=reference.algorithm_id,
                    axis=dict(reference.axis),
                    perturbation={**marker, "env": {str(k): str(v) for k, v in overlay.items()}},
                ))
            elif axis == "configuration":
                overrides = value if isinstance(value, Mapping) else {}
                if not overrides:
                    raise SpecError(
                        "a configuration perturbation value must be a mapping of "
                        "parameter overrides",
                        "reproducibility.perturbations.configuration",
                    )
                merged = dict(reference.config)
                merged.update(overrides)
                conditions.append(Condition(
                    name=name, role=Role.REFERENCE, config=merged,
                    ablated=reference.ablated, group=f"perturbation:{axis}",
                    dataset_id=reference.dataset_id, model_id=reference.model_id,
                    algorithm_id=reference.algorithm_id, axis=dict(reference.axis),
                    perturbation=marker,
                ))
            elif axis == "dataset":
                other = _resolve_dataset(EntityRef.parse(value, "reproducibility.dataset"),
                                         registry, root=root)
                if other is None:
                    continue
                extra_datasets[other.id] = other
                conditions.append(Condition(
                    name=name, role=Role.REFERENCE, config=dict(reference.config),
                    ablated=reference.ablated, group=f"perturbation:{axis}",
                    dataset_id=other.id, model_id=reference.model_id,
                    algorithm_id=reference.algorithm_id, axis=dict(reference.axis),
                    perturbation={**marker, "dataset": other.ref},
                ))
            elif axis == "model":
                other_model = _resolve_model(EntityRef.parse(value, "reproducibility.model"), registry)
                if other_model is None:
                    continue
                extra_models[other_model.id] = other_model
                conditions.append(Condition(
                    name=name, role=Role.REFERENCE, config=dict(reference.config),
                    ablated=reference.ablated, group=f"perturbation:{axis}",
                    dataset_id=reference.dataset_id, model_id=other_model.id,
                    algorithm_id=reference.algorithm_id, axis=dict(reference.axis),
                    perturbation={**marker, "model": other_model.ref},
                ))
            elif axis == "implementation":
                other_algorithm = _resolve_algorithm(
                    EntityRef.parse(value, "reproducibility.implementation"),
                    registry, root=root, strict=strict, spec_paths=exclude_from_digest,
                )
                if other_algorithm is None:
                    continue
                extra_algorithms[other_algorithm.id] = other_algorithm
                conditions.append(Condition(
                    name=name, role=Role.REFERENCE, config=dict(reference.config),
                    ablated=reference.ablated, group=f"perturbation:{axis}",
                    dataset_id=reference.dataset_id, model_id=reference.model_id,
                    algorithm_id=other_algorithm.id, axis=dict(reference.axis),
                    perturbation={**marker, "implementation": other_algorithm.ref},
                ))
    return conditions, extra_seeds


def _perturbation_label(value: Any, index: int) -> str:
    """A short, readable name for one perturbation value.

    Prefers an explicit ``label``, then an entity reference, and only
    falls back to dumping the mapping when neither is available -- a
    label appears in every table and chart, so it has to stay short.
    """
    if isinstance(value, Mapping):
        if "label" in value:
            return str(value["label"])
        if "name" in value:
            version = value.get("version")
            return f"{value['name']}@{version}" if version else str(value["name"])
        pairs = [(k, v) for k, v in sorted(value.items()) if not isinstance(v, (dict, list))]
        return ",".join(f"{k}={v}" for k, v in pairs)[:48] or f"variant-{index}"
    if isinstance(value, str):
        return value
    return f"{index}:{value}"


def _build_runs(
    spec: ExperimentSpec,
    experiment_id: str,
    conditions: Sequence[Condition],
    perturbation_seeds: Sequence[int],
    dataset: Dataset | None,
    model: Model | None,
) -> list[Run]:
    """Multiply conditions by seeds (and by determinism replicates)."""
    runs: list[Run] = []
    base_seeds = list(spec.seeds.values)
    replicates = max(0, spec.reproducibility.determinism_replicates
                     if spec.reproducibility.enabled else spec.execution.determinism_replicates)

    reference_id = next((c.id for c in conditions if c.role == Role.TREATMENT), "")

    for condition in conditions:
        seeds = list(base_seeds)
        if condition.is_perturbed and condition.perturbation.get("axis") == "seed":
            seeds = [int(condition.perturbation["value"])]
        for seed in seeds:
            runs.append(_make_run(experiment_id, condition, seed, 0))
        # Extra seeds from a seed-axis perturbation attach to the reference
        # condition: changing the seed *is* rerunning the same condition.
        if condition.id == reference_id:
            for seed in perturbation_seeds:
                if seed not in seeds:
                    runs.append(_make_run(experiment_id, condition, seed, 0))
            for replicate in range(1, replicates + 1):
                # Identical inputs, executed again, to test whether the
                # pipeline is deterministic at all.
                runs.append(_make_run(experiment_id, condition, base_seeds[0], replicate))
    return runs


def _make_run(experiment_id: str, condition: Condition, seed: int, replicate: int) -> Run:
    return Run(
        id=run_id_for(experiment_id, condition.id, seed, replicate),
        experiment_id=experiment_id,
        condition_id=condition.id,
        condition_name=condition.name,
        seed=seed,
        role=condition.role,
        status=RunStatus.PENDING,
        config=dict(condition.config),
        dataset_id=condition.dataset_id,
        model_id=condition.model_id,
        algorithm_id=condition.algorithm_id,
        ablated=condition.ablated,
        replicate=replicate,
        extra={"perturbation": condition.perturbation} if condition.perturbation else {},
    )


# --------------------------------------------------------------------------
# graph construction
# --------------------------------------------------------------------------


def build_graph(
    experiment: Experiment,
    spec: ExperimentSpec,
    conditions: Sequence[Condition],
    runs: Sequence[Run],
    dataset: Dataset | None,
    model: Model | None,
    algorithm: Algorithm | None,
    extra_datasets: Mapping[str, Dataset] | None = None,
    extra_models: Mapping[str, Model] | None = None,
    extra_algorithms: Mapping[str, Algorithm] | None = None,
) -> ExperimentGraph:
    """Wire entities, conditions and runs into the experiment graph."""
    graph = ExperimentGraph(experiment.name)
    graph.add_node(experiment.id, "experiment", experiment.name,
                   research_question=experiment.research_question)

    datasets = dict(extra_datasets or {})
    if dataset:
        datasets[dataset.id] = dataset
    models = dict(extra_models or {})
    if model:
        models[model.id] = model
    algorithms = dict(extra_algorithms or {})
    if algorithm:
        algorithms[algorithm.id] = algorithm

    for item in datasets.values():
        graph.add_node(item.id, "dataset", item.ref, digest=item.digest,
                       n_examples=item.n_examples)
    for item in models.values():
        graph.add_node(item.id, "model", item.ref, weights_digest=item.weights_digest,
                       precision=item.precision)
    for item in algorithms.values():
        graph.add_node(item.id, "algorithm", item.ref, commit=item.commit,
                       entrypoint=item.entrypoint)

    for condition in conditions:
        graph.add_node(condition.id, "condition", condition.name,
                       role=condition.role.value, config=condition.config,
                       ablated=list(condition.ablated), group=condition.group,
                       perturbation=condition.perturbation)
        graph.add_edge(experiment.id, condition.id, g.CONTAINS)
        for entity_id in (condition.dataset_id, condition.model_id, condition.algorithm_id):
            if entity_id and graph.has_node(entity_id):
                graph.add_edge(entity_id, condition.id, g.USES)
        if condition.is_perturbed:
            reference_name = condition.perturbation.get("reference")
            for other in conditions:
                if other.name == reference_name:
                    graph.add_edge(other.id, condition.id, g.PERTURBS,
                                   axis=condition.perturbation.get("axis"))
                    break

    for run in runs:
        graph.add_node(run.id, "run", f"{run.condition_name} seed={run.seed}"
                       + (f" rep={run.replicate}" if run.replicate else ""),
                       seed=run.seed, status=run.status.value, replicate=run.replicate)
        graph.add_edge(run.condition_id, run.id, g.REPLICATES)
    return graph
