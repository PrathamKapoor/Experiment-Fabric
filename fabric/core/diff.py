"""Structured difference between two experiments.

The question this answers is the one a researcher asks constantly and
usually resolves by eye: *these two runs disagree -- what is actually
different between them?*

Doing it by eye is where reproducibility failures hide.  The two
experiments differ in fourteen places, thirteen of them irrelevant, and
the fourteenth is a dataset version bump nobody mentioned.  Fabric
already records every one of those fourteen things by content, so the
comparison can be mechanical.

The output is deliberately grouped by *what kind of thing changed*,
because that is what determines whether a difference explains a result:
a different dataset digest can move a metric, a different report
timestamp cannot.  Each difference also carries whether Fabric considers
it capable of changing a number, so the summary can lead with the ones
that matter instead of the ones that are merely visible.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from fabric.util.hashing import content_id

DIFF_SCHEMA_VERSION = "1.0"

#: Categories in the order a reader should consider them: the ones that
#: can move a number first, bookkeeping last.
CATEGORY_ORDER = (
    "dataset",
    "model",
    "algorithm",
    "benchmark",
    "configuration",
    "arms",
    "seeds",
    "metrics",
    "analysis",
    "execution",
    "environment",
    "hardware",
    "results",
)

#: Categories whose contents can change a measured number, as opposed to
#: changing how it is reported. Used to sort and to phrase the summary;
#: never to hide anything.
#:
#: ``execution`` is in here because it deserves to be: running inline
#: rather than in a subprocess changes the thread configuration a library
#: inherits, and that is enough to move a floating-point result.
#: ``analysis`` is not, because it decides how a number is *read* -- a
#: wider interval is not a different measurement.
MATERIAL = frozenset({
    "dataset", "model", "algorithm", "benchmark", "configuration",
    "arms", "seeds", "environment", "hardware", "execution",
})

#: Categories that are neither an input nor a reading, but the outcome.
OUTCOME = frozenset({"results"})


@dataclass(frozen=True)
class Difference:
    """One field that differs, or is worth reporting as identical."""

    category: str
    field: str
    left: Any
    right: Any
    note: str = ""

    @property
    def material(self) -> bool:
        """Whether a change here is capable of moving a metric."""
        return self.category in MATERIAL

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "field": self.field,
            "left": self.left,
            "right": self.right,
            "material": self.material,
            "note": self.note,
        }

    def describe(self) -> str:
        left = "absent" if self.left is None else str(self.left)
        right = "absent" if self.right is None else str(self.right)
        text = f"{self.category}.{self.field}: {left} -> {right}"
        return f"{text} ({self.note})" if self.note else text


@dataclass
class ExperimentDiff:
    """Everything that differs between two experiments."""

    left_id: str = ""
    right_id: str = ""
    left_name: str = ""
    right_name: str = ""
    differences: list[Difference] = field(default_factory=list)
    schema_version: str = DIFF_SCHEMA_VERSION

    def __bool__(self) -> bool:
        return bool(self.differences)

    def for_category(self, category: str) -> list[Difference]:
        return [d for d in self.differences if d.category == category]

    @property
    def categories(self) -> list[str]:
        present = {d.category for d in self.differences}
        return [c for c in CATEGORY_ORDER if c in present]

    @property
    def material(self) -> list[Difference]:
        """Differences capable of changing a measured number."""
        return [d for d in self.differences if d.material]

    @property
    def identical(self) -> bool:
        return not self.differences

    def describe(self) -> str:
        """One sentence, leading with whether anything could explain a gap."""
        if not self.differences:
            return (
                f"{self.left_id} and {self.right_id} are identical in every field "
                "Fabric records. Any difference in their results comes from "
                "non-determinism, not from the specification."
            )
        material = self.material
        if not material:
            return (
                f"{len(self.differences)} difference(s), none of them capable of "
                "changing a measured number: the two experiments differ only in "
                "reporting and bookkeeping."
            )
        heads = ", ".join(sorted({d.category for d in material}))
        return (
            f"{len(self.differences)} difference(s), {len(material)} of which could "
            f"move a number ({heads})."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "left": {"experiment_id": self.left_id, "name": self.left_name},
            "right": {"experiment_id": self.right_id, "name": self.right_name},
            "identical": self.identical,
            "summary": self.describe(),
            "differences": [d.to_dict() for d in self.differences],
            "by_category": {
                category: [d.to_dict() for d in self.for_category(category)]
                for category in self.categories
            },
        }


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------


def _add(out: list[Difference], category: str, name: str,
         left: Any, right: Any, note: str = "") -> None:
    if left != right:
        out.append(Difference(category, name, left, right, note))


def _entity_fields(entity: Any, names: Sequence[str]) -> dict[str, Any]:
    return {name: getattr(entity, name, None) for name in names} if entity else {}


def _compare_entity(out: list[Difference], category: str,
                    left: Any, right: Any, names: Sequence[str]) -> None:
    if left is None and right is None:
        return
    if left is None or right is None:
        out.append(Difference(
            category, "present", left is not None, right is not None,
            f"one experiment declares a {category} and the other does not",
        ))
        return
    a, b = _entity_fields(left, names), _entity_fields(right, names)
    for name in names:
        _add(out, category, name, a.get(name), b.get(name))


def _config_diff(out: list[Difference], left: Mapping[str, Any],
                 right: Mapping[str, Any]) -> None:
    for key in sorted(set(left) | set(right)):
        _add(out, "configuration", key, left.get(key), right.get(key),
             "present in only one experiment" if key not in left or key not in right
             else "")


def _config_fingerprint(configuration: Mapping[str, Any]) -> str:
    return content_id({str(k): configuration[k] for k in sorted(configuration)},
                      length=12)


def _config_set_diff(out: list[Difference], left: Sequence[Mapping[str, Any]],
                     right: Sequence[Mapping[str, Any]]) -> None:
    """Report configurations that exist on one side only.

    ``expand_configurations`` orders the grid deterministically with the
    first declared value of each swept parameter first, so position 0 is
    the reference configuration and cannot reveal a change to any later
    value. This compares the whole set, which can.
    """
    left_fingerprints = {_config_fingerprint(c): c for c in left}
    right_fingerprints = {_config_fingerprint(c): c for c in right}
    only_left = [left_fingerprints[k] for k in left_fingerprints.keys()
                 - right_fingerprints.keys()]
    only_right = [right_fingerprints[k] for k in right_fingerprints.keys()
                  - left_fingerprints.keys()]
    if not only_left and not only_right:
        return
    _add(out, "configuration", "swept configurations",
         [dict(sorted(c.items())) for c in sorted(only_left, key=repr)],
         [dict(sorted(c.items())) for c in sorted(only_right, key=repr)],
         "only these configurations appear in one of the two experiments")


def diff_plans(left_plan: Any, right_plan: Any) -> ExperimentDiff:
    """Compare two plans: everything decided before a run happens."""
    out: list[Difference] = []
    left_spec, right_spec = left_plan.spec, right_plan.spec

    # -- benchmark ---------------------------------------------------------
    left_benchmark = getattr(left_spec, "benchmark", None)
    right_benchmark = getattr(right_spec, "benchmark", None)
    if left_benchmark or right_benchmark:
        _add(out, "benchmark", "ref",
             left_benchmark.ref if left_benchmark else None,
             right_benchmark.ref if right_benchmark else None)
        _add(out, "benchmark", "digest",
             left_benchmark.digest if left_benchmark else None,
             right_benchmark.digest if right_benchmark else None,
             "the benchmark definition itself changed")

    # -- scientific inputs -------------------------------------------------
    _compare_entity(out, "dataset", left_plan.dataset, right_plan.dataset,
                    ("name", "version", "digest", "n_examples"))
    _compare_entity(out, "model", left_plan.model, right_plan.model,
                    ("name", "version", "weights_digest", "architecture",
                     "precision", "quantization", "tokenizer"))
    _compare_entity(out, "algorithm", left_plan.algorithm, right_plan.algorithm,
                    ("name", "version", "entrypoint", "commit", "source_digest"))

    # -- configuration -----------------------------------------------------
    left_configs = left_spec.configurations
    right_configs = right_spec.configurations
    _config_diff(out, left_configs[0] if left_configs else {},
                 right_configs[0] if right_configs else {})
    # The first configuration is the reference point, but a sweep can differ
    # further out and still be invisible: changing the *second* value of
    # [0.05, 0.2] to [0.05, 0.3] leaves configurations[0] identical and
    # n_configurations unchanged, so the whole sweep change would go
    # unreported. Compare the sets of configurations as a whole.
    _config_set_diff(out, left_configs, right_configs)
    _add(out, "configuration", "n_configurations",
         len(left_configs), len(right_configs),
         "the sweeps expand to different numbers of configurations")

    # -- arms --------------------------------------------------------------
    _add(out, "arms", "treatment", left_spec.treatment_name, right_spec.treatment_name)
    _add(out, "arms", "baselines",
         sorted(b.name for b in left_spec.baselines),
         sorted(b.name for b in right_spec.baselines))
    _add(out, "arms", "ablations",
         sorted(a.name for a in left_spec.ablations),
         sorted(a.name for a in right_spec.ablations))
    _add(out, "arms", "components",
         sorted(left_spec.components), sorted(right_spec.components))

    # -- seeds -------------------------------------------------------------
    _add(out, "seeds", "values",
         list(left_spec.seeds.values), list(right_spec.seeds.values))
    _add(out, "seeds", "strategy",
         left_spec.seeds.strategy, right_spec.seeds.strategy)

    # -- metrics -----------------------------------------------------------
    left_metrics = {m.name: m for m in left_spec.metrics}
    right_metrics = {m.name: m for m in right_spec.metrics}
    for name in sorted(set(left_metrics) | set(right_metrics)):
        a, b = left_metrics.get(name), right_metrics.get(name)
        if a is None or b is None:
            out.append(Difference("metrics", name,
                                  "declared" if a else None,
                                  "declared" if b else None,
                                  "measured by only one of the two"))
            continue
        _add(out, "metrics", f"{name}.direction",
             a.direction.value, b.direction.value)
        _add(out, "metrics", f"{name}.tolerance", a.tolerance, b.tolerance)

    # -- analysis and execution --------------------------------------------
    for name in ("primary_metric", "alpha", "confidence_level", "correction",
                 "paired", "bootstrap_resamples", "bootstrap_seed",
                 "permutation_resamples", "permutation_seed", "omnibus"):
        _add(out, "analysis", name,
             getattr(left_spec.analysis, name, None),
             getattr(right_spec.analysis, name, None))
    for name in ("isolation", "max_workers", "timeout_seconds", "retries",
                 "cache", "determinism_replicates", "energy_watts"):
        _add(out, "execution", name,
             getattr(left_spec.execution, name, None),
             getattr(right_spec.execution, name, None))

    return ExperimentDiff(
        left_id=left_plan.experiment.id,
        right_id=right_plan.experiment.id,
        left_name=left_spec.name,
        right_name=right_spec.name,
        differences=out,
    )


def diff_experiments(
    left_plan: Any, left_runs: Sequence[Any], left_analysis: Any,
    right_plan: Any, right_runs: Sequence[Any], right_analysis: Any,
) -> ExperimentDiff:
    """Compare two executed experiments: the plans, plus what happened.

    The environment and hardware comparison is the reason this exists as
    well as :func:`diff_plans`. Two identical specifications that produced
    different numbers differ *somewhere*, and if it is not in the plan then
    it is in the machine.
    """
    result = diff_plans(left_plan, right_plan)
    out = result.differences

    # -- the machines the runs actually happened on ------------------------
    for category, attribute in (("environment", "environment_id"),
                                ("hardware", "hardware_id")):
        left_ids = sorted({getattr(r, attribute, "") for r in left_runs} - {""})
        right_ids = sorted({getattr(r, attribute, "") for r in right_runs} - {""})
        _add(out, category, "ids", left_ids, right_ids,
             "the runs executed under different captured "
             f"{category}s, which Fabric records but cannot attribute blame to")

    # -- what came out -----------------------------------------------------
    _add(out, "results", "runs", len(left_runs), len(right_runs))
    _add(out, "results", "succeeded",
         sum(1 for r in left_runs if r.succeeded),
         sum(1 for r in right_runs if r.succeeded))

    metric = left_analysis.primary_metric or right_analysis.primary_metric
    if metric:
        left_results, right_results = left_analysis.results, right_analysis.results
        names = set()
        for results in (left_results, right_results):
            if results is not None:
                names |= set(results.conditions)
        for name in sorted(names):
            a = _condition_mean(left_results, name, metric)
            b = _condition_mean(right_results, name, metric)
            if a is None and b is None:
                continue
            if a is None or b is None:
                out.append(Difference("results", f"{name}.{metric}", a, b,
                                      "the condition ran in only one experiment"))
            elif abs(a - b) > 1e-12:
                out.append(Difference(
                    "results", f"{name}.{metric}", round(a, 6), round(b, 6),
                    f"mean {metric} moved by {b - a:+.6f}",
                ))

    result.differences = sorted(
        out, key=lambda d: (CATEGORY_ORDER.index(d.category)
                            if d.category in CATEGORY_ORDER else len(CATEGORY_ORDER),
                            d.field))
    return result


def _condition_mean(results: Any, name: str, metric: str) -> float | None:
    if results is None or name not in results.conditions:
        return None
    condition = results.conditions[name]
    if not condition.series(metric):
        return None
    return condition.summary(metric).mean
