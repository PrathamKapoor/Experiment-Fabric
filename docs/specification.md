# The experiment specification

An experiment specification is a YAML file. It is the only thing you write
by hand, and it is the thing Fabric hashes: change it and you have a
different experiment, with a different identifier and a different set of
runs.

The rule the schema exists to enforce: **the analysis is declared before
the results exist.** Which metric is primary, what counts as a meaningful
difference, which correction is applied, which arms are compared — all of
it is fixed in the file, so none of it can be chosen after seeing the
numbers.

```bash
fabric plan experiment.yaml    # expand it, run nothing
fabric run experiment.yaml     # execute it
```

---

## A minimal specification

Everything not listed here has a default.

```yaml
experiment:
  name: my-first-experiment
  research_question: Does method X beat method Y on benchmark Z?

algorithm:
  name: my-method
  entrypoint: python:mypackage.train:run

metrics:
  - accuracy

seeds:
  values: [1, 2, 3, 4, 5]
```

Fabric refuses a specification with no research question. An experiment
without a question it answers is a script, and Fabric is not a script
runner.

---

## `experiment`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `name` | string | required | Human name. Appears in every report. |
| `research_question` | string | required | The question this experiment answers. |
| `hypothesis` | string | `""` | What you expect. Recorded before the runs, and printed beside the result whether or not it held. |
| `treatment_name` | string | `proposed` | What the main arm is called in tables. |
| `tags` | list | `[]` | Free-form labels. |
| `description` | string | `""` | Longer prose for the report. |
| `ablation_scope` | `reference` \| `all` | `reference` | Whether ablations run at the reference configuration only, or across every swept configuration. |

`research_question` and `hypothesis` may be written at the top level
instead of nested under `experiment`; both forms are accepted.

---

## `dataset`, `model`, `algorithm`

Each may be a reference to something already registered:

```yaml
dataset: benchmark-x@v3        # or: {name: benchmark-x, version: v3}
```

or an inline definition, which Fabric registers as it plans:

```yaml
dataset:
  name: benchmark-x
  version: v3
  path: data/benchmark-x       # hashed, and the hash becomes the identity
  source: https://example.org/benchmark
  preprocessing: deduplicated, lowercased
  n_examples: 12000
  license: CC-BY-4.0
  splits:
    train: {n: 10000, path: train.jsonl}
    test:  {n: 2000,  path: test.jsonl, role: evaluation}
```

Giving each split a `path` is what lets Fabric check for train/test
contamination. Without it, the check reports that it could not run rather
than passing silently.

```yaml
model:
  name: foo-7b
  version: v4
  weights_digest: sha256:...
  architecture: decoder-only transformer
  precision: bf16
  quantization: none
  tokenizer: v2
  parameters: 7000000000
  config: {capability: 0.0}    # anything your code needs

algorithm:
  name: my-method
  entrypoint: python:mypackage.train:run
  commit: auto                 # read from git; or pin a hash explicitly
  description: What the method does.
```

**A dataset's identity is its content.** Two registrations of the same
bytes under the same `name@version` are the same dataset however
differently they are annotated; two different sets of bytes under the same
version are a conflict, and Fabric refuses it by name.

### Entrypoint forms

| Form | Meaning |
|---|---|
| `python:package.module:function` | Import and call. The fast path. |
| `script:relative/path.py:function` | The same, for a file not on the import path. |
| `command:./train.sh --config {config_path}` | Any executable. Fabric writes the run description to JSON, passes its path in `$FABRIC_RUN`, and reads `metrics.json` back from `$FABRIC_OUTPUT_DIR`. |

See [`docs/writing-experiments.md`](writing-experiments.md) for the
contract your code has to satisfy.

---

## `configuration`

Every key is a parameter. **A bare list means "sweep these values."**

```yaml
configuration:
  learning_rate: [1e-4, 5e-5]                            # swept: 2 values
  batch_size: 32                                          # fixed
  layers: {fixed: [64, 64]}                               # a list that is one value
  dropout: {sweep: linear, low: 0.0, high: 0.5, count: 3} # 0.0, 0.25, 0.5
  temperature: {sweep: log, low: 1e-3, high: 1e-1, count: 3}
  width: {sweep: range, start: 64, stop: 512, step: 128}
  optimizer: {values: [adam, sgd]}                        # explicit form
```

Grid order is deterministic — parameters sorted by name, values in
declaration order — so the same file produces the same run identifiers on
every machine.

### `sweep`

```yaml
sweep:
  strategy: grid      # or: random
  samples: 12         # required for random
  seed: 7             # the random subsample is itself reproducible
```

Bayesian and population-based search are deliberately absent. Adaptive
search makes runs depend on each other, which breaks the clean provenance
that everything else here relies on; it is worth adding only once
deterministic execution is proven, not before.

---

## `seeds`

```yaml
seeds:
  strategy: fixed     # fixed | range | random
  values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

```yaml
seeds: {strategy: range, count: 10, start: 1}
seeds: {strategy: random, count: 10, seed: 1234}   # drawn once, then recorded
seeds: [1, 2, 3]                                    # shorthand
```

A "random" seed set is still written into the specification digest and the
run records. A seed set you cannot reproduce is not a seed set.

Fabric warns below five seeds and warns harder below three. It will also
tell you when your sample size makes significance *arithmetically
impossible* — with five paired runs, an exact test cannot produce a
p-value below 0.0625, so no result can clear `alpha = 0.05`.

---

## `baselines`

```yaml
baselines:
  - name: logistic-regression
    description: The conventional comparison.
    algorithm:
      name: logistic-regression
      entrypoint: python:mypackage.baselines:logreg
    configuration: {alpha: 1.0}        # merged over the reference configuration
    reference_value: {accuracy: 0.918} # a published number to reproduce
    source: Smith et al. 2024, Table 3
```

Each baseline may override the `algorithm`, `model` or `dataset`. When a
baseline overrides a *swept* parameter, Fabric says so in the report: it
means the arms are not being compared at the same operating point, which
may be correct but should not be silent.

---

## `components` and `ablations`

Declare the parts of the system, then remove them:

```yaml
components: [retrieval, reranking, memory, planning, verification]

ablations:
  - remove_retrieval
  - remove_memory
  - name: remove_planning+verification
    remove: [planning, verification]
    description: The two most expensive components together.
    configuration: {timeout: 30}       # optional overrides
```

Shorthand names (`remove_x`, `no-x`, `-x`, `remove_a+b`) are parsed into
component sets and checked against `components`, so a typo is an error
rather than an ablation that removes nothing.

Your code sees this through `ctx.enabled("retrieval")`. Write the system
once with those guards and every ablation becomes runnable without
touching it.

---

## `metrics`

```yaml
metrics:
  - accuracy                       # direction inferred: maximize
  - latency_ms                     # inferred: minimize (name contains "latency")
  - name: f1
    direction: maximize
    primary: true
    unit: ""
    tolerance: 0.01
    description: Macro F1 on the held-out split.
```

**`tolerance` is the most important field in the file.** It is the
difference you would consider practically meaningful, declared in advance.
With it, Fabric can run equivalence tests and answer *"is this the same
within X?"* — which is the question a reproduction actually asks. Without
it, the only available statement is "we failed to detect a difference,"
which is much weaker and is often misread as agreement.

Direction is inferred from the name when not given: names containing
`loss`, `error`, `latency`, `time`, `cost`, `memory`, `perplexity` or
`regret` are treated as lower-is-better.

---

## `analysis`

Declared before the runs, and applied unchanged afterwards.

```yaml
analysis:
  primary_metric: accuracy
  alpha: 0.05
  confidence_level: 0.95    # defaults to 1 - alpha
  correction: holm          # none | bonferroni | holm | benjamini-hochberg | sidak
  paired: auto              # auto | always | never
  bootstrap_resamples: 5000
  bootstrap_seed: 12345
  permutation_resamples: 10000
  permutation_seed: 12345
  omnibus: true             # ANOVA gate before pairwise tests
```

| Field | Type | Default | Meaning |
|---|---|---|---|
| `primary_metric` | metric name | the metric marked `primary`, else the first | The one the headline, the reproducibility axes and the decision-stability estimate are read on. Must be one of `metrics`. |
| `alpha` | `0 < x < 1` | `0.05` | The error rate the *decision* accepts. Drives every verdict. |
| `confidence_level` | `0 < x < 1` | `1 - alpha` | The width of every reported **interval**. |
| `correction` | enum | `holm` | Multiplicity correction within each metric's family. |
| `paired` | enum | `auto` | `auto` pairs by seed where the seed sets match. |
| `bootstrap_resamples` | `>= 1` | `5000` | Resamples for the BCa interval on a difference. |
| `bootstrap_seed` | int | `12345` | Makes that interval reproducible. |
| `permutation_resamples` | `>= 1` | `10000` | Budget for a permutation test. |
| `permutation_seed` | int | `12345` | Makes a sampled permutation p-value reproducible. |
| `omnibus` | bool | `true` | Run an ANOVA across all arms before the pairwise tests. |

Fabric chooses the *test* itself, from the structure of the data, and
records why. See [`docs/statistics.md`](statistics.md).

### `alpha` and `confidence_level` are not the same knob

They answer different questions, so Fabric keeps them apart. Alpha is the
error rate the decision accepts; the confidence level is how wide an
interval the reader is shown. A study may quite reasonably test at 0.05
and report 99% intervals, and this is how:

```yaml
analysis:
  alpha: 0.05
  confidence_level: 0.99
```

Every interval in the report, the CLI and the JSON then carries
`"level": 0.99` and is correspondingly wider; every verdict is still
decided at 0.05. Omit `confidence_level` and it follows alpha, which is
the conventional pairing.

### `permutation_resamples` decides how the null is explored

Not merely how long it takes. Fabric enumerates the permutation
distribution exactly when the budget can afford it and samples when it
cannot, so raising this value can promote a test from Monte-Carlo to
exact:

```
permutation_resamples: 200     ->  Monte-Carlo label shuffling, 200 resamples, seed 12345
permutation_resamples: 20000   ->  exact enumeration over 12870 label assignments
```

Which one ran is recorded in `decision.parameters` and in the test's
`method_detail`, so a p-value is never anonymous. A sampled p-value uses
the `(r+1)/(B+1)` convention and therefore can never be reported as
exactly zero.

---

## `reproducibility`

The perturbation study. Each axis changes exactly one thing.

```yaml
reproducibility:
  enabled: true
  determinism_replicates: 3      # execute an identical run this many extra times
  reference_condition: full-agent
  perturbations:
    environment:
      values:
        - {label: single-threaded, OMP_NUM_THREADS: "1", MKL_NUM_THREADS: "1"}
        - {label: hash-randomised, PYTHONHASHSEED: random}
    hardware:
      values:
        - {label: cpu-only, CUDA_VISIBLE_DEVICES: ""}
    dataset:
      values: [{name: benchmark-x, version: v4, path: data/benchmark-x-v4}]
    model:
      values: [{name: foo-7b, version: v5, weights_digest: sha256:...}]
    implementation:
      values: [{name: method-v2, entrypoint: python:mypackage.train:run_v2}]
    configuration:
      values: [{label: hot, temperature: 0.8}]
    seed:
      values: [101, 102, 103]     # extra seeds for the reference condition
```

`environment` and `hardware` values are environment-variable overlays, so
both axes are executable on one machine — which is what makes the
perturbation study something you can actually run rather than something
you plan to do across a cluster one day.

---

## `execution`

```yaml
execution:
  max_workers: 4
  timeout_seconds: 600
  isolation: process        # process (crash-isolated) | inline (faster)
  cache: true
  retries: 0                # only infrastructure failures are retried
  continue_on_failure: true
  determinism_replicates: 0
  energy_watts: 65          # optional; see below
```

| Field | Type | Default | Meaning |
|---|---|---|---|
| `max_workers` | `>= 1` | `1` | Runs executed concurrently. |
| `timeout_seconds` | `>= 0` | `0` (none) | A run exceeding this is killed and classified as a timeout. |
| `isolation` | `process` \| `inline` | `process` | See below. |
| `cache` | bool | `true` | Reuse a previous run whose every input matches. |
| `retries` | `>= 0` | `0` | Retries, and only for infrastructure failures. |
| `continue_on_failure` | bool | `true` | A failed run does not abandon the experiment. |
| `determinism_replicates` | `>= 0` | `0` | Extra identical runs, to test whether the pipeline is deterministic at all. |
| `energy_watts` | `> 0` | unset | Declared average power draw. See below. |

`process` runs each execution in a fresh interpreter, so a segfault costs
one run rather than the sweep. `inline` is faster and appropriate for
lightweight pure-Python experiments.

### `energy_watts`: an estimate, labelled as one

Fabric cannot measure power. There is no portable way to read a machine's
draw, and inventing a figure from CPU time would be worse than reporting
nothing. So energy is reported **only** when you declare what the machine
draws, and everything derived from it is labelled an estimate:

```yaml
execution:
  energy_watts: 65        # or: fabric run spec.yaml --energy-watts 65
```

```
Energy    0.83 Wh -- ESTIMATED, not measured: estimated: declared 65 W x wall time
```

Without it the report says `not measured -- no energy_watts was declared,
and Fabric will not invent one`, and `energy_joules` is `null` rather than
a plausible-looking number nobody stood behind. The declared wattage
travels with the estimate in every run record, so a reader can check the
assumption instead of trusting the product of it.

The same rule governs cost: `--gpu-cost` and `--cpu-cost` produce an
estimate from a declared hourly price, and the report says `ESTIMATED`.

---

## `integrity`

```yaml
integrity:
  require_dataset_digest: true
  forbid_test_in_train: true
  detect_duplicate_examples: true
  max_train_test_overlap: 0.0
  freeze_baselines: true
```

These control which checks run. They do not control what the checks
*say*: a check that cannot run reports that it could not run.

---

## What Fabric will tell you before spending compute

`fabric plan` prints advisories. They are the things a good reviewer would
raise before the cluster time is gone rather than after:

- fewer than five seeds, or fewer than three
- no baselines, so no comparative claim is possible
- a primary metric with no `tolerance`
- a dataset with no digest
- a dirty git tree, so the commit does not describe the code
- ablations declared without a `components` list to check them against
- an expansion above 5000 runs
- a design whose smallest attainable p-value is above `alpha`
