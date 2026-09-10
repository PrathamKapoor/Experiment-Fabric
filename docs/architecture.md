# Architecture

How the pieces fit, and why the seams are where they are.

```
                    experiment.yaml
                          │
                    ┌─────▼─────┐
                    │   spec    │  parse, validate, advise, digest
                    └─────┬─────┘
                          │
                    ┌─────▼─────┐     ┌──────────┐
                    │  planner  │◄────┤ registry │  content-addressed entities
                    └─────┬─────┘     └──────────┘
                          │
             conditions × seeds  ──►  experiment graph
                          │
                    ┌─────▼─────┐     ┌──────────┐
                    │  runner   │◄────┤  cache   │
                    └─────┬─────┘     └──────────┘
                          │            ┌──────────────┐
                     run records ◄─────┤ environment  │
                          │            │ + hardware   │
                    ┌─────▼─────┐      └──────────────┘
                    │ analysis  │  baselines, ablations, sweeps,
                    └─────┬─────┘  failures, determinism, reproducibility
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
         provenance  integrity   statistics
              │           │           │
              └───────────┼───────────┘
                          ▼
                    ┌───────────┐
                    │ reporting │  one document → md / html / tex / json
                    └─────┬─────┘
                          ▼
                 certificate, bundle
```

## Dependency direction

Strictly upward. `util` imports nothing of Fabric's; `core` imports
`util`; `execution` imports `core`; `analysis` imports `statistics` and
`core`; `reporting` imports everything below it, and nothing imports
`reporting` except the CLI.

`statistics` is deliberately independent of everything else in the
project: it takes lists of floats and returns results, so it is usable --
and testable -- entirely on its own.

## Layers

| Package | Responsibility | Depends on |
|---|---|---|
| `fabric/util` | Hashing, canonical JSON, atomic writes, file locking, YAML | stdlib only |
| `fabric/core` | Entities, specifications, graphs, registry, workspace, provenance | `util` |
| `fabric/environments` | Environment and hardware capture, differencing | `core` |
| `fabric/execution` | Planner, runner, adapters, worker, cache, resources, failures | `core`, `environments` |
| `fabric/statistics` | Distributions, tests, effect sizes, intervals, power, selection | stdlib only |
| `fabric/analysis` | Baselines, ablations, sweeps, failures, determinism, reproducibility | `statistics`, `execution` |
| `fabric/integrity` | Contamination, drift and effort checks | `analysis` |
| `fabric/reporting` | Document model, renderers, charts, certificates | everything below |
| `fabric/cli` | The `fabric` command | everything |

## The identity rules

Everything is content-addressed, but *what counts as content* differs by
kind, and the difference is deliberate:

- **Dataset** -- identity is `(name, version, digest)` when a digest
  exists. Annotations (source, preprocessing, licence, schema) describe
  the data; they do not change which data it is. Without a digest, every
  declared field participates, because there is nothing else to address it
  by.
- **Model** -- identity is the weights digest plus everything that changes
  inference: precision, quantization, tokenizer, config.
- **Algorithm** -- identity is the entrypoint, commit, source digest,
  version and params. Description and *dirtiness* are excluded: what pins
  a method is the hash of its source file, and if dirtiness changed
  identity, editing any unrelated file in the repository would invalidate
  every cached run. Dirtiness is recorded and flagged instead, and the
  certificate downgrades the `code` check when it is set.
- **Condition** -- everything that defines an arm: algorithm, model,
  dataset, configuration, ablation set, perturbation.
- **Run** -- `(experiment, condition, seed, replicate)`.
- **Experiment** -- the specification digest.

Determinism of these identifiers is what makes caching, resumption and
cross-machine comparison possible at all.

## On-disk layout

```
.fabric/
  config.json
  registry/<kind>/<id>.json      content-addressed entities
  registry/index.json            name@version -> id
  experiments/index.json
  experiments/<EXP>/
    experiment.yaml              the specification, as it was
    plan.json  graph.json
    runs/<RUN>/run.json
    runs/<RUN>/artifacts/
    analysis.json  provenance.json  execution.json
    reports/  certificate.json  reproduction.json
  cache/index.json
```

Plain JSON, everywhere. A scientific record that has to outlive its
tooling must be readable without it.

## Concurrency

Runs execute in a thread pool, and each run is a separate process by
default, so parallelism is real and a segfault costs one run rather than
the sweep. The three shared index files are guarded by a cross-process
lock (`O_CREAT | O_EXCL`, with stale-lock breaking and re-entrancy), and
every write is atomic: write to a temporary file in the destination
directory, then rename, with a short retry because Windows refuses a
rename while another handle has the target open.

## Extension points

- **A new entrypoint scheme**: `fabric/execution/adapters.py`.
- **A new statistical test**: add it to `hypothesis.py` and teach
  `selection.py` when to choose it. The decision procedure is one
  function, not conditionals scattered through the analysis.
- **A new perturbation axis**: add it to `AXES` in `core/spec.py` and give
  the planner a branch that builds the counterfactual condition.
- **A new report format**: consume `reporting/document.py`. Renderers
  never touch the analysis, which is why the four existing formats cannot
  disagree with each other.
- **A new integrity check**: a function in `integrity/checks.py` that
  appends a `Finding`.
