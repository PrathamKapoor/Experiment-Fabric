# Changelog

All notable changes to Experiment Fabric. This project follows
[semantic versioning](https://semver.org/); before 1.0, minor versions may
change the on-disk layout, and the layout version in `.fabric/config.json`
records which one a workspace was written by.

## [0.3.1]

Closing the gaps between what the project claimed and what it had
verified. Every item below was found by building the verification, not by
reading the code.

### Added

- **CI** across 3 platforms x 4 Python versions, with a bare install that
  proves the core still needs nothing, the suite run with and without
  PyYAML, a wheel built into a clean virtualenv, a container build, and a
  cross-version job that exports a bundle on 3.10 and reproduces it on
  3.13. Verified locally on 3.10 through 3.14.
- `scripts/check_no_dependencies.py`, which fails if anything third-party
  enters the core, and `scripts/smoke.sh`, which executes every command
  the README claims and asserts its output.
- **A real PyTorch example** (`examples/torch_mlp/`): an MLP trained with
  autograd on a spiral task, with its own perturbation study. The first
  thing in the project that is not a simulation.
- **An interactive provenance graph** in the HTML report and via
  `fabric graph --format html`: click a node and its lineage lights up,
  which is the `fabric why` query performed with a mouse. Inline SVG and
  vanilla JavaScript, no dependencies.
- `ExperimentGraph.collapsed()`, which folds a 913-node graph to 45 while
  keeping every structural relationship, and drops nodes orphaned by the
  fold rather than drawing relationships it cannot show.
- **An evidence-based related-work survey** (`docs/related-work.md`)
  checked against primary sources, including corrections to this
  project's own earlier claims.
- **A pilot study that ran** (`paper/run_study.py`, `paper/results/`):
  two real pipelines, one simulated and excluded, with certified bundles.
- A **Dockerfile** whose purpose is a pinned environment for reproduction,
  plus `scripts/container_check.sh` and a CI job that reproduces a bundle
  inside the image.

### Fixed

- **Experiment identity depended on whether PyYAML was installed.**
  PyYAML implements YAML 1.1, where `1e-4` is a string rather than a
  float; the bundled parser read it as a float. Since the specification
  digest is computed from the parsed document, the same file had two
  different experiment ids in different environments, silently breaking
  caching, resumption and cross-machine comparison. The README's own
  example used `1e-4`. Fabric now defines the canonical reading and
  applies it to both parsers, mapping keys are always strings, and the
  emitter writes floats both parsers read back as floats.
- **Cross-site scripting in the generated report.** Node labels come from
  dataset and condition names; a label containing a closing script tag
  terminated the embedded JSON block and let the rest execute as HTML in
  anyone who opened the report.
- **A stopwatch made pipelines look non-deterministic.** The determinism
  check judged `train_seconds`, which varies between identical runs by
  construction. Metrics now carry a `deterministic` expectation, inferred
  from the name and overridable, and timing metrics are reported but
  excluded from the verdict.
- `MetricSpec` inferred direction and determinism only through
  `from_any()`, so the direct constructor disagreed with it. Both are
  resolved in `__post_init__`.
- An experiment with no baselines reported "No treatment condition
  produced results" above a table of its own successful runs.
- A metric that reproduced bit-identically was reported INCONCLUSIVE
  because it had no declared tolerance. Exact equality needs no margin
  and now reports PASS.

### Corrected

- The comparison table claimed established trackers do not version
  datasets. **MLflow does**: `mlflow.data` computes a digest
  automatically. The real difference is narrower and is now stated as
  such. DVC is stronger at data versioning than Fabric, not weaker.
- The paper's research question is not novel, and `paper/` now says so:
  prior work already demonstrates a non-seed axis exceeding the seed
  axis. H1 is framed as a replication at greater breadth.

## [0.3.0]

The research-grade layer. Everything the MVP promised, plus the
reproducibility instrument the project exists for.

### Added

- **Reproducibility measurement** on six perturbation axes (environment,
  hardware, dataset, model, implementation, configuration), each changing
  exactly one thing. Stability is `1 - |Cliff's delta|` calibrated against
  the relabelling null, reported with an interval, a permutation p-value,
  the shift in units of seed noise, and a three-way tolerance verdict.
- **Determinism testing** by executing identical runs and comparing them,
  with the resulting noise floor propagated into the perturbation
  analysis.
- **Decision stability**: the probability a half-sized replication reaches
  the same conclusion, estimated by seed subsampling.
- **Minimum detectable effect** on every comparison, significant or not,
  plus `required_replications()` for the follow-up question.
- **Design-limit detection**: Fabric refuses to let an experiment run
  quietly when its smallest attainable p-value is above alpha.
- **Failure analysis**: classification into ten causes, per-condition
  rates, chi-squared attribution to configuration values, and separation
  of infrastructure failures from method failures.
- **Ablation interactions**: joint removals compared against the sum of
  individual removals, labelled additive, synergistic or redundant.
- **Integrity checks**: dataset digest verification, exact-match
  train/test contamination detection, seed alignment, metric
  completeness, replication depth, specification drift, and baseline
  tuning asymmetry.
- **Reproduction certificates**, content-addressed and tamper-evident.
- **Portable bundles** (`fabric export`, `verify`, `reproduce`) carrying
  the specification, runs, analysis, provenance, report and certificate as
  plain text.
- **Reports** in Markdown, HTML, LaTeX and JSON from one document model,
  with dependency-free inline SVG charts that follow the reader's light or
  dark theme.
- **Provenance queries**: `fabric why` traces a number to everything that
  produced it; `fabric impact` traces forwards to what a change breaks.
- **Cost accounting**: wall, CPU and GPU time, peak memory, and optional
  cost, with a quality-against-compute comparison.
- A second worked example using real algorithms and no `fabric` import.

### Changed

- Entity identity is now **content**, not annotation. A dataset with a
  digest is identified by `(name, version, digest)`; describing it more
  fully in a second experiment no longer mints a second dataset.
- A dirty git tree no longer changes an algorithm's identity. It is
  recorded, reported as an advisory, and downgrades the `code` check on
  the certificate — but it does not invalidate every cached run because an
  unrelated file was edited.
- The seed axis is reported as the noise floor rather than scored. An
  earlier split-half score measured its own estimator's sampling noise and
  reported a perfectly stable experiment as 0.69.
- Cache hits now materialise their artifacts in the cached run's own
  directory, so every run is self-contained and verifiable.

### Fixed

- A quoted YAML scalar containing a colon (`- "note: here"`) was parsed as
  a mapping.
- Concurrent workers could corrupt the shared cache, registry and
  experiment indexes; all three are now guarded by a re-entrant
  cross-process lock, and writes retry on the Windows rename contention
  that made this fatal rather than merely lossy.
- A corrupt JSON record raised a decoder traceback instead of naming the
  damaged file.
- A zero-variance paired comparison reported a standardised effect of 0.0
  where the correct answer is unbounded.

## [0.2.0]

- Experiment graphs, provenance capture, environment and hardware
  fingerprinting, the execution cache, and the statistical engine
  (distributions, tests, effect sizes, intervals, corrections) implemented
  without third-party dependencies.

## [0.1.0]

- The MVP: specifications, expansion into conditions and runs, execution
  with seeds and sweeps, baselines, ablations, the registry, the
  workspace, and the `fabric` command line.
