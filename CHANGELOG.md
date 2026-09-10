# Changelog

All notable changes to Experiment Fabric. This project follows
[semantic versioning](https://semver.org/); before 1.0, minor versions may
change the on-disk layout, and the layout version in `.fabric/config.json`
records which one a workspace was written by.

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
