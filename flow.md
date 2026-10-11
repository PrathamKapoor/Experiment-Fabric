# System flow

What Fabric actually does, in the order it does it, and why each stage
exists. Every function named here was read in the source; where a claim is
about behaviour that was executed rather than read, it says so.

Names, paths and line numbers reflect the state at the end of this work.

---

## The shape of it

```
experiment.yaml ──► parse ──► plan ──► execute ──► store
                                    │
                                    ├─ cache? ──► reuse a completed run
                                    └─ worker subprocess / inline
                                              │
                                              ▼
                                    metrics + artifacts
                                              │
                                    ┌─────────┴─────────┐
                                    ▼                   ▼
                              statistics          integrity
                                    └─────────┬─────────┘
                                              ▼
                                     report / certificate
                                              │
                                    ┌─────────┴─────────┐
                                    ▼                   ▼
                                 export              reproduce
```

Everything below expands one of these boxes with the real call order.

---

## 1. Parse — `ExperimentSpec.from_yaml()` (`fabric/core/spec.py`)

**WHAT.** Turns a YAML document into validated dataclasses.

**WHEN.** `fabric run`, `fabric plan`, `fabric analyze` (from the stored copy),
and the HTTP server's submit endpoint.

**ORDER, in `from_dict()`:**

1. Benchmark resolution, if `benchmark:` is declared —
   `resolve_benchmark()` then `apply_benchmark()` (`spec.py:838-842`). The
   benchmark's metrics, dataset and baselines are merged into the raw
   mapping *before* any parser sees them, which is why a benchmark-injected
   baseline behaves identically to a declared one downstream.
2. `experiment`, `dataset`, `model`, `algorithm`, `configuration`, `seeds`,
   `baselines`, `ablations`, `metrics`, `analysis`, `execution`,
   `reproducibility`.
3. `digest` — the content hash excluding `source_path` and `raw`
   (`spec.py:754-771`). The benchmark reference and its digest are embedded
   here, so swapping a benchmark for a different version changes the
   experiment's identity.

**FAILURE.** `SpecError` naming the exact field path. The CLI turns it into
exit code 2.

**EVIDENCE.** `tests/test_core.py`, and `fabric plan` printing the resolved
benchmark with its digest.

**NOTE.** `spec.n_conditions` / `spec.n_runs` are computed *before*
perturbations and determinism replicates exist, so they are a prediction, not
the truth. `examples/agent_pipeline/reproducibility.yaml` predicts 20 and
plans 93. Only `len(plan.conditions)` and `len(plan.runs)` are authoritative,
and those are what `fabric plan` prints.

---

## 2. Plan — `plan_experiment()` (`fabric/execution/planner.py:327`)

**WHAT.** Expands one specification into every run it implies, plus the
provenance graph.

**ORDER — this matters, because it decides what the run counts mean:**

| Stage | Lines |
|---|---|
| Resolve dataset, model, algorithm | 339-343 |
| Advisories (dirty tree, no git, missing digest) | 345-362 |
| `experiment_id_for(spec)` | 364-375 |
| **Treatment arms** — one per configuration | 392-402 |
| `reference_config = configurations[0]` | 404 |
| **Ablation arms** — against the reference configuration | 410-425 |
| **Baseline arms** | 428-463 |
| **Perturbations** — only if `reproducibility.enabled` | 466-483 |
| Dedupe / `PlanError` on a duplicate condition name | 485-502 |
| `_build_runs()` — cross every condition with every seed | 505 |
| `build_graph()` | 507 |

**WHY this order.** Ablations are defined relative to the full system, so
they need the reference configuration to exist first. Baselines are
independent of the treatment, so they come after. Perturbations come last
because they are extra arms derived from whichever arms exist.

**IDENTITY.** `experiment_id_for` = `"EXP-" + content_id({spec, name}, 10)`;
`run_id_for` = `"RUN-" + content_id({experiment, condition, seed, replicate}, 12)`.
Both are content-addressed, so the same specification produces the same ids
on any machine — which is what makes caching and resumption possible at all.

**EXAMPLE, verified by running `fabric plan`:**
`examples/text_classification/experiment.yaml` declares one baseline
(`naive-bayes`) and references `sentiment-small@v1`, which contributes
`majority-class`. 2 configurations + 2 ablations + 2 baselines = 6 conditions
× 10 seeds = **60 runs**, and the plan prints all six conditions with
`majority-class` attributed to the benchmark. The documented example and the
executed plan agree; the second baseline is injected at parse time by
`apply_benchmark()` and is disclosed in the output.

---

## 3. Execute — `Runner.execute()` (`fabric/execution/runner.py:186`)

**WHAT.** Runs the pending runs, honouring the cache and the worker limit.

**PER RUN — `_execute_one` → `_run_once`:**

```
cache_key(run, algorithm, source_digest, environment, policy)   runner.py:241
  │
  ├─ is_cacheable and not --rerun ──► cache_lookup(key) ──hit─► reuse
  │                                                            store.py:221
  └─ otherwise run:
       RunContext(seed=run.seed, ...)                            runner.py:277
       ResourceMonitor(...)                                       runner.py:295
       isolation == "inline" ? _run_inline : _run_subprocess      runner.py:306
         WorkerCrash   -> keep the worker's own classification     runner.py:310
         TimeoutError  -> Failure(TIMEOUT)                        runner.py:313
         EntrypointError -> Failure(INVALID_OUTPUT, + traceback)   runner.py:315
       _collect_artifacts(output_dir)  -- hashes everything left  runner.py:322
       any NaN metric      -> Failure(NAN)                        runner.py:331
       save_run(experiment_id, completed)                         runner.py:367
       cache_store(key, ...)                                     runner.py:369
```

**SEEDS reach user code through five channels** — `ctx.rng()`, `PYTHONHASHSEED`,
`random.seed()` for inline, `FABRIC_SEED` for `command:` entrypoints, and
`_seed_global_rngs()` which seeds numpy/torch only if already imported
(`fabric/execution/worker.py:24`).

**CACHE IDENTITY** — `cache_key()` (`fabric/execution/cache.py:70`) hashes:
algorithm id, source digest, dataset id, model id, configuration, seed,
ablated components, and an environment key (python version, git commit, git
dirty flag, numerically-relevant package versions, accelerator runtime, env
vars). Hardware is included only when `policy.include_hardware`. A
determinism replicate is never cacheable, because its whole purpose is to
execute again (`cache.py:102`).

**FAILURE CLASSIFICATION** — `fabric/execution/failures.py`. `classify()`
orders its patterns so the first match wins: OOM, TIMEOUT, NAN, DIVERGENCE,
SEGFAULT, DEPENDENCY, NETWORK, INVALID_OUTPUT, METRIC_FAILURE, USER_ERROR,
UNKNOWN. Only NETWORK and DEPENDENCY are retried. A failed run is recorded
with `metrics={}` — nothing is invented.

**IDEMPOTENCE.** `pending_runs()` drops runs that already have a `succeeded`
record, so re-running resumes rather than repeats.

---

## 4. Analyse — `analyse()` (`fabric/analysis/engine.py:96`)

**ORDER — verified by reading the call sites, and the order is load-bearing:**

| # | Stage | Why here |
|---|---|---|
| 1 | `ResultSet.build()` | everything downstream reads this |
| 2 | `analyse_baselines()` | needs the omnibus test first |
| 3 | `analyse_ablations()` | comparison family is separate |
| 4 | `analyse_references()` | published values, no hypothesis test |
| 5 | `analyse_sweep()` | |
| 6 | `analyse_failures()` | |
| 7 | `analyse_determinism()` | |
| 8 | `_strongest_baseline()` | picks the comparator |
| 9 | `analyse_reproducibility()` | **consumes determinism and the comparator** |
| 10 | `run_checks()` — integrity | last of the checks, before interpretation |
| 11 | `cost_effectiveness()`, `aggregate()` | |
| 12 | `_claims()`, `_limitations()` | read everything above |

Reproducibility comes after determinism because it needs `analysis.determinism`
as input. Integrity is last of the checks because it compares the stored
specification against the one that was executed.

**TEST SELECTION** — `select_test()` (`fabric/statistics/selection.py:94`):

1. Fewer than two replications per condition → **refuse to test**, report why.
2. Pairs by seed when the seed sets match.
3. Anderson–Darling at n ≥ 8; below that the assumption is recorded as
   *unverified* rather than assumed either way.
4. Choose: normality not rejected → t-test; unverifiable or rejected at
   n < 12 → exact permutation; rejected at n ≥ 12 → rank test.
5. Record every assumption, including the ones that could not be checked.

**The design limit is reported before the result** —
`minimum_attainable_p()` states the smallest p-value the sample sizes could
produce, so five paired runs are told they cannot reach p < 0.0625 before
the compute is spent, not after.

**MULTIPLICITY** — `adjust()` (`fabric/statistics/multiple.py`). Correction is
applied within each metric's family: the number of baselines on that metric,
not the number of numbers in the report. **The family size is the declared
family**, so a comparison whose test could not run still counts; only the
ranking uses the p-values that exist. See D-001.

**VERDICTS** are derived, not asserted: `improvement` / `regression` when the
adjusted p clears alpha and the difference has a sign; `equivalent` when TOST
against the declared tolerance succeeds; `inconclusive` otherwise;
`not_testable` when no test ran. `equivalent` and `inconclusive` are
deliberately distinct — a failed test is not evidence of equivalence.

---

## 5. Store

**Everything is plain JSON under `.fabric/`**, written through
`atomic_write_text()`, which writes a temporary file in the destination
directory and renames it into place. `read_json()` retries the Windows
sharing violation that a rename causes (D-002), and raises `CorruptRecordError`
by name when a record exists but cannot be parsed — a damaged scientific
record is not the same as a missing one.

**Non-finite floats become `null`.** `sanitise()` converts NaN and ±Inf to
`null` because they are not in the JSON grammar, and `dumps()` sets
`allow_nan=False` so anything `sanitise` missed raises rather than producing a
file no other language can read. Every such field travels beside a `verdict`
or a narrative that says which case it was.

---

## 6. Report, certify, export, reproduce

| Command | What it does | Guarantee it makes |
|---|---|---|
| `fabric report` | Markdown, HTML, LaTeX, JSON from one document model | presentation only |
| `fabric certify` | A content-addressed certificate over code, data, environment, hardware, determinism | **what it checks**, and which checks could not run |
| `fabric export` | Spec, runs, analysis, provenance, report, manifest into a portable bundle | complete: `fabric verify` re-hashes every file |
| `fabric verify` | Re-hash artifacts, entities and the certificate | *integrity*: recorded data matches its expected hashes |
| `fabric reproduce` | Re-execute with the cache disabled, then compare | *reproduction*: fresh execution matches within declared tolerances |

**These are three different things and none implies another.** A valid hash
does not prove a scientific conclusion is right; a reproduction does not
prove the original was right; a passing certificate is tamper-evidence, not a
signature. `reproduce()` runs with `cache=CachePolicy(enabled=False)` and
`rerun=True` — a reproduction that reads its answer from the cache has
reproduced nothing.

**Verdicts:** `PASS`, `PASS WITH DEVIATION` (environment differences named and
ranked), `INCONCLUSIVE` (a sample-size limit, reported as one), `FAIL`.

---

## 7. Serve — `fabric serve` (`fabric/server/`)

**WHAT.** A local HTTP API and browser dashboard over the same workspace.

**Default posture:** bound to `127.0.0.1`, execution **off**. `--execute`
requires `--i-understand` as well, because it means "this server will import
and call whatever a submitted specification names, as this user, with no
sandbox".

**REQUEST PATH:**

```
HTTP request
  -> urlparse + unquote (before routing, so %2f is already a separator)
  -> _cross_origin_refusal()      POST only, 403 on a cross-origin drive   D-009
  -> body size check              1 MiB, drained then refused with 413
  -> dispatch(): first route whose method and pattern match
       ServiceError -> its own status
       anything else -> 500, traceback to the log and never to the client
  -> _respond(): nosniff and DENY on every path
```

**EXECUTION PATH:**

```
POST /experiments/{id}/run      requires --execute, else 403
  -> JobStore.enqueue()          row state=queued
  -> Worker._run_experiment()
       _options_from(spec, payload)     isolation forced to process  D-010
                                       workers capped at 8            D-011
       Runner.execute()           each run in its own subprocess
       store.progress(), store.finish()
```

**JOB STATES.** `queued → running → succeeded|failed|cancelled`. `claim()` is a
conditional `UPDATE ... WHERE state='queued'` inside a `BEGIN IMMEDIATE`
transaction; `rowcount` decides the winner, so two workers can never both
believe they own a job. Verified by a test with six threads and forty jobs
asserting exactly forty unique claims.

**CANCELLATION** applies to queued jobs only. A running job is not killed — it
is left alone, and the API says so with a 409 rather than pretending.

**CRASH RECOVERY** requeues a job only when the process that claimed it is
gone, decided by the pid already in the worker identity. `os.kill(pid, 0)` is
*not* safe on Windows — it terminates the process — so Windows is asked
through `OpenProcess` + `GetExitCodeProcess` (D-012).

**What this is not.** `ThreadingHTTPServer` is not a production web server;
there is no authentication, no TLS, and subprocess isolation is a crash
boundary, not a sandbox. `docs/deployment.md` states the threat model and the
gaps.

---

## What is enforced, and what is merely reported

**Enforced** — refuses to proceed:

- a specification that does not validate
- a duplicate condition name
- fewer than two replications, for any inference
- an out-of-range `confidence_level`, `permutation_resamples`,
  `bootstrap_resamples`
- a pinned benchmark metric whose `direction`/`tolerance`/`unit` is redefined
- a benchmark digest that does not match the pinned one
- a non-finite value reaching JSON
- `isolation: inline` and an unbounded worker count, under the server only
- a cross-origin `POST`, under the server only
- a path escaping the UI directory

**Reported, never blocking** — a bad design is made visible rather than
forbidden, because a tool that refuses to run teaches people to bypass it:

- unsatisfied benchmark constraints
- fewer replications than a constraint asks for
- a treatment swept over configurations while baselines were not
- low power, and the design's minimum attainable p-value
- determinism not tested
- a dirty working tree

---

## Verification performed

| Claim | How | Result |
|---|---|---|
| 537 tests pass | `python -m pytest` | 537 passed, 1 skipped |
| lint clean | `python -m ruff check fabric tests examples` | All checks passed |
| core has no dependencies | `python scripts/check_no_dependencies.py` | 69 modules, 1 guarded exception |
| every documented command works | `bash scripts/smoke.sh` | all 35 checks passed |
| statistics are correct | `tests/test_statistics_against_reference.py` | 68 cases vs SciPy 1.17.1 and enumeration |
| the wheel installs and works | `uv build`, install into an empty venv, `fabric benchmarks` | 3 definitions resolve |
| **reproduction in a fresh environment** | wheel-only venv → `fabric init` → run → export → verify → reproduce | `INTACT`, then `PASS`: accuracy, f1 and vocabulary_size exact; `train_seconds` differs by 0.158 s and still passes its own 0.5 s tolerance |
| an experiment runs end to end | `fabric run examples/agent_pipeline/experiment.yaml` | 120 runs in 2.3 s |
| analysis is coherent | `fabric analyze` | 12 arms, 3 baselines, 7 limitations |
| a bundle reproduces | `fabric reproduce bundle` | PASS, every metric exact |
| tampering is detected | edit `results.json`, `fabric verify` | `MODIFIED`, file named |
| the server works | real socket, `fabric serve` | 21 routes, UI served, traversal refused |
| a page cannot drive it | `POST` with `Sec-Fetch-Site: cross-site` | 403 |

Not verified here: Linux and macOS behaviour, and the container image — this
machine is Windows-only and no Docker daemon was available. CI covers both.