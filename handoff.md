# Handoff

Verified state at the end of this work. Numbers here come from commands
that were run; nothing is projected.

---

## 1. Current work

**Objective.** Take the repository to a defensible, demonstrable state:
scientific correctness first, then reliability, then evidence, then
usability.

**Status: complete for the scope below.** Seven defects were found and
fixed, each with a regression test confirmed to fail beforehand. The
remaining items in §9 are known and bounded, not deferred.

---

## 2. What was done

### 2.1 The multiple-comparison family shrank (highest severity)

`adjust()` computed its multiplier from the number of p-values that were
*usable*, so a comparison whose test could not run shrank the family the
survivors were corrected against.

Three declared baselines at p = 0.020 and 0.030, with the third untestable,
reported **two** rejections under the old behaviour and **zero** under the
correct one. Shrinking a family because a test happened to fail is deleting
a comparison after seeing the results — the bias these corrections exist to
prevent.

Fixed: the family size is `len(pairs)`; only the ranking uses usable
p-values; untestable entries are still reported, unrejected, with a NaN
adjusted p. D-001.

### 2.2 A flaky crash in the persistence layer

`read_json` caught `FileNotFoundError` but not the `PermissionError` that
Windows raises when a rename lands while a handle is open — which is what
`cache_lookup` does on every run, concurrently, while the runner writes the
same index. `atomic_write_text` already retried this; the reader did not.

Observed as a non-deterministic failure: `test_analyze_json_is_machine_readable`
on one run, `test_graph_exports_every_format` on the next. A stress harness
(one writer, four readers, 6 s) recorded **4 reader failures and 0 writer
failures**, isolating the reader as the side that could not cope. After the
fix: **0 reader failures**, successful reads 23,789 → 78,758. D-002.

### 2.3 A self-inflicted regression, caught immediately

Converting an unread `open()` call to satisfy `SIM115` kept the reader
handle alive across the write in `fabric/reproduce.py`. Since reproduction
recreates the same experiment id, source and destination are the same path —
and Windows refuses `os.replace` onto an open file. It failed **12 runs out
of 12**, deterministically. Fixed by reading, closing, then writing. D-003.

This is recorded because it is the clearest case in this work of a linter
being right about the context manager and wrong about where the line had to
go.

### 2.4 Shipped benchmarks were unreachable from an installed wheel

The search path's last entry was built from two `dirname` calls and pointed
at `fabric/benchmarks`, which did not exist; the definitions lived at the
repository root, in neither artefact. On a clean install, `fabric benchmarks`
printed `none found` and every shipped example failed at parse time.

`scripts/smoke.sh` copied `examples/` but never `benchmarks/`, so the CI job
named "Run every command the README claims" died on its second command
having asserted nothing.

Fixed: definitions ship as package data at `fabric/benchmarks/`. Verified by
building the wheel, installing into an empty virtualenv, and listing
benchmarks from an unrelated directory — all three resolve with digests
matching the checkout. D-004, D-005.

### 2.5 An unversioned benchmark reference resolved the wrong version

Versions were sorted as strings, so `v2` beat `v10`, and the *first* match
was returned. The resolved benchmark supplies metrics, baselines and
constraints and contributes its digest to the experiment's identity — so the
wrong one is analysed against a contract the author did not mean, invisibly.
Fixed: numeric ordering with a string fallback; resolve to the newest. D-006.

### 2.6 `fabric diff` could not see a sweep change

It compared `configurations[0]` and the configuration count. Index 0 is the
reference configuration by construction, so sweeping `scale` from `[1.0,
2.0]` to `[1.0, 9.0]` left both identical — and two experiments differing in
one of the two configurations they run were reported as *identical in every
field Fabric records*. Fixed by comparing the whole set by content
fingerprint. Verified directly against the real planner. D-008.

### 2.7 Four ways the server trusted data arriving over the network

| | Before | After |
|---|---|---|
| A web page could queue its own specification | loopback binding, no origin check | every `POST` refused unless `Sec-Fetch-Site` says same-origin or `Origin`/`Referer` matches `Host`; a `POST` with neither is refused |
| A submitted spec could choose where code runs | `isolation: inline` honoured, contradicting the module's own docstring | the server forces `process` |
| A submitted spec could size the thread pool | `max_workers: 100000` → 100,000 threads serving HTTP | capped at 8, server-side |
| A second server stole live jobs | every `RUNNING` job requeued at startup | requeued only if the claiming process is gone |

The last one needed care: `os.kill(pid, 0)` is **not** safe on Windows — any
signal other than the two console events is delivered by calling
`TerminateProcess`, so the naive liveness probe would kill the sibling
process the check exists to protect. A bare `OpenProcess` was also verified
to return a plausible handle for an already-exited process. Windows is asked
through `OpenProcess` + `GetExitCodeProcess`, which only read state. D-009
through D-012.

### 2.8 Documentation that did not match the code

- README claimed 251 tests (there were 442, now 537), claimed CI covered
  Python 3.14 (it did not), and stated a web UI was "deliberately not built"
  while shipping one.
- `docs/deployment.md` claimed every run is a subprocess (false for a
  submitted `isolation: inline`) and that runs are time-bounded (true only
  when a non-default field is set).
- The container `CMD` passed `--workspace` after the subcommand, which
  argparse rejects, so `docker compose up` could not start the service.

All corrected. CI now covers Python 3.14, and the container job checks that
the image's default command actually serves — nothing else in that job would
have noticed.

---

## 3. Files changed

| Path | Why |
|---|---|
| `fabric/statistics/multiple.py` | family size is the declared family |
| `fabric/util/jsonio.py` | shared retry for Windows sharing violations |
| `fabric/reproduce.py` | read before rewriting the same path |
| `fabric/core/benchmark.py` | shipped definitions resolve; numeric version order |
| `fabric/core/diff.py` | whole configuration set compared |
| `fabric/server/app.py` | cross-origin `POST` refused |
| `fabric/server/worker.py` | forced process isolation, worker cap, scoped recovery |
| `fabric/server/jobs.py` | liveness-scoped recovery |
| `fabric/cli/main.py` | corrected the `--workspace` error message |
| `pyproject.toml`, `Dockerfile`, `fabric/benchmarks/` | definitions ship |
| `scripts/smoke.sh` | copies what it depends on |
| `tests/test_statistics.py` | family-size regression tests |
| `tests/test_core.py` | sharing-violation and read-before-write tests |
| `tests/test_benchmarks.py` | shipped-definition and version-order tests |
| `tests/test_diff.py` | sweep-change tests |
| `tests/test_server.py` | cross-origin, isolation, cap, recovery tests |
| `tests/test_statistics_against_reference.py` | **new**, 68 cases |
| `.github/workflows/ci.yml` | Python 3.14; container health check |
| `README.md`, `docs/statistics.md`, `docs/deployment.md`, `docs/benchmarks.md` | claims matched to behaviour |
| `decisions.md`, `flow.md`, `handoff.md` | **new** |

---

## 4. Architecture, as verified

- `fabric/core/` entities, specifications, graphs, registry, provenance, store
- `fabric/execution/` planner, runner, adapters, cache, resources, failures
- `fabric/statistics/` distributions, tests, effect sizes, intervals, power, selection, multiplicity
- `fabric/analysis/` baselines, ablations, sweep, failures, determinism, reproducibility, reference
- `fabric/integrity/` contamination, drift, effort
- `fabric/reporting/` document model, renderers, plots, interactive graph, certificates
- `fabric/server/` HTTP API, SQLite job store, workers, UI assets
- `fabric/cli/` the `fabric` command

`fabric/` has **no third-party dependencies**. PyYAML is used when present
and a bundled parser is used when it is not; both are exercised by CI.

Call order for every workflow is in [`flow.md`](flow.md).

---

## 5. Decisions

`decisions.md` holds D-001 through D-012, each with context, root cause,
alternatives considered, consequences, files, and the regression tests. The
short version:

- D-001 correct the declared family, not the testable subset
- D-002 retry sharing violations on read as well as write
- D-003 read a file before rewriting it
- D-004 ship the benchmark definitions inside the package
- D-005 smoke.sh copies what it depends on
- D-006 order versions numerically
- D-007 `--workspace` precedes the subcommand
- D-008 compare the whole configuration set
- D-009 refuse cross-origin writes
- D-010 the server decides where submitted code runs
- D-011 the server caps its own workers
- D-012 scope crash recovery by liveness

---

## 6. Rules that must keep holding

1. **`dependencies = []` in `pyproject.toml`.** `scripts/check_no_dependencies.py`
   is in CI for a reason: a tool that measures reproducibility should not
   perturb the environment it measures.
2. **No fabricated numbers.** Where Fabric cannot compute something it says
   `unknown` or reports the limitation. `reference.py` deliberately declines
   to run a hypothesis test against a published scalar.
3. **No false success.** A failed run is recorded with its classification and
   empty metrics. Nothing is invented to fill a gap.
4. **"Unverified" over a green tick.**
5. **Local-first, trusted execution.** No remote execution, no
   authentication, no TLS — and the boundary is documented rather than
   implied.
6. **`decisions.md` and `flow.md` accompany behaviour changes.**
7. **Commits use `PrathamKapoor <prathamkapoor027@gmail.com>`** as both author
   and committer. No co-authors, no attribution trailers.
8. **`paper/`, `research/`, `.tex` are not committed.** `paper/` is tracked
   from before this work and its regeneration is left uncommitted (§9).

---

## 7. Testing and verification

### Commands run, and their outcomes

| Command | Outcome |
|---|---|
| `python -m pytest` (3.14.7) | **537 passed, 1 skipped** |
| `python -m pytest` (3.11.16) | 442 passed, 1 skipped |
| `python -m ruff check fabric tests examples` | All checks passed |
| `python scripts/check_no_dependencies.py` | 69 core modules, 1 guarded exception |
| `bash scripts/smoke.sh` | all 35 checks passed |
| SciPy cross-validation (separate venv, SciPy 1.17.1) | 68 cases; 4 apparent disagreements all explained and pinned as tests |
| `uv build` + install into an empty venv | wheel and sdist built; benchmarks resolve |
| `fabric run examples/agent_pipeline/experiment.yaml` | 120 runs, 2.3 s |
| `fabric analyze EXP-4607d0cd00` | 12 arms, 3 baselines, 7 limitations |
| `fabric certify` / `export` / `verify` | `INTACT`; tampering → `MODIFIED`, file named |
| `fabric reproduce bundle` | `PASS`, all five metrics exact |
| **fresh-environment reproduction** | wheel-only venv, no source tree: `init` → `run` → `export` → `verify` → `reproduce`. `INTACT`, then `PASS` — accuracy/f1/vocabulary_size exact, `train_seconds` +0.158 s within its 0.5 s tolerance. This is the strongest reproduction evidence here: a different installation, not a rerun in the same one |
| `fabric why --metric accuracy --seed 3` | full lineage to dataset digest |
| `fabric report --format all` | md, html, tex, json + 22 LaTeX snippets |
| `fabric serve` + real HTTP | 21 routes; UI served; traversal 403/404 |
| cross-origin `POST` over HTTP | **403**; same-origin 201 |

The one skip is `test_yaml_parity.py`, which requires PyYAML; CI runs a
second pass with it installed.

### Regression tests, each confirmed to fail before its fix

- `test_the_family_size_is_the_declared_family_not_the_testable_subset` (+2)
- `test_a_transient_sharing_violation_does_not_fail_the_reader`
- `test_reading_a_file_before_rewriting_it_works` (via 12/12 reproduction)
- `test_the_definitions_that_ship_with_fabric_are_on_the_search_path` (+2)
- `test_an_unversioned_reference_resolves_to_the_newest_version` (+2)
- `test_a_change_to_a_later_swept_value_is_reported` (+1)
- four cross-origin `POST` tests, `test_a_submitted_spec_cannot_choose_to_run_in_the_server_process`,
  `test_a_submitted_spec_cannot_ask_for_an_unbounded_thread_pool`,
  `test_starting_a_second_server_does_not_steal_a_running_job`,
  `test_a_job_whose_process_died_is_recovered`

### Not executed here

- **CI itself.** Never run; the workflow file was read and extended.
- **Linux and macOS.** This machine is Windows-only. CI covers both.
- **The container image.** No Docker daemon available. The `CMD` fix was
  verified against the real CLI's argparse behaviour, not by running the
  image; the new CI job is what will actually prove it.
- **Python 3.10, 3.12, 3.13.** Only 3.14.7 and 3.11.16 were available locally.
  The README no longer claims otherwise.

---

## 8. Known issues and risks

**Confirmed defects: none outstanding** in the areas this work touched. The
suite is green and every fix above has a test that fails without it.

**Risks, stated as risks:**

1. **The `fabric` console script collides with an existing PyPI package.**
   `fabric` (fabfile.org, SSH deployment, 131 releases) installs its own
   `fabric` command. Installing both means last-writer-wins on the
   executable, and `import fabric` resolves to whichever is first on
   `sys.path`. The distribution name `experiment-fabric` is unclaimed and
   correct. This is documented nowhere in the repo and should be — it is a
   real user-facing surprise, not a hypothetical.
2. **`agent-qa@v1`'s `require_determinism_check` constraint is unmet by the
   shipped example.** Deliberate — it demonstrates a non-conforming design
   producing a warning rather than a refusal. It is visible in the integrity
   report and in the limitations list.
3. **Wilcoxon's normal approximation above n = 20** overstates the p-value in
   the far tail by roughly an order of magnitude. So does SciPy's. It is an
   approximation, labelled as one, with the exact value computed in the same
   test file so the error is visible.
4. **A worker identity without a parseable pid is never recovered.** Chosen
   deliberately: sweeping an unparseable identity would duplicate a live
   sibling's job. The cost is that a crashed worker keeps its job, which is
   visible in the queue rather than silent.
5. **`timeout_seconds` defaults to 0**, meaning no timeout. A specification
   that does not set it can hang a worker indefinitely, and running jobs
   cannot be cancelled. Documented in `docs/deployment.md`; the default was
   not changed because it is a behavioural default the specification owns.
6. **`ConstraintResult.fatal` is vestigial** — serialized, read by nothing.
7. **`pyproject.toml` emits setuptools deprecation warnings** for the
   `license = { text = ... }` table and the License classifier. Removal is
   required by 2027-02-18.

---

## 9. Unfinished work

Ordered by severity. None of these is a release blocker for a trusted local
release; items 1 and 2 are the first things a careful user would hit.

1. **Document the `fabric` name collision** in the README. A paragraph
   stating that the distribution is `experiment-fabric`, that the import name
   and console script collide with an existing SSH tool, and what to do about
   it. Undocumented at present.
2. **Decide `timeout_seconds`' default.** Either set a finite default for the
   server, or add an explicit warning when it is 0 under `--execute`. Right
   now a hung run pins a worker with no bound.
3. **Give a submitted specification a bounded resource envelope** — output
   directory size, wall-clock across the whole job, not just per run. The
   per-run timeout and the worker cap exist; the aggregate does not.
4. **Add a `Content-Length`-less `POST` guard.** A chunked or
   `Transfer-Encoding` request currently reads to EOF. Harmless at the 1 MiB
   cap for the JSON path, but worth bounding explicitly.
5. **Remove `ConstraintResult.fatal`**, or make something read it.
6. **Clean the packaging deprecations** before 2027-02-18.
7. **Regenerate `paper/results/`** and commit it deliberately. It is
   currently modified but uncommitted (see below).
8. **`smoke.sh` runs under Git Bash on Windows**, not WSL. CI runs it on
   Linux and Windows, so this is only a local-ergonomics note.

---

## 10. Next actions

1. Add the name-collision note to the README (§9.1). Ten minutes, and it is
   the first thing a new user trips over.
2. Decide the timeout default (§9.2). It changes behaviour, so it needs a
   decision entry, not just a code change.
3. Run the suite on Python 3.10, 3.12 and 3.13 locally if interpreters are
   available; CI covers it otherwise.
4. Build and run the container once a daemon is available, to close the D-007
   verification gap the new CI job was written for.

---

## 11. Critical context

**The uncommitted `paper/` changes are deliberate and should stay that way
until someone chooses otherwise.** `paper/results/` is regenerated output
from the study in `paper/run_study.py`. It currently reflects runs made
against commit `749c511` with the working tree in a mid-session state, so its
experiment ids and spec digests do not match anything reproducible from the
current tree. Committing it would publish results that cannot be
reproduced from the commit they live in — which is the exact failure this
project exists to prevent. Regenerate it against a clean commit and commit
that instead.

**SciPy is not installed by Fabric and must not become a dependency.** It was
used as an independent oracle in a throwaway virtualenv. The values it
produced are frozen in `tests/test_statistics_against_reference.py`.

**`bash` on this machine resolves to WSL, which is not installed.** Use
`"C:\Program Files\Git\bin\bash.exe"` for `scripts/*.sh`.

**The benchmark definitions are duplicated on purpose** — at the repository
root for the examples' relative paths and the Dockerfile, and inside the
package for the wheel. They must stay byte-identical, or the digests will
diverge. `fabric/benchmarks/` is the copy that ships.

**The server's `--i-understand` flag is not ceremony.** Removing it removes
the only thing between a submitted specification and code execution as the
serving user.

---

## 12. Instructions for continuing work

1. Read `decisions.md` and `flow.md` before changing code. Both describe
   behaviour that was verified, not intended behaviour.
2. Every fix here follows the same shape: reproduce, add a regression test,
   confirm the test fails, fix, confirm it passes, run the full suite,
   document. Keep doing it.
3. Commit as `PrathamKapoor <prathamkapoor027@gmail.com>`, author *and*
   committer. No co-authors, no trailers, no phase-named branches or
   directories.
4. Do not claim a test passed because another document says it did. Run it.
5. If a number in a report, README or doc cannot be reproduced from the
   current tree, say so rather than updating it to look consistent.
6. Keep `dependencies = []`. If something genuinely needs a library, that is
   a decision entry and a discussion, not an import.
7. Prefer a bad result reported honestly to a good result obtained by
   adjusting a threshold. The project's value depends on it.