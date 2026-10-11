# Engineering decisions

Chronological. Each entry records one decision, why it was made, and what
it cost. Entries are append-only; a reversed decision says so and names the
entry it supersedes.

Identifiers are `D-NNN` and are never reused.

---

### D-001: A comparison that could not be tested still counts towards the family

- Date / work area / commit: 2026-10-11, `fabric/statistics/multiple.py`, uncommitted
- Context: `adjust()` computed the multiplicity multiplier from the number
  of p-values that were *usable*, so a comparison whose test could not run
  shrank the family the surviving comparisons were corrected against. With
  three declared baselines where one produced no p-value, Holm applied the
  multiplier sequence 2, 1 instead of 3, 2.
- Decision: the family size is `len(pairs)` — the declared family. Only the
  *ranking* uses the usable p-values. Untestable entries are still reported,
  with a NaN adjusted p-value and `rejected=False`.
- Why: this is the same manoeuvre as deleting a comparison after seeing the
  results, which is the bias multiple-comparison control exists to prevent.
  It was not conservative in the abstract — it was anti-conservative in the
  case that matters. Three declared baselines at p = 0.020 and 0.030 with one
  untestable reported **two** rejections under the old behaviour and **zero**
  under the corrected one. R's `p.adjust` treats the declared `n` the same
  way: it filters the working vector but keeps the original family size.
- Alternatives considered: (a) drop untestable comparisons from the report
  entirely — rejected, because then a reader cannot tell the difference
  between "we tested two baselines" and "one baseline produced no number".
  (b) refuse to correct at all when the family is incomplete — rejected as
  needlessly brittle; the corrected values are the honest ones.
- Consequences: adjusted p-values can rise, and comparisons that previously
  read as significant may now read as inconclusive. That is the correct
  direction for an error rate. `Correction.family_size` now means the
  declared family, which is what callers and reports display.
- Regression test: `tests/test_statistics.py`,
  `test_the_family_size_is_the_declared_family_not_the_testable_subset`,
  `test_an_untested_hypothesis_makes_the_correction_more_conservative`,
  `test_a_family_of_only_untestable_hypotheses_reports_all_of_them`.
  All three failed against the previous implementation and pass now.
- Files: `fabric/statistics/multiple.py`, `tests/test_statistics.py`.

---

### D-002: Read retries the Windows sharing violations that write already retries

- Date / work area / commit: 2026-10-11, `fabric/util/jsonio.py`, uncommitted
- Context: `atomic_write_text` already retried `PermissionError` from
  `os.replace`, because Windows refuses to rename onto a file any handle has
  open. `read_json` caught only `FileNotFoundError`, so a reader could catch
  the mirror-image violation and fail the whole call. `Workspace.cache_lookup`
  reads the cache index unlocked from every worker thread while the runner
  writes it, which is exactly that shape.
- Observed symptom: `PermissionError: [Errno 13] Permission denied:
  ...\.fabric\cache\index.json` at `fabric/util/jsonio.py:140`, from
  `tests/test_cli.py::test_analyze_json_is_machine_readable` and
  `tests/test_cli.py::test_graph_exports_every_format` — different tests on
  different runs, i.e. non-deterministic.
- Verified root cause: a stress harness with one writer and four readers on
  one path recorded 4 reader failures and 0 writer failures over 6 seconds,
  with the writer's `os.replace` succeeding every time. The reader was the
  only side that could not cope.
- Decision: both directions go through one helper,
  `_retry_on_sharing_violation`, with the same backoff and the same
  exhausted-attempts behaviour — raise, never return a silent default.
- Alternatives considered: (a) make `cache_lookup` take the file lock —
  rejected, it would serialise every read behind every write for no
  correctness gain, and the index is small enough to read without one.
  (b) let `read_json` return the default on `PermissionError` — rejected,
  because that converts a transient lock into silent data loss, which is
  strictly worse than a visible error.
- Consequences: a reader now tolerates the same transient windows the writer
  always did. Measured effect on the stress harness: reader failures 4 → 0,
  successful reads 23,789 → 78,758 over the same window.
- Regression test: `tests/test_core.py`,
  `test_a_transient_sharing_violation_does_not_fail_the_reader`,
  `test_a_persistent_sharing_violation_is_still_reported`,
  `test_a_missing_file_is_still_a_missing_file_not_a_sharing_violation`.
  The first was confirmed to fail against the previous implementation.
- Files: `fabric/util/jsonio.py`, `tests/test_core.py`.

---

### D-003: Read a file before rewriting it, rather than holding the handle across the write

- Date / work area / commit: 2026-10-11, `fabric/reproduce.py`, uncommitted
- Context: converting
  `workspace.write_text(..., open(spec_path).read())` into a `with` block to
  satisfy `SIM115` kept the reader handle alive across `write_text`. When
  the source and the destination are the same path — which they are for
  `fabric reproduce`, because `plan_experiment` derives the same experiment
  id from the same spec — `os.replace` then targets a file the process still
  has open.
- Observed symptom: `PermissionError: [WinError 5] Access is denied:
  ...\experiment.yaml`, failing 12 runs out of 12, not intermittently.
- Verified root cause: on Windows a handle opened for reading does not grant
  `FILE_SHARE_DELETE`, so `os.replace` onto it is refused. Confirmed by
  isolated harness: writing onto a file with a live reader handle raises
  `PermissionError: Access is denied`; closing first succeeds.
- Decision: read into a variable, leave the `with`, then write.
- Alternatives considered: (b) writing to a distinct path — rejected, it
  changes the on-disk layout to work around a local scoping mistake.
- Consequences: `fabric reproduce` works again on Windows. This also means
  the lint fix that introduced it was verified by running the affected test
  rather than by trusting the linter — the linter was right about the
  context manager and wrong about where the line had to go.
- Regression test: `tests/test_core.py`,
  `test_reading_a_file_before_rewriting_it_works`.
- Files: `fabric/reproduce.py`, `tests/test_core.py`.

---

### D-004: Ship the benchmark definitions inside the package

- Date / work area / commit: 2026-10-11, `pyproject.toml`,
  `fabric/core/benchmark.py`, `fabric/benchmarks/`, uncommitted
- Context: `search_paths()` ends with the directory holding the definitions
  that ship with Fabric. That entry was built from two `dirname` calls on
  `fabric/core/benchmark.py`, which reaches `fabric/` — a directory that did
  not exist. `benchmarks/` lived at the repository root and was in neither
  the wheel nor the sdist.
- Observed symptom: on a clean install of the built wheel, `fabric benchmarks`
  printed `none found`, and every shipped example failed at parse time with
  `no benchmark named 'agent-qa@v1' was found`. `scripts/smoke.sh` failed
  the same way in CI, because it copied `examples/` but not `benchmarks/`.
- Decision: the definitions ship as package data at `fabric/benchmarks/`,
  the search path resolves there, and the repository-root `benchmarks/` is
  kept because the examples' relative paths and the Dockerfile expect it.
  Both copies are byte-identical, so the digests agree.
- Why this location: it is the only one that works for a wheel. Shipping at
  the distribution root would need `[tool.setuptools.data-files]`, which
  installs outside the package and is not importable.
- Alternatives considered: (a) require `FABRIC_BENCHMARK_PATH` in every
  environment — rejected, it makes a fresh install non-functional.
  (b) drop the benchmark-in-package idea and ship nothing — rejected, the
  examples are the product's front door.
- Consequences: `fabric benchmarks` lists the shipped definitions on a bare
  install. Verified by building the wheel, installing it into an empty
  virtualenv, and running `fabric benchmarks` from an unrelated directory:
  all three definitions resolve with matching digests.
- Files: `pyproject.toml`, `fabric/core/benchmark.py`, `fabric/benchmarks/`,
  `docs/benchmarks.md`, `scripts/smoke.sh`.
- Regression tests: `tests/test_benchmarks.py`,
  `test_the_definitions_that_ship_with_fabric_are_on_the_search_path`,
  `test_a_shipped_definition_resolves_from_an_installed_package`.

---

### D-005: The smoke test copies the benchmark definitions it depends on

- Date / work area / commit: 2026-10-11, `scripts/smoke.sh`, uncommitted
- Context: `scripts/smoke.sh` is the CI job named "Run every command the
  README claims". It copied `examples/` into a scratch workspace and never
  copied `benchmarks/`, and set no `FABRIC_BENCHMARK_PATH`.
- Observed symptom: the script died on its second command with
  `BenchmarkError: benchmark: no benchmark named 'agent-qa@v1' was found`,
  having asserted nothing. Under WSL-free Windows the same script could not
  even be invoked via `bash`, because `bash` resolves to WSL.
- Verified root cause: adding `cp -r "$ROOT/benchmarks" .` plus
  `FABRIC_BENCHMARK_PATH` made the script reach all 35 assertions and print
  `all smoke checks passed`. That was the only missing piece; the
  smoke assertions themselves were sound.
- Decision: copy the definitions and set the variable, with a comment
  explaining why, so the next reader does not "simplify" the copy away.
- Consequences: the CI `demo` job can go green again on Linux and Windows.
- Files: `scripts/smoke.sh`.

---

### D-006: An unversioned benchmark reference resolves to the newest version

- Date / work area / commit: 2026-10-11, `fabric/core/benchmark.py`, uncommitted
- Context: `available()` sorted definitions by `(name, version)` as strings,
  and `resolve()` returned the first match. With `v2` and `v10` both on the
  search path, `benchmark: foo` resolved `v2`.
- Why it matters: the resolved benchmark contributes its digest to the
  experiment's identity and supplies metrics, baselines and constraints.
  Resolving the wrong one means a run is analysed against a contract
  different from the one its author meant, and the discrepancy is invisible
  because both are valid definitions.
- Decision: order versions numerically where possible, and resolve to the
  **last** match rather than the first. A version with no digits in it falls
  back to string comparison so it stays orderable instead of disappearing
  from the catalogue.
- Alternatives considered: requiring an explicit `@version` — rejected as a
  breaking change to specifications that already work.
- Consequences: the ordering is now numeric (`v1 < v2 < v10`), which matches
  how versions are read. An explicitly versioned reference is unaffected:
  it still resolves exactly what it names.
- Regression tests: `tests/test_benchmarks.py`,
  `test_an_unversioned_reference_resolves_to_the_newest_version`,
  `test_an_explicitly_versioned_reference_still_wins_over_a_newer_one`,
  `test_a_non_numeric_version_is_still_orderable`. All three failed against
  the previous implementation.
- Files: `fabric/core/benchmark.py`, `tests/test_benchmarks.py`,
  `docs/benchmarks.md`.

---

### D-007: `--workspace` precedes the subcommand

- Date / work area / commit: 2026-10-11, `Dockerfile`, `fabric/cli/main.py`,
  uncommitted
- Context: `--workspace` is a global option on the top-level parser, so
  argparse rejects it after a subcommand. The container `CMD` was
  `fabric serve --host 0.0.0.0 --port 8080 --workspace /data --workers 2`.
- Observed symptom: `fabric serve --workspace /data` exits 2 with
  `unrecognized arguments: --workspace /data`, so `docker compose up`
  could not start the service. CI never caught it: the container job runs
  `fabric --version`, the dependency guard and `scripts/container_check.sh`,
  but never `fabric serve`.
- Decision: reorder the `CMD` to put `--workspace` before `serve`, and state
  why in a comment. Also corrected the `CommandError` text that told a user
  to "pass --workspace" without saying it is a pre-subcommand option.
- Alternatives considered: adding `--workspace` to the `serve` subparser
  too — rejected as a second source of truth for the same value; a global
  option misplaced after the subcommand is a user error worth reporting
  precisely, and the message now does.
- Consequences: the image starts. Not executed here — no Docker daemon was
  available in this environment — so the argparse behaviour was confirmed
  against the real CLI rather than by running the image.
- Files: `Dockerfile`, `fabric/cli/main.py`.

---

### D-008: Compare the whole configuration set, not only the reference configuration

- Date / work area / commit: 2026-10-11, `fabric/core/diff.py`, uncommitted
- Context: `diff_plans()` compared `spec.configurations[0]` against the other
  side's `configurations[0]`, then reported `n_configurations`.
- Why it matters: `expand_configurations` orders the grid deterministically
  with the first declared value of each swept parameter first, so index 0 is
  the reference configuration. Changing a *later* swept value leaves index 0
  byte-identical and leaves the count unchanged. Sweeping
  `scale: [1.0, 2.0]` against `[1.0, 9.0]` was therefore reported as
  "identical in every field Fabric records" — for two experiments that
  differ in one of the two configurations they actually run.
- Decision: additionally compare the sets of configurations, reporting
  configurations that exist on one side only. Configurations are
  fingerprinted with `content_id` over their sorted items, so the
  comparison is order-independent and does not depend on expansion order.
- Alternatives considered: (a) compare by index across the whole list —
  rejected, it reports a difference whenever expansion order shifts without
  the set changing, which is noise. (b) report only that the sets differ —
  rejected, naming the configurations is what makes the finding actionable.
- Consequences: a sweep change anywhere in the grid is now reported as a
  material difference. Verified directly: `[1.0, 2.0]` vs `[1.0, 9.0]`
  previously produced no configuration difference and now reports
  `swept configurations: [{'scale': 2.0}] -> [{'scale': 9.0}]`.
- Regression tests: `tests/test_diff.py`,
  `test_a_change_to_a_later_swept_value_is_reported` and
  `test_an_added_configuration_is_reported` were both confirmed to fail
  without the change;
  `test_two_specifications_with_the_same_sweep_are_identical` guards against
  over-reporting.
- Files: `fabric/core/diff.py`, `tests/test_diff.py`.

---

### D-009: Refuse state-changing HTTP requests that a web page initiated

- Date / work area / commit: 2026-10-11, `fabric/server/app.py`, uncommitted
- Context: `fabric serve` binds `127.0.0.1` and has no authentication. There
  was no check on where a request came from.
- Why it matters: binding to loopback is not a trust boundary against a
  browser. Any page the researcher visits can POST to
  `http://127.0.0.1:8080/experiments`, and a plain HTML form needs no
  preflight. On a server started with `--execute` that is a drive-by page
  supplying its own specification and queuing it — code execution on the
  researcher's machine, with no user action beyond visiting a page. The
  reachability is `--execute`-gated, which limits it; it does not remove it.
- Decision: every `POST` is refused unless `Sec-Fetch-Site` says
  `same-origin`/`none`, or, when that header is absent, `Origin`/`Referer`
  matches the `Host` header. A `POST` carrying neither is refused rather
  than allowed. Reads are never restricted.
- Why refuse rather than allow on missing headers: an API client can always
  send `Origin`. One that cannot should be the command line, which does not
  go through HTTP at all. Defaulting open would make the absent-header case
  the dangerous one.
- Alternatives considered: (a) requiring `Content-Type: application/json`,
  which forces a preflight — rejected as a single point of failure; an
  attacker who finds a form-post gap walks straight through it. (b) a CSRF
  token — rejected as unnecessary complexity for a local trusted tool, and
  it would have to be threaded through the whole interface. (c) refusing to
  bind anything but loopback — already the default, and demonstrably not
  sufficient.
- Consequences: `Sec-Fetch-Site` alone is honoured when present, so the
  check holds even against a client that forges `Origin`. The refusal
  response does not read the body, so a refused `POST` cannot be used to tie
  up a connection.
- Files: `fabric/server/app.py`, `docs/deployment.md`,
  `tests/test_server.py`.
- Regression tests: `test_a_page_cannot_queue_a_specification_of_its_own`,
  `test_sec_fetch_site_is_honoured_whenever_the_browser_sends_it`,
  `test_a_cross_origin_header_is_refused_when_sec_fetch_is_absent`,
  `test_a_post_with_no_provenance_headers_at_all_is_refused`,
  `test_reads_are_never_refused`. The four refusal tests were confirmed to
  fail without the change.

---

### D-010: The server, not the specification, decides where submitted code runs

- Date / work area / commit: 2026-10-11, `fabric/server/worker.py`,
  `fabric/server/worker.py` docstring, `docs/deployment.md`, uncommitted
- Context: `_options_from()` copied `spec.execution.isolation` straight
  through, and a specification may declare `isolation: inline`, which runs
  the entrypoint by importing it **in the current process**.
- Why it matters: the module's own docstring claimed "each run goes through
  the same subprocess isolation the command line uses", and
  `docs/deployment.md` claimed "each run is a subprocess". Neither was true
  for a submitted specification. Inline execution has no crash boundary and
  no timeout, so one submitted specification can permanently stop a
  long-lived server serving — a strictly worse failure than the one a
  subprocess contains.
- Decision: the server forces `isolation="process"` regardless of what the
  specification asks for. A *user-specified* run at the command line is
  untouched: this is a property of arriving over the network, not a
  restriction on the tool.
- Alternatives considered: rejecting submissions that declare `inline` —
  rejected, because the submitted specification is frequently a copy of one
  the author legitimately runs inline locally, and refusing it would break a
  workflow that is otherwise fine.
- Consequences: the documented claims become true. A run that expected
  inline execution takes marginally longer to start.
- Regression test: `tests/test_server.py`,
  `test_a_submitted_spec_cannot_choose_to_run_in_the_server_process`.
- Files: `fabric/server/worker.py`, `docs/deployment.md`.

---

### D-011: The server caps its own worker count

- Date / work area / commit: 2026-10-11, `fabric/server/worker.py`, uncommitted
- Context: `_options_from()` clamped the *request* to the specification's
  `execution.max_workers`, and the specification validated only
  `max_workers >= 1`. A specification declaring `max_workers: 100000`
  therefore built a `ThreadPoolExecutor` with 100,000 threads inside the
  process that also serves HTTP.
- Decision: `MAX_SERVER_WORKERS = 8`, applied to the specification as well
  as the request. The command line is unchanged — there, the author is the
  person running the experiment and owns the consequence.
- Alternatives considered: (a) refusing the submission — rejected, the run
  is legitimate work and only the executor size is unreasonable.
  (b) making the bound configurable — deferred; 8 is a default, not a
  promise, and a flag would need the same justification as any other
  unbounded setting.
- Why 8: a seed-level experiment is not I/O-bound on the server; its
  parallelism matters because each run is a subprocess. Eight concurrent
  subprocesses is already far past what one machine's cores use.
- Regression tests: `tests/test_server.py`,
  `test_a_submitted_spec_cannot_ask_for_an_unbounded_thread_pool` and
  `test_an_over_eager_worker_count_is_reported_rather_than_obeyed`, the
  latter confirming a clamped run still completes all 18 runs.
- Files: `fabric/server/worker.py`, `docs/deployment.md`.

---

### D-012: Crash recovery is scoped by worker liveness

- Date / work area / commit: 2026-10-11, `fabric/server/jobs.py`,
  `fabric/server/worker.py`, uncommitted
- Context: `WorkerPool.start()` called `requeue_running()`, which swept
  every `RUNNING` row unconditionally. `docs/deployment.md` recommends
  several worker processes sharing one SQLite file — under which starting a
  second process requeued the first process's *live* jobs. Two workers then
  executed the same runs into the same `runs/<id>/run.json` concurrently,
  which is exactly the corruption `JobStore.cancel()`'s docstring says it
  avoids.
- Decision: the sweep requeues a job only when the process that claimed it
  is gone. Worker identities are already written `worker-<n>@<pid>`, so
  liveness is a pid check. `stale_after` is available for the case liveness
  cannot see — a worker that is alive but wedged.
- A critical constraint: `os.kill(pid, 0)` is **not** safe on Windows.
  There, any signal other than the two console-control events is delivered
  by unconditionally calling `TerminateProcess`, so the naive POSIX liveness
  probe would kill the sibling process this check exists to protect.
  Windows is asked through `OpenProcess` + `GetExitCodeProcess`, which only
  read state. Verified directly: with a bare `OpenProcess` and no exit-code
  check, a process that had already exited still returned a plausible
  handle; the exit code is what distinguishes `STILL_ACTIVE` (259) from
  gone.
- Alternatives considered: (a) matching on worker *slot* (`worker-1` in any
  process) — rejected after testing, because two concurrent servers both
  use slot `worker-1` and would steal each other's jobs, which is the bug
  being fixed. (b) a heartbeat column — rejected as more machinery than a
  pid check needs, given the identity already carries one.
- Consequences: a crashed server's jobs are still recovered on restart
  (verified by a test that spawns a process, waits for it, and asserts the
  sweep requeues its job). A live sibling's jobs are not. An identity with
  no parseable pid is left alone, so an unfamiliar format cannot cause
  duplication — at the cost of a crashed worker keeping its job, which is
  visible in the queue rather than silent.
- Regression tests: `tests/test_server.py`,
  `test_starting_a_second_server_does_not_steal_a_running_job`,
  `test_a_job_whose_process_died_is_recovered`,
  `test_a_stale_job_can_be_recovered_by_age`,
  `test_jobs_left_running_by_a_crash_are_requeued`.
- Files: `fabric/server/jobs.py`, `fabric/server/worker.py`,
  `docs/deployment.md`.
---

### D-013: One authoritative answer to "did this experiment produce evidence?"

- Date / work area / commit: 2026-10-11, `fabric/core/outcome.py`, uncommitted
- Context: Fabric answers "did this experiment produce evidence?" in five
  places -- `cmd_run`, `cmd_certify`, the certificate issuer, the integrity
  checks and the job queue -- and each answered independently. All five
  agreed on the one answer that mattered least: they all reported an
  experiment in which *every run failed* as a success.
- Observed symptom, reproduced before the fix: an experiment whose entrypoint
  raised on every one of its 8 runs produced

      fabric run      exit 0   "ok 0 completed, 8 failed"
      fabric certify  exit 0   VERDICT PASS WITH GAPS
      integrity       pass    "All 1 declared metrics were produced by every arm"
      certificate     provenance  complete
                        -> "8 runs recorded (8 failed); every successful run
                            carries its environment, hardware and code identity"
      fabric report   exit 0

  Every line of that is false. Three of them are false because of
  `all()` over an empty or wholly-excluded set:

  - `certificate.py:302` computed `all(... for r in runs if r.succeeded)`.
    With no successful runs the generator is empty and `all([])` is `True`.
  - `integrity/checks.py:307` counted only arms that had succeeded
    (`if c.successful and not c.series(...)`), so arms that produced
    nothing were excluded from the completeness test rather than counted
    as incomplete.
  - `cmd_run` returned a literal `0` and `cmd_certify` returned `0` unless
    the verdict was already `FAIL`, which a total failure never produced.

- Decision: a single classifier, `fabric.core.outcome.outcome_of()`, is now
  the only place that decides. It distinguishes four states -- `no_runs`,
  `total_failure`, `partial`, `complete` -- and exposes `has_evidence`,
  which is true for `partial` as well as `complete`. Every layer calls it:

  | Layer | Before | After |
  |---|---|---|
  | `fabric run` exit code | always `0` | `1` when there is no evidence |
  | `fabric certify` exit code | `0` unless the verdict was `FAIL` | follows the verdict, which is now `FAIL` |
  | `certificate` `provenance` | `complete` via `all([])` | `failed`, with a sentence saying nothing ran |
  | `integrity` `metrics.complete` | `pass` via an excluded empty set | `metrics.absent` warn, distinct from `metrics.missing` |
  | job queue terminal state | `succeeded` whenever execution returned | `failed` when no run succeeded |

- Why `partial` still counts as evidence: some measurements exist, and the
  user asked for the runs to be attempted. Failing them would train users to
  ignore the exit code, and `fabric analyze` already reports partial failure
  as a finding and a limitation. The failures stay in the record either way;
  `test_a_failed_run_is_never_silently_dropped` pins that.
- Why `no_runs` is separate from `total_failure`: "you have not run this
  yet" and "you ran it and nothing worked" call for different user actions,
  but neither is success.
- Alternatives considered: (a) leaving exit codes alone and only fixing the
  certificate -- rejected, because the CI failure mode is the exit code,
  not the printed verdict. (b) A new public verdict string such as
  `NO EVIDENCE` -- rejected; `FAIL` already expresses it, and the
  certificate's existing check vocabulary (`verified` / `incomplete` /
  `unverified` / `failed`) is expressive enough. Adding a verdict would
  break every consumer for no gain. (c) Reusing `metrics.missing` for the
  empty case -- rejected, because "some arms produced the metric and some
  did not" and "no arm produced anything" call for different remedies, and
  conflating them would hide the difference.
- Consequences: an all-failed experiment now exits `1` from `run` and
  `certify`, certifies as `FAIL`, and reports `metrics.absent`. Diagnostic
  reports, the failure analysis and the run records are all still produced
  and still readable -- `test_a_diagnostic_report_is_still_producible` pins
  that, because refusing to analyse a failure would hide it.
  Verified before and after on a disposable failing entrypoint:

      before: run=0 certify=0 VERDICT PASS WITH GAPS  provenance complete
      after : run=1 certify=1 VERDICT FAIL             provenance failed

- Compatibility: `fabric run` and `fabric certify` return `1` where they
  returned `0`. Any wrapper that treated "exit 0" as "the experiment
  produced a result" was previously reading a false signal and is now
  correct. Successful and partially failed experiments are unaffected.
  `certify --json` also gained the verdict-derived exit code it previously
  lacked.
- Regression tests: `tests/test_failed_evidence.py`, 15 tests. Eight assert
  the corrected behaviour and were confirmed to fail before the change; the
  other seven pin the cases that must not change, including that an
  *inconclusive* result from valid runs is not an execution failure.
- Remaining limitation: a run that succeeds while producing nothing
  measurable is caught downstream as `invalid_output` and so becomes a
  failed run, but a run that returns a single plausible number for the wrong
  arm is not detectable by Fabric and is out of scope.
- Files: `fabric/core/outcome.py`, `fabric/cli/main.py`,
  `fabric/reporting/certificate.py`, `fabric/integrity/checks.py`,
  `fabric/server/worker.py`, `tests/test_failed_evidence.py`,
  `tests/evidence.py`.
---

### D-014: Reject unexpected Host values before routing

- Date / work area / commit: 2026-10-11, `fabric/server/app.py`, committed
- Context: The server binds to `127.0.0.1` by default, but `GET /health` with
  `Host: evil.example` returned 200. Without an explicit allowed-host policy,
  a malicious page could attempt a DNS-rebinding attack against the local
  service, especially when `--execute` is enabled.
- Decision: Every request checks the `Host` header before routing. The hostname
  (port stripped) must be `localhost`, `127.0.0.1`, `[::1]`, `::1`, any `127.*`
  loopback, or a value that resolves to loopback. Requests with missing,
  empty, or unexpected Host values receive `400` with an explanation instead
  of being processed.
- Why refuse rather than ignore: allowing arbitrary Host values makes the
  missing-validation gap invisible; refusing it produces a clear, testable
  boundary that appears in server logs and API responses.
- Consequences: Legitimate loopback usage continues to work (`localhost`,
  `127.0.0.1`, ports included). Cross-origin `POST` refusal (`D-009`) now
  operates on requests that have already passed the host check, so the two
  controls are independent layers.
- Regression tests: `tests/test_server.py`, `test_an_invalid_host_is_refused`,
  `test_localhost_and_loopback_hosts_are_allowed`.
- Files: `fabric/server/app.py`, `tests/test_server.py`.
- Remaining limitation: IPv6 loopback addresses beyond `[::1]` are accepted
  by the `127.*` and `[::1]` rules; broader IPv6 ranges are not explicitly
  handled but are unlikely in practice for this deployment mode.

---

### D-015: Cache identity includes the project source snapshot

- Date / work area / commit: 2026-10-11, `fabric/execution/adapters.py`, committed
- Context: The previous cache fingerprint (`entrypoint_digest`) hashed only
  the entrypoint file. Changing an imported helper module (e.g., from `0.80`
  to `0.95`) did not change the digest, so the cache served stale results
  while the provenance records showed no code change.
- Decision: `entrypoint_digest` now combines the entrypoint file digest with
  a `tree_digest` of the project root, excluding artifact directories
  (`.fabric`, `build`, `dist`, `.git`, etc.). Any change to any source file
  in the workspace invalidates the cache.
- Why a broad snapshot: perfect static dependency tracking (parsing all
  imports, tracking dynamic imports, resolving package versions) is complex
  and still incomplete. A broader but understandable fingerprint is safer than
  an incomplete precise one, because the failure mode of a broad fingerprint
  is a false miss (re-running unnecessarily) rather than a false hit (stale
  data presented as fresh).
- Consequences: Cache reuse is now conservative. Existing caches that were
  built before this change will naturally miss, which is the correct behaviour
  when source identity rules have tightened.
- Regression test: verified by changing a helper file and observing that the
  next experiment does not reuse the old cached result.
- Files: `fabric/execution/adapters.py`.
