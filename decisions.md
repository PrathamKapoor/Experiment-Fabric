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