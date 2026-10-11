# Benchmarks

A benchmark is the part of an experiment that is *not* the researcher's
contribution: the task, the data, the metrics, the arms everyone compares
against, and the conditions a result has to satisfy before it means
anything.

Today that part gets retyped into every specification. Two papers report
"accuracy on benchmark-x" having measured different things — one counted
abstentions as wrong, the other dropped them; one used v1 of the data,
the other v2 — and nothing in either paper makes the difference visible.

So Fabric makes it an object with its own identity:

```yaml
benchmark: sentiment-small@v1
```

That line does three things. It supplies the metrics, the dataset and the
standard baselines. It applies the benchmark's rules to the design and
says which ones are unmet. And it puts the benchmark's **digest** into the
experiment's identity, so a benchmark that changes makes every experiment
that used the old one provably a different experiment.

---

## Using one

```bash
fabric benchmarks                          # what is available
fabric benchmarks show sentiment-small     # the full contract
fabric benchmarks validate benchmarks/mine.yaml
fabric benchmarks --json                   # machine-readable
```

In a specification:

```yaml
benchmark: sentiment-small@v1     # name@version
benchmark: sentiment-small        # newest version found
benchmark: ../shared/task.yaml    # an explicit path
benchmark:                        # pinned, and refused if it has changed
  name: sentiment-small
  version: v1
  digest: "sha256:2130047a3661ae99..."
```

Definitions are found, nearest first, in:

1. a `benchmarks/` directory beside the specification,
2. the specification's own directory,
3. `benchmarks/` in the working tree,
4. every directory in `$FABRIC_BENCHMARK_PATH`,
5. the definitions that ship with Fabric, inside the installed package
   (`fabric/benchmarks/`).

An unversioned name resolves to the **highest** version found. Versions are
ordered as numbers, not strings, so a path holding both `v2` and `v10`
resolves `v10` — a lexicographic sort would pick `v2` and quietly serve the
older contract.

A name found earlier shadows a later one, so a project can pin its own
copy without touching the installation. When resolution fails, the error
lists where it looked and what it found.

The three definitions Fabric ships — `agent-qa`, `sentiment-small` and
`spiral` — live in the package, so they resolve from an installed wheel as
well as from a checkout. `fabric benchmarks` on a fresh install lists them
without any environment variable set. The copies at the repository root
are the same files, kept where the examples' relative paths expect them;
`fabric/benchmarks/` is the copy that ships.

---

## Writing one

```yaml
benchmark:
  name: sentiment-small
  version: v1
  schema_version: "1.0"
  description: >
    What this measures, and what it does not.
  license: CC0-1.0
  citation: >
    Where it came from. Say plainly if it is synthetic.

tasks:
  - name: polarity
    description: Predict the binary sentiment label of a held-out review.
    dataset: sentiment
    metrics: [accuracy, f1]

datasets:
  - name: sentiment
    version: v1
    path: examples/text_classification/data/sentiment-v1   # or: uri:
    digest: "sha256:..."          # optional if Fabric can hash the path
    split: test
    n_examples: 1600
    license: CC0-1.0

metrics:
  - name: accuracy
    direction: maximize
    aggregation: mean
    primary: true
    tolerance: 0.01
    description: What it counts, and what it does not.
  - name: train_seconds
    direction: minimize
    unit: s
    required: false      # an experiment need not report it
    pinned: false        # an experiment may redefine it

output_schema:           # what a conforming run must return
  accuracy: float
  f1: float

baselines:               # in the experiment specification's own shape
  - name: majority-class
    description: The floor. A method that does not clear this has done nothing.
    algorithm:
      name: majority-class
      entrypoint: python:examples.text_classification.model:majority

constraints:
  - kind: min_seeds
    value: 5
    reason: >
      Why this bar exists. The reason is shown wherever the constraint is.
```

### Fields

| Field | Meaning |
|---|---|
| `name`, `version` | Identity. `name@version` is how experiments refer to it. |
| `schema_version` | Which benchmark schema this file speaks. A major version Fabric does not understand is refused rather than guessed at. |
| `description`, `citation`, `license` | Prose. Excluded from the digest: a clearer description is not a different benchmark. |
| `tasks[]` | `name`, `description`, `dataset`, `metrics`. Most benchmarks have one. Those that do not need metrics attributed to the right task or the aggregate means nothing. |
| `datasets[]` | `name`, `version`, `digest`, `path` or `uri`, `split`, `n_examples`, `license`, `source`. Fabric tracks *which* data, not the bytes: a `uri` is the right answer for anything large. |
| `metrics[]` | `name`, `direction`, `aggregation`, `unit`, `tolerance`, `primary`, `required`, `pinned`, `description`. |
| `output_schema` | `{metric: type}`. What a conforming run returns. |
| `baselines[]` | Arms every experiment on this benchmark inherits, in the same shape `baselines:` takes in a specification. |
| `constraints[]` | Rules, checked. See below. |

### `pinned` and `required`

These are the two flags that make a benchmark a contract rather than a
default.

**`pinned: true`** (the default) means an experiment may not redefine the
metric's direction, tolerance or unit. Try it and Fabric refuses, with the
reason:

```
metrics.accuracy.tolerance: benchmark spiral@v1 pins accuracy.tolerance to
0.01, and this experiment declares 0.02. Two experiments that disagree
here are not measuring the same thing, so 'evaluated on spiral' would mean
nothing. Either match the benchmark, mark the metric 'pinned: false' in
the definition, or rename the metric to make the difference explicit.
```

Mark a metric `pinned: false` when its meaning legitimately varies —
`train_seconds` is a property of the machine as much as of the method, so
every shipped definition leaves it unpinned.

**`required: true`** (the default) means an experiment claiming this
benchmark must report the metric. A missing one is an unmet constraint,
not a silent omission.

### Constraints

Checked against the parsed specification, reported everywhere, and
**never** enforced.

| Kind | Value | Checks |
|---|---|---|
| `min_seeds` | int | enough replications for the variance estimate to mean something |
| `max_alpha` | float | the decision threshold is no looser than the benchmark allows |
| `require_baselines` | `true` or a list of names | the arms a result is uninterpretable without |
| `require_dataset_digest` | bool | the data is identified by content, not by a name |
| `require_determinism_check` | bool | identical runs were executed and compared |
| `required_metrics` | list | these metrics are declared |

An unmet constraint appears as a planning advisory, as a row in the
report's benchmark table, and as an integrity warning:

```
benchmark agent-qa@v1 constraint 'require_determinism_check' is not met:
0 determinism replicate(s) declared, agent-qa@v1 requires at least 2
(The pipeline draws randomness at five points. An experiment that has not
checked whether identical runs agree cannot say which of its differences
are real.)
```

The experiment still runs. Section 20 of the design is explicit about
this: Fabric's job is to make a weak design *visible*, not to forbid
research someone has a reason to do. What it will not do is let the claim
pass unremarked — the report says, in the summary, that calling these
"agent-qa results" claims a comparability this design does not have.

---

## What a benchmark is not

**Not a place for scripts.** It declares what must be measured and against
what; the code that measures it stays in the experiment's algorithm, which
is the part that varies. A benchmark that shipped its own runner would be
a framework, and the project does not need one.

**Not a data host.** Fabric tracks dataset identity and provenance. For
anything large, `uri` points at where the data actually lives and `digest`
says which version it was. Putting a corpus in the definition would make
the definition unreadable and the digest meaningless.

**Not a leaderboard.** There is no ranking, no submission flow and no
scoreboard. The output of running on a benchmark is an evidence package,
not a position.

---

## The definitions that ship here

All three are synthetic, generated deterministically from a version
string, and say so in their own `citation`. They exist to exercise the
machinery in seconds, not to rank anything.

| Benchmark | What it is | Why it is here |
|---|---|---|
| `sentiment-small@v1` | 1600 short reviews with deliberate negation | Real algorithms, no framework; the fastest end-to-end path |
| `agent-qa@v1` | 400 questions answered by a **simulated** five-component agent | Exercises ablation, interaction analysis and all six perturbation axes in seconds. Simulated: no result on it is evidence about any real agent. |
| `spiral@v1` | Two interleaved spirals, 1400 points | The one whose reference implementation trains a real network with real autograd |
