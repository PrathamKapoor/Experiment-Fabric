# Experiment Fabric

**A provenance-first scientific experimentation platform: it turns
computational experiments into reproducible, statistically auditable,
versioned evidence graphs.**

An experiment is not a script that produces a number. It is a structured
scientific claim backed by a chain of evidence — and Fabric's job is to
keep that chain intact from the research question to the published figure,
so that any number in a report can be traced back to the exact code, data,
model, environment, hardware, seed and raw artifact that produced it.

```bash
fabric run experiment.yaml       # one file becomes 120 runs that know how they relate
fabric analyze EXP-07fd093d8e    # tested, corrected, effect-sized, with limitations
fabric why    EXP-07fd093d8e --metric accuracy     # trace one number back to its source
fabric report EXP-07fd093d8e --format html         # a paper-shaped scientific report
fabric export EXP-07fd093d8e --out bundle/         # a portable, verifiable artifact
```

No dependencies. Python 3.10-3.14. `pip install experiment-fabric`.

537 tests. CI runs them on Linux, macOS and Windows across Python 3.10 to
3.14, with and without PyYAML installed, and adds a bare-install job that
proves the core still imports with nothing else on the machine. The
statistics engine is additionally cross-checked against SciPy and against
brute-force enumeration of the exact null distributions — see
[`docs/statistics.md`](docs/statistics.md).

---

## Why

You have a table of results. Six months later, someone asks a reasonable
question and you cannot answer it:

- *Which exact version of the benchmark produced row three?*
- *Was the baseline tuned as hard as the proposed method?*
- *Is +0.2 points larger than the seed-to-seed noise?*
- *Would this survive a different CUDA version?*
- *Does the model still reproduce, and how far off is it now?*

Every one of those is a query in Fabric, because the information needed to
answer them is captured at the moment the number is produced rather than
reconstructed afterwards.

---

## What it does that a tracker does not

Checked against primary documentation in September 2026, not vendor
comparison posts. The full survey, with citations and corrections, is in
[`docs/related-work.md`](docs/related-work.md).

| | Established trackers | Experiment Fabric |
|---|---|---|
| Log metrics and parameters | yes | yes |
| Store artifacts, model registry | yes | yes, content-addressed |
| Dataset digest recorded | yes (MLflow computes one automatically) | yes |
| Dataset digest **is** the identity, and a conflicting re-registration is refused | no | yes |
| Experiment graphs | limited | the core abstraction, traversable both ways |
| Automated ablations | limited | native, with interaction analysis |
| Statistical analysis | limited; specialised tools exist (NLPStatTest) | native, domain-agnostic |
| Test selection | manual | automatic, with the reasoning and assumptions stored |
| Minimum detectable effect | no | on every comparison |
| Determinism testing | no | executes identical runs and checks |
| Perturbation study across six axes | no | native, executable on one machine |
| Decision stability under seed resampling | no | native |
| Failure analysis | partial | classified, attributed, tested |
| Reproduction certificates | no | content-addressed, tamper-evident |
| Paper generation | limited | Markdown, HTML, LaTeX, JSON from one model |
| Statistics checked against an independent reference | no | 68 cases vs SciPy and exact enumeration, in the suite |
| Refuses a conclusion the evidence cannot support | no | an inconclusive result stays inconclusive, and the report says why |

Two corrections worth making explicitly, because an earlier version of
this table got them wrong:

- **MLflow does version datasets.** `mlflow.data` records a digest and
  computes one automatically. The real difference is narrower: there, the
  digest is metadata attached to a run; here it *is* the dataset's
  identity, and registering the same version with different bytes is
  refused with a field-level diff.
- **DVC is stronger than Fabric at data versioning**, not weaker. It
  handles large remote storage, which Fabric does not attempt at all.

The honest version of the difference: a tracker records **what happened**.
Fabric records **what would have to be true for the result to mean what
you say it means**, and then checks as much of it as it can.

Three of the examples below lose. That is the point of shipping them: a
tool that only ever confirms its user is not being used to check anything,
and the failure mode of an experiment tool is not a crash, it is a number
that looked fine.

### A note on the name

The distribution is `experiment-fabric`, which is unclaimed on PyPI. The
import package and the console script are both `fabric` — and `fabric` is
also [the standard Python SSH/deployment tool](https://fabfile.org), which
installs its own `fabric` command. Installing both means one overwrites the
other on `PATH`, and `import fabric` resolves to whichever comes first on
`sys.path`. Nothing here can be done about that without renaming, which
would be a larger change than it looks; it is worth knowing before you
install it into an environment that already has the other one.

---

## Five minutes

```bash
pip install experiment-fabric
fabric init
```

Write `experiment.yaml`:

```yaml
experiment:
  name: my-first-experiment
  research_question: Does method X beat the standard baseline on benchmark Z?

dataset:
  name: benchmark-z
  version: v1
  path: data/benchmark-z       # hashed; the hash becomes its identity

algorithm:
  name: method-x
  entrypoint: python:mypackage.train:run
  commit: auto

metrics:
  - name: accuracy
    primary: true
    tolerance: 0.01            # what counts as a meaningful difference

seeds:
  values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

baselines:
  - name: standard
    algorithm:
      name: standard
      entrypoint: python:mypackage.baselines:standard
```

Write the method:

```python
def run(ctx):
    rng = ctx.rng()
    model = train(ctx.dataset, lr=ctx.get("learning_rate", 1e-4), rng=rng)
    ctx.write_json_artifact("predictions.json", model.predict(test))
    return {"accuracy": evaluate(model)}
```

Then:

```bash
fabric plan experiment.yaml    # see the expansion, and the advisories, before spending compute
fabric run  experiment.yaml
fabric analyze <EXP>
```

---

## The worked examples

Three ship with the project. Two run in seconds on a laptop with no GPU;
the third needs PyTorch.

```bash
python examples/agent_pipeline/make_data.py
fabric run examples/agent_pipeline/experiment.yaml          # 120 runs, ~3s
fabric run examples/agent_pipeline/reproducibility.yaml     # 93 runs, ~2s
fabric report <EXP> --format html --open
```

| Example | What it is | Why it is here |
|---|---|---|
| `agent_pipeline/` | A five-component retrieval agent, **simulated** | Exercises ablation, interaction analysis and all six perturbation axes in seconds |
| `text_classification/` | Naive Bayes and logistic regression, hand-written, **no `import fabric`** | Real algorithms; proves the core is framework-agnostic |
| `torch_mlp/` | An MLP trained with **real PyTorch autograd** | The pipeline that is not a simulation |

### The interesting part: the hypotheses lose

Each specification states a hypothesis *before* the runs. In two of the
three, the data contradicts it, and the report says so — because the
analysis was declared in the specification and could not be adjusted
afterwards.

**`text_classification`** predicted logistic regression would edge out
naive Bayes, that bigrams would help, and that lowercasing would not
matter:

```
logistic-regression scored 0.6490 +/- 0.0134 on accuracy over 10 seeds,
beating 1 of 2 baselines. Against naive-bayes the difference is not
distinguishable from noise.

  baseline        difference  95% CI             test       adj. p     g  verdict
  naive-bayes        -0.0102  [-0.0207, 0.0002]  paired-t   0.0542  -0.6  inconclusive
  majority-class     +0.1550  [0.1422, 0.1678]   paired-t  <0.0001   7.9  improvement

  component                          drop in accuracy  detectable
  lowercasing  ####################           +0.0145  yes
  bigrams      ####................           +0.0028  no
```

All three predictions wrong: the proposed method is *behind* the
baseline, lowercasing matters and bigrams do not.

**`torch_mlp`** predicted batch normalisation would carry the gain. Over
ten seeds of real training, none of the four components — batchnorm,
dropout, weight decay, cosine schedule — is detectable at all, while the
MLP beats a linear model by 0.51 accuracy.

That is the whole point of the tool.

## The core abstraction: experiment graphs

An experiment is a DAG whose nodes are addressable scientific objects.

```
      dataset@v1 ────┐
      model@v4 ──────┼──► condition ──► run(seed=3) ──► accuracy=0.7175
   algorithm@abc123 ─┘         │              │
                          config, seed   environment, hardware
                                                │
                                          predictions.json
```

Because the graph is explicit, two questions become mechanical:

```bash
fabric why    <EXP> --metric accuracy --seed 3   # why does this number exist?
fabric impact <EXP> --node <dataset-id>          # what breaks if this changes?
fabric graph  <EXP> --format html --open         # ...or click through it
```

```
accuracy = 0.7175

  metric       accuracy=0.7175
  run          full-agent[top_k=3] seed=3        status: completed
  condition    full-agent[top_k=3]               role: treatment
                 config: {top_k: 3, temperature: 0.2, ...}
  algorithm    full-agent@5d2a903                commit: 5d2a903cd548...
  model        sim-reasoner@v1                   precision: float64
  dataset      agent-benchmark@v1                digest: sha256:4cb166de...
  environment  python 3.11.16 on Windows
  hardware     1x NVIDIA GeForce GTX 1650
  experiment   agent-scaffolding-ablation

  raw artifacts
    predictions.json  sha256:fa8ff681633a  113931 bytes
```

### And the same query, with a mouse

The HTML report embeds an interactive version: **click any node and its
lineage lights up**, everything else recedes. Drag to pan, scroll to
zoom, hover for digests and configuration.

It is inline SVG plus about eighty lines of vanilla JavaScript — no
library, no network, no build step. A 913-node graph folds to 45 by
grouping each condition's replications into one node, and the caption
says exactly how much it folded and where the full graph lives.

---

## Reproducibility as a measured property

Not a badge. A measurement, on six axes, each changing exactly one thing:

```
  axis            perturbation           stability  noise    delta  tolerance
  configuration   temperature-0.8   #.        0.05   3.0x  -0.0632  beyond
  dataset         agent-benchmark@v2 #####.   0.38   1.4x  +0.0302  beyond
  model           sim-reasoner@v2   ######.   0.48   1.2x  +0.0252  beyond
  implementation  full-agent-reordered ####   1.00   0.1x  +0.0027  inconclusive
  environment     single-threaded   ########  1.00   0.0x  +0.0000  within
  hardware        no-accelerator    ########  1.00   0.0x  +0.0000  within
```

`stability` is `1 - |Cliff's delta|`, calibrated against random
relabelling of the same runs, so 1.00 means "indistinguishable from
noise" rather than "we did not look hard enough". `noise` is the shift in
units of seed standard deviation — the most interpretable column, because
an axis that moves the result 3× further than the randomness you already
report is a finding. `tolerance` is a three-way verdict: `within`,
`beyond`, or `inconclusive`, because a failed equivalence test means
equivalence was not established, not that the difference is real.

The seed axis is the **yardstick, not a score**. The methodology, its
calibration, and seven named limitations are in
[`docs/reproducibility.md`](docs/reproducibility.md).

Determinism is tested rather than assumed: Fabric executes the same run
again and checks. A fixed seed does not make a pipeline deterministic, and
an effect smaller than a pipeline's own jitter cannot be measured by it.

---

## Statistics that show their work

Fabric chooses the test from the structure of the data and records why:

```
Test:     paired-t
Reason:   normality was not rejected, so a t-test was used. Pairing by seed
          removes seed-to-seed variance from the comparison.
Assumptions:
  ok          paired differences is approximately normal
              Anderson-Darling A2*=0.295, p=0.5973
  ok          observations are independent across replications
  unverified  pairs are independent of each other
```

It will also tell you when your design cannot work *before* you spend the
compute:

```
design limit: with these sample sizes the smallest two-sided p-value
permutation can produce is 0.0625, which is above alpha=0.05. No result of
this experiment can be significant; add replications.
```

And it reports what the design could have detected, whether or not the
result was significant:

> With 10 replications and the observed spread, this design could detect a
> difference of 0.0243 in accuracy or larger 80% of the time at alpha=0.05.

Details in [`docs/statistics.md`](docs/statistics.md).

---

## Portable, verifiable artifacts

```bash
fabric certify <EXP>                  # a content-addressed certificate
fabric export  <EXP> --out bundle/    # spec, runs, analysis, provenance, report
fabric verify  bundle/                # every file re-hashed against the manifest
fabric reproduce bundle/              # re-execute and compare, honestly
```

```
REPRODUCTION COMPLETE

  original result       0.7478
  reproduced result     0.7478
  difference            +0.0000 (+0.00%)
  95% CI on difference  [0.0000, 0.0000]
  equivalence (TOST)    p=0.0000 against a margin of 0.01

  reproducibility       PASS
```

The verdict has four levels, and the middle two are the point: `PASS`,
`PASS WITH DEVIATION` (with the environment differences named and ranked
by how likely each is to explain it), `INCONCLUSIVE` (neither equivalence
nor difference established — usually a sample-size limit, reported as
one), and `FAIL` (the numbers differ, and either they exceed the declared tolerance or none was declared to judge them against).

Tampering is detectable: edit a bundled file and `fabric verify` names it;
edit a certificate and it no longer hashes to its own identifier.

---

## Command reference

| Command | What it does |
|---|---|
| `fabric init` | Create a workspace. |
| `fabric register dataset\|model\|algorithm` | Register an entity by content. |
| `fabric registry ls\|verify` | Inspect the registry, or re-hash all of it. |
| `fabric plan <spec>` | Expand a specification and print advisories. Runs nothing. |
| `fabric run <spec>` | Execute. Resumable, cached, crash-isolated. |
| `fabric status <exp>` | Per-condition progress. |
| `fabric ls` | Every experiment in the workspace. |
| `fabric analyze <exp>` | Full analysis; `--json` for the machine-readable form. |
| `fabric report <exp>` | Markdown, HTML, LaTeX, JSON, or `--format all`. |
| `fabric why <exp>` | Trace a number to everything that produced it. |
| `fabric graph <exp>` | Export provenance as an interactive HTML page, mermaid, DOT, JSON or ASCII. |
| `fabric impact <exp>` | What depends on a node, and what claims it supports. |
| `fabric reproduce <exp\|bundle>` | Re-execute and compare against the original. |
| `fabric verify <exp\|bundle>` | Re-hash artifacts, entities and certificates. |
| `fabric certify <exp>` | Issue a reproduction certificate. |
| `fabric export <exp>` | Write a portable bundle. |
| `fabric benchmarks\|show\|validate` | The benchmark definitions on the search path. |
| `fabric diff <a> <b>` | What differs between two experiments, and which of it could move a number. |
| `fabric serve` | A local HTTP API and browser dashboard over the workspace. |
| `fabric cache stats\|clear` | Inspect or clear the execution cache. |
| `fabric doctor` | Show captured environment and hardware. |

`fabric analyze` is also spelled `fabric analyse`. `fabric benchmarks` is also
spelled `fabric benchmark`.

---

## Looking at it in a browser

```bash
fabric serve            # http://127.0.0.1:8080
```

The dashboard reads the same workspace the command line does, so anything it
shows can be reproduced with the commands above. Execution is **off** unless
you ask for it:

```bash
fabric serve --execute --i-understand     # and read docs/deployment.md first
```

`--i-understand` exists because that flag means "this server will import and
call whatever code a submitted specification names, as this user, with no
sandbox". Fabric will not put that behind a flag you can pass without
reading what it says.

---

## Design commitments

**No dependencies in the core.** Specifications, execution, statistics,
provenance, reporting and the server run on the standard library alone —
including the incomplete beta and gamma functions behind every p-value, and
a bundled YAML parser. A tool that measures reproducibility should not
itself perturb the environment it is measuring. PyYAML is used
automatically when present.

**A declared comparison that could not run still counts.** The
multiple-comparison family is the one you declared, not the subset that
happened to produce a p-value. Dropping a test because it failed is
deleting a comparison after seeing the results — and it is
anti-conservative in exactly the case that matters, turning *not
significant* into *significant*. See [`docs/statistics.md`](docs/statistics.md).

**The statistics are checked against something else.** 68 cases compare
every distribution, test, effect size, interval and correction against
SciPy 1.17.1 and against brute-force enumeration of the exact null
distributions. SciPy is not a dependency and is not installed by Fabric;
the reference values are frozen into the suite as literals, so they
outlive the library that produced them.

**Framework-agnostic.** The core understands experiments, not neural
networks. Nothing in `fabric/` imports PyTorch, JAX, NumPy or
scikit-learn, and the `command:` adapter runs anything at all.

**Plain text on disk.** Everything lives in `.fabric/` as readable JSON. A
results database you cannot read without the tool that wrote it is a
liability for a record that has to outlive the tool.

**Deterministic identity.** The same specification produces the same
experiment id, condition ids and run ids on any machine — which is what
makes caching, resumption and cross-machine comparison possible at all.

**Say "unverified" rather than nothing.** A green tick beside a check that
never ran is worse than no tick.

---

## Project layout

```
fabric/
  core/          entities, specifications, graphs, registry, provenance, storage
  execution/     planner, runner, adapters, cache, resources, failures
  environments/  environment and hardware capture, and differencing
  statistics/    distributions, tests, effect sizes, intervals, power, selection
  analysis/      baselines, ablations, sweeps, failures, determinism, reproducibility
  integrity/     contamination, drift and effort checks
  reporting/     document model, renderers, SVG charts, interactive graph, certificates
  cli/           the fabric command
examples/        three runnable experiments; one trains with real PyTorch
docs/            specification, statistics, reproducibility, related work, architecture
paper/           the pre-registered study, and a pilot that actually ran
scripts/         dependency guard, smoke test, container check
tests/           537 tests, including reference-value checks for every distribution
server/          a local HTTP API and browser dashboard (stdlib only)
.github/         CI: 3 platforms x 5 Python versions, plus container and cross-version reproduction
```

```bash
python -m pytest              # 537 tests, ~60s
bash scripts/smoke.sh         # every documented command, output asserted
bash scripts/demo.sh          # the full walkthrough, on a throwaway workspace
python paper/run_study.py     # the pilot study, regenerated
```

---

## Status and scope

Version 0.3. The MVP and the research-grade layer are complete:
expansion, execution, caching, statistics, ablations, sweeps, failure
analysis, determinism, perturbation studies, provenance, integrity
checks, reporting, certificates, bundles and reproduction.

**What has actually been verified**, as opposed to claimed:

| | Evidence |
|---|---|
| 537 tests pass | on Python 3.14.7 and 3.11.16 locally; the CI matrix adds Linux, macOS and Python 3.10, 3.12 and 3.13 |
| The statistics are independently checked | every distribution, test, effect size, interval and correction cross-checked against SciPy 1.17 — see [`docs/statistics.md`](docs/statistics.md) |
| The core has no dependencies | `scripts/check_no_dependencies.py`, enforced in CI |
| Both YAML parsers agree | every shipped specification parsed by each and compared |
| Every documented command works | `scripts/smoke.sh` asserts the output of each one |
| It runs real ML | `examples/torch_mlp/` trains with PyTorch autograd |
| The wheel installs clean | built and run in an empty virtualenv, with `fabric benchmarks` resolving the definitions that ship inside the package |
| Bundles reproduce | `fabric reproduce` on an exported bundle, and across Python versions in CI. Re-verified here in a **fresh environment**: a wheel-only virtualenv, run → export → verify → reproduce. `INTACT`, then `PASS` — accuracy, f1 and vocabulary_size reproduce exactly, and `train_seconds` differs by 0.158 s while still passing its own declared 0.5 s tolerance |
| The container reproduces | CI builds the image and runs `scripts/container_check.sh` inside it |
| The server refuses what a page drove it to do | a cross-origin `POST` is refused; the isolation and worker caps are covered by tests over a real socket |

Deliberately **not** built: distributed orchestration, a cloud GPU
marketplace, Bayesian optimisation, autonomous experiment generation, dataset
or model hosting, multi-tenant authentication. Each would bury the
contribution under infrastructure that already exists elsewhere — and
[`docs/related-work.md`](docs/related-work.md) names the tools that do those
things better.

The web interface that *is* built is deliberately local and unprivileged:
`fabric serve` binds `127.0.0.1`, refuses state-changing requests a web page
initiated, and refuses to execute anything unless you pass `--execute` *and*
`--i-understand`. Its threat model, and the limits it does not claim to
cover, are in [`docs/deployment.md`](docs/deployment.md).

Known limitations are documented rather than hidden — see the end of
[`docs/reproducibility.md`](docs/reproducibility.md).

## The research question behind it

> How sensitive are machine-learning conclusions to controlled
> perturbations in randomness, software environments, hardware, model
> versions, dataset versions, and implementation details — and how does
> that sensitivity compare to the seed variance that papers already
> report?

This is **not a new question**, and the project does not pretend it is.
Prior work already shows that non-seed axes can dominate the seed axis
(arXiv:2206.12353) and that fixed seeds do not guarantee determinism on
GPU. What is missing is comparability: no existing study puts all six
axes on one scale for the same experiment.

A **pilot** has been run — two real pipelines, one simulated and
excluded. Genuine results, in [`paper/results/findings.md`](paper/results/findings.md):

- the dataset version moved results by **7.5× and 80.5×** the seed noise
- the environment axis measured **exactly 0.00** across four perturbations
- **0 of 2** real pipelines were non-deterministic, once timing metrics
  were correctly excluded from the verdict
- **all three** were floor-limited at half the seeds: an exact test on
  five paired runs cannot reach p < 0.05 whatever it measures

Two pipelines cannot support a claim about the field, and the findings
say so on every page. [`paper/`](paper) holds the pre-registration and
the design the pilot is a pilot for.

## License

Apache-2.0.
