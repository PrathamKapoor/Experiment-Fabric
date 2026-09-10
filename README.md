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

No dependencies. Python 3.10+. `pip install experiment-fabric`.

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

| | Typical experiment tracker | Experiment Fabric |
|---|---|---|
| Log metrics and parameters | yes | yes |
| Store artifacts, model registry | yes | yes, content-addressed |
| Dataset versioning | partial | first-class; identity **is** the content hash |
| Experiment graphs | limited | the core abstraction |
| Automated ablations | limited | native, with interaction analysis |
| Statistical analysis | limited | native, with recorded assumptions |
| Test selection | manual | automatic, with the reasoning stored |
| Reproducibility testing | limited | measured on six perturbation axes |
| Determinism testing | no | executes identical runs and checks |
| Hardware / environment perturbation | no | executable on one machine |
| Failure analysis | partial | classified, attributed, tested |
| Minimum detectable effect | no | on every comparison |
| Provenance | partial | end to end, traversable both ways |
| Reproduction certificates | no | content-addressed, tamper-evident |
| Paper generation | limited | Markdown, HTML, LaTeX, JSON from one model |

The honest version of the difference: a tracker records **what happened**.
Fabric records **what would have to be true for the result to mean what
you say it means**, and then checks as much of it as it can.

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

## The worked example

Two examples ship with the project and both run in seconds on a laptop
with no GPU.

```bash
python examples/agent_pipeline/make_data.py
fabric run examples/agent_pipeline/experiment.yaml          # 120 runs, ~3s
fabric run examples/agent_pipeline/reproducibility.yaml     # 93 runs, ~2s
fabric report <EXP> --format html --open
```

The first is a five-component retrieval agent (retrieval, reranking,
memory, planning, verification) evaluated against three baselines with
seven ablations over ten seeds. The second perturbs six axes to measure
how far the result travels.

The second example, `examples/text_classification/`, is the more
interesting one, because **the result does not go the author's way**:

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

The specification's stated hypothesis was that logistic regression would
edge out naive Bayes, that bigrams would help, and that lowercasing would
not matter. All three are contradicted, and the report says so — because
the analysis was declared before the runs and could not be adjusted
afterwards.

That is the whole point of the tool.

---

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
one), and `FAIL`.

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
| `fabric graph <exp>` | Export provenance as mermaid, DOT, JSON or ASCII. |
| `fabric impact <exp>` | What depends on a node, and what claims it supports. |
| `fabric reproduce <exp\|bundle>` | Re-execute and compare against the original. |
| `fabric verify <exp\|bundle>` | Re-hash artifacts, entities and certificates. |
| `fabric certify <exp>` | Issue a reproduction certificate. |
| `fabric export <exp>` | Write a portable bundle. |
| `fabric cache stats\|clear` | Inspect or clear the execution cache. |
| `fabric doctor` | Show captured environment and hardware. |

---

## Design commitments

**No dependencies in the core.** Specifications, execution, statistics,
provenance and reporting run on the standard library alone — including the
incomplete beta and gamma functions behind every p-value, and a bundled
YAML parser. A tool that measures reproducibility should not itself
perturb the environment it is measuring. PyYAML is used automatically when
present.

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
  reporting/     document model, markdown/html/latex renderers, SVG charts, certificates
  cli/           the fabric command
examples/        two runnable experiments with real data
docs/            specification, statistics, reproducibility, writing experiments
paper/           the reproducibility study this instrument was built for
tests/           180 tests, including reference-value checks for every distribution
```

```bash
python -m pytest        # 180 tests, ~18s
```

---

## Status and scope

Version 0.3. The MVP and the research-grade layer are complete: expansion,
execution, caching, statistics, ablations, sweeps, failure analysis,
determinism, perturbation studies, provenance, integrity checks,
reporting, certificates, bundles and reproduction.

Deliberately **not** built: distributed orchestration, a cloud GPU
marketplace, a web UI, Bayesian optimisation, autonomous experiment
generation, dataset or model hosting. Each would bury the contribution
under infrastructure that already exists elsewhere.

Known limitations are documented rather than hidden — see the end of
[`docs/reproducibility.md`](docs/reproducibility.md).

---

## The research question behind it

> How sensitive are machine-learning conclusions to controlled
> perturbations in randomness, software environments, hardware, model
> versions, dataset versions, and implementation details — and how does
> that sensitivity compare to the seed variance that papers already
> report?

Fabric is the instrument. [`paper/`](paper) is the study design.

## License

Apache-2.0.
