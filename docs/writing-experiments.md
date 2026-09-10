# Writing an experiment

What your code has to do to run under Fabric, and what it gets in return.

The short answer: write a function that takes a context and returns a
dictionary of numbers.

```python
def run(ctx):
    return {"accuracy": 0.91, "latency_ms": 120.0}
```

Point a specification at it and everything else — seeds, sweeps,
baselines, ablations, caching, provenance, statistics, reports — is
Fabric's problem.

---

## The contract

```yaml
algorithm:
  entrypoint: python:mypackage.train:run
```

The function receives a `RunContext` and must either return a mapping of
metric names to numbers, or record them on the context. A run that
produces no metrics is a **failure**, not an empty success: Fabric never
substitutes a default.

### What the context gives you

```python
def run(ctx):
    lr = ctx.get("learning_rate", 1e-4)   # configuration values
    ctx.seed                              # this run's seed
    ctx.dataset                           # the resolved dataset record
    ctx.model                             # the resolved model record
    ctx.condition                         # this arm's name
    ctx.ablated                           # components removed from this arm
```

### Randomness

```python
rng = ctx.rng()            # seeded from ctx.seed
data_rng = ctx.rng("data") # an independent, still-reproducible stream
init_rng = ctx.rng("init")
```

Named streams let different parts of a pipeline draw independently
without interfering. Fabric also seeds `random`, and NumPy and PyTorch
when they are already imported, and sets `PYTHONHASHSEED` before your code
runs.

### Ablations

This is the contract worth internalising:

```python
def run(ctx):
    documents = load(ctx.dataset)

    if ctx.enabled("retrieval"):
        documents = retrieve(documents, top_k=ctx.get("top_k", 5))
    if ctx.enabled("reranking"):
        documents = rerank(documents)
    if ctx.enabled("verification"):
        answer = verify(answer)

    return {"accuracy": score(answer)}
```

Write the system once with those guards, declare the components in the
specification, and every ablation — including combinations you have not
thought of yet — becomes runnable without touching this file.

### Artifacts

```python
ctx.write_json_artifact("predictions.json", predictions)
ctx.write_artifact("trace.txt", text)
path = ctx.artifact_path("checkpoint.pt")   # write it yourself
```

Everything in the artifact directory is hashed at the moment the run
finishes and recorded in the manifest, which is what makes a predictions
file citable evidence rather than a file that happens to be on disk.

### Logging

```python
ctx.log("epoch 3: loss 0.214")
```

Captured into the run record. `print()` also works; stdout and stderr are
captured per run.

---

## What your code cannot see

Deliberately: other runs, results so far, and **which arm is the
"proposed" one**. `ctx.role` exists but carries no privileged path, and
nothing exposes the comparison. Code that cannot tell whether it is the
treatment cannot be accidentally tuned to win.

---

## Running something that is not Python

```yaml
algorithm:
  entrypoint: "command:./train.sh --config {config_path}"
```

Fabric writes the run description to JSON and passes its path in
`$FABRIC_RUN`, along with `$FABRIC_OUTPUT_DIR`, `$FABRIC_SEED`,
`$FABRIC_RUN_ID` and `$FABRIC_CONDITION`. Write a JSON object of metric
names to numbers at `$FABRIC_OUTPUT_DIR/metrics.json` and exit zero.

```bash
#!/usr/bin/env bash
set -euo pipefail
SEED="$FABRIC_SEED"
LR=$(python -c "import json,os;print(json.load(open(os.environ['FABRIC_RUN']))['config']['learning_rate'])")

Rscript analysis.R --seed "$SEED" --lr "$LR" > out.txt

python - <<'PY'
import json, os
accuracy = float(open("out.txt").read().strip())
json.dump({"accuracy": accuracy}, open(os.path.join(os.environ["FABRIC_OUTPUT_DIR"], "metrics.json"), "w"))
PY
```

PyTorch, JAX, scikit-learn, R, a compiled binary, a shell script — all the
same to Fabric. Nothing in the core imports a machine-learning framework,
and nothing ever will.

---

## Keeping Fabric out of your codebase

The context is a structural interface, not a base class. Declare what you
need and your method never imports Fabric at all:

```python
from typing import Any, Protocol

class Context(Protocol):
    seed: int
    config: dict[str, Any]
    def get(self, key: str, default: Any = None) -> Any: ...
    def enabled(self, component: str) -> bool: ...

def run(ctx: Context) -> dict[str, float]:
    ...
```

`examples/text_classification/model.py` does exactly this: real
classifiers, no `import fabric` anywhere, callable from Fabric, a
notebook, or a plain script.

---

## Failures are results

Do not catch exceptions to keep a run "green". A method that OOMs above
batch size 64 has told you something about the method, and Fabric
classifies it (`oom`, `timeout`, `nan`, `divergence`, `segfault`,
`dependency`, `network`, `invalid_output`, `metric_failure`,
`user_error`), attributes it to configuration values, tests whether
failures cluster, and separates infrastructure failures from method
failures in the report.

A NaN metric is treated as a failed run rather than a recorded NaN.

---

## Making runs cacheable

The cache key covers the dataset, model, configuration, seed, ablation
set, the **source digest of the entrypoint file**, and the numerically
relevant part of the environment. Editing your method invalidates its
cached runs — which is the property that makes an experiment cache
trustworthy rather than a source of stale numbers.

Two knobs, both off by default:

```bash
fabric run experiment.yaml --strict-environment   # any package change invalidates
fabric run experiment.yaml --no-cache             # ignore the cache entirely
```

Determinism replicates never come from the cache: their whole purpose is
to execute identical inputs again.

---

## A checklist before the compute

```bash
fabric plan experiment.yaml
```

Read the advisories. Then:

- Are there at least five seeds? Ten if the effect might be small.
- Is there a baseline? Without one there is no comparative claim.
- Does the primary metric have a `tolerance`? Without one, reproduction
  cannot be tested, only failed-to-refute.
- Is the git tree clean? A dirty tree means the commit does not describe
  the code.
- Does the dataset have a digest?
- Is the smallest attainable p-value below your alpha?
