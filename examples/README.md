# Examples

Two complete experiments. Both run in seconds on a laptop with no GPU, no
API key and no downloads.

```bash
python examples/agent_pipeline/make_data.py
python examples/text_classification/make_data.py
fabric init
fabric run examples/agent_pipeline/experiment.yaml
```

Or run everything, including reports, certificates and a reproduction:

```bash
./demo.sh
```

---

## 1. `agent_pipeline/` — ablation and perturbation

A five-component retrieval-QA agent: retrieval, reranking, memory,
planning, verification. Three baselines, seven ablations, ten seeds, five
metrics.

| File | What it is |
|---|---|
| `make_data.py` | Generates two versions of a 400-question benchmark, deterministically. |
| `pipeline.py` | The system under study, written against `ctx.enabled(...)`. |
| `experiment.yaml` | The ablation study: 12 conditions, 120 runs. |
| `reproducibility.yaml` | The perturbation study: 6 axes, 93 runs. |

### It is a simulation, and says so

There is no language model behind `pipeline.py`. It is a deterministic
log-odds model of an agent, chosen so that the demonstration runs in
seconds and exercises every part of Fabric on genuinely noisy, genuinely
structured data. The numbers are real measurements *of this code*; they
are not predictions about any real agent, and the shipped example is
explicitly excluded from the study sample in [`paper/`](../paper).

What *is* real: the benchmark is loaded from content-addressed files on
disk, every component genuinely changes the outcome, the seed variance is
real sampling variance over 400 questions, and the accuracy/latency
trade-offs are consequences of the model rather than hand-written
constants.

### What it demonstrates

Removing retrieval costs 0.53 accuracy; removing memory costs 0.03.
Reranking without retrieval changes **nothing** — a built-in sanity check
that the interaction analysis detects and labels as redundant. `top_k` is
swept, and the sweep analysis reports it moves accuracy by 0.04× the seed
noise, which is to say not at all.

The perturbation study finds that the environment and the accelerator do
not move the result, while the benchmark version, the model version and
the temperature move it by 1.2× to 3.0× the seed standard deviation —
past the tolerance declared in advance.

---

## 2. `text_classification/` — real algorithms, no Fabric import

Multinomial naive Bayes, logistic regression trained by minibatch gradient
descent, and a majority-class floor, over a bag-of-words sentiment corpus
with negation and label noise.

| File | What it is |
|---|---|
| `make_data.py` | Generates a 1600-review corpus with disjoint splits. |
| `model.py` | The classifiers. **Does not import `fabric`.** |
| `experiment.yaml` | 6 conditions, 60 runs. |

`model.py` declares a `Protocol` for the two or three attributes it needs
from the context, so it runs under Fabric, in a notebook, or from a plain
script without knowing which. That is the framework-agnosticism claim,
demonstrated rather than asserted.

### It is the more interesting example, because the result goes badly

The specification states a hypothesis before the runs:

> Logistic regression edges out naive Bayes on accuracy; bigrams help
> because the corpus contains negation, and lowercasing does almost
> nothing.

All three parts are wrong:

```
logistic-regression scored 0.6490 +/- 0.0134 over 10 seeds, beating 1 of 2
baselines. Against naive-bayes the difference is not distinguishable from noise.

  baseline        difference  95% CI             test       adj. p     g  verdict
  naive-bayes        -0.0102  [-0.0207, 0.0002]  paired-t   0.0542  -0.6  inconclusive
  majority-class     +0.1550  [0.1422, 0.1678]   paired-t  <0.0001   7.9  improvement

  component                          drop in accuracy  detectable
  lowercasing  ####################           +0.0145  yes
  bigrams      ####................           +0.0028  no
```

The proposed method is *behind* the baseline, the difference is
inconclusive rather than a loss, lowercasing matters and bigrams do not.
Because the analysis was declared in the specification before the runs,
none of it could be adjusted after the fact — which is the entire argument
for declaring it there.

---

## Writing your own

Start from either `experiment.yaml`, and read
[`docs/writing-experiments.md`](../docs/writing-experiments.md). The whole
contract is:

```python
def run(ctx):
    return {"accuracy": 0.91}
```

Then check it before spending compute:

```bash
fabric plan my-experiment.yaml
```
