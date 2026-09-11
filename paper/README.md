# The reproducibility study

Experiment Fabric is infrastructure, and infrastructure is not a research
contribution. The contribution is the methodology it makes executable and
the study it makes possible.

This directory holds the design for that study. It is a plan, not a
result: nothing here reports findings that have not been produced.

---

## Research question

> How sensitive are machine-learning conclusions to controlled
> perturbations in randomness, software environments, hardware, model
> versions, dataset versions, and implementation details — and how does
> that sensitivity compare to the seed variance that papers already
> report?

## What is already known, and what is not

This question does not arrive on empty ground, and the paper must not
pretend it does. See [`../docs/related-work.md`](../docs/related-work.md)
for the survey; the short version:

- That random seeds move ML results is **established**, across NLP, deep
  RL, computer vision and LLM fine-tuning.
- That a *non-seed* axis can exceed the seed axis is **already
  demonstrated**: Quantifying Inherent Randomness in Machine Learning
  Algorithms (arXiv:2206.12353) finds random train/test splitting
  produces larger variation than training randomness.
- That fixed seeds do not guarantee determinism on GPU is **documented**,
  including the CUDA mechanisms behind it.

So H1 below is, in part, a **replication at greater breadth** rather than
a discovery, and it is reported as one.

What is not established, and is what this study is for:

1. **Comparability.** Every existing study fixes one axis, one domain and
   one model family. There is no profile that places seed, environment,
   hardware, dataset version, model version and implementation on a
   single scale, for the same experiment, so that they can be ranked.
2. **Decision stability.** The literature reports variance in point
   estimates. Whether the *conclusion* flips — which is what a reader of
   a paper actually relies on — is reported far less often.
3. **Re-runnable evidence.** Reproducibility measurements are published
   as tables. Here each one is a bundle that re-executes and re-compares
   on someone else's machine.

The contribution is therefore **an instrument and a comparable
methodology**, plus whatever the breadth turns up. That is a narrower
claim than "we discovered that ML is irreproducible", and it is the one
the evidence can carry.

## Hypotheses, stated in advance

**H1.** For a substantial fraction of published-style experiments, at
least one non-seed axis moves the primary metric by more than one seed
standard deviation. *(Partially anticipated by prior work; treated as a
replication at breadth.)*

**H2.** Dataset version and model version dominate; environment and
hardware are usually — but not always — below the seed noise floor.

**H3.** A non-trivial fraction of pipelines are not deterministic under a
fixed seed, and where they are not, the non-determinism floor exceeds
effect sizes that are reported as findings.

**H4.** Conclusions are substantially less stable than point estimates
under seed resampling: many experiments that report a significant result
would not have reported one with half the seeds.

H4 is the most consequential and the least anticipated by prior work, and
it is the cheapest to test, because it needs no extra runs — only the
seeds already collected.

## Design

For each experiment in the sample:

```
                        experiment
                             │
        ┌──────────┬─────────┼─────────┬──────────┬──────────┐
      seeds    environment  hardware  dataset   model   implementation
      30x         k envs     m devices  versions  versions   variants
        └──────────┴─────────┼─────────┴──────────┴──────────┘
                             ▼
                   per-axis stability, in units
                     of the seed noise floor
```

One thing changes per condition. Every arm is evaluated on the same seed
set so comparisons stay paired. The analysis is declared in the
specification before execution and cannot be adjusted afterwards; the
specification digest in each certificate is the evidence of that.

## Measurements

Per experiment, per axis:

- Δ in the primary metric, with a confidence interval
- Δ expressed in seed standard deviations (the headline quantity)
- calibrated stability, `1 − |Cliff's delta|` against the relabelling null,
  with its interval and permutation p-value
- an equivalence verdict against the tolerance declared in advance
- the determinism floor from repeated identical runs
- decision stability under seed subsampling
- the minimum detectable effect of the design

Definitions and their limitations: [`../docs/reproducibility.md`](../docs/reproducibility.md).

## Sample

The claim is about the field, so the sample has to look like the field,
not like whatever was convenient:

- classical ML on tabular and text data (cheap, high-n, many seeds)
- small vision and sequence models trained from scratch
- fine-tuning runs on a fixed pretrained checkpoint
- retrieval and agent pipelines with component structure
- LLM evaluation harnesses, where seeds are only partly controllable

Stratifying by compute cost matters: the axes that dominate for a
five-second scikit-learn fit are unlikely to be the ones that dominate for
a multi-GPU fine-tune, and a study drawn only from the cheap end would
reach a confident and wrong conclusion.

## Deliverables

1. **A reproducibility profile per experiment** — the per-axis table, in a
   comparable form, from a certified bundle.
2. **A cross-experiment analysis** — which axes dominate, how often the
   non-seed axes exceed the seed floor, how often conclusions flip.
3. **A benchmark** — the specifications themselves, so the study can be
   re-run on new hardware or new framework versions and the answer
   tracked over time. This is the part with a long half-life.
4. **Guidance** — what a paper should report so a reader can judge how far
   its conclusions travel.

## Threats to validity

Stated here because a study of reproducibility that hides its own
weaknesses would be self-refuting:

- **Sample bias.** Experiments that are cheap to run many times are not
  representative of experiments that are expensive to run once, and the
  expensive ones are where the field's conclusions actually live.
- **Main effects only.** The design measures one axis at a time. Whether
  CUDA version matters *more* on a particular dataset version is a
  question this design cannot answer.
- **Environment perturbation is partial.** Environment-variable overlays
  cover thread counts and determinism flags; genuinely different builds
  require executing the specification on a different machine, which the
  reproduction workflow supports but which does not scale to a large
  sample without cluster access.
- **Tolerances are chosen.** Equivalence verdicts depend on the margin
  declared in each specification. Declared in advance, but still a
  judgement, and it should be reported alongside every verdict.
- **The stability score is an estimate.** Its interval is wide at ten runs
  per side. Conclusions should rest on the shift in seed-noise units and
  the equivalence verdict; the score is a summary, not the evidence.
- **Simulated components in the shipped example.** The agent example is a
  simulation, chosen so the demonstration runs in seconds. It exercises
  the instrument; it is not evidence about real agents, and the study
  sample must not include it.

## Reproducing this study

Every experiment in the study is a specification plus a certified bundle.

```bash
fabric reproduce study/exp-0001-bundle/
fabric verify    study/exp-0001-bundle/
```

Each bundle carries the specification, every run, the analysis, the
provenance graph, the certificate and the report — in plain text, readable
without Fabric installed.

## Files

| File | Contents |
|---|---|
| `README.md` | This document. |
| `protocol.md` | The pre-registration: hypotheses, sample, analysis plan, stopping rules. |
| `outline.md` | The paper's section structure and which artifact fills each one. |
