# Related work

What already exists, what it does, and — stated plainly — what of this
project is genuinely new versus what is a better-packaged version of an
idea other people have had.

Sources were checked in September 2026 against primary documentation and
published papers rather than vendor comparison posts. Where a claim could
not be verified from a primary source, it says so instead of asserting.

---

## The honest summary first

**The research question is not novel.** That randomness moves ML results
is well established, and at least one published study has already shown a
*non-seed* axis dominating the seed axis: Quantifying Inherent Randomness
in Machine Learning Algorithms finds that random dataset splitting
produces larger variation than training randomness does, for the
algorithms it studied. Fabric's H1 is, in that specific case, already
answered.

**The statistical machinery is not novel either.** Cliff's delta,
permutation calibration, TOST equivalence, Holm correction and
minimum-detectable-effect are all standard. NLPStatTest already automates
significance testing with effect sizes for a specific domain.

What is different is the **composition and the plumbing**:

1. A single tool where the *specification* fixes the analysis before the
   runs, and the provenance, statistics, perturbation design and
   reporting are the same object rather than four disconnected steps.
2. Non-seed axes measured **on the same scale as the seed**, so
   "environment" and "dataset version" and "seed" are directly
   comparable, in one table, per experiment.
3. **Decision stability** — whether the *conclusion* survives, not just
   whether the number moves. Existing randomness studies mostly report
   variance in point estimates.
4. Reproducibility measurements that are themselves **portable and
   verifiable** artifacts (bundles, content-addressed certificates).

That is an infrastructure contribution with a methodological wrapper. It
is not a new statistical result, and the paper should not pretend to be
one.

---

## 1. Experiment trackers

**MLflow** ([docs](https://mlflow.org/docs/latest/)) — tracking, model
registry, projects, deployment. Relevant detail often missed in
comparisons: MLflow *does* compute dataset digests. The `mlflow.data`
module records datasets against runs with "the dataset's name, digest
(hash), schema, and profile", and computes a digest automatically when
one is not supplied
([Dataset Tracking](https://mlflow.org/docs/latest/ml/dataset/)).

So "MLflow does not version datasets" would be **false**, and an earlier
draft of this project's comparison table implied it. The accurate
difference is narrower and worth stating precisely:

| | MLflow | Fabric |
|---|---|---|
| Dataset digest computed | yes | yes |
| Digest is *metadata attached to a run* | yes | — |
| Digest **is the dataset's identity** | no | yes |
| Re-registering the same version with different bytes | permitted | refused, with a field-level diff |

The distinction matters for exactly one scenario, which is the scenario
this project exists for: six months later, did the benchmark change
underneath the results? A recorded digest lets you answer that if you
think to look. An enforced identity makes it impossible not to notice.

**Weights & Biases** — managed SaaS tracking, visual run comparison, and
W&B Sweeps for hyperparameter search, which Fabric deliberately does not
attempt (see "Not built" in the README). Acquired by CoreWeave in May
2025.

**DVC** — git-centric versioning of data and pipelines, content-addressed
by design. On the data-versioning axis specifically, DVC is *stronger*
than Fabric: it handles large remote storage, which Fabric does not do at
all. Fabric's registry is a local record of identity, not a data store.

**Sacred**, **Aim**, **Neptune**, **ClearML** — the same broad category.
Sacred's configuration capture and run database anticipate a good deal of
what specification-driven tools do.

**What none of them appear to provide**, based on their primary
documentation: automatic selection of a statistical test with its
assumptions recorded, automated ablation expansion with interaction
analysis, determinism testing by re-executing identical runs, controlled
perturbation studies across environment/hardware/dataset/model/
implementation axes, or minimum-detectable-effect reporting. If any of
them has added these, this document is out of date and the comparison
should be corrected rather than defended.

---

## 2. Replicability frameworks

**MLXP** ([arXiv:2402.13831](https://arxiv.org/abs/2402.13831)) — the
closest neighbour in intent: an open-source Python tool for managing
experiments "while ensuring reproducibility and replicability", explicitly
positioned as simple and lightweight against tools whose complexity
"hinders adoption within the research community".

Fabric shares the goal and takes the opposite bet on complexity: it adds
a statistics engine and a perturbation-study design, which is more
machinery, not less. MLXP's critique is a fair one to hold this project
to, and the mitigation is that the extra machinery is *declarative* —
`fabric run experiment.yaml` — rather than API surface a researcher has
to learn.

**Hydra** — configuration composition and multirun sweeps. Fabric's
`configuration` block covers a strict subset of what Hydra does, and
Hydra users should keep using Hydra; the expansion is not the
contribution.

---

## 3. Provenance systems

**W3C PROV** — the standard vocabulary for provenance (entities,
activities, agents). Fabric's graph is PROV-shaped without claiming
conformance; exporting to PROV-O would be a genuinely useful addition and
is not implemented.

**noWorkflow**, **ReproZip** — provenance capture at the language and
system level respectively. ReproZip in particular packs an execution
environment so it can be re-run elsewhere, which is stronger than
Fabric's environment *fingerprinting*: Fabric records what the
environment was and can tell you how two differ, but it does not package
one.

---

## 4. Statistical tooling for ML

**NLPStatTest** ([arXiv:2011.13231](https://arxiv.org/pdf/2011.13231)) —
automates comparing system performance by running appropriate
significance tests and estimating effect size. This is prior art for
Fabric's statistics layer and should be cited as such. The differences
are scope (Fabric is domain-agnostic and covers the whole experiment
lifecycle) and the recorded decision procedure, not the idea of
automating test selection.

The broader methodological literature on significance testing in ML —
including long-standing critiques of single-run comparisons and of
p-values without effect sizes — is the foundation the statistics module
implements rather than extends.

---

## 5. Empirical studies of randomness and reproducibility

This is the literature the project's research question sits inside, and
where it must be most careful about claiming novelty.

**Quantifying Inherent Randomness in Machine Learning Algorithms**
([arXiv:2206.12353](https://arxiv.org/pdf/2206.12353)) — compares
variation from training randomness against variation from random
train/test splitting, across random forests, gradient boosting machines
and feedforward networks. Finds that **dataset splitting produces higher
variation than training randomness**, and that network training varies
more than tree methods.

This is the closest thing to a direct partial answer to H1 and has to be
treated as such: a non-seed axis exceeding the seed axis is a *documented
finding*, not a hypothesis this project originated. Fabric's contribution
on that axis is generalisation (more axes, more experiment types, one
comparable scale) rather than discovery.

**Investigating the Impact of Randomness on Reproducibility in Computer
Vision** ([arXiv:2410.02806](https://arxiv.org/abs/2410.02806)) — studies
randomness in applied CV, including CUDA-induced nondeterminism from
non-deterministic operation implementations, floating-point precision,
and parallel execution order. Directly supports the design decision to
*test* determinism rather than assume a fixed seed provides it.

**Studies of seed effects in NLP and deep RL** — random-seed choice
changing reported results for sequence-tagging systems, and the variance
and reporting problems in deep reinforcement learning, are long-standing
results in this literature.

**Assessing the Macro and Micro Effects of Random Seeds on Fine-Tuning
LLMs** ([arXiv:2503.07329](https://arxiv.org/html/2503.07329v2)) — the
same question for the current model scale.

### What is left for this project to contribute

Given the above, the defensible framing is narrow and should stay narrow:

- Existing studies each fix one axis, one domain, one set of models.
  There is no **comparable, cross-experiment profile** that puts seed,
  environment, hardware, dataset version, model version and
  implementation on one scale.
- Existing studies report **variance in estimates**. Whether the
  *conclusion* flips — decision stability — is reported far less often,
  and is what a reader of a paper actually depends on.
- The measurements are not usually shipped as **re-runnable artifacts**.
  A bundle that re-executes and re-compares on someone else's machine is
  a different kind of claim than a table in a PDF.

If the study finds what the prior literature predicts, that is a
replication at greater breadth, and the paper should present it as one.

---

## 6. What this project does not do

Stated so the comparison is not read as a claim of superiority:

- **No large-file storage.** DVC and git-lfs do this; Fabric records
  identity and expects the bytes to live somewhere else.
- **No hyperparameter optimisation.** W&B Sweeps, Optuna and Ray Tune do
  this well. Fabric supports grid and random expansion only, deliberately.
- **No environment packaging.** ReproZip and containers do this; Fabric
  fingerprints and diffs environments but cannot rebuild one.
- **No hosted service, no collaboration features, no dashboards.**
- **No distributed orchestration.**

---

## How to correct this document

Every claim above is meant to be checkable. If a tool has a capability
this document says it lacks, the fix is to change the document — and the
README table it backs — rather than to defend the original wording. The
comparison table in the README is generated from this file's conclusions
by hand; when this changes, that changes.

Last verified: September 2026.
