# sentiment-reproducibility

**Research question.** How far does a pure-Python text classifier move under controlled perturbations of the corpus version, the feature space, the learning rate and the software environment - relative to how far it moves when only the random seed changes?

`EXP-f694df8930` · 83 runs · primary metric `accuracy` · alpha 0.05 with holm correction · generated 2026-09-11 04:22:14 UTC

Specification digest `sha256:cdd116b3b2ce93b341b2595778d9a53bc264e677aedcdacef3edda7f224cf5ba`

## Contents

- [Summary](#summary)
- [Experimental setup](#experimental-setup)
  - [Dataset](#dataset)
  - [Model](#model)
  - [Method](#method)
  - [Conditions](#conditions)
- [Results](#results)
- [Statistical analysis](#statistical-analysis)
  - [accuracy](#accuracy)
  - [f1](#f1)
  - [vocabulary_size](#vocabulary-size)
  - [train_seconds](#train-seconds)
  - [Multiple comparisons](#multiple-comparisons)
- [Cost of the result](#cost-of-the-result)
- [Failure analysis](#failure-analysis)
- [Reproducibility](#reproducibility)
  - [Determinism](#determinism)
  - [Seed noise (the reference)](#seed-noise-the-reference)
  - [Perturbation axes](#perturbation-axes)
  - [Fingerprint](#fingerprint)
- [Evaluation integrity](#evaluation-integrity)
- [Limitations](#limitations)
- [Provenance](#provenance)


## Summary

*The claim this experiment supports, and what it rests on.*

> **Note** — Research question
>
> How far does a pure-Python text classifier move under controlled perturbations of the corpus version, the feature space, the learning rate and the software environment - relative to how far it moves when only the random seed changes?

> **Note** — Hypothesis, stated before the runs
>
> The corpus version dominates because it changes the label noise. The environment does nothing at all, because this pipeline has no threads and no floating-point reductions that could reorder.


logistic-regression scored 0.6490 +/- 0.0134 on accuracy over 10 seeds, which is not distinguishable from any of the 1 baselines at this sample size.

*At a glance*

- **Experiment**: EXP-f694df8930
- **Runs**: 83 completed of 83 planned
- **Conditions**: 8
- **Seeds**: 10
- **Primary metric**: accuracy
- **Significance level**: 0.05 with holm correction
- **Specification digest**: sha256:cdd116b3b2ce93b341b25957

*Claims about accuracy, holm-corrected*

| Compared against | Verdict | Difference | p | Test |
| :--- | :--- | ---: | ---: | :--- |
| naive-bayes | undecided | -0.0102 | 0.0542 | paired-t |

> Difference is signed so that positive always means better, whichever direction the metric runs.

> **OK** — Determinism
>
> Deterministic: 4 executions of an identical run produced identical values on every result metric (accuracy, f1, vocabulary_size). train_seconds varied, as expected: those measure the machine rather than the method.

> **Warning** — Reproducibility
>
> Across 6 perturbation axes, the most fragile is 'dataset' (stability 0.00); the geometric mean is 0.02. Changing configuration, dataset moves accuracy beyond the tolerance declared for it, so a reproduction differing on those axes should not be expected to match. On implementation the data cannot decide whether the shift stays within tolerance; more runs would be needed to say.


## Experimental setup

*Exactly what was run, and on what.*


### Dataset

- **Name**: sentiment@v1
- **Content digest**: sha256:dc7654fbfa35ef9c3b338f8a997b912339861cccbe4c677d167fd5722519e82f
- **Examples**: 1600
- **Source**: not declared
- **Preprocessing**: not declared
- **Splits**: train, test
- **Identifier**: ds-fa210bbbee52f073


### Model

- **Name**: bag-of-words@v1
- **Architecture**: sparse linear model over token counts
- **Weights digest**: not declared
- **Precision**: float64
- **Quantization**: none
- **Tokenizer**: not declared
- **Identifier**: mdl-75ebab4675446f1c


### Method

- **Name**: logistic-regression
- **Entrypoint**: python:examples.text_classification.model:logistic_regression
- **Source commit**: 2078734d7f4bad150b08e975ec7094fcc444139f
- **Source digest**: sha256:822ca4c09cfa4cd0581969a6e0771b6a4784d66117b70e8e022daa680cd42be9
- **Identifier**: alg-f583466bc695ec84


### Conditions

The specification expanded to 8 conditions across 1 configuration(s) and 10 seeds, giving 83 runs.

*Every experimental arm*

| Condition | Role | Components removed | Configuration |
| :--- | :--- | :--- | :--- |
| logistic-regression | treatment | - | alpha=1, batch_size=16, epochs=8, l2=0, learning_rate=0.05 |
| naive-bayes | baseline | - | alpha=1, batch_size=16, epochs=8, l2=0, learning_rate=0.05 |
| logistic-regression~environment:single-threaded | reference | - | alpha=1, batch_size=16, epochs=8, l2=0, learning_rate=0.05 |
| logistic-regression~environment:hash-randomised | reference | - | alpha=1, batch_size=16, epochs=8, l2=0, learning_rate=0.05 |
| logistic-regression~implementation:naive-bayes-as-implementation | reference | - | alpha=1, batch_size=16, epochs=8, l2=0, learning_rate=0.05 |
| logistic-regression~dataset:sentiment@v2 | reference | - | alpha=1, batch_size=16, epochs=8, l2=0, learning_rate=0.05 |
| logistic-regression~configuration:lr-0.2 | reference | - | alpha=1, batch_size=16, epochs=8, l2=0, label=lr-0.2, learning_rate=0.2 |
| logistic-regression~configuration:more-epochs | reference | - | alpha=1, batch_size=16, epochs=16, l2=0, label=more-epochs, learning_rate=0.05 |

- **Seed strategy**: fixed
- **Seed values**: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
- **Sweep strategy**: grid
- **Ablation scope**: reference


## Results

*Every condition, every metric, with its spread.*

*Mean ± standard deviation across seeds*

| Condition | Role | n | accuracy | f1 | vocabulary_size | train_seconds (s) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **logistic-regression** | **treatment** | **10** | **0.6490 ± 0.0134** | **0.6496 ± 0.0215** | **1280.0000 ± 5.3748** | **0.1415 ± 0.0037** |
| naive-bayes | baseline | 10 | 0.6593 ± 0.0149 | 0.6492 ± 0.0167 | 1280.0000 ± 5.3748 | 0.0372 ± 0.0052 |

> A standard deviation is across seeds, not across examples: it is the run-to-run variability of the whole pipeline.


*Every run of every condition on accuracy. One dot per seed; the vertical rule is the mean.*

*Confidence intervals on the mean accuracy*

| Condition | Mean | 95% interval | SD | n |
| :--- | ---: | ---: | ---: | ---: |
| logistic-regression | 0.6490 | [0.6394, 0.6586] | 0.0134 | 10 |
| naive-bayes | 0.6593 | [0.6486, 0.6699] | 0.0149 | 10 |

> Intervals are Student-t on the seed distribution.


## Statistical analysis

*4 comparisons, holm-corrected at alpha=0.05.*


### accuracy

*Treatment versus each baseline on accuracy*

| Baseline | Treatment | Baseline mean | Difference | 95% CI | Test | p | adjusted p | Effect size | Verdict |
| :--- | ---: | ---: | ---: | :--- | :--- | ---: | ---: | :--- | :--- |
| naive-bayes | 0.6490 | 0.6593 | -0.0102 | [-0.0207, 0.0002] | paired-t | 0.0542 | 0.0542 | -0.64 (medium) | inconclusive |

```
naive-bayes  ################################  -0.0102
```

*Estimated difference in accuracy with 95% intervals. An interval crossing zero means the difference is unresolved.*

*What this design could resolve on accuracy*

| Baseline | Smallest detectable difference | Measured difference | Below the floor |
| :--- | ---: | ---: | ---: |
| naive-bayes | 0.0146 | -0.0102 | yes |

> The smallest difference the design would detect 80% of the time at this alpha, given the spread actually observed. A measured difference below that floor cannot be separated from noise by this experiment however the test comes out.


#### Why these tests

Fabric selects the test from the experimental structure rather than letting the author pick after seeing the data. The reasoning for each comparison is recorded below, including the assumptions it could not check.


##### logistic-regression vs naive-bayes

normality was not rejected, so a t-test was used. Pairing by seed removes seed-to-seed variance from the comparison.

*Assumptions of the chosen test*

| Assumption | Status | Evidence |
| :--- | :--- | :--- |
| paired differences is approximately normal | ok | Anderson-Darling A2*=0.639, p=0.0957 |
| observations are independent across replications | ok | each replication is a separate process with a distinct seed |
| pairs are matched | ok | conditions were evaluated on an identical seed set |
| pairs are independent of each other | unverified | structural |

Equivalence test (TOST) against a margin of 0.01: p=0.5209, equivalence not established (which is not the same as a difference having been shown).


### f1

*Treatment versus each baseline on f1*

| Baseline | Treatment | Baseline mean | Difference | 95% CI | Test | p | adjusted p | Effect size | Verdict |
| :--- | ---: | ---: | ---: | :--- | :--- | ---: | ---: | :--- | :--- |
| naive-bayes | 0.6496 | 0.6492 | 0.0004 | [-0.0199, 0.0207] | paired-t | 0.9668 | 0.9668 | 0.01 (negligible) | inconclusive |

```
naive-bayes  ################################  0.0004
```

*Estimated difference in f1 with 95% intervals. An interval crossing zero means the difference is unresolved.*

*What this design could resolve on f1*

| Baseline | Smallest detectable difference | Measured difference | Below the floor |
| :--- | ---: | ---: | ---: |
| naive-bayes | 0.0282 | 0.0004 | yes |

> The smallest difference the design would detect 80% of the time at this alpha, given the spread actually observed. A measured difference below that floor cannot be separated from noise by this experiment however the test comes out.


### vocabulary_size

*Treatment versus each baseline on vocabulary_size*

| Baseline | Treatment | Baseline mean | Difference | 95% CI | Test | p | adjusted p | Effect size | Verdict |
| :--- | ---: | ---: | ---: | :--- | :--- | ---: | ---: | :--- | :--- |
| naive-bayes | 1280.0000 | 1280.0000 | -0.0000 | [-0.0000, -0.0000] | permutation | 1.0000 | 1.0000 | 0.00 (negligible) | equivalent |

```
naive-bayes                                    0
```

*Estimated difference in vocabulary_size with 95% intervals. An interval crossing zero means the difference is unresolved.*

*What this design could resolve on vocabulary_size*

| Baseline | Smallest detectable difference | Measured difference | Below the floor |
| :--- | ---: | ---: | ---: |
| naive-bayes | 0.0000 | -0.0000 | no |

> The smallest difference the design would detect 80% of the time at this alpha, given the spread actually observed. A measured difference below that floor cannot be separated from noise by this experiment however the test comes out.


### train_seconds

*Treatment versus each baseline on train_seconds*

| Baseline | Treatment | Baseline mean | Difference | 95% CI | Test | p | adjusted p | Effect size | Verdict |
| :--- | ---: | ---: | ---: | :--- | :--- | ---: | ---: | :--- | :--- |
| naive-bayes | 0.1415 | 0.0372 | -0.1043 | [-0.1099, -0.0988] | paired-t | <0.0001 | <0.0001 | 12.38 (large) | regression |

```
naive-bayes  ################################  -0.1043
```

*Estimated difference in train_seconds with 95% intervals. An interval crossing zero means the difference is unresolved.*

*What this design could resolve on train_seconds*

| Baseline | Smallest detectable difference | Measured difference | Below the floor |
| :--- | ---: | ---: | ---: |
| naive-bayes | 0.0077 | -0.1043 | no |

> The smallest difference the design would detect 80% of the time at this alpha, given the spread actually observed. A measured difference below that floor cannot be separated from noise by this experiment however the test comes out.


### Multiple comparisons

1 comparisons on accuracy were corrected with the holm method. Holm-Bonferroni controls the family-wise error rate with uniformly more power than Bonferroni and no extra assumptions.


## Cost of the result

*What each condition's quality cost in compute.*

Accuracy without cost is half a comparison. The table below puts the quality of each condition beside the compute it took to get it.

*Quality against compute, summed over all runs of each condition*

| Condition | Role | accuracy | Wall time | Compute x | Δ vs naive-bayes | Cost (USD) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| naive-bayes | baseline | 0.6593 | 2.9 s | 1.00 | 0.0000 | not measured |
| logistic-regression | treatment | 0.6490 | 4.1 s | 1.39 | -0.0102 | not measured |

> Wall time is the total across every run of the condition, measured on the machine in the setup section. Cost is only shown when a price was declared; Fabric does not invent one.


*Up and to the left is better. A condition far to the right has to earn its position with height.*


## Failure analysis

*0 failures in 83 runs*

> **OK** — No failures
>
> All 83 runs completed and produced every declared metric.


## Reproducibility

*Across 6 perturbation axes, the most fragile is 'dataset' (stability 0.00); the geometric mean is 0.02. Changing configuration, dataset moves accuracy beyond the tolerance declared for it, so a reproduction differing on those axes should not be expected to match. On implementation the data cannot decide whether the shift stays within tolerance; more runs would be needed to say.*


### Determinism

Deterministic: 4 executions of an identical run produced identical values on every result metric (accuracy, f1, vocabulary_size). train_seconds varied, as expected: those measure the machine rather than the method.

*Repeat executions of identical runs*

| Condition | Metric | Seed | Replicates | Identical | Max deviation | Counts toward the verdict |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic-regression | accuracy | 1 | 4 | yes | 0.00000000 | yes |
| logistic-regression | f1 | 1 | 4 | yes | 0.00000000 | yes |
| logistic-regression | vocabulary_size | 1 | 4 | yes | 0.00000000 | yes |
| logistic-regression | train_seconds | 1 | 4 | no | 0.01438450 | no (measures the machine) |

> A determinism replicate is the same run executed again: same seed, same code, same data, same machine. Any difference on a result metric is a floor on the effects this pipeline can measure. Timing and memory metrics vary between identical runs by construction and are reported but excluded from the verdict.


### Seed noise (the reference)

seed: 10 seeds of 'logistic-regression' give accuracy = 0.6490 +/- 0.0134 (CV 2.07%), mean [0.6394, 0.6586]. A replication using 5 of these seeds would reach the same conclusion 100% of the time. Read that with care: with 5 runs the smallest attainable p-value is 0.0625, above alpha, so the agreement reflects a shared inability to detect anything rather than a shared detection.

- **Condition**: logistic-regression
- **Seeds**: 10
- **Mean**: 0.6490
- **Standard deviation**: 0.0134
- **Coefficient of variation**: 2.07%
- **Interval on the mean**: [0.6394, 0.6586]
- **Decision stability**: 100%


### Perturbation axes

Each axis below changes exactly one thing and leaves everything else identical. Stability is calibrated against random relabelling of the same runs, so 1.00 means the perturbed runs are indistinguishable from the reference ones.

*Stability, defined*

```
stability(axis) = 1 - max(0, (|Cliff's delta| - E|delta|_null) / (1 - E|delta|_null)), where E|delta|_null is the mean |Cliff's delta| under random relabelling of the same samples. The seed axis is the reference noise floor and is not scored.
```

*Effect of each perturbation on accuracy*

| Axis | Perturbation | Δ mean | Interval | × seed noise | Stability | Stability CI | Permutation p | Tolerance |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dataset | sentiment@v2 | -0.1010 | [-0.1205, -0.0815] | 7.51 | 0.00 | - | 0.0025 | beyond |
| configuration | lr-0.2 | -0.0215 | [-0.0264, -0.0166] | 1.60 | 0.31 | [0.19, 0.43] | 0.0075 | beyond |
| implementation | naive-bayes-as-implementation | +0.0102 | [-0.0002, 0.0207] | 0.76 | 0.69 | [0.50, 0.87] | 0.0898 | inconclusive |
| environment | hash-randomised | +0.0000 | [0.0000, 0.0000] | 0.00 | 1.00 | [1.00, 1.00] | 1.0000 | within |
| environment | single-threaded | +0.0000 | [0.0000, 0.0000] | 0.00 | 1.00 | [1.00, 1.00] | 1.0000 | within |
| configuration | more-epochs | +0.0015 | [-0.0036, 0.0066] | 0.11 | 1.00 | [0.92, 1.00] | 0.7157 | within |

> '× seed noise' is the shift divided by the seed standard deviation: above 1 means the axis moves the result more than the randomness already reported does. The tolerance column is a three-way verdict; 'inconclusive' means the interval straddles the declared margin, which is different from exceeding it.

```
dataset: sentiment@v2                                                           0
configuration: lr-0.2                         ##########                        0.307
implementation: naive-bayes-as-implementatio  ######################            0.6892
environment: hash-randomised                  ################################  1
environment: single-threaded                  ################################  1
configuration: more-epochs                    ################################  1
```

*Calibrated stability per axis, with its interval. The chip on the right is the tolerance verdict.*

- dataset (sentiment@v2): moved accuracy by -0.1010 [-0.1205, -0.0815], 7.5x the seed noise; stability 0.00, beyond the declared tolerance of 0.01.
- configuration (lr-0.2): moved accuracy by -0.0215 [-0.0264, -0.0166], 1.6x the seed noise; stability 0.31 [0.19, 0.43], beyond the declared tolerance of 0.01.
- implementation (naive-bayes-as-implementation): moved accuracy by +0.0102 [-0.0002, 0.0207], 0.8x the seed noise; stability 0.69 [0.50, 0.87] (not distinguishable from relabelling noise, permutation p=0.090), and whether it stays within the declared tolerance of 0.01 is undecided: the interval straddles that boundary.
- environment (hash-randomised): moved accuracy by +0.0000 [0.0000, 0.0000], 0.0x the seed noise; stability 1.00 [1.00, 1.00] (not distinguishable from relabelling noise, permutation p=1.000), within the declared tolerance of 0.01.
- environment (single-threaded): moved accuracy by +0.0000 [0.0000, 0.0000], 0.0x the seed noise; stability 1.00 [1.00, 1.00] (not distinguishable from relabelling noise, permutation p=1.000), within the declared tolerance of 0.01.
- configuration (more-epochs): moved accuracy by +0.0015 [-0.0036, 0.0066], 0.1x the seed noise; stability 1.00 [0.92, 1.00] (not distinguishable from relabelling noise, permutation p=0.716), within the declared tolerance of 0.01.


### Fingerprint

- **Metric**: accuracy
- **Axes measured**: 6
- **Worst-case stability**: 0.00
- **Most fragile axis**: dataset
- **Geometric mean**: 0.02
- **Verdict**: fails_tolerance

The worst case is the headline because a result is only as reproducible as its most fragile dependency. Neither number should be quoted without the per-axis table above.

- Axes from most to least fragile: dataset (0.00), configuration (0.31), implementation (0.69), environment (1.00), environment (1.00), configuration (1.00)
- These axes move the result further than one seed standard deviation: dataset (7.5x), configuration (1.6x). A paper reporting seed variance alone would understate its own uncertainty on them.


## Evaluation integrity

*6 passed, 0 informational, 0 warnings, 0 failures*

*Automated integrity checks*

| Check | Result | Finding |
| :--- | :--- | :--- |
| dataset.verified | pass | Dataset sentiment@v1 matches its registered digest |
| splits.disjoint | pass | Splits contain no exactly duplicated examples |
| seeds.aligned | pass | All 2 arms completed the same 10 seeds |
| metrics.complete | pass | All 4 declared metrics were produced by every arm |
| replication.sufficient | pass | Every arm completed at least 10 replications |
| spec.unchanged | pass | The specification is unchanged since these runs were executed |


## Limitations

*What this experiment does not establish.*

These are generated from the experiment's own structure and results. They are the objections a careful reviewer would raise, stated before the reviewer has to.

1. With 10 replications and the observed spread, this design could detect a difference of 0.0146 in accuracy or larger 80% of the time at alpha=0.05. The difference actually measured (-0.0102) is smaller than that floor, so a null result here is a statement about the experiment's resolution, not about the methods.
2. Changing the configuration moves accuracy by -0.0215, beyond the declared tolerance of 0.01. Results should not be expected to reproduce across configuration changes.
3. Changing the dataset moves accuracy by -0.1010, beyond the declared tolerance of 0.01. Results should not be expected to reproduce across dataset changes.
4. Whether a change of implementation keeps accuracy within its declared tolerance of 0.01 is undecided: the interval [-0.0002, 0.0207] straddles that boundary. More runs on that axis would resolve it.


## Provenance

*Where every number came from.*

*Artifact manifest*

- **Artifacts**: 83
- **Total size**: 0.25 MB
- **Location**: .fabric/experiments/EXP-f694df8930/runs/

*First 12 of 83 artifacts*

| Run | Condition | Seed | Artifact | Digest | Bytes |
| :--- | :--- | ---: | :--- | :--- | ---: |
| RUN-025aa1692d99 | logistic-regression~environment:single-threaded | 6 | predictions.json | f4fef31a5b1a | 2979 |
| RUN-038bc7cfde37 | logistic-regression | 3 | predictions.json | 3184ed307c12 | 2979 |
| RUN-0e670b4b614d | logistic-regression~implementation:naive-bayes-as-implementation | 2 | predictions.json | 4cf9ec53d5b0 | 2979 |
| RUN-10d5b3d289d5 | logistic-regression | 4 | predictions.json | d1732a0a02ea | 2979 |
| RUN-10ef7cebbc35 | logistic-regression~configuration:more-epochs | 1 | predictions.json | b5039fbebb37 | 3008 |
| RUN-11418761fb3a | logistic-regression~implementation:naive-bayes-as-implementation | 5 | predictions.json | d81e77cf9b0e | 2979 |
| RUN-12acffc69942 | naive-bayes | 10 | predictions.json | 5bf667f2edf1 | 2979 |
| RUN-1a015305debd | logistic-regression~configuration:lr-0.2 | 5 | predictions.json | 34bbaff03578 | 3001 |
| RUN-20194b9af644 | logistic-regression~configuration:lr-0.2 | 7 | predictions.json | cf1e1cf47026 | 3001 |
| RUN-21c35f884711 | logistic-regression | 1 | predictions.json | 65af2d1f7410 | 2979 |
| RUN-237586bb1fc6 | naive-bayes | 3 | predictions.json | ad0d4b1bf284 | 2979 |
| RUN-237ff9d26d51 | logistic-regression~configuration:more-epochs | 6 | predictions.json | d2aeb4047177 | 3008 |

> Every artifact is hashed at the moment it is written; the digest is what makes it citable evidence rather than a file that happens to be there.

*Identity*

- **Experiment identifier**: EXP-f694df8930
- **Specification digest**: sha256:cdd116b3b2ce93b341b2595778d9a53bc264e677aedcdacef3edda7f224cf5ba
- **Analysis version**: 1.0
- **Runs recorded**: 83


---

Generated by [Experiment Fabric](https://github.com/) — every number above is addressable back to the run, seed, code, data, environment and hardware that produced it.
