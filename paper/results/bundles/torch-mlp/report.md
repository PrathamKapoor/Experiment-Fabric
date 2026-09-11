# torch-mlp-reproducibility

**Research question.** How far does a real PyTorch training run move under controlled perturbations of the thread count, the loss implementation, the model width, the dataset version and the batch size - relative to how far it moves when only the random seed changes?

`EXP-24d9c76f72` · 83 runs · primary metric `test_loss` · alpha 0.05 with holm correction · generated 2026-09-11 04:22:17 UTC

Specification digest `sha256:b7aee5d9804ec49f4f388e80e2e7cc9b3629d12a319edeb0415642dfc5ae5114`

## Contents

- [Summary](#summary)
- [Experimental setup](#experimental-setup)
  - [Dataset](#dataset)
  - [Model](#model)
  - [Method](#method)
  - [Conditions](#conditions)
- [Results](#results)
- [Statistical analysis](#statistical-analysis)
  - [test_loss](#test-loss)
  - [accuracy](#accuracy)
  - [train_loss](#train-loss)
  - [train_seconds](#train-seconds)
  - [threads](#threads)
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
> How far does a real PyTorch training run move under controlled perturbations of the thread count, the loss implementation, the model width, the dataset version and the batch size - relative to how far it moves when only the random seed changes?

> **Note** — Hypothesis, stated before the runs
>
> Changing the CPU thread count perturbs floating-point reduction order and so moves the loss slightly, while leaving accuracy untouched; the model width, the dataset version and the batch size move both. Fixing the seed does not make training deterministic across thread counts.


mlp scored 0.1639 +/- 0.0023 on test_loss over 10 seeds, beating every baseline; the narrowest margin is +0.5056 over logistic-regression [0.5038, 0.5075] (paired-t, adjusted p=0.0000).

*At a glance*

- **Experiment**: EXP-24d9c76f72
- **Runs**: 83 completed of 83 planned
- **Conditions**: 8
- **Seeds**: 10
- **Primary metric**: test_loss
- **Significance level**: 0.05 with holm correction
- **Specification digest**: sha256:b7aee5d9804ec49f4f388e80

*Claims about test_loss, holm-corrected*

| Compared against | Verdict | Difference | p | Test |
| :--- | :--- | ---: | ---: | :--- |
| logistic-regression | supported | 0.5056 | <0.0001 | paired-t |

> Difference is signed so that positive always means better, whichever direction the metric runs.

> **OK** — Determinism
>
> Deterministic: 4 executions of an identical run produced identical values on every result metric (accuracy, test_loss, threads, train_loss). train_seconds varied, as expected: those measure the machine rather than the method.

> **Warning** — Reproducibility
>
> Across 6 perturbation axes, the most fragile is 'dataset' (stability 0.00); the geometric mean is 0.03. Changing dataset moves test_loss beyond the tolerance declared for it, so a reproduction differing on that axis should not be expected to match.


## Experimental setup

*Exactly what was run, and on what.*


### Dataset

- **Name**: spiral@v1
- **Content digest**: sha256:879697e9da556f271c2672c48ac804606acb69b0ffaf314253c010a1d56c6241
- **Examples**: 1400
- **Source**: not declared
- **Preprocessing**: not declared
- **Splits**: train, test
- **Identifier**: ds-37348658f32407cd


### Model

- **Name**: mlp-2x64@v1
- **Architecture**: 2-64-64-2 multi-layer perceptron
- **Weights digest**: not declared
- **Precision**: float32
- **Quantization**: none
- **Tokenizer**: not declared
- **Identifier**: mdl-5b618fcec10cb0af


### Method

- **Name**: torch-mlp
- **Entrypoint**: python:examples.torch_mlp.model:mlp
- **Source commit**: 2078734d7f4bad150b08e975ec7094fcc444139f
- **Source digest**: sha256:302b9adb0a86d51754650cd8ce35dc90ed8837a3162458f5f346c940493eb689
- **Identifier**: alg-162d73e9e42e776f


### Conditions

The specification expanded to 8 conditions across 1 configuration(s) and 10 seeds, giving 83 runs.

*Every experimental arm*

| Condition | Role | Components removed | Configuration |
| :--- | :--- | :--- | :--- |
| mlp | treatment | - | batch_size=64, dropout_rate=0.1, epochs=40, learning_rate=0.01, weight_decay=0.0001 |
| logistic-regression | baseline | - | batch_size=64, dropout_rate=0.1, epochs=40, learning_rate=0.01, weight_decay=0.0001 |
| mlp~environment:single-threaded | reference | - | batch_size=64, dropout_rate=0.1, epochs=40, learning_rate=0.01, weight_decay=0.0001 |
| mlp~environment:two-threaded | reference | - | batch_size=64, dropout_rate=0.1, epochs=40, learning_rate=0.01, weight_decay=0.0001 |
| mlp~implementation:torch-mlp-unfused-loss | reference | - | batch_size=64, dropout_rate=0.1, epochs=40, learning_rate=0.01, weight_decay=0.0001 |
| mlp~model:mlp-2x128@v2 | reference | - | batch_size=64, dropout_rate=0.1, epochs=40, learning_rate=0.01, weight_decay=0.0001 |
| mlp~dataset:spiral@v2 | reference | - | batch_size=64, dropout_rate=0.1, epochs=40, learning_rate=0.01, weight_decay=0.0001 |
| mlp~configuration:batch-32 | reference | - | batch_size=32, dropout_rate=0.1, epochs=40, label=batch-32, learning_rate=0.01, weight_decay=0.0001 |

- **Seed strategy**: fixed
- **Seed values**: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
- **Sweep strategy**: grid
- **Ablation scope**: reference


## Results

*Every condition, every metric, with its spread.*

*Mean ± standard deviation across seeds*

| Condition | Role | n | test_loss | accuracy | train_loss | train_seconds (s) | threads |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **mlp** | **treatment** | **10** | **0.1639 ± 0.0023** | **0.9629 ± 0.0000** | **0.3050 ± 0.2230** | **3.3665 ± 0.1039** | **6.0000 ± 0.0000** |
| logistic-regression | baseline | 10 | 0.6695 ± 0.0015 | 0.4583 ± 0.0097 | 0.6450 ± 0.0562 | 2.1108 ± 0.1698 | 6.0000 ± 0.0000 |

> A standard deviation is across seeds, not across examples: it is the run-to-run variability of the whole pipeline.


*Every run of every condition on test_loss. One dot per seed; the vertical rule is the mean.*

*Confidence intervals on the mean test_loss*

| Condition | Mean | 95% interval | SD | n |
| :--- | ---: | ---: | ---: | ---: |
| mlp | 0.1639 | [0.1622, 0.1656] | 0.0023 | 10 |
| logistic-regression | 0.6695 | [0.6685, 0.6706] | 0.0015 | 10 |

> Intervals are Student-t on the seed distribution.


## Statistical analysis

*5 comparisons, holm-corrected at alpha=0.05.*


### test_loss

*Treatment versus each baseline on test_loss*

| Baseline | Treatment | Baseline mean | Difference | 95% CI | Test | p | adjusted p | Effect size | Verdict |
| :--- | ---: | ---: | ---: | :--- | :--- | ---: | ---: | :--- | :--- |
| logistic-regression | 0.1639 | 0.6695 | 0.5056 | [0.5038, 0.5075] | paired-t | <0.0001 | <0.0001 | -182.09 (large) | improvement |

```
logistic-regression  ################################  0.5056
```

*Estimated difference in test_loss with 95% intervals. An interval crossing zero means the difference is unresolved.*

*What this design could resolve on test_loss*

| Baseline | Smallest detectable difference | Measured difference | Below the floor |
| :--- | ---: | ---: | ---: |
| logistic-regression | 0.0025 | 0.5056 | no |

> The smallest difference the design would detect 80% of the time at this alpha, given the spread actually observed. A measured difference below that floor cannot be separated from noise by this experiment however the test comes out.


#### Why these tests

Fabric selects the test from the experimental structure rather than letting the author pick after seeing the data. The reasoning for each comparison is recorded below, including the assumptions it could not check.


##### mlp vs logistic-regression

normality was not rejected, so a t-test was used. Pairing by seed removes seed-to-seed variance from the comparison.

*Assumptions of the chosen test*

| Assumption | Status | Evidence |
| :--- | :--- | :--- |
| paired differences is approximately normal | ok | Anderson-Darling A2*=0.348, p=0.4774 |
| observations are independent across replications | ok | each replication is a separate process with a distinct seed |
| pairs are matched | ok | conditions were evaluated on an identical seed set |
| pairs are independent of each other | unverified | structural |

Equivalence test (TOST) against a margin of 0.02: p=1.0000, equivalence not established (which is not the same as a difference having been shown).


### accuracy

*Treatment versus each baseline on accuracy*

| Baseline | Treatment | Baseline mean | Difference | 95% CI | Test | p | adjusted p | Effect size | Verdict |
| :--- | ---: | ---: | ---: | :--- | :--- | ---: | ---: | :--- | :--- |
| logistic-regression | 0.9629 | 0.4583 | 0.5046 | [0.4976, 0.5115] | paired-t | <0.0001 | <0.0001 | 47.41 (large) | improvement |

```
logistic-regression  ################################  0.5046
```

*Estimated difference in accuracy with 95% intervals. An interval crossing zero means the difference is unresolved.*

*What this design could resolve on accuracy*

| Baseline | Smallest detectable difference | Measured difference | Below the floor |
| :--- | ---: | ---: | ---: |
| logistic-regression | 0.0097 | 0.5046 | no |

> The smallest difference the design would detect 80% of the time at this alpha, given the spread actually observed. A measured difference below that floor cannot be separated from noise by this experiment however the test comes out.


### train_loss

*Treatment versus each baseline on train_loss*

| Baseline | Treatment | Baseline mean | Difference | 95% CI | Test | p | adjusted p | Effect size | Verdict |
| :--- | ---: | ---: | ---: | :--- | :--- | ---: | ---: | :--- | :--- |
| logistic-regression | 0.3050 | 0.6450 | 0.3401 | [0.1818, 0.4983] | paired-t | 0.0009 | 0.0009 | -1.41 (large) | improvement |

```
logistic-regression  ################################  0.3401
```

*Estimated difference in train_loss with 95% intervals. An interval crossing zero means the difference is unresolved.*

*What this design could resolve on train_loss*

| Baseline | Smallest detectable difference | Measured difference | Below the floor |
| :--- | ---: | ---: | ---: |
| logistic-regression | 0.2201 | 0.3401 | no |

> The smallest difference the design would detect 80% of the time at this alpha, given the spread actually observed. A measured difference below that floor cannot be separated from noise by this experiment however the test comes out.


### train_seconds

*Treatment versus each baseline on train_seconds*

| Baseline | Treatment | Baseline mean | Difference | 95% CI | Test | p | adjusted p | Effect size | Verdict |
| :--- | ---: | ---: | ---: | :--- | :--- | ---: | ---: | :--- | :--- |
| logistic-regression | 3.3665 | 2.1108 | -1.2557 | [-1.3292, -1.1823] | paired-t | <0.0001 | <0.0001 | 11.18 (large) | regression |

```
logistic-regression  ################################  -1.2557
```

*Estimated difference in train_seconds with 95% intervals. An interval crossing zero means the difference is unresolved.*

*What this design could resolve on train_seconds*

| Baseline | Smallest detectable difference | Measured difference | Below the floor |
| :--- | ---: | ---: | ---: |
| logistic-regression | 0.1021 | -1.2557 | no |

> The smallest difference the design would detect 80% of the time at this alpha, given the spread actually observed. A measured difference below that floor cannot be separated from noise by this experiment however the test comes out.


### threads

*Treatment versus each baseline on threads*

| Baseline | Treatment | Baseline mean | Difference | 95% CI | Test | p | adjusted p | Effect size | Verdict |
| :--- | ---: | ---: | ---: | :--- | :--- | ---: | ---: | :--- | :--- |
| logistic-regression | 6.0000 | 6.0000 | 0.0000 | [0.0000, 0.0000] | permutation | 1.0000 | 1.0000 | 0.00 (negligible) | inconclusive |

```
logistic-regression                                    0
```

*Estimated difference in threads with 95% intervals. An interval crossing zero means the difference is unresolved.*

*What this design could resolve on threads*

| Baseline | Smallest detectable difference | Measured difference | Below the floor |
| :--- | ---: | ---: | ---: |
| logistic-regression | 0.0000 | 0.0000 | no |

> The smallest difference the design would detect 80% of the time at this alpha, given the spread actually observed. A measured difference below that floor cannot be separated from noise by this experiment however the test comes out.


### Multiple comparisons

1 comparisons on test_loss were corrected with the holm method. Holm-Bonferroni controls the family-wise error rate with uniformly more power than Bonferroni and no extra assumptions.


## Cost of the result

*What each condition's quality cost in compute.*

Accuracy without cost is half a comparison. The table below puts the quality of each condition beside the compute it took to get it.

*Quality against compute, summed over all runs of each condition*

| Condition | Role | test_loss | Wall time | Compute x | Δ vs logistic-regression | Cost (USD) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| mlp | treatment | 0.1639 | 65.1 s | 1.26 | 0.5056 | not measured |
| logistic-regression | baseline | 0.6695 | 51.7 s | 1.00 | -0.0000 | not measured |

> Wall time is the total across every run of the condition, measured on the machine in the setup section. Cost is only shown when a price was declared; Fabric does not invent one.


*Up and to the left is better. A condition far to the right has to earn its position with height.*


## Failure analysis

*0 failures in 83 runs*

> **OK** — No failures
>
> All 83 runs completed and produced every declared metric.


## Reproducibility

*Across 6 perturbation axes, the most fragile is 'dataset' (stability 0.00); the geometric mean is 0.03. Changing dataset moves test_loss beyond the tolerance declared for it, so a reproduction differing on that axis should not be expected to match.*


### Determinism

Deterministic: 4 executions of an identical run produced identical values on every result metric (accuracy, test_loss, threads, train_loss). train_seconds varied, as expected: those measure the machine rather than the method.

*Repeat executions of identical runs*

| Condition | Metric | Seed | Replicates | Identical | Max deviation | Counts toward the verdict |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| mlp | test_loss | 1 | 4 | yes | 0.00000000 | yes |
| mlp | accuracy | 1 | 4 | yes | 0.00000000 | yes |
| mlp | train_loss | 1 | 4 | yes | 0.00000000 | yes |
| mlp | train_seconds | 1 | 4 | no | 0.90835560 | no (measures the machine) |
| mlp | threads | 1 | 4 | yes | 0.00000000 | yes |

> A determinism replicate is the same run executed again: same seed, same code, same data, same machine. Any difference on a result metric is a floor on the effects this pipeline can measure. Timing and memory metrics vary between identical runs by construction and are reported but excluded from the verdict.


### Seed noise (the reference)

seed: 10 seeds of 'mlp' give test_loss = 0.1639 +/- 0.0023 (CV 1.43%), mean [0.1622, 0.1656]. A replication using 5 of these seeds would reach the same conclusion 0% of the time, but that is a limit of the test rather than of the effect: with 5 runs the smallest attainable p-value is 0.0625, which is above alpha, so no 5-seed replication could reach significance whatever it measured.

- **Condition**: mlp
- **Seeds**: 10
- **Mean**: 0.1639
- **Standard deviation**: 0.0023
- **Coefficient of variation**: 1.43%
- **Interval on the mean**: [0.1622, 0.1656]
- **Decision stability**: 0%


### Perturbation axes

Each axis below changes exactly one thing and leaves everything else identical. Stability is calibrated against random relabelling of the same runs, so 1.00 means the perturbed runs are indistinguishable from the reference ones.

*Stability, defined*

```
stability(axis) = 1 - max(0, (|Cliff's delta| - E|delta|_null) / (1 - E|delta|_null)), where E|delta|_null is the mean |Cliff's delta| under random relabelling of the same samples. The seed axis is the reference noise floor and is not scored.
```

*Effect of each perturbation on test_loss*

| Axis | Perturbation | Δ mean | Interval | × seed noise | Stability | Stability CI | Permutation p | Tolerance |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dataset | spiral@v2 | +0.1884 | [0.1857, 0.1911] | 80.47 | 0.00 | - | 0.0025 | beyond |
| model | mlp-2x128@v2 | -0.0008 | [-0.0021, 0.0004] | 0.36 | 0.91 | [0.70, 1.00] | 0.3092 | within |
| implementation | torch-mlp-unfused-loss | +0.0000 | [0.0000, 0.0000] | 0.00 | 1.00 | [1.00, 1.00] | 1.0000 | within |
| environment | single-threaded | +0.0000 | [-0.0000, 0.0001] | 0.01 | 1.00 | [0.98, 1.00] | 0.8579 | within |
| environment | two-threaded | +0.0000 | [-0.0000, 0.0001] | 0.01 | 1.00 | [1.00, 1.00] | 0.9177 | within |
| configuration | batch-32 | -0.0001 | [-0.0026, 0.0024] | 0.05 | 1.00 | [1.00, 1.00] | 0.9302 | within |

> '× seed noise' is the shift divided by the seed standard deviation: above 1 means the axis moves the result more than the randomness already reported does. The tolerance column is a three-way verdict; 'inconclusive' means the interval straddles the declared margin, which is different from exceeding it.

```
dataset: spiral@v2                                                        0
model: mlp-2x128@v2                     #############################     0.9138
implementation: torch-mlp-unfused-loss  ################################  1
environment: single-threaded            ################################  1
environment: two-threaded               ################################  1
configuration: batch-32                 ################################  1
```

*Calibrated stability per axis, with its interval. The chip on the right is the tolerance verdict.*

- dataset (spiral@v2): moved test_loss by +0.1884 [0.1857, 0.1911], 80.5x the seed noise; stability 0.00, beyond the declared tolerance of 0.02.
- model (mlp-2x128@v2): moved test_loss by -0.0008 [-0.0021, 0.0004], 0.4x the seed noise; stability 0.91 [0.70, 1.00] (not distinguishable from relabelling noise, permutation p=0.309), within the declared tolerance of 0.02.
- implementation (torch-mlp-unfused-loss): moved test_loss by +0.0000 [0.0000, 0.0000], 0.0x the seed noise; stability 1.00 [1.00, 1.00] (not distinguishable from relabelling noise, permutation p=1.000), within the declared tolerance of 0.02.
- environment (single-threaded): moved test_loss by +0.0000 [-0.0000, 0.0001], 0.0x the seed noise; stability 1.00 [0.98, 1.00] (not distinguishable from relabelling noise, permutation p=0.858), within the declared tolerance of 0.02.
- environment (two-threaded): moved test_loss by +0.0000 [-0.0000, 0.0001], 0.0x the seed noise; stability 1.00 [1.00, 1.00] (not distinguishable from relabelling noise, permutation p=0.918), within the declared tolerance of 0.02.
- configuration (batch-32): moved test_loss by -0.0001 [-0.0026, 0.0024], 0.1x the seed noise; stability 1.00 [1.00, 1.00] (not distinguishable from relabelling noise, permutation p=0.930), within the declared tolerance of 0.02.


### Fingerprint

- **Metric**: test_loss
- **Axes measured**: 6
- **Worst-case stability**: 0.00
- **Most fragile axis**: dataset
- **Geometric mean**: 0.03
- **Verdict**: fails_tolerance

The worst case is the headline because a result is only as reproducible as its most fragile dependency. Neither number should be quoted without the per-axis table above.

- Axes from most to least fragile: dataset (0.00), model (0.91), implementation (1.00), environment (1.00), environment (1.00), configuration (1.00)
- These axes move the result further than one seed standard deviation: dataset (80.5x). A paper reporting seed variance alone would understate its own uncertainty on them.


## Evaluation integrity

*6 passed, 0 informational, 0 warnings, 0 failures*

*Automated integrity checks*

| Check | Result | Finding |
| :--- | :--- | :--- |
| dataset.verified | pass | Dataset spiral@v1 matches its registered digest |
| splits.disjoint | pass | Splits contain no exactly duplicated examples |
| seeds.aligned | pass | All 2 arms completed the same 10 seeds |
| metrics.complete | pass | All 5 declared metrics were produced by every arm |
| replication.sufficient | pass | Every arm completed at least 10 replications |
| spec.unchanged | pass | The specification is unchanged since these runs were executed |


## Limitations

*What this experiment does not establish.*

These are generated from the experiment's own structure and results. They are the objections a careful reviewer would raise, stated before the reviewer has to.

1. With 10 replications and the observed spread, this design could detect a difference of 0.0025 in test_loss or larger 80% of the time at alpha=0.05.
2. Changing the dataset moves test_loss by +0.1884, beyond the declared tolerance of 0.02. Results should not be expected to reproduce across dataset changes.


## Provenance

*Where every number came from.*

*Artifact manifest*

- **Artifacts**: 73
- **Total size**: 0.19 MB
- **Location**: .fabric/experiments/EXP-24d9c76f72/runs/

*First 12 of 73 artifacts*

| Run | Condition | Seed | Artifact | Digest | Bytes |
| :--- | :--- | ---: | :--- | :--- | ---: |
| RUN-00b74b312096 | mlp~configuration:batch-32 | 8 | predictions.json | 236b63639502 | 2688 |
| RUN-01a90b51a86c | mlp~environment:two-threaded | 5 | predictions.json | c2a5336a23fe | 2663 |
| RUN-0736dca580f2 | mlp~configuration:batch-32 | 5 | predictions.json | 236b63639502 | 2688 |
| RUN-0eb6e6ec08ab | mlp~dataset:spiral@v2 | 6 | predictions.json | 5826ff6337e0 | 2663 |
| RUN-23bb21391921 | mlp~environment:two-threaded | 3 | predictions.json | c2a5336a23fe | 2663 |
| RUN-27d72c70581c | mlp~configuration:batch-32 | 3 | predictions.json | 236b63639502 | 2688 |
| RUN-305f851041fa | mlp~model:mlp-2x128@v2 | 3 | predictions.json | 14e3f07485c0 | 2663 |
| RUN-40de29ea752a | mlp | 10 | predictions.json | 14e3f07485c0 | 2663 |
| RUN-4aabb0b998e3 | mlp~implementation:torch-mlp-unfused-loss | 6 | predictions.json | 14e3f07485c0 | 2663 |
| RUN-5046d333f8bc | mlp~configuration:batch-32 | 7 | predictions.json | b62f2ad87e2e | 2688 |
| RUN-5279eb69fb7d | mlp~model:mlp-2x128@v2 | 1 | predictions.json | 14e3f07485c0 | 2663 |
| RUN-5663f1aea827 | mlp | 4 | predictions.json | 14e3f07485c0 | 2663 |

> Every artifact is hashed at the moment it is written; the digest is what makes it citable evidence rather than a file that happens to be there.

*Identity*

- **Experiment identifier**: EXP-24d9c76f72
- **Specification digest**: sha256:b7aee5d9804ec49f4f388e80e2e7cc9b3629d12a319edeb0415642dfc5ae5114
- **Analysis version**: 1.0
- **Runs recorded**: 83


---

Generated by [Experiment Fabric](https://github.com/) — every number above is addressable back to the run, seed, code, data, environment and hardware that produced it.
