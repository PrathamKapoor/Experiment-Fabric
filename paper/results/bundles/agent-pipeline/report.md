# agent-reproducibility-profile

**Research question.** How far does the accuracy of the retrieval agent move under controlled perturbations of the software environment, the accelerator, the benchmark version, the model version, the implementation, and the temperature - relative to how far it moves when only the random seed changes?

`EXP-832b6cabc2` · 93 runs · primary metric `accuracy` · alpha 0.05 with holm correction · generated 2026-09-11 04:22:17 UTC

Specification digest `sha256:c3033e55b7f4364c319f80b212e03c2947b6987bb7961b864af9a3c7e1c90e9f`

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
  - [latency_ms](#latency-ms)
  - [robustness](#robustness)
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
> How far does the accuracy of the retrieval agent move under controlled perturbations of the software environment, the accelerator, the benchmark version, the model version, the implementation, and the temperature - relative to how far it moves when only the random seed changes?

> **Note** — Hypothesis, stated before the runs
>
> Seed variance dominates the environment and hardware axes, but the dataset version and the model version move the result by more than one seed standard deviation, which means a reproduction that differs on those axes should not be expected to match.


full-agent scored 0.7478 +/- 0.0214 on accuracy over 10 seeds, beating every baseline; the narrowest margin is +0.2200 over rag-classic [0.2032, 0.2368] (paired-t, adjusted p=0.0000).

*At a glance*

- **Experiment**: EXP-832b6cabc2
- **Runs**: 93 completed of 93 planned
- **Conditions**: 9
- **Seeds**: 10
- **Primary metric**: accuracy
- **Significance level**: 0.05 with holm correction
- **Specification digest**: sha256:c3033e55b7f4364c319f80b2

*Claims about accuracy, holm-corrected*

| Compared against | Verdict | Difference | p | Test |
| :--- | :--- | ---: | ---: | :--- |
| rag-classic | supported | 0.2200 | <0.0001 | paired-t |

> Difference is signed so that positive always means better, whichever direction the metric runs.

> **OK** — Determinism
>
> Deterministic: 4 executions of an identical run produced identical values on every result metric (accuracy, robustness). latency_ms varied, as expected: those measure the machine rather than the method.

> **Warning** — Reproducibility
>
> Across 7 perturbation axes, the most fragile is 'configuration' (stability 0.05); the geometric mean is 0.51. Changing configuration, dataset, model moves accuracy beyond the tolerance declared for it, so a reproduction differing on those axes should not be expected to match. On implementation the data cannot decide whether the shift stays within tolerance; more runs would be needed to say.


## Experimental setup

*Exactly what was run, and on what.*


### Dataset

- **Name**: agent-benchmark@v1
- **Content digest**: sha256:4cb166de973fceda83a082f989f603e0cdd7b47f9ffc6be559842d92acf9f45a
- **Examples**: 400
- **Source**: not declared
- **Preprocessing**: not declared
- **Splits**: not declared
- **Identifier**: ds-79847b591f21d45d


### Model

- **Name**: sim-reasoner@v1
- **Architecture**: simulated log-odds responder
- **Weights digest**: not declared
- **Precision**: float64
- **Quantization**: none
- **Tokenizer**: not declared
- **Identifier**: mdl-b8d5d6ce0e3e86c2


### Method

- **Name**: full-agent
- **Entrypoint**: python:examples.agent_pipeline.pipeline:run
- **Source commit**: 2078734d7f4bad150b08e975ec7094fcc444139f
- **Source digest**: sha256:def10dd51a0902c76cc38f86d150f9076a2d341ee02118b6e1ff20992e74f794
- **Identifier**: alg-98bac261fa29a0a8


### Conditions

The specification expanded to 9 conditions across 1 configuration(s) and 10 seeds, giving 93 runs.

*Every experimental arm*

| Condition | Role | Components removed | Configuration |
| :--- | :--- | :--- | :--- |
| full-agent | treatment | - | memory_size=32, rerank_depth=10, temperature=0.2, top_k=8, verification_threshold=0.45 |
| rag-classic | baseline | - | memory_size=32, rerank_depth=10, temperature=0.2, top_k=8, verification_threshold=0.45 |
| full-agent~environment:single-threaded | reference | - | memory_size=32, rerank_depth=10, temperature=0.2, top_k=8, verification_threshold=0.45 |
| full-agent~environment:hash-randomised | reference | - | memory_size=32, rerank_depth=10, temperature=0.2, top_k=8, verification_threshold=0.45 |
| full-agent~hardware:no-accelerator | reference | - | memory_size=32, rerank_depth=10, temperature=0.2, top_k=8, verification_threshold=0.45 |
| full-agent~dataset:agent-benchmark@v2 | reference | - | memory_size=32, rerank_depth=10, temperature=0.2, top_k=8, verification_threshold=0.45 |
| full-agent~model:sim-reasoner@v2 | reference | - | memory_size=32, rerank_depth=10, temperature=0.2, top_k=8, verification_threshold=0.45 |
| full-agent~implementation:full-agent-reordered | reference | - | memory_size=32, rerank_depth=10, temperature=0.2, top_k=8, verification_threshold=0.45 |
| full-agent~configuration:temperature-0.8 | reference | - | label=temperature-0.8, memory_size=32, rerank_depth=10, temperature=0.8, top_k=8, verification_threshold=0.45 |

- **Seed strategy**: fixed
- **Seed values**: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
- **Sweep strategy**: grid
- **Ablation scope**: reference


## Results

*Every condition, every metric, with its spread.*

*Mean ± standard deviation across seeds*

| Condition | Role | n | accuracy | latency_ms (ms) | robustness |
| :--- | :--- | ---: | ---: | ---: | ---: |
| **full-agent** | **treatment** | **10** | **0.7478 ± 0.0214** | **145.7519 ± 0.4401** | **0.7123 ± 0.0223** |
| rag-classic | baseline | 10 | 0.5277 ± 0.0249 | 90.2741 ± 0.2727 | 0.4728 ± 0.0294 |

> A standard deviation is across seeds, not across examples: it is the run-to-run variability of the whole pipeline.


*Every run of every condition on accuracy. One dot per seed; the vertical rule is the mean.*

*Confidence intervals on the mean accuracy*

| Condition | Mean | 95% interval | SD | n |
| :--- | ---: | ---: | ---: | ---: |
| full-agent | 0.7478 | [0.7324, 0.7631] | 0.0214 | 10 |
| rag-classic | 0.5277 | [0.5099, 0.5456] | 0.0249 | 10 |

> Intervals are Student-t on the seed distribution.


## Statistical analysis

*3 comparisons, holm-corrected at alpha=0.05.*


### accuracy

*Treatment versus each baseline on accuracy*

| Baseline | Treatment | Baseline mean | Difference | 95% CI | Test | p | adjusted p | Effect size | Verdict |
| :--- | ---: | ---: | ---: | :--- | :--- | ---: | ---: | :--- | :--- |
| rag-classic | 0.7478 | 0.5277 | 0.2200 | [0.2032, 0.2368] | paired-t | <0.0001 | <0.0001 | 8.59 (large) | improvement |

```
rag-classic  ################################  0.22
```

*Estimated difference in accuracy with 95% intervals. An interval crossing zero means the difference is unresolved.*

*What this design could resolve on accuracy*

| Baseline | Smallest detectable difference | Measured difference | Below the floor |
| :--- | ---: | ---: | ---: |
| rag-classic | 0.0233 | 0.2200 | no |

> The smallest difference the design would detect 80% of the time at this alpha, given the spread actually observed. A measured difference below that floor cannot be separated from noise by this experiment however the test comes out.


#### Why these tests

Fabric selects the test from the experimental structure rather than letting the author pick after seeing the data. The reasoning for each comparison is recorded below, including the assumptions it could not check.


##### full-agent vs rag-classic

normality was not rejected, so a t-test was used. Pairing by seed removes seed-to-seed variance from the comparison.

*Assumptions of the chosen test*

| Assumption | Status | Evidence |
| :--- | :--- | :--- |
| paired differences is approximately normal | ok | Anderson-Darling A2*=0.536, p=0.1703 |
| observations are independent across replications | ok | each replication is a separate process with a distinct seed |
| pairs are matched | ok | conditions were evaluated on an identical seed set |
| pairs are independent of each other | unverified | structural |

Equivalence test (TOST) against a margin of 0.01: p=1.0000, equivalence not established (which is not the same as a difference having been shown).


### latency_ms

*Treatment versus each baseline on latency_ms*

| Baseline | Treatment | Baseline mean | Difference | 95% CI | Test | p | adjusted p | Effect size | Verdict |
| :--- | ---: | ---: | ---: | :--- | :--- | ---: | ---: | :--- | :--- |
| rag-classic | 145.7519 | 90.2741 | -55.4778 | [-55.5980, -55.3575] | paired-t | <0.0001 | <0.0001 | 301.75 (large) | regression |

```
rag-classic  ################################  -55.4778
```

*Estimated difference in latency_ms with 95% intervals. An interval crossing zero means the difference is unresolved.*

*What this design could resolve on latency_ms*

| Baseline | Smallest detectable difference | Measured difference | Below the floor |
| :--- | ---: | ---: | ---: |
| rag-classic | 0.1672 | -55.4778 | no |

> The smallest difference the design would detect 80% of the time at this alpha, given the spread actually observed. A measured difference below that floor cannot be separated from noise by this experiment however the test comes out.


### robustness

*Treatment versus each baseline on robustness*

| Baseline | Treatment | Baseline mean | Difference | 95% CI | Test | p | adjusted p | Effect size | Verdict |
| :--- | ---: | ---: | ---: | :--- | :--- | ---: | ---: | :--- | :--- |
| rag-classic | 0.7123 | 0.4728 | 0.2395 | [0.2285, 0.2505] | paired-t | <0.0001 | <0.0001 | 14.22 (large) | improvement |

```
rag-classic  ################################  0.2395
```

*Estimated difference in robustness with 95% intervals. An interval crossing zero means the difference is unresolved.*

*What this design could resolve on robustness*

| Baseline | Smallest detectable difference | Measured difference | Below the floor |
| :--- | ---: | ---: | ---: |
| rag-classic | 0.0153 | 0.2395 | no |

> The smallest difference the design would detect 80% of the time at this alpha, given the spread actually observed. A measured difference below that floor cannot be separated from noise by this experiment however the test comes out.


### Multiple comparisons

1 comparisons on accuracy were corrected with the holm method. Holm-Bonferroni controls the family-wise error rate with uniformly more power than Bonferroni and no extra assumptions.


## Cost of the result

*What each condition's quality cost in compute.*

Accuracy without cost is half a comparison. The table below puts the quality of each condition beside the compute it took to get it.

*Quality against compute, summed over all runs of each condition*

| Condition | Role | accuracy | Wall time | Compute x | Δ vs rag-classic | Cost (USD) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| full-agent | treatment | 0.7478 | 291 ms | 1.33 | 0.2200 | not measured |
| rag-classic | baseline | 0.5277 | 219 ms | 1.00 | 0.0000 | not measured |

> Wall time is the total across every run of the condition, measured on the machine in the setup section. Cost is only shown when a price was declared; Fabric does not invent one.


*Up and to the left is better. A condition far to the right has to earn its position with height.*


## Failure analysis

*0 failures in 93 runs*

> **OK** — No failures
>
> All 93 runs completed and produced every declared metric.


## Reproducibility

*Across 7 perturbation axes, the most fragile is 'configuration' (stability 0.05); the geometric mean is 0.51. Changing configuration, dataset, model moves accuracy beyond the tolerance declared for it, so a reproduction differing on those axes should not be expected to match. On implementation the data cannot decide whether the shift stays within tolerance; more runs would be needed to say.*


### Determinism

Deterministic: 4 executions of an identical run produced identical values on every result metric (accuracy, robustness). latency_ms varied, as expected: those measure the machine rather than the method.

*Repeat executions of identical runs*

| Condition | Metric | Seed | Replicates | Identical | Max deviation | Counts toward the verdict |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| full-agent | accuracy | 1 | 4 | yes | 0.00000000 | yes |
| full-agent | latency_ms | 1 | 4 | yes | 0.00000000 | no (measures the machine) |
| full-agent | robustness | 1 | 4 | yes | 0.00000000 | yes |

> A determinism replicate is the same run executed again: same seed, same code, same data, same machine. Any difference on a result metric is a floor on the effects this pipeline can measure. Timing and memory metrics vary between identical runs by construction and are reported but excluded from the verdict.


### Seed noise (the reference)

seed: 10 seeds of 'full-agent' give accuracy = 0.7478 +/- 0.0214 (CV 2.87%), mean [0.7324, 0.7631]. A replication using 5 of these seeds would reach the same conclusion 0% of the time, but that is a limit of the test rather than of the effect: with 5 runs the smallest attainable p-value is 0.0625, which is above alpha, so no 5-seed replication could reach significance whatever it measured.

- **Condition**: full-agent
- **Seeds**: 10
- **Mean**: 0.7478
- **Standard deviation**: 0.0214
- **Coefficient of variation**: 2.87%
- **Interval on the mean**: [0.7324, 0.7631]
- **Decision stability**: 0%


### Perturbation axes

Each axis below changes exactly one thing and leaves everything else identical. Stability is calibrated against random relabelling of the same runs, so 1.00 means the perturbed runs are indistinguishable from the reference ones.

*Stability, defined*

```
stability(axis) = 1 - max(0, (|Cliff's delta| - E|delta|_null) / (1 - E|delta|_null)), where E|delta|_null is the mean |Cliff's delta| under random relabelling of the same samples. The seed axis is the reference noise floor and is not scored.
```

*Effect of each perturbation on accuracy*

| Axis | Perturbation | Δ mean | Interval | × seed noise | Stability | Stability CI | Permutation p | Tolerance |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| configuration | temperature-0.8 | -0.0632 | [-0.0754, -0.0511] | 2.95 | 0.05 | [0.01, 0.09] | 0.0025 | beyond |
| dataset | agent-benchmark@v2 | +0.0302 | [0.0223, 0.0382] | 1.41 | 0.38 | [0.24, 0.52] | 0.0200 | beyond |
| model | sim-reasoner@v2 | +0.0252 | [0.0194, 0.0311] | 1.18 | 0.48 | [0.33, 0.64] | 0.0299 | beyond |
| implementation | full-agent-reordered | +0.0027 | [-0.0071, 0.0126] | 0.13 | 1.00 | [0.91, 1.00] | 0.7157 | inconclusive |
| environment | hash-randomised | +0.0000 | [0.0000, 0.0000] | 0.00 | 1.00 | [1.00, 1.00] | 1.0000 | within |
| environment | single-threaded | +0.0000 | [0.0000, 0.0000] | 0.00 | 1.00 | [1.00, 1.00] | 1.0000 | within |
| hardware | no-accelerator | +0.0000 | [0.0000, 0.0000] | 0.00 | 1.00 | [1.00, 1.00] | 1.0000 | within |

> '× seed noise' is the shift divided by the seed standard deviation: above 1 means the axis moves the result more than the randomness already reported does. The tolerance column is a three-way verdict; 'inconclusive' means the interval straddles the declared margin, which is different from exceeding it.

```
configuration: temperature-0.8        ##                                0.0508
dataset: agent-benchmark@v2           ############                      0.3828
model: sim-reasoner@v2                ###############                   0.4831
implementation: full-agent-reordered  ################################  1
environment: hash-randomised          ################################  1
environment: single-threaded          ################################  1
hardware: no-accelerator              ################################  1
```

*Calibrated stability per axis, with its interval. The chip on the right is the tolerance verdict.*

- configuration (temperature-0.8): moved accuracy by -0.0632 [-0.0754, -0.0511], 3.0x the seed noise; stability 0.05 [0.01, 0.09], beyond the declared tolerance of 0.01.
- dataset (agent-benchmark@v2): moved accuracy by +0.0302 [0.0223, 0.0382], 1.4x the seed noise; stability 0.38 [0.24, 0.52], beyond the declared tolerance of 0.01.
- model (sim-reasoner@v2): moved accuracy by +0.0252 [0.0194, 0.0311], 1.2x the seed noise; stability 0.48 [0.33, 0.64], beyond the declared tolerance of 0.01.
- implementation (full-agent-reordered): moved accuracy by +0.0027 [-0.0071, 0.0126], 0.1x the seed noise; stability 1.00 [0.91, 1.00] (not distinguishable from relabelling noise, permutation p=0.716), and whether it stays within the declared tolerance of 0.01 is undecided: the interval straddles that boundary.
- environment (hash-randomised): moved accuracy by +0.0000 [0.0000, 0.0000], 0.0x the seed noise; stability 1.00 [1.00, 1.00] (not distinguishable from relabelling noise, permutation p=1.000), within the declared tolerance of 0.01.
- environment (single-threaded): moved accuracy by +0.0000 [0.0000, 0.0000], 0.0x the seed noise; stability 1.00 [1.00, 1.00] (not distinguishable from relabelling noise, permutation p=1.000), within the declared tolerance of 0.01.
- hardware (no-accelerator): moved accuracy by +0.0000 [0.0000, 0.0000], 0.0x the seed noise; stability 1.00 [1.00, 1.00] (not distinguishable from relabelling noise, permutation p=1.000), within the declared tolerance of 0.01.


### Fingerprint

- **Metric**: accuracy
- **Axes measured**: 7
- **Worst-case stability**: 0.05
- **Most fragile axis**: configuration
- **Geometric mean**: 0.51
- **Verdict**: fails_tolerance

The worst case is the headline because a result is only as reproducible as its most fragile dependency. Neither number should be quoted without the per-axis table above.

- Axes from most to least fragile: configuration (0.05), dataset (0.38), model (0.48), implementation (1.00), environment (1.00), environment (1.00), hardware (1.00)
- These axes move the result further than one seed standard deviation: configuration (3.0x), dataset (1.4x), model (1.2x). A paper reporting seed variance alone would understate its own uncertainty on them.


## Evaluation integrity

*5 passed, 0 informational, 0 warnings, 0 failures*

*Automated integrity checks*

| Check | Result | Finding |
| :--- | :--- | :--- |
| dataset.verified | pass | Dataset agent-benchmark@v1 matches its registered digest |
| seeds.aligned | pass | All 2 arms completed the same 10 seeds |
| metrics.complete | pass | All 3 declared metrics were produced by every arm |
| replication.sufficient | pass | Every arm completed at least 10 replications |
| spec.unchanged | pass | The specification is unchanged since these runs were executed |


## Limitations

*What this experiment does not establish.*

These are generated from the experiment's own structure and results. They are the objections a careful reviewer would raise, stated before the reviewer has to.

1. With 10 replications and the observed spread, this design could detect a difference of 0.0233 in accuracy or larger 80% of the time at alpha=0.05.
2. Changing the configuration moves accuracy by -0.0632, beyond the declared tolerance of 0.01. Results should not be expected to reproduce across configuration changes.
3. Changing the dataset moves accuracy by +0.0302, beyond the declared tolerance of 0.01. Results should not be expected to reproduce across dataset changes.
4. Changing the model moves accuracy by +0.0252, beyond the declared tolerance of 0.01. Results should not be expected to reproduce across model changes.
5. Whether a change of implementation keeps accuracy within its declared tolerance of 0.01 is undecided: the interval [-0.0071, 0.0126] straddles that boundary. More runs on that axis would resolve it.


## Provenance

*Where every number came from.*

*Artifact manifest*

- **Artifacts**: 93
- **Total size**: 10.59 MB
- **Location**: .fabric/experiments/EXP-832b6cabc2/runs/

*First 12 of 93 artifacts*

| Run | Condition | Seed | Artifact | Digest | Bytes |
| :--- | :--- | ---: | :--- | :--- | ---: |
| RUN-009573ced6d7 | full-agent~implementation:full-agent-reordered | 4 | predictions.json | 5dc214fc3238 | 113923 |
| RUN-039bd81531ae | full-agent~environment:single-threaded | 3 | predictions.json | 60c3ff9bbe18 | 113958 |
| RUN-07b063ecd991 | full-agent~hardware:no-accelerator | 3 | predictions.json | 60c3ff9bbe18 | 113958 |
| RUN-09de9d2ee139 | full-agent~configuration:temperature-0.8 | 10 | predictions.json | 5c0cb66824d2 | 113950 |
| RUN-0a180505d8f3 | full-agent~environment:hash-randomised | 9 | predictions.json | 3daafb144616 | 113919 |
| RUN-0d30015d985f | full-agent~model:sim-reasoner@v2 | 2 | predictions.json | 584e964f97e2 | 113908 |
| RUN-13df23239b49 | full-agent~configuration:temperature-0.8 | 2 | predictions.json | 3a97b6c61ab1 | 113937 |
| RUN-15f50676b65f | full-agent~environment:single-threaded | 10 | predictions.json | 9c86f637c81b | 113937 |
| RUN-1a0e2eef492f | full-agent~model:sim-reasoner@v2 | 4 | predictions.json | 9d003df76230 | 113898 |
| RUN-1dda4365e2e3 | full-agent | 1 | predictions.json | 1fbe3b66a42a | 113929 |
| RUN-1f0c81991561 | full-agent~environment:single-threaded | 9 | predictions.json | 3daafb144616 | 113919 |
| RUN-20e88eaa4664 | full-agent~environment:single-threaded | 8 | predictions.json | 5d2a0554b28d | 113937 |

> Every artifact is hashed at the moment it is written; the digest is what makes it citable evidence rather than a file that happens to be there.

*Identity*

- **Experiment identifier**: EXP-832b6cabc2
- **Specification digest**: sha256:c3033e55b7f4364c319f80b212e03c2947b6987bb7961b864af9a3c7e1c90e9f
- **Analysis version**: 1.0
- **Runs recorded**: 93


---

Generated by [Experiment Fabric](https://github.com/) — every number above is addressable back to the run, seed, code, data, environment and hardware that produced it.
