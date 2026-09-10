# Pre-registration protocol

Written before data collection. Its purpose is to remove the degrees of
freedom that would otherwise let the analysis be shaped by the results.

## Primary hypotheses

| ID | Statement | Test | Decision rule |
|---|---|---|---|
| H1 | At least one non-seed axis exceeds one seed standard deviation | Proportion of experiments with `max(noise_multiples) > 1` | Wilson interval; supported if the lower bound exceeds 0.25 |
| H2 | Dataset and model versions dominate the other axes | Friedman test across axes, ranked within experiment | Supported if dataset/model rank highest and the omnibus rejects at 0.05 |
| H3 | A material fraction of pipelines are non-deterministic under a fixed seed | Proportion failing the determinism check | Wilson interval; reported, not thresholded |
| H4 | Conclusions are less stable than point estimates under seed resampling | Decision stability at half the seeds | Supported if median decision stability is below 0.90 among originally-significant results |

H4 is the most consequential and the cheapest to test, because it needs no
extra runs -- only the seeds already collected.

## Fixed in advance

- **Primary metric**: each experiment's declared `primary` metric. Not
  chosen by the study.
- **Alpha**: 0.05, two-sided.
- **Correction**: Holm, within each experiment's comparison family.
- **Seeds**: 30 per condition where compute allows, never fewer than 10.
- **Determinism replicates**: 3.
- **Equivalence margins**: declared per metric in each specification
  before execution, and reported with every verdict.
- **Exclusions**: an experiment is excluded only if more than 20% of its
  runs fail for *infrastructure* reasons (network, dependency). Method
  failures are data and are never grounds for exclusion.

## Stopping rule

The sample size is fixed in advance per stratum. No interim analysis, and
no adding experiments after seeing results. If compute runs out, the
shortfall is reported by stratum rather than quietly absorbed.

## What would falsify the thesis

If non-seed axes are consistently *below* the seed noise floor across the
sample, the thesis is wrong and the paper says so: the field's current
practice of reporting seed variance alone would be adequate, and the
instrument's value would be confined to the cases where it is not.

This outcome is reported with the same prominence as the alternative.

## Analysis code

The analysis is the released version of `fabric.analysis`, pinned by
commit in every certificate. No bespoke analysis scripts: if a number in
the paper cannot be produced by `fabric analyze`, it does not go in the
paper.
