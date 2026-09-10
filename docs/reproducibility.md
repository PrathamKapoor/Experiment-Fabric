# Measuring reproducibility

This document defines what Fabric means by "reproducible", how the
numbers in the reproducibility section are computed, and — at least as
importantly — what they do not mean.

It is written to be argued with. If the methodology is wrong, it should be
wrong in a way a reader can identify from this page.

---

## The problem with the usual claim

"This experiment is reproducible" is not a falsifiable statement. It has
no units, no threshold, and no stated conditions. Reproducible by whom, on
what hardware, with which package versions, to how many decimal places?

Fabric replaces the claim with measurements. For each *axis* along which
an experiment could differ, it asks: **how far does the result move when
only that axis changes, compared to how far it moves when nothing
changes?**

The axes are the seed, the software environment, the hardware, the dataset
version, the model version, the implementation, and the configuration.

---

## The seed axis is the yardstick, not a score

Changing the random seed is not a perturbation to be scored. It *is* the
experiment's own noise, and it is the reference every other axis is
measured against.

An earlier version of this metric scored the seed axis like any other, by
splitting the seed set in half and measuring the separation between the
halves. That was wrong, and instructively so: because both halves are
drawn from the same distribution, the procedure measures the *estimator's*
sampling noise and nothing about the experiment. A perfectly stable
experiment scored 0.69, and the number moved when you added seeds.

So the seed row reports what it actually knows:

- the standard deviation across seeds, which is the noise floor
- the coefficient of variation
- the confidence interval on the mean
- **decision stability**: the probability that a replication using half as
  many seeds would reach the same conclusion about the headline
  comparison, estimated by subsampling the seeds actually run

Decision stability answers the question researchers really have — *if I
had run five seeds instead of ten, would I have concluded the same
thing?* — and it carries a warning when the smaller design could not have
reached significance at all, so a high agreement rate is not misread as
robustness when it is really a shared inability to detect anything.

---

## The stability score, for every other axis

For an axis `A` there are two samples of the same metric: the reference
runs `R` and the perturbed runs `P`, which differ in `A` and in nothing
else. Start from **Cliff's delta**:

```
delta = P(p > r) - P(p < r)
```

for a random perturbed value `p` and a random reference value `r`.

Cliff's delta is used because metric distributions across seeds are
bounded, frequently skewed, and occasionally bimodal. It is
non-parametric, bounded in [-1, 1], scale-free — so accuracy and latency
sit on one axis without inventing a normalisation — and it is an existing
statistic with published magnitude thresholds rather than something
invented for a dashboard.

### Why the raw delta will not do

Two samples drawn from the *same* distribution do not give `delta = 0`.
With five runs per side, relabelling noise alone produces `|delta|` around
0.3. Scoring against that would report a perfectly reproducible axis as
30% irreproducible, and the number would depend on how many runs you did.

So the score is calibrated against its own null. Pool the reference and
perturbed values, relabel them at random into groups of the observed
sizes, and take `e_null`, the mean `|delta|` under relabelling. Then:

```
excess    = max(0, (|delta| - e_null) / (1 - e_null))
stability = 1 - excess
```

- **1.0** — the perturbed runs are indistinguishable from a random
  relabelling of the reference runs. Knowing which set a value came from
  tells you nothing.
- **0.0** — complete separation, beyond anything relabelling produces.

The same relabelling procedure yields a **permutation p-value** for the
observed delta, which is reported beside the score: it is the rigorous
statement, and the score is the readable one. An interval on the score is
derived from Cliff's delta's own confidence interval, so it is never shown
as a bare point estimate.

### How it behaves

Measured over repeated synthetic trials (`tests/test_analysis_and_reporting.py`):

| Case | n per side | median stability | fraction with permutation p < 0.05 |
|---|---|---|---|
| identical distributions | 10 | 1.00 | ~0.07 |
| identical distributions | 20 | 1.00 | ~0.10 |
| shift of 1 SD | 10 | 0.57 | ~0.53 |
| shift of 3 SD | 10 | 0.03 | 1.00 |

The false-positive rate sits slightly above the nominal 5% because the
correction subtracts the *mean* of the null rather than a quantile of it.
That is a deliberate trade: it keeps the score readable and conservative
about claiming stability. The permutation p-value beside it is the
calibrated test.

---

## What the score deliberately does not tell you

**It says nothing about the size of a shift in metric units.** A
perturbation that moves every run by a completely negligible +0.0001 in
the same direction separates the two samples completely and scores 0.
That is not a defect being apologised for; it is the definition. The score
measures *detectability*.

Which is why it is never reported alone. Every axis also carries:

- the raw difference in means, with a confidence interval
- the shift **in units of seed noise** — the single most interpretable
  column in the table, because "this axis moves the result 3.0× further
  than the randomness you are already reporting" needs no calibration
- an **equivalence verdict**, when the metric declares a `tolerance`

Read **stability as detectability** and **equivalence as acceptability**.
They answer different questions and can disagree. When they do, the report
says so explicitly.

---

## The three-way tolerance verdict

A failed equivalence test does not mean "outside the tolerance". It means
equivalence was not established, which is weaker and different. Conflating
the two is the equivalence-testing counterpart of reading `p > 0.05` as
proof of no effect, so Fabric keeps the third answer:

| Verdict | Meaning |
|---|---|
| `within` | Equivalence established: the difference is smaller than the declared tolerance (TOST rejects at `alpha`). |
| `beyond` | The confidence interval lies entirely outside the tolerance band. The difference is real and larger than acceptable. |
| `inconclusive` | The interval straddles the tolerance boundary. More runs would settle it; this data does not. |

---

## Determinism is tested, not assumed

Most people believe a fixed seed makes a pipeline deterministic. Very
often it does not: thread scheduling, non-associative floating-point
reduction, dictionary iteration order, and non-deterministic GPU kernels
all survive a fixed seed.

A **determinism replicate** is the same run executed again — same seed,
same code, same data, same machine. Any difference between replicates is a
lower bound on noise that no amount of seed-fixing removes, and it matters
downstream: an effect smaller than a pipeline's own non-determinism cannot
be measured by that pipeline, however many seeds you run. Fabric reports
that floor and flags any perturbation effect below it.

```yaml
reproducibility:
  determinism_replicates: 3
```

---

## The fingerprint

The headline number is the **minimum** across the measured non-seed axes,
because a result is only as reproducible as its most fragile dependency.
The geometric mean is reported beside it.

Neither should be quoted without the per-axis table. A single number that
compresses "the environment does not matter but the benchmark version
does" into one figure has destroyed the finding.

---

## Known limitations

Stated here rather than left for a reader to discover:

1. **Stability is estimated from few runs.** With ten runs per side,
   Cliff's delta has a standard error around 0.26. The reported interval
   reflects this; the point estimate alone does not.
2. **The null calibration is Monte Carlo.** 400 relabellings, seeded, so
   it is reproducible — but it is an estimate, not an exact value.
3. **One axis at a time.** Fabric measures main effects, not interactions.
   Whether a CUDA change matters *more* on a different dataset version is
   not answered by this design.
4. **`environment` and `hardware` are environment-variable overlays.**
   They capture thread counts, determinism flags and device visibility.
   They do not capture a genuinely different machine or a different CUDA
   build; for that, run the same specification elsewhere and use
   `fabric reproduce`, which diffs the captured environments and names the
   differences.
5. **The perturbations are the ones you declared.** An axis nobody thought
   to vary is not measured, and the report says "unknown" rather than
   "fine". Untested is not the same as stable.
6. **`source_digest` hashes the entrypoint file, not its imports.** A dirty
   working tree is recorded and flagged, and the certificate downgrades
   the `code` check accordingly, but a change in a helper module the
   entrypoint imports is not individually detected. Commit before the runs
   that matter.
7. **Contamination detection is exact-match.** Duplicated examples are
   found; near-duplicates and paraphrases are not.

---

## The research question this enables

The instrument exists to answer a question about the field, not about one
experiment:

> How sensitive are machine-learning conclusions to controlled
> perturbations in randomness, software environments, hardware, model
> versions, dataset versions, and implementation details — and how does
> that sensitivity compare to the seed variance that papers already
> report?

Fabric makes each of those a measured quantity on a comparable scale. See
[`paper/`](../paper) for the study design.
