# Statistics that show their work

Most experiment tooling either hard-codes a t-test or leaves the choice to
the author. Fabric does neither: it inspects the experimental structure,
applies a written-down decision procedure, and records the whole chain —
what it checked, what it found, what it chose, and what it was worried
about — next to the p-value.

Everything here is implemented from scratch in the standard library. Not
for purity: so that a p-value computed on a laptop and one computed on a
cluster node come from the same code, bit for bit, and so that installing
Fabric into a locked scientific environment cannot perturb it.

---

## The test-selection procedure

In order, every time:

1. **Is the design paired?** Two conditions evaluated on the *same* seed
   set are paired; anything else is independent. Pairing removes
   seed-to-seed variance from the comparison and typically multiplies
   power several-fold, so Fabric pairs whenever the seed sets match.
2. **Is there enough data to say anything?** Below two replications per
   condition, Fabric refuses to test and says so instead of returning a
   number.
3. **Does normality hold?** Anderson-Darling at n ≥ 8. Below that the
   question is not answerable, and Fabric records it as *unverified*
   rather than assuming either way.
4. **Choose:**

   | Normality | n | Test |
   |---|---|---|
   | not rejected | any | paired t / Welch's t |
   | unverifiable | < 12 | exact permutation test |
   | unverifiable | ≥ 12 | paired t / Welch's t (robustness applies) |
   | rejected | < 12 | exact permutation test |
   | rejected | ≥ 12 | Wilcoxon signed-rank / Mann-Whitney U |

5. **Record every assumption**, including the ones that could not be
   checked.

Welch's is the default for independent groups rather than Student's,
because equal variances are an assumption almost no ML comparison earns
and Welch costs almost no power when they happen to hold.

Below n = 12 a rank test is leaning on asymptotics that have not arrived;
an exact permutation test assumes only exchangeability and, at those
sizes, runs in milliseconds.

---

## Every result carries its assumptions

```
Test:     paired-t
Reason:   normality was not rejected, so a t-test was used. Pairing by seed
          removes seed-to-seed variance from the comparison.

Assumptions:
  ok          paired differences is approximately normal
              Anderson-Darling A2*=0.295, p=0.5973
  ok          observations are independent across replications
              each replication is a separate process with a distinct seed
  ok          pairs are matched
              conditions were evaluated on an identical seed set
  unverified  pairs are independent of each other
              structural
```

`unverified` is a first-class state. A tool that prints a tick beside an
assumption it never checked is worse than one that prints nothing.

---

## The design limit

Exact tests draw from a finite null distribution, so a small experiment
has a hard floor beneath which no p-value can go. With five paired runs the
smallest attainable two-sided p-value is 2/2⁵ = 0.0625 — above `alpha =
0.05`. **No result of that experiment can be significant, whatever it
measures.**

Fabric computes this before execution and says so:

```
design limit: with these sample sizes the smallest two-sided p-value
permutation can produce is 0.0625, which is above alpha=0.05. No result of
this experiment can be significant; add replications.
```

---

## Minimum detectable effect

Every non-significant result has a companion question that is almost never
asked: *how big would the effect have had to be for this design to find
it?* Without it, "no significant difference" is uninterpretable.

Fabric computes it for every comparison, significant or not:

```
MDE = (t_{1-alpha/2, df} + t_{power, df}) * SE
```

with the standard error taken from the data actually collected, and
reports it as a limitation:

> With 10 replications and the observed spread, this design could detect a
> difference of 0.0243 in accuracy or larger 80% of the time at
> alpha=0.05.

When the measured difference is *below* that floor, the report says so
plainly: a null result there describes the experiment's resolution, not
the methods.

`required_replications()` answers the follow-up — how many more seeds
would be needed to detect an effect of a given size.

---

## Effect sizes, always

A p-value answers "could this be noise?". It does not answer "does this
matter?". Every comparison carries both, and every magnitude is labelled
with the convention it came from so nobody has to guess whether 0.3 is
big.

| Effect size | What it says |
|---|---|
| absolute difference | The change, in the metric's own units. |
| relative improvement | Percentage change, signed so positive is always better. |
| Hedges' g | Cohen's d with the small-sample bias correction. At ten seeds, d overstates by ~5%. |
| Cohen's d | The uncorrected standardised difference (d_z when paired). |
| Cliff's delta | Non-parametric dominance, bounded in [-1, 1]. |
| Vargha-Delaney A | P(treatment > baseline). The most readable of the set, and the one worth putting in an abstract. |

Every interval Fabric reports names its own construction method: `t`,
`welch`, `paired-t`, `bootstrap-bca`, `bootstrap-percentile`, `wilson`,
`cliff-consistent`, `normal-approx`. BCa is the bootstrap default because
the bounded metrics that dominate ML papers (accuracy, F1) make percentile
intervals misbehave exactly where they matter — an interval on a mean
accuracy of 0.98 should not extend past 1.0.

Bootstrap seeds are recorded. A confidence interval that changes when you
rerun the analysis is not evidence.

---

## Equivalence, not just difference

`p > 0.05` means "we failed to detect a difference". It does not mean the
two are the same. When a metric declares a `tolerance`, Fabric runs **two
one-sided tests (TOST)** and can make the positive claim instead:

> The two are statistically equivalent within the declared margin of 0.01
> (TOST p=0.0004).

This is the test a reproduction actually needs. `tolerance` is the single
highest-value field in a specification.

---

## Multiple comparisons

Twenty baselines at `alpha = 0.05` buys one significant result for free.
Fabric knows how many comparisons an experiment implies, because it
generated them, so correction is automatic rather than left to the
author's conscience.

Correction is applied **within each metric's family** — the multiplicity
that matters is "how many baselines on this metric", not the total count
of numbers in the report. Available: `none`, `bonferroni`, `holm`
(default), `benjamini-hochberg`, `sidak`. Each records its own rationale
in the output.

An omnibus ANOVA runs first when there are three or more arms. When it
fails to reject, the report labels the pairwise tests **exploratory**
rather than confirmatory.

---

## What is checked against what

The distributions are verified against published critical-value tables,
not against another implementation of the same algorithms:

- Student's t at df = 1, 2, 5, 10, 30, 100, 1000 to 1e-6
- chi-squared and F critical values to 1e-5
- the normal distribution against `statistics.NormalDist` to 1e-13
- ANOVA against a textbook worked example (F = 9.2647)
- Mann-Whitney and Wilcoxon against exactly enumerated null distributions
- the incomplete beta and gamma against closed forms

See `tests/test_statistics.py`.

---

## What is deliberately absent

- **Bayesian analysis.** Priors are a modelling decision that belongs to
  the researcher, not to infrastructure that would have to choose one.
- **Sequential testing / early stopping.** Valid only under a
  pre-registered stopping rule, and the failure mode when misapplied is
  invisible.
- **Automatic outlier removal.** Dropping runs that look wrong is how a
  method's failures become someone else's noise. Failures are recorded and
  analysed instead.
- **A "significance" badge without an effect size.** Never rendered
  anywhere in the reporting layer.
