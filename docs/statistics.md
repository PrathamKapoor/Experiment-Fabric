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

Two independent references, because neither alone is enough.

**In the test suite**, against published critical-value tables rather than
another implementation of the same algorithms:

- Student's t at df = 1, 2, 5, 10, 30, 100, 1000 to 1e-6
- chi-squared and F critical values to 1e-5
- the normal distribution against `statistics.NormalDist` to 1e-13
- ANOVA against a textbook worked example (F = 9.2647)
- Mann-Whitney and Wilcoxon against exactly enumerated null distributions
- the incomplete beta and gamma against closed forms

See `tests/test_statistics.py`.

**Against SciPy 1.17**, in a separate environment, so a passing table cannot
come from the same assumption written twice:

The expected values live in
[`../tests/test_statistics_against_reference.py`](../tests/test_statistics_against_reference.py)
as literals. SciPy is **not** a dependency and that module does not import it
— the values have to outlive the library that produced them. Where a closed
form or an exact enumeration exists, the expected value is the enumerated
one, so the test inherits neither SciPy's code nor its conventions.

| What | Worst observed agreement |
|---|---|
| `norm_cdf`, `norm_sf` | 1e-12 relative |
| `norm_ppf` | 1e-8 relative (a far-tail quantile is the least determined quantity here) |
| `t_cdf`, `t_ppf`, df 1–1000 | 1e-12 / 1e-8 |
| `chi2_cdf`, `f_cdf`, `betainc`, `gammainc` | 1e-12 |
| Welch's and Student's t | statistic, p and df to 1e-10 |
| paired t | statistic and p to 1e-10 |
| Mann-Whitney, exact branch | **exact** to 1e-12 against all 252 label assignments |
| Wilcoxon, exact branch | **exact** to 1e-12 against all 2⁸ sign assignments |
| one-way ANOVA, chi-squared, Brown–Forsythe | 1e-10 |
| Cohen's d, Hedges' g, Cliff's delta, Vargha–Delaney A | 1e-12 against their closed forms |
| t intervals at 80/90/95/99% | both endpoints to 1e-10 |
| Holm, Bonferroni, Sidak, Benjamini–Hochberg | 1e-5, checked against their definitions by hand |

Four apparent disagreements turned out to be correct behaviour, and are
recorded because they look like bugs:

- **Anderson–Darling.** Fabric reports `A2*`, the Stephens (1986)
  modification; SciPy reports unmodified `A²`. They differ by
  `1 + 0.75/n + 2.25/n²` — confirmed numerically rather than trusted:
  for the test's sample SciPy's raw value is `0.12059711600456957` and
  Fabric's is `0.1323553348150151`, a ratio of `1.097499`. Comparing the two
  numbers directly would fail and would say nothing about either.
- **Mann-Whitney's statistic.** Fabric reports `min(U₁, U₂)`, which is what
  the exact two-sided tail needs; SciPy reports `U₁`.
- **Wilcoxon above n = 20** takes the normal approximation, as does SciPy.
  Both approximate, and in the far tail the approximation is worth an order
  of magnitude: for the sample in the test, the exact p-value by enumeration
  is `1.335e-05`, Fabric reports `1.515e-04`, and SciPy reports `1.413e-04`.
  Fabric's approximation is further from the truth here. What the test pins
  is that Fabric reports the approximation it says it reports — the error is
  the method's, not a defect in it — and that the exact value is computed in
  the same file so the size of the error is visible rather than asserted.
- **The Wilson interval** is checked against its closed form rather than
  SciPy, because SciPy has no Wilson: `binom.interval` is exact and behaves
  very differently at the boundary.

---

## The family size is the declared one

A comparison whose test could not run is still a comparison somebody
planned. Fabric corrects against the number of hypotheses **declared**, not
the number that produced a p-value, so a test that fails to run cannot
shrink the family and inflate the survivors' significance. Only the ranking
uses the p-values that exist.

This is not a detail. Three declared baselines at p = 0.020 and 0.030, with
the third untestable, are **two** rejections when the family shrinks to two
and **zero** when it does not. See `D-001` in [`../decisions.md`](../decisions.md).

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
