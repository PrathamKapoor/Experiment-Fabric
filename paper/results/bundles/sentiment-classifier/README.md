# Reproduction bundle: sentiment-reproducibility

`EXP-f694df8930`

## What this is

A complete, self-describing record of one computational experiment: the
specification that defined it, every run it produced, the statistical analysis,
the provenance graph connecting each number to what produced it, and a
certificate of what was verified.

Everything here is plain text. You can read it without installing anything.

## The claim

> How far does a pure-Python text classifier move under controlled perturbations of the corpus version, the feature space, the learning rate and the software environment - relative to how far it moves when only the random seed changes?

logistic-regression scored 0.6490 +/- 0.0134 on accuracy over 10 seeds, which is not distinguishable from any of the 1 baselines at this sample size.

## How to reproduce it

With [Experiment Fabric](https://github.com/) installed:

```bash
fabric reproduce sentiment-classifier
```

That re-executes the specification in your environment and compares the result
against `results.json`, reporting the difference with a confidence interval, an
equivalence test against the tolerance declared in the specification, and a list
of the environment differences that plausibly explain any deviation.

Without Fabric, `reproduce.sh` lists the commands, and `manifest.json` records
the exact dataset digest, model version, source commit and package versions the
original runs used.

## Verifying this bundle has not changed

```bash
fabric verify sentiment-classifier
```

Every file is hashed in `manifest.json`. The certificate is content-addressed:
an edited certificate no longer matches its own identifier.

## What was measured

- 83 runs across 8 conditions and 10 seeds
- Deterministic: 4 executions of an identical run produced identical values on every result metric (accuracy, f1, vocabulary_size). train_seconds varied, as expected: those measure the machine rather than the method.
- Across 6 perturbation axes, the most fragile is 'dataset' (stability 0.00); the geometric mean is 0.02. Changing configuration, dataset moves accuracy beyond the tolerance declared for it, so a reproduction differing on those axes should not be expected to match. On implementation the data cannot decide whether the shift stays within tolerance; more runs would be needed to say.

## Limitations

- With 10 replications and the observed spread, this design could detect a difference of 0.0146 in accuracy or larger 80% of the time at alpha=0.05. The difference actually measured (-0.0102) is smaller than that floor, so a null result here is a statement about the experiment's resolution, not about the methods.
- Changing the configuration moves accuracy by -0.0215, beyond the declared tolerance of 0.01. Results should not be expected to reproduce across configuration changes.
- Changing the dataset moves accuracy by -0.1010, beyond the declared tolerance of 0.01. Results should not be expected to reproduce across dataset changes.
- Whether a change of implementation keeps accuracy within its declared tolerance of 0.01 is undecided: the interval [-0.0002, 0.0207] straddles that boundary. More runs on that axis would resolve it.

Generated 2026-09-11 by Experiment Fabric 1.0.
