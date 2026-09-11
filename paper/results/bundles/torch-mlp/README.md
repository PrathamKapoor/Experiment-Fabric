# Reproduction bundle: torch-mlp-reproducibility

`EXP-24d9c76f72`

## What this is

A complete, self-describing record of one computational experiment: the
specification that defined it, every run it produced, the statistical analysis,
the provenance graph connecting each number to what produced it, and a
certificate of what was verified.

Everything here is plain text. You can read it without installing anything.

## The claim

> How far does a real PyTorch training run move under controlled perturbations of the thread count, the loss implementation, the model width, the dataset version and the batch size - relative to how far it moves when only the random seed changes?

mlp scored 0.1639 +/- 0.0023 on test_loss over 10 seeds, beating every baseline; the narrowest margin is +0.5056 over logistic-regression [0.5038, 0.5075] (paired-t, adjusted p=0.0000).

## How to reproduce it

With [Experiment Fabric](https://github.com/) installed:

```bash
fabric reproduce torch-mlp
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
fabric verify torch-mlp
```

Every file is hashed in `manifest.json`. The certificate is content-addressed:
an edited certificate no longer matches its own identifier.

## What was measured

- 83 runs across 8 conditions and 10 seeds
- Deterministic: 4 executions of an identical run produced identical values on every result metric (accuracy, test_loss, threads, train_loss). train_seconds varied, as expected: those measure the machine rather than the method.
- Across 6 perturbation axes, the most fragile is 'dataset' (stability 0.00); the geometric mean is 0.03. Changing dataset moves test_loss beyond the tolerance declared for it, so a reproduction differing on that axis should not be expected to match.

## Limitations

- With 10 replications and the observed spread, this design could detect a difference of 0.0025 in test_loss or larger 80% of the time at alpha=0.05.
- Changing the dataset moves test_loss by +0.1884, beyond the declared tolerance of 0.02. Results should not be expected to reproduce across dataset changes.

Generated 2026-09-11 by Experiment Fabric 1.0.
