# Reproduction bundle: agent-reproducibility-profile

`EXP-832b6cabc2`

## What this is

A complete, self-describing record of one computational experiment: the
specification that defined it, every run it produced, the statistical analysis,
the provenance graph connecting each number to what produced it, and a
certificate of what was verified.

Everything here is plain text. You can read it without installing anything.

## The claim

> How far does the accuracy of the retrieval agent move under controlled perturbations of the software environment, the accelerator, the benchmark version, the model version, the implementation, and the temperature - relative to how far it moves when only the random seed changes?

full-agent scored 0.7478 +/- 0.0214 on accuracy over 10 seeds, beating every baseline; the narrowest margin is +0.2200 over rag-classic [0.2032, 0.2368] (paired-t, adjusted p=0.0000).

## How to reproduce it

With [Experiment Fabric](https://github.com/) installed:

```bash
fabric reproduce agent-pipeline
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
fabric verify agent-pipeline
```

Every file is hashed in `manifest.json`. The certificate is content-addressed:
an edited certificate no longer matches its own identifier.

## What was measured

- 93 runs across 9 conditions and 10 seeds
- Deterministic: 4 executions of an identical run produced identical values on every result metric (accuracy, robustness). latency_ms varied, as expected: those measure the machine rather than the method.
- Across 7 perturbation axes, the most fragile is 'configuration' (stability 0.05); the geometric mean is 0.51. Changing configuration, dataset, model moves accuracy beyond the tolerance declared for it, so a reproduction differing on those axes should not be expected to match. On implementation the data cannot decide whether the shift stays within tolerance; more runs would be needed to say.

## Limitations

- With 10 replications and the observed spread, this design could detect a difference of 0.0233 in accuracy or larger 80% of the time at alpha=0.05.
- Changing the configuration moves accuracy by -0.0632, beyond the declared tolerance of 0.01. Results should not be expected to reproduce across configuration changes.
- Changing the dataset moves accuracy by +0.0302, beyond the declared tolerance of 0.01. Results should not be expected to reproduce across dataset changes.
- Changing the model moves accuracy by +0.0252, beyond the declared tolerance of 0.01. Results should not be expected to reproduce across model changes.
- Whether a change of implementation keeps accuracy within its declared tolerance of 0.01 is undecided: the interval [-0.0071, 0.0126] straddles that boundary. More runs on that axis would resolve it.

Generated 2026-09-11 by Experiment Fabric 1.0.
