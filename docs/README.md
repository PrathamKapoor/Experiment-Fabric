# Documentation

| Document | What it covers |
|---|---|
| [specification.md](specification.md) | Every key of an `experiment.yaml`, what it defaults to, and what Fabric will warn you about before spending compute. |
| [writing-experiments.md](writing-experiments.md) | The contract your code has to satisfy, the ablation guard pattern, running non-Python methods, and keeping Fabric out of your codebase. |
| [statistics.md](statistics.md) | The test-selection procedure, recorded assumptions, effect sizes, equivalence testing, multiplicity, minimum detectable effect, and what is deliberately absent. |
| [reproducibility.md](reproducibility.md) | What "reproducible" is defined to mean here, how the stability score is calibrated, why the seed axis is a yardstick rather than a score, and seven named limitations. |
| [related-work.md](related-work.md) | What already exists, checked against primary sources, and what of this is genuinely new versus better-packaged. Includes corrections to earlier claims. |
| [architecture.md](architecture.md) | Layers, dependency direction, identity rules, on-disk layout, concurrency, and extension points. |

Start with the [README](../README.md) for the five-minute version, or run
`./demo.sh` to see the whole system work end to end.
