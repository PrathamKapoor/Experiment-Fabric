# Paper outline

Each section names the artifact that fills it. Nothing is written by hand
that a bundle can generate.

| Section | Content | Generated from |
|---|---|---|
| 1. Introduction | Results are reported; the conditions that produced them are not preserved. The gap, and what it costs. | prose |
| 2. Related work | Experiment trackers, workflow provenance (W3C PROV), reproducibility studies, statistical practice in ML. | prose |
| 3. The problem | "Reproducible" is not falsifiable as usually stated. What a falsifiable version requires. | prose |
| 4. Method | Experiment graphs; the perturbation design; calibrated stability; the seed axis as yardstick. | `docs/reproducibility.md` |
| 5. Instrument | Fabric: expansion, execution, provenance, statistics, integrity. Enough detail to reproduce the instrument itself. | `README.md`, `docs/` |
| 6. Study design | Sample, strata, pre-registration. | `paper/protocol.md` |
| 7. Results | Per-axis stability across the sample; how often non-seed axes exceed the seed floor; determinism failures; decision stability. | `fabric analyze --json` over the bundles |
| 8. Case studies | Two or three experiments in depth, including at least one where a published-style conclusion does not survive a perturbation. | individual bundles |
| 9. Discussion | What a paper should report. What the field currently under-reports, and by how much. | prose |
| 10. Threats to validity | Sample bias, main effects only, partial environment perturbation, estimator variance. | `paper/README.md` |
| 11. Conclusion | Reproducibility is a measurable property with an experimental design attached, not a badge. | prose |

## Figures

| Figure | Content | Produced by |
|---|---|---|
| 1 | The experiment graph for one experiment, metric back to inputs. | `fabric graph <EXP> --format dot` |
| 2 | Per-axis stability across the sample, ordered by fragility. | stability chart over all bundles |
| 3 | Shift in seed-noise units, per axis, as a distribution. | analysis JSON |
| 4 | Decision stability against seed count. | seed subsampling |
| 5 | An ablation matrix with interaction analysis, as an instrument demonstration. | `fabric report` |

## Tables

Every results table in the paper is `\input{}` from the LaTeX snippets
that `fabric report --format latex` regenerates, so the paper cannot drift
out of step with the evidence.

## Artifact appendix

- the specifications, one file per experiment
- one certified bundle per experiment, verifiable offline
- the instrument at the commit named in every certificate
- `fabric reproduce <bundle>` as the single reproduction instruction
