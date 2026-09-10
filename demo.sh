#!/usr/bin/env bash
# The whole system, end to end, in about a minute on a laptop.
#
#   ./demo.sh
#
# Runs three experiments, analyses them, traces a number back to its
# source, generates a report in four formats, issues a certificate,
# exports a portable bundle, verifies it, tampers with it to show the
# verification working, and reproduces it.
set -euo pipefail

FABRIC="${FABRIC:-python -m fabric}"
BOLD=$(printf '\033[1m'); DIM=$(printf '\033[2m'); OFF=$(printf '\033[0m')

step() { printf "\n%s==> %s%s\n" "$BOLD" "$1" "$OFF"; }
note() { printf "%s    %s%s\n" "$DIM" "$1" "$OFF"; }

cd "$(dirname "${BASH_SOURCE[0]}")"

step "1. Generate the benchmark data"
note "deterministic from the version string, so the digests match anywhere"
python examples/agent_pipeline/make_data.py
python examples/text_classification/make_data.py

step "2. Create a workspace"
$FABRIC init --force --name experiment-fabric-demo

step "3. Plan without running"
note "expansion, inputs, and the advisories a reviewer would raise"
$FABRIC plan examples/agent_pipeline/experiment.yaml

step "4. Run the ablation study"
note "1 treatment x 2 configurations + 3 baselines + 7 ablations x 10 seeds = 120 runs"
$FABRIC run examples/agent_pipeline/experiment.yaml
AGENT=$($FABRIC ls | awk '/agent-scaffolding-ablation/ {print $1}')

step "5. Run the reproducibility perturbation study"
note "six axes, one thing changed per condition, plus determinism replicates"
$FABRIC run examples/agent_pipeline/reproducibility.yaml
REPRO=$($FABRIC ls | awk '/agent-reproducibility-profile/ {print $1}')

step "6. Run the second example: real algorithms, no fabric import"
$FABRIC run examples/text_classification/experiment.yaml
TEXT=$($FABRIC ls | awk '/sentiment-classifier-comparison/ {print $1}')

step "7. Analyse the ablation study"
$FABRIC analyze "$AGENT"

step "8. Analyse the reproducibility profile"
$FABRIC analyze "$REPRO"

step "9. The text classifier, where the hypothesis does not survive"
note "the proposed method loses to naive Bayes, and the report says so"
$FABRIC analyze "$TEXT"

step "10. Trace one number back to everything that produced it"
$FABRIC why "$AGENT" --metric accuracy --seed 3

step "11. What depends on the dataset?"
$FABRIC impact "$AGENT"

step "12. Generate the scientific report in every format"
$FABRIC report "$AGENT" --format all
$FABRIC report "$REPRO" --format html

step "13. Issue a reproduction certificate"
$FABRIC certify "$REPRO"

step "14. Export a portable bundle"
rm -rf demo-bundle
$FABRIC export "$REPRO" --out demo-bundle

step "15. Verify the bundle"
$FABRIC verify demo-bundle

step "16. Tamper with it, and verify again"
note "one appended byte is enough"
printf '\n' >> demo-bundle/results.json
$FABRIC verify demo-bundle || true

step "17. Restore and reproduce it"
rm -rf demo-bundle
$FABRIC export "$REPRO" --out demo-bundle >/dev/null
$FABRIC reproduce demo-bundle

step "Done"
note "HTML report: .fabric/experiments/$AGENT/reports/report.html"
note "Bundle:      demo-bundle/"
note "Everything above is regenerable: fabric reproduce demo-bundle"
