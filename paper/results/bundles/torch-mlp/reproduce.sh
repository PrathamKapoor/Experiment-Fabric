#!/usr/bin/env bash
# Regenerate this experiment from its specification.
#
# The specification is the source of truth: it names the dataset version, the
# model version, the source commit and the seeds. Anything this script cannot
# fetch for you is listed in manifest.json.
set -euo pipefail

BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Experiment:    torch-mlp-reproducibility"
echo "Identifier:    EXP-24d9c76f72"
echo "Specification: $BUNDLE/experiment.yaml"
echo

# 1. Recreate the workspace (safe to run in an existing one).
fabric init --quiet || true

# 2. Re-execute every run declared by the specification.
fabric run "$BUNDLE/experiment.yaml"

# 3. Compare what you got against what was originally recorded.
fabric reproduce "$BUNDLE"

# 4. Regenerate the report from your own results.
fabric report EXP-24d9c76f72 --format html
