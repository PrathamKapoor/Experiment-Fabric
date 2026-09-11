#!/usr/bin/env bash
# Prove the reproduction container actually reproduces.
#
# Run inside the image, with a writable directory mounted at /out:
#
#   docker run --rm -v "$PWD/out:/out" experiment-fabric bash scripts/container_check.sh
#
# Kept as a file rather than inlined into the CI YAML because a shell
# script nested inside YAML inside a docker invocation is three levels of
# quoting and exactly one of them will be wrong.
set -euo pipefail

OUT="${1:-/out}"
mkdir -p "$OUT"

echo "== the core is still dependency-free =="
python scripts/check_no_dependencies.py

echo "== generate data and run a perturbation study =="
python examples/agent_pipeline/make_data.py
fabric init --quiet
fabric run examples/agent_pipeline/reproducibility.yaml

EXP="$(fabric ls | awk '/agent-reproducibility-profile/ {print $1}')"
if [ -z "$EXP" ]; then
  echo "FAIL: the experiment was not registered"
  exit 1
fi
echo "experiment: $EXP"

echo "== analyse =="
fabric analyze "$EXP" | tail -n 25

echo "== certify and export =="
fabric certify "$EXP" | tail -n 6
rm -rf "$OUT/bundle"
fabric export "$EXP" --out "$OUT/bundle"

echo "== verify =="
fabric verify "$OUT/bundle" | tail -n 4

echo "== reproduce, inside the pinned environment =="
fabric reproduce "$OUT/bundle" | tee "$OUT/reproduction.txt" | tail -n 12
grep -q "PASS" "$OUT/reproduction.txt"

echo
echo "container check passed; bundle written to $OUT/bundle"
