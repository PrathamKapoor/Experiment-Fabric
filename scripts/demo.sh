#!/usr/bin/env bash
# The whole system, end to end, in about a minute on a laptop.
#
#   scripts/demo.sh
#
# One YAML becomes an experiment graph, thirty runs, a baseline, an
# ablation, statistics, a reproducibility measurement, a provenance chain,
# a certificate and a paper-ready report -- and then the certificate
# catches a single appended byte.
#
# It runs in a TEMPORARY WORKSPACE and touches nothing you own. Set
# FABRIC_DEMO_DIR to keep the output somewhere you choose.
set -euo pipefail

FABRIC="${FABRIC:-python -m fabric}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  BOLD=$(printf '\033[1m'); DIM=$(printf '\033[2m'); OFF=$(printf '\033[0m')
else
  BOLD=""; DIM=""; OFF=""
fi

step() { printf "\n%s==> %s%s\n" "$BOLD" "$1" "$OFF"; }
note() { printf "%s    %s%s\n" "$DIM" "$1" "$OFF"; }

# A temporary workspace, cleaned up on exit unless the caller named one.
if [ -n "${FABRIC_DEMO_DIR:-}" ]; then
  WORK="$FABRIC_DEMO_DIR"
  mkdir -p "$WORK"
  KEEP=1
else
  WORK="$(mktemp -d 2>/dev/null || mktemp -d -t fabric-demo)"
  KEEP=0
fi
cleanup() { [ "$KEEP" = "1" ] || rm -rf "$WORK"; }
trap cleanup EXIT

step "0. A scratch workspace, so nothing you own is touched"
note "$WORK"
# The examples and benchmarks are copied in rather than referenced, so the
# demo has its own data, its own registry and its own results. Relative
# paths inside a specification resolve against the workspace root, which
# is exactly why this is a copy and not a --workspace flag.
cp -r "$REPO/examples" "$WORK/examples"
cp -r "$REPO/benchmarks" "$WORK/benchmarks"

# Under Git Bash / MSYS the shell speaks POSIX paths and the Python it
# launches is a native Windows build that does not. Translate, and use the
# separator that Python expects on this platform.
case "$(uname -s 2>/dev/null || echo unknown)" in
  MINGW*|MSYS*|CYGWIN*) PSEP=";"; native() { cygpath -w "$1"; } ;;
  *)                    PSEP=":"; native() { printf '%s' "$1"; } ;;
esac
# $WORK so the copied examples import; $REPO so a checkout that has not
# been installed still finds fabric itself.
export PYTHONPATH="$(native "$WORK")$PSEP$(native "$REPO")${PYTHONPATH:+$PSEP$PYTHONPATH}"
export FABRIC_BENCHMARK_PATH="$(native "$WORK/benchmarks")"
cd "$WORK"

step "1. Generate the benchmark data"
note "deterministic from the version string, so the digests match anywhere"
python examples/agent_pipeline/make_data.py
python examples/text_classification/make_data.py

$FABRIC init --name experiment-fabric-demo >/dev/null

step "2. The benchmarks these experiments are reported on"
note "metrics, tolerances and baselines are defined once, not per experiment"
$FABRIC benchmarks

FAB="$FABRIC"

step "3. Plan without running"
note "expansion, inputs, benchmark conformance, and the advisories a reviewer would raise"
$FAB plan examples/agent_pipeline/experiment.yaml

step "4. Run the ablation study"
note "1 treatment x 2 configurations + 3 baselines + 7 ablations x 10 seeds = 120 runs"
$FAB run examples/agent_pipeline/experiment.yaml
AGENT=$($FAB ls | awk '/agent-scaffolding-ablation/ {print $1}')

step "5. Run the reproducibility perturbation study"
note "six axes, one thing changed per condition, plus determinism replicates"
$FAB run examples/agent_pipeline/reproducibility.yaml
REPRO=$($FAB ls | awk '/agent-reproducibility-profile/ {print $1}')

step "6. Run the second example: real algorithms, no fabric import"
note "hand-written naive Bayes and logistic regression; model.py never imports fabric"
$FAB run examples/text_classification/experiment.yaml
TEXT=$($FAB ls | awk '/sentiment-classifier-comparison/ {print $1}')

step "7. Analyse the ablation study"
$FAB analyze "$AGENT"

step "8. Analyse the reproducibility profile"
$FAB analyze "$REPRO"

step "9. The text classifier, where the hypothesis does not survive"
note "the proposed method loses to naive Bayes, and the report says so"
$FAB analyze "$TEXT"

step "10. Trace one number back to everything that produced it"
$FAB why "$AGENT" --metric accuracy --seed 3

step "11. What depends on the dataset?"
$FAB impact "$AGENT"

step "12. What is different between two experiments?"
note "one command instead of diffing two YAML files by eye"
$FAB diff "$AGENT" "$TEXT" --material

step "13. The provenance graph, as something you can click"
note "click a node and its lineage lights up; that is fabric why, with a mouse"
$FAB graph "$AGENT" --format html

step "14. Generate the scientific report in every format"
$FAB report "$AGENT" --format all
$FAB report "$REPRO" --format html

step "15. Issue a reproduction certificate"
$FAB certify "$REPRO"

step "16. Export a portable bundle"
rm -rf demo-bundle
$FAB export "$REPRO" --out demo-bundle

step "17. Verify the bundle"
$FAB verify demo-bundle

step "18. Tamper with it, and verify again"
note "one appended byte is enough"
printf '\n' >> "$WORK/demo-bundle/results.json"
$FAB verify demo-bundle || true

step "19. Restore and reproduce it"
rm -rf demo-bundle
$FAB export "$REPRO" --out demo-bundle >/dev/null
$FAB reproduce demo-bundle

step "Done"
note "HTML report:  $WORK/.fabric/experiments/$AGENT/reports/report.html"
note "Graph:        $WORK/.fabric/experiments/$AGENT/reports/graph.html"
note "Bundle:       $WORK/demo-bundle/"
note "Everything above is regenerable: fabric reproduce $WORK/demo-bundle"
if [ "$KEEP" = "0" ]; then
  note ""
  note "This workspace is about to be deleted. Re-run with"
  note "  FABRIC_DEMO_DIR=./demo-output scripts/demo.sh"
  note "to keep the reports."
fi
note ""
note "The service, with the same experiments in a browser:"
note "  fabric serve                 # http://127.0.0.1:8080"
note "  docker compose up --build"
note ""
note "Not shown, because it needs PyTorch and three minutes:"
note "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
note "  fabric run examples/torch_mlp/reproducibility.yaml   # real training"
note "  python paper/run_study.py                            # the pilot study"
