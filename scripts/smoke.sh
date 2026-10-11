#!/usr/bin/env bash
# Every command the README claims, executed, with its output asserted.
#
# Documentation that has never been run is a promise, not a fact. This runs
# in CI on Linux and Windows so the promises stay facts.
set -euo pipefail

WORK="${1:-$(mktemp -d)}"
mkdir -p "$WORK"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Use the installed console script when there is one (CI), and the source
# tree otherwise, so this file is runnable both ways.
if ! command -v fabric >/dev/null 2>&1; then
  fabric() { python -m fabric "$@"; }
fi

pass() { printf '  ok   %s\n' "$1"; }
fail() { printf '  FAIL %s\n' "$1"; exit 1; }

expect() {  # expect <description> <needle> <file>
  grep -qF -- "$2" "$3" || { printf '\n--- output ---\n'; cat "$3"; fail "$1"; }
  pass "$1"
}

echo "smoke test in $WORK"
python examples/agent_pipeline/make_data.py >/dev/null
python examples/text_classification/make_data.py >/dev/null

cd "$WORK"
cp -r "$ROOT/examples" .
# The shipped examples declare `benchmark: <name>@<version>`, and the
# definitions those names resolve against live in benchmarks/. Without this
# the first real command fails with "no benchmark named 'agent-qa@v1' was
# found" and the smoke test never reaches a single assertion.
cp -r "$ROOT/benchmarks" .
export PYTHONPATH="$ROOT"
export FABRIC_NO_COLOR=1
export FABRIC_BENCHMARK_PATH="$WORK/benchmarks"

fabric init --quiet
pass "fabric init"

fabric plan examples/agent_pipeline/experiment.yaml > plan.txt
expect "fabric plan expands the specification" "conditions" plan.txt
expect "fabric plan runs nothing" "nothing was executed" plan.txt

fabric run examples/agent_pipeline/experiment.yaml > run.txt
expect "fabric run executes every run" "120 completed" run.txt

EXP=$(fabric ls | awk '/agent-scaffolding-ablation/ {print $1}')
[ -n "$EXP" ] || fail "fabric ls did not list the experiment"
pass "fabric ls"

fabric run examples/agent_pipeline/experiment.yaml > rerun.txt
expect "fabric run is idempotent" "already complete" rerun.txt

fabric status "$EXP" > status.txt
expect "fabric status reports progress" "complete" status.txt

fabric analyze "$EXP" > analyze.txt
expect "analysis compares baselines" "baseline comparisons" analyze.txt
expect "analysis ranks ablations" "ablations" analyze.txt
expect "analysis reports limitations" "limitations" analyze.txt
expect "analysis names the load-bearing component" "retrieval" analyze.txt

fabric analyze "$EXP" --json > analysis.json
python -c "
import json
data = json.load(open('analysis.json'))
assert data['primary_metric'] == 'accuracy', data['primary_metric']
assert data['baselines']['comparisons'], 'no comparisons'
assert data['claims'], 'no claims'
assert data['limitations'], 'no limitations'
"
pass "fabric analyze --json is machine-readable"

fabric why "$EXP" --metric accuracy --seed 3 > why.txt
for needle in run condition algorithm dataset environment hardware; do
  expect "provenance includes $needle" "$needle" why.txt
done

fabric impact "$EXP" > impact.txt
expect "fabric impact reports dependents" "depends on it" impact.txt

for format in mermaid dot json; do
  fabric graph "$EXP" --format "$format" > "graph.$format"
  [ -s "graph.$format" ] || fail "fabric graph --format $format produced nothing"
done
expect "mermaid export" "flowchart" graph.mermaid
expect "dot export" "digraph" graph.dot

fabric report "$EXP" --format all > report.txt
for artefact in report.md report.html report.tex report.json; do
  [ -s ".fabric/experiments/$EXP/reports/$artefact" ] || fail "missing $artefact"
done
pass "fabric report produces all four formats"

python -c "
import xml.etree.ElementTree as ET, pathlib, re
html = pathlib.Path('.fabric/experiments/$EXP/reports/report.html').read_text(encoding='utf-8')
assert html.startswith('<!doctype html>')
charts = re.findall(r'<svg.*?</svg>', html, re.S)
assert charts, 'the report contains no charts'
for chart in charts:
    ET.fromstring(chart)
print(f'  {len(charts)} charts, all well-formed')
"
pass "the HTML report is self-contained and its charts parse"

fabric certify "$EXP" > certificate.txt
expect "certificate is issued" "EXPERIMENT CERTIFICATE" certificate.txt
expect "certificate has a verdict" "VERDICT" certificate.txt

fabric export "$EXP" --out bundle > export.txt
for artefact in manifest.json results.json analysis.json report.md reproduce.sh README.md; do
  [ -s "bundle/$artefact" ] || fail "the bundle is missing $artefact"
done
pass "fabric export writes a complete bundle"

fabric verify bundle > verify.txt
expect "an untouched bundle verifies" "INTACT" verify.txt

printf '\n' >> bundle/results.json
if fabric verify bundle > tampered.txt; then
  fail "verification passed on a tampered bundle"
fi
expect "a tampered bundle is detected" "MODIFIED" tampered.txt
pass "tampering is detected"

rm -rf bundle
fabric export "$EXP" --out bundle >/dev/null
fabric reproduce bundle > reproduce.txt
expect "reproduction completes" "REPRODUCTION COMPLETE" reproduce.txt
expect "reproduction of identical runs passes" "PASS" reproduce.txt

fabric registry verify > registry.txt
expect "registry entities verify" "hashes to its own identifier" registry.txt

fabric cache stats > cache.txt
expect "cache statistics" "entries" cache.txt

fabric doctor > doctor.txt
expect "doctor reports the environment" "fingerprint" doctor.txt

echo
echo "all smoke checks passed"
