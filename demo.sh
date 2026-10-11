#!/usr/bin/env bash
# The demo lives in scripts/demo.sh, next to the other scripts.
#
# This wrapper stays because the README told people to run ./demo.sh, and
# a documented command that stops working is its own small reproducibility
# failure.
set -euo pipefail
exec "$(dirname "${BASH_SOURCE[0]}")/scripts/demo.sh" "$@"
