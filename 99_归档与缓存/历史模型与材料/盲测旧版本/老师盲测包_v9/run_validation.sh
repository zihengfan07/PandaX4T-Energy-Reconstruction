#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 /path/to/new_run_scalar.txt" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
rm -rf "${SCRIPT_DIR}/validation_output"

PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/apply_grouped_physics_v9.py" \
  --model "${SCRIPT_DIR}/model/candidate_v9.joblib" \
  --input "$1" \
  --output "${SCRIPT_DIR}/validation_output"

echo "Validation finished: ${SCRIPT_DIR}/validation_output"
