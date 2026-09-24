#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 /path/to/new_run_scalar.txt" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
rm -rf "${SCRIPT_DIR}/validation_output"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ -d "${SCRIPT_DIR}/python_lib" ]; then
  export PYTHONPATH="${SCRIPT_DIR}/python_lib${PYTHONPATH:+:${PYTHONPATH}}"
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/apply_foundation_environment_v11.py" \
  --model "${SCRIPT_DIR}/model/candidate_v11.joblib" \
  --input "$1" \
  --output "${SCRIPT_DIR}/validation_output"

echo "Validation finished: ${SCRIPT_DIR}/validation_output"
