#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 1 ]; then
  echo "Usage: ./run_validation.sh NEW_SCALAR.txt"
  exit 2
fi
python validate_new_data.py \
  --input "$1" \
  --model model/candidate_v8.joblib \
  --output-dir validation_output
