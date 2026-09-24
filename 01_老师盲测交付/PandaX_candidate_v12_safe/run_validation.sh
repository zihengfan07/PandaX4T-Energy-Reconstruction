#!/bin/bash
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ "$#" -ne 1 ]; then
  echo "Usage: ./run_validation.sh /path/to/input_scalar.txt"
  exit 2
fi
rm -rf "$HERE/validation_output"
mkdir -p "$HERE/validation_output"
/usr/bin/python3 "$HERE/apply_v12_safe.py" \
  --model "$HERE/model/candidate_v12_safe.json" \
  --input "$1" \
  --output "$HERE/validation_output"
echo "Result: $HERE/validation_output/validation_events.txt"
