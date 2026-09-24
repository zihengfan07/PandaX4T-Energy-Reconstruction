#!/bin/bash
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: ./run.sh /full/path/input_scalar.txt [output_directory]"
  exit 2
fi
OUT="${2:-$PWD/v13_output}"
mkdir -p "$OUT"
/usr/bin/python3 "$HERE/apply_v13.py" \
  --model "$HERE/model/v13_portable.json" \
  --input "$1" \
  --output "$OUT"
echo "Output: $OUT/energy_before_after.txt"
