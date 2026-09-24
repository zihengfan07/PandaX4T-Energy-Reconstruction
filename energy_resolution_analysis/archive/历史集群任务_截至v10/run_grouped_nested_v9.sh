#!/bin/bash
set -euo pipefail

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd /home/researcher/eres_sigma_search
rm -rf results/run10972_grouped_nested_v9_soft.tmp
mkdir -p results/run10972_grouped_nested_v9_soft.tmp

python3 grouped_nested_physics_v9.py \
  --input /home/researcher/light_ana_run10972_finalSS_Egt2MeV_scalar.txt \
  --output results/run10972_grouped_nested_v9_soft.tmp \
  --bootstrap 1000

rm -rf results/run10972_grouped_nested_v9_soft
mv results/run10972_grouped_nested_v9_soft.tmp \
  results/run10972_grouped_nested_v9_soft
