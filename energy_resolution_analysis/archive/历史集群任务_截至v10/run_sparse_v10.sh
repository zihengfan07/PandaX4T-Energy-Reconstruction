#!/bin/bash
set -euo pipefail

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd /home/researcher/eres_sigma_search
rm -rf results/run10972_sparse_grouped_v10.tmp
mkdir -p results/run10972_sparse_grouped_v10.tmp

python3 sparse_grouped_physics_v10.py \
  --input /home/researcher/light_ana_run10972_finalSS_Egt2MeV_scalar.txt \
  --output results/run10972_sparse_grouped_v10.tmp

rm -rf results/run10972_sparse_grouped_v10
mv results/run10972_sparse_grouped_v10.tmp \
  results/run10972_sparse_grouped_v10
