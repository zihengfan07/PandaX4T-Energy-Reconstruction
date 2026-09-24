#!/bin/bash
#SBATCH --job-name=eres_v11
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=06:00:00
#SBATCH --output=/home/researcher/eres_sigma_search/logs/v11_%j.out
#SBATCH --error=/home/researcher/eres_sigma_search/logs/v11_%j.err

set -euo pipefail
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd /home/researcher/eres_sigma_search
mkdir -p logs
rm -rf results/run10972_foundation_environment_v11.tmp
mkdir -p results/run10972_foundation_environment_v11.tmp

python3 foundation_environment_v11.py \
  --input /home/researcher/light_ana_run10972_finalSS_Egt2MeV_scalar.txt \
  --output results/run10972_foundation_environment_v11.tmp

rm -rf results/run10972_foundation_environment_v11
mv results/run10972_foundation_environment_v11.tmp \
  results/run10972_foundation_environment_v11
