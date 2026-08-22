#!/usr/bin/env bash
#SBATCH --job-name=eventtok-noise
#SBATCH --partition=shared
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%A_%a.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%A_%a.err
set -uo pipefail
CACHE=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache
PY=/n/lab_storage/ydu_lab/wfy/.conda/envs/robomme/bin/python
cd "${REPO:-/n/home04/wfy/repos/eventtok_mem}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
mkdir -p "${CACHE}/slurm" "${CACHE}/eval"
TASKS=(ButtonUnmask ButtonUnmask SwingXtimes SwingXtimes)
KS=(16 64 16 64)
I=${SLURM_ARRAY_TASK_ID:-0}
echo "=== $(date -Is) task=${TASKS[$I]} k=${KS[$I]} ==="
"${PY}" -m eventtok.scripts.noise_floor --task "${TASKS[$I]}" --k "${KS[$I]}" --seeds 10 \
  --out "${CACHE}/eval/noise_${TASKS[$I]}_k${KS[$I]}.json"
echo "=== $(date -Is) done ==="
