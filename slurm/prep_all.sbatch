#!/usr/bin/env bash
#SBATCH --job-name=eventtok-prepall
#SBATCH --partition=shared
#SBATCH --array=0-12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%A_%a.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%A_%a.err

# Prep the 13 RoboMME tasks not already cached, so the tokenizer can be evaluated on
# all 16 rather than the 2-3 it was developed on. Every decision in this project that
# had to be undone came from measuring one task and generalising.
#
# PickXtimes prepped in 90 seconds, so the whole array is minutes of wall clock.
# Both steps are idempotent and skip what exists.

set -uo pipefail

CACHE=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache
PY=/n/lab_storage/ydu_lab/wfy/.conda/envs/robomme/bin/python
REPO=${REPO:-/n/home04/wfy/repos/eventtok_mem}
SCALE=${SCALE:-2x2}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
mkdir -p "${CACHE}/slurm"
cd "${REPO}"

# SwingXtimes, ButtonUnmask and PickXtimes are already cached and omitted.
TASKS=(BinFill ButtonUnmaskSwap InsertPeg MoveCube PatternLock PickHighlight \
       RouteStick StopCube VideoPlaceButton VideoPlaceOrder VideoRepick \
       VideoUnmask VideoUnmaskSwap)
TASK=${TASKS[${SLURM_ARRAY_TASK_ID:-0}]}

echo "=== $(date -Is) prep ${TASK} scale=${SCALE} ==="
"${PY}" -m eventtok.scripts.repack_task --task "${TASK}" --scale "${SCALE}" \
    --workers "${SLURM_CPUS_PER_TASK:-16}" | tail -3
"${PY}" -m eventtok.scripts.build_meta --task "${TASK}" \
    --workers "${SLURM_CPUS_PER_TASK:-16}" | tail -3
echo "=== $(date -Is) done ${TASK} ==="
