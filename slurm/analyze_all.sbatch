#!/usr/bin/env bash
#SBATCH --job-name=eventtok-anal
#SBATCH --partition=shared
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%A_%a.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%A_%a.err

# The same pipeline, the same split and the same fixed hyperparameters on all 16
# RoboMME tasks. Fixed settings on purpose: this is a cross-task comparison, so a
# per-task hyperparameter search would confound "where does the method work" with
# "where did the search get lucky".
#
# 128G because VideoPlaceOrder is 112k frames and the vision block is 2 x 4 x 2048
# per frame in fp32 before PCA.

set -uo pipefail

CACHE=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache
PY=/n/lab_storage/ydu_lab/wfy/.conda/envs/robomme/bin/python
REPO=${REPO:-/n/home04/wfy/repos/eventtok_mem}
K=${K:-16}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
mkdir -p "${CACHE}/slurm" "${CACHE}/eval"
cd "${REPO}"

TASKS=(BinFill ButtonUnmask ButtonUnmaskSwap InsertPeg MoveCube PatternLock \
       PickHighlight PickXtimes RouteStick StopCube SwingXtimes VideoPlaceButton \
       VideoPlaceOrder VideoRepick VideoUnmask VideoUnmaskSwap)
TASK=${TASKS[${SLURM_ARRAY_TASK_ID:-0}]}

echo "=== $(date -Is) analyze ${TASK} k=${K} ==="
"${PY}" -m eventtok.scripts.analyze_task --task "${TASK}" --k "${K}" \
    --out "${CACHE}/eval/alltasks_k${K}_${TASK}.json"
echo "=== $(date -Is) done ${TASK} ==="
