#!/usr/bin/env bash
#SBATCH --job-name=pi05evt
#SBATCH --partition=shared
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%A_%a.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%A_%a.err

# Per-frame causal event logs for the pi0.5 integration, one cache per task.
#
# CPU partition on purpose: this is k-means + BPE, no GPU. It needs the memory though --
# SwingXtimes (43k frames) is OOM-killed on a login node.
#
# Idempotent: a task whose cache exists is skipped, so a requeue after preemption
# resumes at task granularity rather than restarting the array.

set -uo pipefail

CACHE=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache
PY=/n/lab_storage/ydu_lab/wfy/.conda/envs/robomme/bin/python
REPO=${REPO:-/n/home04/wfy/repos/eventtok_mem}
TAG=${TAG:-default}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd "${REPO}"

TASKS=(BinFill ButtonUnmask ButtonUnmaskSwap InsertPeg MoveCube PatternLock \
       PickHighlight PickXtimes RouteStick StopCube SwingXtimes VideoPlaceButton \
       VideoPlaceOrder VideoRepick VideoUnmask VideoUnmaskSwap)
TASK=${TASKS[${SLURM_ARRAY_TASK_ID:-0}]}

OUT="${CACHE}/pi05_events/${TASK}_${TAG}.npz"
if [ -f "${OUT}" ]; then
    echo "=== ${TASK}: cache exists, skipping ==="
    exit 0
fi

echo "=== $(date -Is) events ${TASK} ==="
"${PY}" -m eventtok.scripts.build_pi05_events --task "${TASK}" --tag "${TAG}" "$@"
status=$?
echo "=== $(date -Is) done ${TASK} exit=${status} ==="
exit ${status}
