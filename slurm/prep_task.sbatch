#!/usr/bin/env bash
#SBATCH --job-name=eventtok-prep
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%j.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%j.err

# One-time data prep for a task: repack per-frame visual features into
# per-episode arrays, and scan the pkls once for actions + subgoal labels.
#
# CPU only and I/O bound -- 32 GB of feature npys and ~21 GB of pkls for
# PickXtimes. No GPU: nothing here touches a model.
#
# Both steps are idempotent and skip existing outputs, so a requeue or a rerun
# after partial completion is safe. That is why this does not need --requeue.
#
#   TASK=PickXtimes sbatch slurm/prep_task.sbatch

set -uo pipefail

CACHE=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache
PY=/n/lab_storage/ydu_lab/wfy/.conda/envs/robomme/bin/python
REPO=${REPO:-/n/home04/wfy/repos/eventtok_mem}
TASK=${TASK:-PickXtimes}
SCALE=${SCALE:-2x2}

mkdir -p "${CACHE}/slurm"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
cd "${REPO}"

echo "=== $(date -Is) prep ${TASK} scale=${SCALE} ==="

echo "[1/2] repacking visual features"
"${PY}" -m eventtok.scripts.repack_task \
    --task "${TASK}" --scale "${SCALE}" \
    --workers "${SLURM_CPUS_PER_TASK:-16}"

echo "[2/2] scanning pkls for actions and subgoal labels"
"${PY}" -m eventtok.scripts.build_meta \
    --task "${TASK}" --workers "${SLURM_CPUS_PER_TASK:-16}"

echo "=== $(date -Is) done ==="
