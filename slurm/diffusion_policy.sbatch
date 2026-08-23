#!/usr/bin/env bash
#SBATCH --job-name=eventtok-dp
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_ydu_lab
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=0-08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%A_%a.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%A_%a.err

# Diffusion Policy with the event log as memory. Eight configurations per task:
# {none, log, wrong, count} x {crossattn, global}.
#
# The second axis is the one worth the GPU time. The research plan asserts that
# pooling the memory into a global_cond destroys the count and reproduces the methods
# this project argues against -- an assertion never tested. crossattn keeps the log a
# sequence; global pools it the way standard Diffusion Policy does. Same data, same
# split, same everything else.
#
# Same four tasks as the behaviour-cloning probe so the two are comparable.

set -uo pipefail

CACHE=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache
PY=/n/lab_storage/ydu_lab/wfy/.conda/envs/robomme/bin/python
REPO=${REPO:-/n/home04/wfy/repos/eventtok_mem}

export MPLCONFIGDIR="/tmp/${USER}/mpl-${SLURM_JOB_ID:-manual}"
export TORCH_HOME="${CACHE}/torch"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
mkdir -p "${CACHE}/slurm" "${CACHE}/eval" "${MPLCONFIGDIR}" "${TORCH_HOME}"
cd "${REPO}"

TASKS=(PickXtimes SwingXtimes ButtonUnmask VideoRepick)
TASK=${TASKS[${SLURM_ARRAY_TASK_ID:-0}]}
EPOCHS=${EPOCHS:-30}

echo "=== $(date -Is) diffusion policy ${TASK} epochs=${EPOCHS} ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true

"${PY}" -m eventtok.scripts.train_diffusion_policy \
    --task "${TASK}" --epochs "${EPOCHS}" \
    --out "${CACHE}/eval/dp_${TASK}.json"
echo "=== $(date -Is) done ${TASK} ==="
