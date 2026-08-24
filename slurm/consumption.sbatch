#!/usr/bin/env bash
#SBATCH --job-name=eventtok-consume
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_ydu_lab
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=0-06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%A_%a.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%A_%a.err

# Does a policy read the event log, and does it care what the log says?
#
# Not rollout success -- there is no simulator on this cluster -- but the offline
# version of the same question: behaviour cloning with and without the memory, plus
# the controls that say whether the memory is being read rather than merely present.
#
# The four tasks are the ones where a memory of what has already happened plausibly
# matters: two counting tasks and two occlusion tasks.
#
#   sbatch slurm/consumption.sbatch

set -uo pipefail

CACHE=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache
PY=/n/lab_storage/ydu_lab/wfy/.conda/envs/robomme/bin/python
REPO=${REPO:-/n/home04/wfy/repos/eventtok_mem}

export MPLCONFIGDIR="/tmp/${USER}/mpl-${SLURM_JOB_ID:-manual}"
export TORCH_HOME="${CACHE}/torch"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
mkdir -p "${CACHE}/slurm" "${CACHE}/eval" "${MPLCONFIGDIR}" "${TORCH_HOME}"
cd "${REPO}"

TASKS=(SwingXtimes VideoRepick PickXtimes ButtonUnmask)
TASK=${TASKS[${SLURM_ARRAY_TASK_ID:-0}]}
EPOCHS=${EPOCHS:-25}

echo "=== $(date -Is) consumption ${TASK} epochs=${EPOCHS} ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true

"${PY}" -m eventtok.scripts.eval_consumption \
    --task "${TASK}" --epochs "${EPOCHS}" \
    --out "${CACHE}/eval/consumption_${TASK}.json"
echo "=== $(date -Is) done ${TASK} ==="
