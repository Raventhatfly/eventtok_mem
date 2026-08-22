#!/usr/bin/env bash
#SBATCH --job-name=eventtok-enck
#SBATCH --partition=shared
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%A_%a.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%A_%a.err

# Is the encoder ordering a property of the encoders, or of k=32?
#
# Every number in this project that had to be retracted came from one setting read
# as a general result. The encoder comparison is only worth reporting if the
# ordering holds across the cluster count, so the whole grid runs and the whole
# grid gets reported -- including cells where DINOv2 loses.
#
#   sbatch slurm/sweep_encoder_k.sbatch
#   then: python -m eventtok.scripts.report_encoder_sweep

set -uo pipefail

CACHE=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache
PY=/n/lab_storage/ydu_lab/wfy/.conda/envs/robomme/bin/python
REPO=${REPO:-/n/home04/wfy/repos/eventtok_mem}

export MPLCONFIGDIR="/tmp/${USER}/mpl-${SLURM_JOB_ID:-manual}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
mkdir -p "${CACHE}/slurm" "${CACHE}/eval" "${MPLCONFIGDIR}"
cd "${REPO}"

TASKS=(SwingXtimes ButtonUnmask)
KS=(16 32 64 128)

I=${SLURM_ARRAY_TASK_ID:-0}
TASK=${TASKS[$(( I / ${#KS[@]} ))]}
K=${KS[$(( I % ${#KS[@]} ))]}

echo "=== $(date -Is) cell ${I}: task=${TASK} k=${K} ==="
"${PY}" -m eventtok.scripts.compare_modalities \
    --task "${TASK}" --encoders siglip dinov2l --cameras image --k "${K}" \
    --out "${CACHE}/eval/enck_${TASK}_k${K}.json"
echo "=== $(date -Is) done ==="
