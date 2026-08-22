#!/usr/bin/env bash
#SBATCH --job-name=eventtok-wrk
#SBATCH --partition=shared
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%A_%a.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%A_%a.err

# Is the wrist camera worth more than the encoder choice? Swept over k, because the
# encoder comparison's headline turned out to be a k=32 artifact and there is no
# reason to expect the camera comparison to be better behaved at one setting.
#
# All four vision sources, so both matched pairs are readable inside each cell:
#   siglip base vs dinov2l base       -> encoder, camera held fixed
#   siglip base vs siglip wrist       -> camera, encoder held fixed
#
# Needs scripts/encode_siglip_wrist.py to have completed for both tasks; submit with
# --dependency=afterok on those jobs.
#
#   sbatch --dependency=afterok:<sgw1>:<sgw2> slurm/sweep_wrist_k.sbatch
#   then: python -m eventtok.scripts.report_encoder_sweep \
#           --pattern '<cache>/eval/wrk_*_k*.json'

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

echo "=== $(date -Is) cell ${I}: task=${TASK} k=${K} (wrist) ==="
"${PY}" -m eventtok.scripts.compare_modalities \
    --task "${TASK}" --encoders siglip dinov2l --cameras image wrist_image \
    --k "${K}" --out "${CACHE}/eval/wrk_${TASK}_k${K}.json"
echo "=== $(date -Is) done ==="
