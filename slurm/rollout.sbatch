#!/usr/bin/env bash
#SBATCH --job-name=eventtok-roll
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_ydu_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --mem=128G
#SBATCH --time=0-08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%j.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%j.err

# Closed-loop success rate with the event log as memory.
#
# H100 pinned deliberately: SAPIEN resets fine on H100 and RTX Pro 6000 and fails on
# A100 with createDeviceUnique: ErrorInitializationFailed. A generic --gres=gpu:1 lands
# on A100s here and the job dies on env reset.
#
# HF_HOME is required, not optional: the DINOv2 encoder loads from the hub by default
# and compute nodes have no outbound network, which is exactly how the first attempt
# failed. HF_HUB_OFFLINE turns a hang or a confusing OSError into an immediate one.

set -uo pipefail
CACHE=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache
PY=/n/lab_storage/ydu_lab/wfy/.conda/envs/robomme/bin/python
cd "${REPO:-/n/home04/wfy/repos/eventtok_mem}"

export HF_HOME=${HF_HOME:-/n/netscratch/ydu_lab/Lab/wfy/hf_cache}
export HF_HUB_OFFLINE=1
export TORCH_HOME="${CACHE}/torch"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MPLCONFIGDIR="/tmp/${USER}/mpl-${SLURM_JOB_ID:-manual}"
mkdir -p "${CACHE}/slurm" "${CACHE}/eval" "${TORCH_HOME}" "${MPLCONFIGDIR}"

TASK=${TASK:-SwingXtimes}
EPISODES=${EPISODES:-20}
EPOCHS=${EPOCHS:-150}
CONDS=${CONDS:-"none log wrong"}
TAG=${TAG:-main}

echo "=== $(date -Is) rollout ${TASK} episodes=${EPISODES} epochs=${EPOCHS} ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true

"${PY}" -m eventtok.scripts.rollout \
    --task "${TASK}" --episodes "${EPISODES}" --epochs "${EPOCHS}" \
    --conditions ${CONDS} \
    --out "${CACHE}/eval/rollout_${TASK}_${TAG}.json"
STATUS=$?
if [[ ${STATUS} -ne 0 ]]; then
  echo "=== $(date -Is) FAILED exit ${STATUS} ==="; exit ${STATUS}
fi
echo "=== $(date -Is) done ==="
