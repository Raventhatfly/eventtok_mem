#!/usr/bin/env bash
#SBATCH --job-name=eventtok-sgw
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_ydu_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=0-04:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --open-mode=append
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%j.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%j.err

# SigLIP features for the wrist camera -- the matched row that makes the wrist
# result an encoder-controlled comparison instead of a two-variable one.
#
# Uses the robomme-openpi env (JAX 0.5.3 + CUDA12 plugin), not robomme, because
# pi0.5's SigLIP tokenizer is Flax and robomme has no JAX. Same weights that built
# the shipped base-camera cache, so camera is the only difference.
#
#   TASK=ButtonUnmask sbatch slurm/encode_siglip_wrist.sbatch
#   TASK=SwingXtimes  sbatch slurm/encode_siglip_wrist.sbatch

set -uo pipefail

REQUEUE_EXIT_CODE=${REQUEUE_EXIT_CODE:-85}
CACHE=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache
PY=/n/lab_storage/ydu_lab/wfy/.conda/envs/robomme-openpi/bin/python
REPO=${REPO:-/n/home04/wfy/repos/eventtok_mem}

export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-/n/netscratch/ydu_lab/Lab/wfy/dataset/robomme/openpi_data}
export PYTHONPATH="/n/home04/wfy/repos/robomme_policy_learning/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="/tmp/${USER}/mpl-${SLURM_JOB_ID:-manual}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
# JAX preallocates 75% of the GPU by default; the forward pass needs far less and
# the cap leaves room for the pkl reader threads.
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.4}
mkdir -p "${CACHE}/slurm" "${MPLCONFIGDIR}"

TASK=${TASK:-ButtonUnmask}
CAMERA=${CAMERA:-wrist_image}
SCALE=${SCALE:-2x2}
BATCH=${BATCH:-64}
READERS=${READERS:-24}

echo "=== $(date -Is) job=${SLURM_JOB_ID:-manual} restarts=${SLURM_RESTART_COUNT:-0} ==="
echo "task=${TASK} camera=${CAMERA} scale=${SCALE} node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
cd "${REPO}"

forward() {
  echo "[sbatch] got $1, forwarding to pid ${CHILD:-none}"
  if [[ -n "${CHILD:-}" ]]; then kill -USR1 "${CHILD}" 2>/dev/null || true; fi
}
trap 'forward SIGUSR1' USR1
trap 'forward SIGTERM' TERM

"${PY}" -m eventtok.scripts.encode_siglip_wrist \
    --task "${TASK}" --camera "${CAMERA}" --scale "${SCALE}" \
    --batch "${BATCH}" --readers "${READERS}" &
CHILD=$!

wait "${CHILD}"; STATUS=$?
while kill -0 "${CHILD}" 2>/dev/null; do wait "${CHILD}"; STATUS=$?; done
echo "[sbatch] encoder exited ${STATUS}"

if [[ "${STATUS}" -eq "${REQUEUE_EXIT_CODE}" ]]; then
  echo "[sbatch] clean episode boundary; requeueing ${SLURM_JOB_ID:-}"
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then scontrol requeue "${SLURM_JOB_ID}"; fi
  exit 0
fi
if [[ "${STATUS}" -ne 0 ]]; then exit "${STATUS}"; fi
echo "=== $(date -Is) done ==="
