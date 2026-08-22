#!/usr/bin/env bash
#SBATCH --job-name=eventtok-dino
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_ydu_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=0-04:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --open-mode=append
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%j.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%j.err

# DINOv2 features for the encoder comparison. Preemptible, and safe to preempt:
# each episode's file is written atomically and the next run skips what exists, so
# requeueing costs at most one episode of redone work.
#
# 16 CPUs and 96 GB because the cost here is reading pkls off netscratch, not the
# forward pass -- ~400 KB per frame per camera, ~28 GB per task. The GPU is idle
# most of the time; that is expected and still the cheapest way to run this.
#
#   TASK=SwingXtimes sbatch slurm/encode_dino.sbatch
#   TASK=ButtonUnmask sbatch slurm/encode_dino.sbatch
#
# Compute nodes have no outbound network, so the weights must already be in
# HF_HOME. Populate it from a login node first:
#   HF_HOME=/n/netscratch/ydu_lab/Lab/wfy/hf_cache python -c \
#     "from huggingface_hub import snapshot_download as d; d('facebook/dinov2-large')"

set -uo pipefail

REQUEUE_EXIT_CODE=${REQUEUE_EXIT_CODE:-85}
CACHE=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache
PY=/n/lab_storage/ydu_lab/wfy/.conda/envs/robomme/bin/python
REPO=${REPO:-/n/home04/wfy/repos/eventtok_mem}

# Nothing on /n/home04 (81% full, 19 GB). HF_HOME especially -- the default is
# ~/.cache/huggingface, and dinov2-large is 1.2 GB.
export HF_HOME=${HF_HOME:-/n/netscratch/ydu_lab/Lab/wfy/hf_cache}
export HF_HUB_OFFLINE=1
export TORCH_HOME="${CACHE}/torch"
export MPLCONFIGDIR="/tmp/${USER}/mpl-${SLURM_JOB_ID:-manual}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
mkdir -p "${CACHE}/slurm" "${TORCH_HOME}" "${MPLCONFIGDIR}"

TASK=${TASK:-SwingXtimes}
MODEL=${MODEL:-dinov2l}
SCALE=${SCALE:-2x2}
BATCH=${BATCH:-128}
READERS=${READERS:-24}
EXTRA=${EXTRA:-}

echo "=== $(date -Is) job=${SLURM_JOB_ID:-manual} restarts=${SLURM_RESTART_COUNT:-0} ==="
echo "task=${TASK} model=${MODEL} scale=${SCALE} node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
cd "${REPO}"

forward() {
  echo "[sbatch] got $1, forwarding to pid ${CHILD:-none}"
  if [[ -n "${CHILD:-}" ]]; then kill -USR1 "${CHILD}" 2>/dev/null || true; fi
}
trap 'forward SIGUSR1' USR1
trap 'forward SIGTERM' TERM

"${PY}" -m eventtok.scripts.encode_dino \
    --task "${TASK}" \
    --model "${MODEL}" \
    --scale "${SCALE}" \
    --batch "${BATCH}" \
    --readers "${READERS}" ${EXTRA} &
CHILD=$!

# `wait` returns 128+signo when a trap fires, so keep waiting for the real status.
wait "${CHILD}"; STATUS=$?
while kill -0 "${CHILD}" 2>/dev/null; do
  wait "${CHILD}"; STATUS=$?
done
echo "[sbatch] encoder exited ${STATUS}"

if [[ "${STATUS}" -eq "${REQUEUE_EXIT_CODE}" ]]; then
  # scancel reaches the job the same way preemption does -- SIGTERM, the trap, a
  # clean stop at a checkpoint boundary, exit 85 -- so requeueing unconditionally
  # here resurrects jobs the user deliberately killed. Observed: a cancelled job
  # came straight back as PENDING and started redoing the work. Ask slurm what
  # actually happened before putting the job back in the queue.
  JOB_STATE=$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null \
    | tr ' ' '\n' | grep -m1 '^JobState=' | cut -d= -f2)
  echo "[sbatch] stopped at a checkpoint boundary; JobState=${JOB_STATE:-unknown}"
  case "${JOB_STATE}" in
    CANCELLED|DEADLINE|TIMEOUT)
      echo "[sbatch] ${JOB_STATE}, not requeueing"
      exit 0
      ;;
  esac
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "[sbatch] requeueing ${SLURM_JOB_ID}"
    scontrol requeue "${SLURM_JOB_ID}"
  fi
  exit 0
fi
if [[ "${STATUS}" -ne 0 ]]; then exit "${STATUS}"; fi

echo "=== $(date -Is) done ==="
