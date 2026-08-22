#!/usr/bin/env bash
#SBATCH --job-name=eventtok
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_ydu_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --open-mode=append
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%j.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%j.err

# kempner_requeue is preemptible, which is why it is cheap. Three things make that
# survivable, and all three have to be present or the cheap partition costs more
# than the expensive one:
#
#   --requeue            slurm puts the job back in the queue after preemption
#   --signal=B:USR1@180  the batch shell is warned 180s before the kill
#   --open-mode=append   each requeue appends to the same log instead of truncating
#
# The signal goes to the batch *shell* (the B: prefix), not to python, so the trap
# below forwards it. The trainer then checkpoints at the next step boundary and
# exits 85, which this script turns into an explicit `scontrol requeue`.
#
# Usage:
#   sbatch slurm/train_tokenizer.sbatch
#   TASK=PickXtimes LEVELS="8 5 5" EPOCHS=40 sbatch slurm/train_tokenizer.sbatch

set -uo pipefail

REQUEUE_EXIT_CODE=${REQUEUE_EXIT_CODE:-85}
CACHE=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache
PY=/n/lab_storage/ydu_lab/wfy/.conda/envs/robomme/bin/python
REPO=${REPO:-/n/home04/wfy/repos/eventtok_mem}

# Nothing goes near /n/home04 (81% full). Scratch dirs are per-job so concurrent
# array tasks do not fight over them.
mkdir -p "${CACHE}/slurm"
export MPLCONFIGDIR="/tmp/${USER}/mpl-${SLURM_JOB_ID:-manual}"
export TORCH_HOME="${CACHE}/torch"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
mkdir -p "${MPLCONFIGDIR}" "${TORCH_HOME}"

TASK=${TASK:-SwingXtimes}
LEVELS=${LEVELS:-"8 8"}
EPOCHS=${EPOCHS:-40}
BATCH=${BATCH:-512}
SAVE_EVERY=${SAVE_EVERY:-200}
STRIDE=${STRIDE:-1}
EXTRA=${EXTRA:-}

TAG=$(echo "${LEVELS}" | tr ' ' 'x')
CKPT="${CACHE}/ckpt/tokenizer_${TASK}_${TAG}.pt"

echo "=== $(date -Is) job=${SLURM_JOB_ID:-manual} restarts=${SLURM_RESTART_COUNT:-0} ==="
echo "task=${TASK} levels=${LEVELS} epochs=${EPOCHS} batch=${BATCH} ckpt=${CKPT}"
cd "${REPO}"

# --- forward the pre-preemption signal to the trainer -----------------------
forward() {
  echo "[sbatch] got $1, forwarding to trainer pid ${CHILD:-none}"
  if [[ -n "${CHILD:-}" ]]; then kill -USR1 "${CHILD}" 2>/dev/null || true; fi
}
trap 'forward SIGUSR1' USR1
trap 'forward SIGTERM' TERM

# --resume auto picks up ${CKPT} if it exists, so a requeued job continues rather
# than starting over. That is the whole point.
"${PY}" -m eventtok.scripts.train_tokenizer \
    --task "${TASK}" \
    --levels ${LEVELS} \
    --epochs "${EPOCHS}" \
    --batch "${BATCH}" \
    --stride "${STRIDE}" \
    --workers "${SLURM_CPUS_PER_TASK:-8}" \
    --save-every "${SAVE_EVERY}" \
    --resume auto \
    --out "${CKPT}" ${EXTRA} &
CHILD=$!

# `wait` returns 128+signo when interrupted by a trap, so keep waiting until the
# child is genuinely gone or the exit status is real.
wait "${CHILD}"; STATUS=$?
while kill -0 "${CHILD}" 2>/dev/null; do
  wait "${CHILD}"; STATUS=$?
done

echo "[sbatch] trainer exited ${STATUS}"

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

if [[ "${STATUS}" -ne 0 ]]; then
  # A real failure. Do not requeue -- that would loop forever on a code bug.
  exit "${STATUS}"
fi

# Training finished. Build the event logs from the final checkpoint, and the
# k-means baseline alongside it: if the learned tokenizer cannot beat Lloyd's
# algorithm, a learned tokenizer that loses to it is not worth its complexity.
echo "[sbatch] building event logs"
"${PY}" -m eventtok.scripts.build_events --ckpt "${CKPT}" --task "${TASK}" || true
"${PY}" -m eventtok.scripts.build_events --kmeans 32 --task "${TASK}" || true
echo "=== $(date -Is) done ==="
