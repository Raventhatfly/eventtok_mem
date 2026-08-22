#!/usr/bin/env bash
#SBATCH --job-name=eventtok-nvk
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_ydu_lab
#SBATCH --array=0-31
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=0-06:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --open-mode=append
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%A_%a.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%A_%a.err

# Train the learned tokenizer and score it with the metrics k-means was scored with.
#
# The whole project has run on k-means. The learned tokenizer (FSQ + register encoder,
# i.e. the QueST/OAT shape) has never been evaluated on label accuracy, token identity
# or boundary F1, so "if a learned tokenizer cannot beat Lloyd's algorithm" has been an
# assertion rather than a measurement.
#
# Codebook-matched on purpose: FSQ levels 8x8 = 64 codes against k-means k=64. A
# learned tokenizer with a bigger codebook would win on label accuracy for the reason
# the k sweep already established -- finer partitions fit labels better -- and that
# would not be evidence about learning. 8 rather than 4 levels per channel because the
# FSQ paper reports subpar behaviour below 5, and the warning fires at 4.
#
# 32 cells: 16 tasks x {action-only, action+vision}. Action-only is the OAT-shaped
# control; the vision variant is the one the project's novelty claim rests on.

set -uo pipefail

REQUEUE_EXIT_CODE=${REQUEUE_EXIT_CODE:-85}
CACHE=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache
PY=/n/lab_storage/ydu_lab/wfy/.conda/envs/robomme/bin/python
REPO=${REPO:-/n/home04/wfy/repos/eventtok_mem}

export MPLCONFIGDIR="/tmp/${USER}/mpl-${SLURM_JOB_ID:-manual}"
export TORCH_HOME="${CACHE}/torch"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
mkdir -p "${CACHE}/slurm" "${CACHE}/ckpt" "${CACHE}/eval" "${MPLCONFIGDIR}" "${TORCH_HOME}"
cd "${REPO}"

TASKS=(BinFill ButtonUnmask ButtonUnmaskSwap InsertPeg MoveCube PatternLock \
       PickHighlight PickXtimes RouteStick StopCube SwingXtimes VideoPlaceButton \
       VideoPlaceOrder VideoRepick VideoUnmask VideoUnmaskSwap)

I=${SLURM_ARRAY_TASK_ID:-0}
TASK=${TASKS[$(( I % 16 ))]}
if [[ $(( I / 16 )) -eq 0 ]]; then
  VISION=""; TAG="a"
else
  VISION="--use-vision"; TAG="av"
fi

LEVELS=${LEVELS:-"8 8"}
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-512}
CKPT="${CACHE}/ckpt/neural_${TASK}_${TAG}.pt"

echo "=== $(date -Is) job=${SLURM_JOB_ID:-manual} restarts=${SLURM_RESTART_COUNT:-0} ==="
echo "task=${TASK} variant=${TAG} levels=${LEVELS} epochs=${EPOCHS}"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true

forward() {
  echo "[sbatch] got $1, forwarding to pid ${CHILD:-none}"
  if [[ -n "${CHILD:-}" ]]; then kill -USR1 "${CHILD}" 2>/dev/null || true; fi
}
trap 'forward SIGUSR1' USR1
trap 'forward SIGTERM' TERM

"${PY}" -m eventtok.scripts.train_tokenizer \
    --task "${TASK}" --levels ${LEVELS} --epochs "${EPOCHS}" --batch "${BATCH}" \
    --workers "${SLURM_CPUS_PER_TASK:-8}" --save-every 500 --resume auto \
    --out "${CKPT}" ${VISION} &
CHILD=$!
wait "${CHILD}"; STATUS=$?
while kill -0 "${CHILD}" 2>/dev/null; do wait "${CHILD}"; STATUS=$?; done
echo "[sbatch] trainer exited ${STATUS}"

if [[ "${STATUS}" -eq "${REQUEUE_EXIT_CODE}" ]]; then
  JOB_STATE=$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null \
    | tr ' ' '\n' | grep -m1 '^JobState=' | cut -d= -f2)
  echo "[sbatch] stopped at a checkpoint boundary; JobState=${JOB_STATE:-unknown}"
  case "${JOB_STATE}" in
    CANCELLED|DEADLINE|TIMEOUT) echo "[sbatch] ${JOB_STATE}, not requeueing"; exit 0 ;;
  esac
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then scontrol requeue "${SLURM_JOB_ID}"; fi
  exit 0
fi
if [[ "${STATUS}" -ne 0 ]]; then exit "${STATUS}"; fi

echo "[sbatch] scoring the checkpoint with the k-means metrics"
"${PY}" -m eventtok.scripts.analyze_task --task "${TASK}" --ckpt "${CKPT}" \
    --variant "neural-${TAG}" \
    --out "${CACHE}/eval/neural_${TASK}_neural-${TAG}.json"
echo "=== $(date -Is) done ${TASK} ${TAG} ==="
