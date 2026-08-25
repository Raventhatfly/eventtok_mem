#!/usr/bin/env bash
#SBATCH --job-name=eventtok-oat
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_ydu_lab
#SBATCH --array=0-23
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

# OAT properly this time, against k-means at matched codebook size.
#
# The earlier verdict -- "the learned tokenizer does not beat k-means" -- tested FSQ plus
# a register encoder WITHOUT nested dropout, the mechanism OAT reports as worth 21 points
# on LIBERO by itself. That was not a test of OAT, and the conclusion was stated more
# broadly than the evidence allowed.
#
# Three arms per task so the ablation is readable:
#   nd-off   FSQ + registers, no ordering pressure   (what was tested before)
#   nd-on    the same plus nested dropout            (OAT)
# and k-means k=64 is already on disk as the matched-codebook baseline.
#
# 8 registers rather than 2: nested dropout needs a register axis long enough for the
# pow2 schedule to have support -- at R=2 it only ever samples {1,2}.

set -uo pipefail
REQUEUE_EXIT_CODE=${REQUEUE_EXIT_CODE:-85}
CACHE=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache
PY=/n/lab_storage/ydu_lab/wfy/.conda/envs/robomme/bin/python
cd "${REPO:-/n/home04/wfy/repos/eventtok_mem}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TORCH_HOME="${CACHE}/torch"
export MPLCONFIGDIR="/tmp/${USER}/mpl-${SLURM_JOB_ID:-manual}"
mkdir -p "${CACHE}/slurm" "${CACHE}/ckpt" "${CACHE}/eval/oat" "${MPLCONFIGDIR}" "${TORCH_HOME}"

TASKS=(BinFill ButtonUnmask ButtonUnmaskSwap InsertPeg MoveCube PatternLock \
       PickHighlight PickXtimes RouteStick StopCube SwingXtimes VideoRepick)
I=${SLURM_ARRAY_TASK_ID:-0}
TASK=${TASKS[$(( I / 2 ))]}
if [[ $(( I % 2 )) -eq 0 ]]; then ND=""; TAG="nd-off"; else ND="--nested-dropout"; TAG="nd-on"; fi

CKPT="${CACHE}/ckpt/oat_${TASK}_${TAG}.pt"
echo "=== $(date -Is) ${TASK} ${TAG} ==="

"${PY}" -m eventtok.scripts.train_tokenizer \
    --task "${TASK}" --levels 8 8 --registers 8 --epochs 30 --batch 512 \
    --workers "${SLURM_CPUS_PER_TASK:-8}" --save-every 500 --resume auto \
    --out "${CKPT}" ${ND}
STATUS=$?
if [[ ${STATUS} -eq ${REQUEUE_EXIT_CODE} ]]; then
  JS=$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | tr ' ' '\n' | grep -m1 '^JobState=' | cut -d= -f2)
  case "${JS}" in CANCELLED|DEADLINE|TIMEOUT) exit 0 ;; esac
  scontrol requeue "${SLURM_JOB_ID}"; exit 0
fi
if [[ ${STATUS} -ne 0 ]]; then echo "FAILED ${STATUS}"; exit ${STATUS}; fi

"${PY}" -m eventtok.scripts.analyze_task --task "${TASK}" --ckpt "${CKPT}" \
    --variant "oat-${TAG}" --out "${CACHE}/eval/oat/${TASK}_${TAG}.json"
echo "=== $(date -Is) done ==="
