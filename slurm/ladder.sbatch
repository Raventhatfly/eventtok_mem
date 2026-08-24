#!/usr/bin/env bash
#SBATCH --job-name=eventtok-ladder
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_ydu_lab
#SBATCH --array=0-7
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

# Where does the tokenizer lose the information?
#
# rawhist (raw executed actions) beat the event tokens by 50+ points on action
# prediction, so the pipeline is discarding almost everything usable. This ladder drops
# one stage at a time to locate it:
#
#   rawhist  raw actions, nothing removed
#   codes    per-frame k-means codes            -> quantisation cost
#   runs     run symbols, repeats collapsed     -> cost of collapsing repeats
#   log      BPE-merged tokens                  -> cost of merging
#
# 4 tasks x 2 seeds. All rungs are causal the same way: a chunk starting at u is only
# available at u+chunk.

set -uo pipefail
CACHE=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache
PY=/n/lab_storage/ydu_lab/wfy/.conda/envs/robomme/bin/python
cd "${REPO:-/n/home04/wfy/repos/eventtok_mem}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MPLCONFIGDIR="/tmp/${USER}/mpl-${SLURM_JOB_ID:-manual}"
mkdir -p "${CACHE}/slurm" "${CACHE}/eval/ladder" "${MPLCONFIGDIR}"

TASKS=(SwingXtimes VideoRepick PickXtimes ButtonUnmask)
SEEDS=(1 2)
I=${SLURM_ARRAY_TASK_ID:-0}
TASK=${TASKS[$(( I / 2 ))]}
SEED=${SEEDS[$(( I % 2 ))]}

echo "=== $(date -Is) ladder ${TASK} seed=${SEED} ==="
"${PY}" -m eventtok.scripts.eval_consumption \
    --task "${TASK}" --seed "${SEED}" --epochs 25 --tokens action+vision \
    --conditions none rawhist codes runs log wrong \
    --out "${CACHE}/eval/ladder/${TASK}_s${SEED}.json"
STATUS=$?
if [[ ${STATUS} -ne 0 ]]; then
  echo "=== $(date -Is) FAILED ${TASK} s${SEED} exit ${STATUS} ==="; exit ${STATUS}
fi
echo "=== $(date -Is) done ==="
