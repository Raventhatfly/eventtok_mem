#!/usr/bin/env bash
#SBATCH --job-name=eventtok-method
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_ydu_lab
#SBATCH --array=0-23
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=0-05:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%A_%a.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%A_%a.err

# The method against its OAT control, with the event tokens as the variable.
#
# Everything measured so far built the tokens from actions alone, which is OAT. This
# crosses {action, action+vision} tokens x 4 tasks x 3 seeds, so the comparison is what
# the tokens are made of, and three seeds because the last seed sweep flipped two of
# four conclusions.
#
# Controls in every cell: wrong (content), rawhist (does the tokenizer beat the raw
# executed actions it summarises), elapsed (is it just episode progress).

set -uo pipefail
CACHE=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache
PY=/n/lab_storage/ydu_lab/wfy/.conda/envs/robomme/bin/python
cd "${REPO:-/n/home04/wfy/repos/eventtok_mem}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MPLCONFIGDIR="/tmp/${USER}/mpl-${SLURM_JOB_ID:-manual}"
mkdir -p "${CACHE}/slurm" "${CACHE}/eval/method" "${MPLCONFIGDIR}"

TASKS=(SwingXtimes VideoRepick PickXtimes ButtonUnmask)
SEEDS=(1 2 3)
TOKENS=(action action+vision)
I=${SLURM_ARRAY_TASK_ID:-0}
TASK=${TASKS[$(( I / 6 ))]}
SEED=${SEEDS[$(( (I / 2) % 3 ))]}
TOK=${TOKENS[$(( I % 2 ))]}
TAG=$(echo "${TOK}" | tr '+' '_')

echo "=== $(date -Is) ${TASK} seed=${SEED} tokens=${TOK} ==="
"${PY}" -m eventtok.scripts.eval_consumption \
    --task "${TASK}" --seed "${SEED}" --epochs 25 --tokens "${TOK}" \
    --conditions none log wrong rawhist elapsed \
    --out "${CACHE}/eval/method/${TASK}_s${SEED}_${TAG}.json"
echo "=== $(date -Is) done ==="
