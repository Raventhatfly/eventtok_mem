#!/usr/bin/env bash
#SBATCH --job-name=eventtok-seeds
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_ydu_lab
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=0-04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%A_%a.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%A_%a.err

# Is a 3-5% memory effect a result or noise? Every number reported so far is one seed,
# and the seed sets the train/test split, the k-means init, the model init and the batch
# order simultaneously. Four tasks x three seeds; the spread of `none` across seeds is
# the yardstick the log-vs-none difference has to clear.

set -uo pipefail
CACHE=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache
PY=/n/lab_storage/ydu_lab/wfy/.conda/envs/robomme/bin/python
cd "${REPO:-/n/home04/wfy/repos/eventtok_mem}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MPLCONFIGDIR="/tmp/${USER}/mpl-${SLURM_JOB_ID:-manual}"
mkdir -p "${CACHE}/slurm" "${CACHE}/eval/seeds" "${MPLCONFIGDIR}"

TASKS=(ButtonUnmask PickXtimes SwingXtimes VideoRepick)
SEEDS=(1 2 3)
I=${SLURM_ARRAY_TASK_ID:-0}
TASK=${TASKS[$(( I / 3 ))]}
SEED=${SEEDS[$(( I % 3 ))]}

echo "=== $(date -Is) ${TASK} seed=${SEED} ==="
"${PY}" -m eventtok.scripts.eval_consumption \
    --task "${TASK}" --seed "${SEED}" --epochs 25 \
    --conditions none log wrong \
    --out "${CACHE}/eval/seeds/consumption_${TASK}_s${SEED}.json"
echo "=== $(date -Is) done ==="
