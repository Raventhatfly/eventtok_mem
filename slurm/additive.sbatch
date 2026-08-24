#!/usr/bin/env bash
#SBATCH --job-name=eventtok-add
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

# Does the event log ADD to the action chunk?
#
# The earlier sweep pitted `log` against `rawhist` as rivals, which mis-states the
# design: memory is high-level and the action chunk still goes in. The log is a lossy
# function of that chunk, so it cannot win a head-to-head, and losing one says nothing.
#
# The scales are complementary. rawhist is 32 steps, about 5% of a 400-800 step
# episode, so it cannot see a repetition from 300 steps ago; the log spans the whole
# episode. The test is rawhist+log against rawhist, with rawhist+wrong as the control
# so any gain has to survive its content being corrupted.
#
# {action, action+vision} tokens x 4 tasks x 3 seeds.

set -uo pipefail
CACHE=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache
PY=/n/lab_storage/ydu_lab/wfy/.conda/envs/robomme/bin/python
cd "${REPO:-/n/home04/wfy/repos/eventtok_mem}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MPLCONFIGDIR="/tmp/${USER}/mpl-${SLURM_JOB_ID:-manual}"
mkdir -p "${CACHE}/slurm" "${CACHE}/eval/additive" "${MPLCONFIGDIR}"

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
    --conditions rawhist rawhist+log rawhist+wrong none log \
    --out "${CACHE}/eval/additive/${TASK}_s${SEED}_${TAG}.json"
STATUS=$?
if [[ ${STATUS} -ne 0 ]]; then
  echo "=== $(date -Is) FAILED exit ${STATUS} ==="; exit ${STATUS}
fi
echo "=== $(date -Is) done ==="
