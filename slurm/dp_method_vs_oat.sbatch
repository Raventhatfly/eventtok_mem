#!/usr/bin/env bash
#SBATCH --job-name=eventtok-dpm
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_ydu_lab
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=0-10:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%A_%a.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%A_%a.err

# The Diffusion Policy with the method's tokens, against the OAT control.
#
# The DP was built and run at 150 epochs already, but with action-only tokens, so it
# measured OAT-as-memory rather than this project's method. This crosses
# {action, action+vision} tokens x 4 tasks x 2 seeds. Two seeds rather than three
# because each cell trains four memory conditions at 150 epochs and samples through
# DDIM; the earlier seed sweep showed two seeds already expose a sign flip where one
# exists.
#
# Conditioning is crossattn only: the pooling control was already run and pooling lost
# on 4/4, so spending GPU on it again buys nothing.

set -uo pipefail
CACHE=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache
PY=/n/lab_storage/ydu_lab/wfy/.conda/envs/robomme/bin/python
cd "${REPO:-/n/home04/wfy/repos/eventtok_mem}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MPLCONFIGDIR="/tmp/${USER}/mpl-${SLURM_JOB_ID:-manual}"
mkdir -p "${CACHE}/slurm" "${CACHE}/eval/dp_method" "${MPLCONFIGDIR}"

TASKS=(SwingXtimes VideoRepick PickXtimes ButtonUnmask)
SEEDS=(1 2)
TOKENS=(action action+vision)
I=${SLURM_ARRAY_TASK_ID:-0}
TASK=${TASKS[$(( I / 4 ))]}
SEED=${SEEDS[$(( (I / 2) % 2 ))]}
TOK=${TOKENS[$(( I % 2 ))]}
TAG=$(echo "${TOK}" | tr '+' '_')

echo "=== $(date -Is) DP ${TASK} seed=${SEED} tokens=${TOK} ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true
"${PY}" -m eventtok.scripts.train_diffusion_policy \
    --task "${TASK}" --seed "${SEED}" --epochs 150 --tokens "${TOK}" \
    --memories none log wrong count --cond crossattn \
    --out "${CACHE}/eval/dp_method/${TASK}_s${SEED}_${TAG}.json"
STATUS=$?
# Without this the array reported COMPLETED 0:0 for 16 cells that had each crashed
# after a full 150-epoch train, because the trailing echo set the exit code.
if [[ ${STATUS} -ne 0 ]]; then
  echo "=== $(date -Is) FAILED ${TASK} s${SEED} ${TOK} exit ${STATUS} ==="
  exit ${STATUS}
fi
echo "=== $(date -Is) done ==="
