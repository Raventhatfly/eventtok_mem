#!/usr/bin/env bash
#SBATCH --job-name=pi05eval
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_ydu_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:2
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%j.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%j.err

# Closed-loop RoboMME evaluation: success rate, the only number that settles anything
# here. Two processes on two GPUs, matching scripts/eval.sh but without tmux, which does
# not survive a batch job.
#
#   GPU 0  serve_policy.py   jax, conda env robomme-openpi
#   GPU 1  examples/robomme/eval.py   ManiSkill 3.0.0b21 + SAPIEN 3.0.3, conda env robomme
#
# NOT requeueable. The rollout writes per-episode results incrementally but the server
# holds all model state in memory, so a preemption mid-episode leaves a half-written
# task. Re-running with the same save dir skips finished tasks instead.
#
# Card choice is not cosmetic: SAPIEN's renderer works on H100 and RTX Pro 6000 and
# fails on A100 here, and pi0.5 inference wants the headroom anyway.

set -uo pipefail

ROBOMME=${ROBOMME:-/n/home04/wfy/repos/robomme_policy_learning}
EVENTTOK=${EVENTTOK:-/n/home04/wfy/repos/eventtok_mem}
CACHE=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache
DATA_ROOT=${DATA_ROOT:-/n/netscratch/ydu_lab/Lab/wfy/dataset/robomme}
PY_SERVE=${PY_SERVE:-/n/lab_storage/ydu_lab/wfy/.conda/envs/robomme-openpi/bin/python}
PY_ENV=${PY_ENV:-/n/lab_storage/ydu_lab/wfy/.conda/envs/robomme/bin/python}

# What to evaluate.
CKPT_DIR=${CKPT_DIR:?set CKPT_DIR to a checkpoint step directory}
CONFIG_TYPE=${CONFIG_TYPE:-mme_vla_suite}
POLICY_NAME=${POLICY_NAME:-$(basename "$(dirname "${CKPT_DIR}")")}
CKPT_ID=${CKPT_ID:-$(basename "${CKPT_DIR}")}
SEED=${SEED:-7}
MAX_STEPS=${MAX_STEPS:-1300}
ONLY_TASKS=${ONLY_TASKS:-}
USE_HISTORY=${USE_HISTORY:-true}
SAVE_DIR=${SAVE_DIR:-${CACHE}/evaluation}

export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-${DATA_ROOT}/openpi_data}
export PYTHONPATH="${EVENTTOK}${PYTHONPATH:+:${PYTHONPATH}}"
# Python block-buffers stdout to a file at 8 KB, so the per-100-step loss and
# mem_enc_norm lines sit invisible for hours while tqdm (stderr) shows progress.
export PYTHONUNBUFFERED=1
export ROBOMME_REPO="${ROBOMME}"
export HF_HOME=${HF_HOME:-${CACHE}/hf}
mkdir -p "${CACHE}/slurm" "${SAVE_DIR}" "${HF_HOME}"

echo "=== $(date -Is) job=${SLURM_JOB_ID:-manual} ==="
echo "ckpt=${CKPT_DIR}"
echo "policy=${POLICY_NAME} ckpt_id=${CKPT_ID} seed=${SEED} config=${CONFIG_TYPE}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true

# SAPIEN's renderer fails on the A100s here, and pi0.5 inference needs the headroom.
MIN_GPU_MB=${MIN_GPU_MB:-70000}
SMALLEST=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
    | sort -n | head -1)
if [ -n "${SMALLEST}" ] && [ "${SMALLEST}" -lt "${MIN_GPU_MB}" ]; then
    echo "GPU has ${SMALLEST} MiB, need ${MIN_GPU_MB}; releasing"
    exit 1
fi
[ -d "${CKPT_DIR}" ] || { echo "no such checkpoint ${CKPT_DIR}"; exit 1; }

PORT=$(${PY_ENV} - <<'PYEOF'
import socket
s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()
PYEOF
)
echo "port=${PORT}"

cd "${ROBOMME}"

SERVE_ARGS=(scripts/serve_policy.py --seed="${SEED}" --port="${PORT}"
            policy:checkpoint --policy.dir="${CKPT_DIR}" --policy.config="${CONFIG_TYPE}")
if [ -n "${EVENT_TAG:-}" ]; then
    EVT=(-m eventtok.scripts.serve_pi05_event
         --event-tag="${EVENT_TAG}" --event-mode="${EVENT_MODE:-log}")
    # Pin the task when evaluating one at a time. Otherwise the server infers it from
    # the prompt, and the action scale it picks is per task -- a wrong guess silently
    # rescales every delta action before quantisation.
    [ -n "${EVENT_TASK:-}" ] && EVT+=(--event-task="${EVENT_TASK}")
    SERVE_ARGS=("${EVT[@]}" "${SERVE_ARGS[@]:1}")
fi

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
    "${PY_SERVE}" "${SERVE_ARGS[@]}" &
SERVER=$!
trap 'kill ${SERVER} 2>/dev/null || true' EXIT

# Wait for the port rather than sleeping a fixed 30 s: loading pi0.5 off netscratch has
# taken anywhere from 40 s to 4 minutes, and a client that starts early dies immediately.
for i in $(seq 1 120); do
    kill -0 ${SERVER} 2>/dev/null || { echo "server died before serving"; wait ${SERVER}; exit 1; }
    ${PY_ENV} -c "
import socket, sys
s = socket.socket(); s.settimeout(1)
sys.exit(0 if s.connect_ex(('127.0.0.1', ${PORT})) == 0 else 1)
" && break
    sleep 5
done
echo "=== $(date -Is) server up after ~$((i * 5))s ==="

CLIENT_ARGS=(examples/robomme/eval.py
             --args.model_seed="${SEED}" --args.port="${PORT}"
             --args.policy_name="${POLICY_NAME}" --args.model_ckpt_id="${CKPT_ID}"
             --args.max_steps="${MAX_STEPS}" --args.save_dir="${SAVE_DIR}")
if [ "${USE_HISTORY}" = "true" ]; then CLIENT_ARGS+=(--args.use-history)
else CLIENT_ARGS+=(--args.no-use-history); fi
[ -n "${ONLY_TASKS}" ] && CLIENT_ARGS+=(--args.only_tasks="${ONLY_TASKS}")

CUDA_VISIBLE_DEVICES=1 "${PY_ENV}" "${CLIENT_ARGS[@]}"
status=$?
echo "=== $(date -Is) eval exit=${status} ==="
exit ${status}
