#!/usr/bin/env bash
#SBATCH --job-name=pi05event
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_ydu_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:2
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%j.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%j.err

# pi0.5 with the event log in its memory slot.
#
# No fork of robomme_policy_learning. The wrapper rebinds one name --
# dataloader.RoboMMEDataset -- so the memory slots are filled from the event cache
# instead of per-frame SigLIP features. Everything else (repack transform,
# HistAugObservation, embed_prefix, the perceptual encoder) is upstream's, which is what
# makes this comparable to the MovieChat and token-dropping baselines: same wiring,
# different memory.
#
# EVENT_MODE=wrong is the control that decides whether any gain is real. If a log from
# a different episode trains to the same success rate, the policy is ignoring the memory.
#
# 80 GB cards, not 40. pi0.5's train state is ~2.8 GB of parameters and jit reported
# 48.7 GB of input/output arguments at init -- params, EMA and both Adam moments -- so
# an A100-40GB dies in init_train_state before the first step, whatever the batch size.
#
# Preemption: --requeue plus RESUME=true. train.py checkpoints every SAVE_INTERVAL
# steps through orbax and initialize_checkpoint_dir picks up the latest, so a requeue
# costs at most SAVE_INTERVAL steps rather than the run.

set -uo pipefail

ROBOMME=${ROBOMME:-/n/home04/wfy/repos/robomme_policy_learning}
EVENTTOK=${EVENTTOK:-/n/home04/wfy/repos/eventtok_mem}
DATA_ROOT=${DATA_ROOT:-/n/netscratch/ydu_lab/Lab/wfy/dataset/robomme}
DATASET_PATH=${DATASET_PATH:-${DATA_ROOT}/robomme_preprocessed_data}
PY=${PY:-/n/lab_storage/ydu_lab/wfy/.conda/envs/robomme-openpi/bin/python}

EVENT_TAG=${EVENT_TAG:-joint}
EVENT_MODE=${EVENT_MODE:-log}
EVENT_DROP=${EVENT_DROP:-0.0}
HISTORY_CONFIG=${HISTORY_CONFIG:-${EVENTTOK}/configs/pi05/eventtok-${EVENT_TAG}-context.yaml}

EXP_NAME=${EXP_NAME:-eventtok_${EVENT_TAG}_${EVENT_MODE}}
# Our own space. The repo's runs/ckpts/mme_vla_suite is a symlink into another lab's
# netscratch tree, and the home disk is 81% full against 11.7 GiB per checkpoint.
CKPT_BASE=${CKPT_BASE:-/n/netscratch/ydu_lab/Lab/wfy/ckpts/robomme_policy_ckpt}
# 32 to match pi05_baseline_2gpu_80k, which is the no-memory control. A different
# batch size would confound the comparison with a trained checkpoint we already have.
BATCH_SIZE=${BATCH_SIZE:-32}
FSDP_DEVICES=${FSDP_DEVICES:-2}
NUM_WORKERS=${NUM_WORKERS:-4}
NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-80000}
SAVE_INTERVAL=${SAVE_INTERVAL:-5000}
WANDB_ENABLED=${WANDB_ENABLED:-true}
RESUME=${RESUME:-true}
OVERWRITE=${OVERWRITE:-false}

export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-${DATA_ROOT}/openpi_data}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}
export PYTHONPATH="${EVENTTOK}${PYTHONPATH:+:${PYTHONPATH}}"
export ROBOMME_REPO="${ROBOMME}"
mkdir -p /n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm

for f in "${HISTORY_CONFIG}" "${DATASET_PATH}/meta/stats.json"; do
    [ -e "${f}" ] || { echo "missing ${f}"; exit 1; }
done

echo "=== $(date -Is) job=${SLURM_JOB_ID:-manual} restarts=${SLURM_RESTART_COUNT:-0} ==="
echo "exp=${EXP_NAME} memory=${EVENT_MODE} tag=${EVENT_TAG} drop=${EVENT_DROP}"
echo "history_config=${HISTORY_CONFIG}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# Bail before python if the card is too small. pi0.5's train state needs ~48 GB at init,
# so a 40 GB A100 -- or worse, a 20 GB MIG slice -- dies inside init_train_state after
# holding the GPU for several minutes at 0% utilisation. Two such jobs earned a Kempner
# idle-GPU warning. Five seconds here instead.
MIN_GPU_MB=${MIN_GPU_MB:-70000}
SMALLEST=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
    | sort -n | head -1)
if [ -n "${SMALLEST}" ] && [ "${SMALLEST}" -lt "${MIN_GPU_MB}" ]; then
    echo "GPU has ${SMALLEST} MiB, need ${MIN_GPU_MB}; releasing rather than OOMing in init"
    exit 1
fi

# A first attempt has nothing to resume from; initialize_checkpoint_dir errors on
# --resume with an empty directory, so only pass it once a checkpoint exists.
CKPT_DIR=${CKPT_DIR:-${CKPT_BASE}/mme_vla_suite/${EXP_NAME}}
ARGS=(
  mme_vla_suite
  --exp-name="${EXP_NAME}"
  --batch-size="${BATCH_SIZE}"
  --num-workers="${NUM_WORKERS}"
  --num-train-steps="${NUM_TRAIN_STEPS}"
  --save-interval="${SAVE_INTERVAL}"
  --fsdp-devices="${FSDP_DEVICES}"
  --dataset-path="${DATASET_PATH}"
  --checkpoint-base-dir="${CKPT_BASE}"
  --model.use_history
  --model.history_config="${HISTORY_CONFIG}"
  --event-tag="${EVENT_TAG}"
  --event-mode="${EVENT_MODE}"
  --event-drop="${EVENT_DROP}"
)
if [ "${WANDB_ENABLED}" = "true" ]; then ARGS+=(--wandb-enabled); else ARGS+=(--no-wandb-enabled); fi
if [ "${OVERWRITE}" = "true" ]; then ARGS+=(--overwrite)
elif [ "${RESUME}" = "true" ] && [ -d "${CKPT_DIR}" ]; then ARGS+=(--resume); fi

cd "${EVENTTOK}"
"${PY}" -m eventtok.scripts.train_pi05_event "${ARGS[@]}"
status=$?
echo "=== $(date -Is) exit=${status} ==="
exit ${status}
