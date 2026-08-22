#!/usr/bin/env bash
#SBATCH --job-name=eventtok-fuse
#SBATCH --partition=shared
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%A_%a.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%A_%a.err

# Are SigLIP and DINOv2 complementary? Matched dims, matched variance.
#
# 160G because "joint" mode concatenates the RAW features before PCA -- SigLIP 16384
# dims plus DINOv2 8192 over 43,419 SwingXtimes rows is 4.3 GB for the concat alone,
# and the train-half copy and the randomised range finder sit alongside it. An
# earlier attempt at this on a login node was killed part-way through, which is what
# the memory figure is sized against.
#
#   sbatch slurm/compare_fusion.sbatch
#   SOURCES="siglip dinov2l siglip_wrist dinov2l_wrist" sbatch slurm/compare_fusion.sbatch

set -uo pipefail

CACHE=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache
PY=/n/lab_storage/ydu_lab/wfy/.conda/envs/robomme/bin/python
REPO=${REPO:-/n/home04/wfy/repos/eventtok_mem}

export MPLCONFIGDIR="/tmp/${USER}/mpl-${SLURM_JOB_ID:-manual}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
mkdir -p "${CACHE}/slurm" "${CACHE}/eval" "${MPLCONFIGDIR}"
cd "${REPO}"

TASKS=(ButtonUnmask SwingXtimes)
TASK=${TASKS[${SLURM_ARRAY_TASK_ID:-0}]}
SOURCES=${SOURCES:-"siglip dinov2l"}
KS=${KS:-"16 32 64 128"}
TAG=${TAG:-pair}

echo "=== $(date -Is) task=${TASK} sources=${SOURCES} ==="
"${PY}" -m eventtok.scripts.compare_fusion \
    --task "${TASK}" --sources ${SOURCES} --ks ${KS} \
    --out "${CACHE}/eval/fusion_${TASK}_${TAG}.json"
echo "=== $(date -Is) done ==="
