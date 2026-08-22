#!/usr/bin/env bash
#SBATCH --job-name=eventtok-mod
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%j.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%j.err

# The modality/encoder table. CPU only -- k-means, PCA and the metric are all CPU
# work, so a GPU would idle.
#
# 96 GB because the feature blocks are held dense: SwingXtimes is 43,419 rows and
# "both" vision form is 2 x 4 x width per row, so SigLIP alone is 2.8 GB in fp32
# and the run holds several blocks plus the k-means distance matrix at once. This
# does not belong on a login node.
#
#   TASK=SwingXtimes sbatch slurm/compare_modalities.sbatch
#   TASK=ButtonUnmask ENCODERS="siglip dinov2l" CAMERAS="image wrist_image" \
#     sbatch slurm/compare_modalities.sbatch

set -uo pipefail

CACHE=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache
PY=/n/lab_storage/ydu_lab/wfy/.conda/envs/robomme/bin/python
REPO=${REPO:-/n/home04/wfy/repos/eventtok_mem}

export MPLCONFIGDIR="/tmp/${USER}/mpl-${SLURM_JOB_ID:-manual}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
mkdir -p "${CACHE}/slurm" "${CACHE}/eval" "${MPLCONFIGDIR}"

TASK=${TASK:-SwingXtimes}
ENCODERS=${ENCODERS:-"siglip dinov2l"}
CAMERAS=${CAMERAS:-"image"}
K=${K:-32}
PCA=${PCA:-64}
FORM=${FORM:-both}
TAG=${TAG:-$(echo "${CAMERAS}" | tr -d ' _image' | sed 's/^$/base/')}

echo "=== $(date -Is) task=${TASK} encoders=${ENCODERS} cameras=${CAMERAS} ==="
cd "${REPO}"

"${PY}" -m eventtok.scripts.compare_modalities \
    --task "${TASK}" \
    --encoders ${ENCODERS} \
    --cameras ${CAMERAS} \
    --k "${K}" \
    --pca "${PCA}" \
    --vision-form "${FORM}" \
    --out "${CACHE}/eval/modalities_${TASK}_k${K}_${TAG}.json"

echo "=== $(date -Is) done ==="
