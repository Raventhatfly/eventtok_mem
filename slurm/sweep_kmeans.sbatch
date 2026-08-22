#!/usr/bin/env bash
#SBATCH --job-name=eventtok-sweep
#SBATCH --partition=shared
#SBATCH --array=0-19
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%A_%a.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%A_%a.err

# The (K, min_frequency) grid for the k-means tokenizer.
#
# CPU only. k-means, BPE and every metric are CPU work, and each cell takes a
# couple of minutes, so this is far cheaper than any GPU allocation. K and
# min_frequency interact -- K sets quantisation granularity, min_frequency sets
# how hard BPE merges the resulting runs -- so aggregate the whole grid rather
# than picking the best cell.
#
#   TASK=SwingXtimes sbatch slurm/sweep_kmeans.sbatch
#   then: python -m eventtok.scripts.report_sweep --task SwingXtimes

set -uo pipefail

CACHE=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache
PY=/n/lab_storage/ydu_lab/wfy/.conda/envs/robomme/bin/python
REPO=${REPO:-/n/home04/wfy/repos/eventtok_mem}
TASK=${TASK:-SwingXtimes}

mkdir -p "${CACHE}/slurm" "${CACHE}/eval"
cd "${REPO}"

CLUSTERS=(16 32 64 128)
MINFREQ=(5 10 15 25 40)

I=${SLURM_ARRAY_TASK_ID:-0}
K=${CLUSTERS[$(( I / ${#MINFREQ[@]} ))]}
MF=${MINFREQ[$(( I % ${#MINFREQ[@]} ))]}

echo "=== $(date -Is) task=${TASK} cell ${I}: K=${K} min_frequency=${MF} ==="

# Each array task appends one row to a shared JSONL, keyed by the grid position
# so a partial sweep is still readable.
"${PY}" -m eventtok.scripts.sweep_kmeans \
    --task "${TASK}" \
    --clusters "${K}" \
    --min-frequency "${MF}" \
    --out "${CACHE}/eval/sweep_kmeans_${TASK}.jsonl"

echo "=== $(date -Is) done ==="
