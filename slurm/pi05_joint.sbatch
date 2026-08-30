#!/usr/bin/env bash
#SBATCH --job-name=pi05joint
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%j.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x-%j.err

# One event vocabulary for all 16 tasks. Cannot be an array job: the tasks share a fit.
#
# Per-task caches give vocabularies of 19-46 symbols, so token 7 means a different
# motion in each. A single pi0.5 trained over all 16 tasks needs one vocabulary or its
# event embedding rows are being asked to be several things at once.
#
# Writes each task's cache as it finishes, so a preempted job resumes by re-running --
# the fit is deterministic given the seed, and completed tasks are simply rewritten.

set -uo pipefail

PY=/n/lab_storage/ydu_lab/wfy/.conda/envs/robomme/bin/python
REPO=${REPO:-/n/home04/wfy/repos/eventtok_mem}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
cd "${REPO}"

echo "=== $(date -Is) joint event vocabulary ==="
"${PY}" -m eventtok.scripts.build_pi05_joint "$@"
status=$?
echo "=== $(date -Is) done exit=${status} ==="
exit ${status}
