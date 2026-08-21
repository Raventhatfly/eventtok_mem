#!/bin/bash
#SBATCH --job-name=eventtok
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm/%x_%j.out
# Nothing writes to /n/home04 (81% full); cache and logs go to netscratch.

set -euo pipefail
mkdir -p /n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/slurm
PY=/n/lab_storage/ydu_lab/wfy/.conda/envs/robomme/bin/python
cd /n/home04/wfy/repos/eventtok_mem

TASK=${TASK:-SwingXtimes}
LEVELS=${LEVELS:-"8 8"}
EPOCHS=${EPOCHS:-20}

echo "task=$TASK levels=$LEVELS epochs=$EPOCHS"
$PY -m eventtok.scripts.train_tokenizer \
    --task "$TASK" --levels $LEVELS --epochs "$EPOCHS" \
    --batch 512 --workers 8 --stride 1

CKPT=/n/netscratch/ydu_lab/Lab/wfy/eventtok_cache/ckpt/tokenizer_${TASK}_$(echo $LEVELS | tr ' ' 'x').pt
$PY -m eventtok.scripts.build_events --ckpt "$CKPT" --task "$TASK"
