#!/bin/bash
#
# Request an interactive Della GPU node and run the attention visualization.
#
# Usage:
#   bash scripts/interactive_vis_attention.sh
#
# This requests 2x A100 GPUs (~80 GB total VRAM) which is needed because
# attn_implementation="eager" materializes full attention matrices across
# all layers/heads during generation.
# salloc --nodes=1 --ntasks=1 --gres=gpu:2 --cpus-per-task=8 --mem=128G --time=01:00:00 --constraint=a100 --account=zhuangl

salloc \
    --nodes=1 \
    --ntasks=1 \
    --gres=gpu:2 \
    --cpus-per-task=8 \
    --mem=128G \
    --time=02:00:00 \
    --constraint=a100 \
    bash -c '
        source ~/.bashrc
        conda activate tam
        cd /scratch/gpfs/ZHUANGL/el5267/thesis-research
        echo "=== Node: $(hostname) ==="
        echo "=== GPUs: $(nvidia-smi -L) ==="
        echo ""
        bash scripts/run_vis_attention.sh
    '
