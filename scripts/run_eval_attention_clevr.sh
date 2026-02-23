#!/bin/bash
#
# Run causal attention analysis over a CLEVR dataset.
#
# Usage:
#   bash scripts/run_eval_attention_clevr.sh [dataset_path]
#
# The dataset path can be passed as:
#   1. First positional argument
#   2. DATASET_PATH environment variable
#   3. Falls back to the default CLEVR valA questions path

DATASET_PATH="${1:-${DATASET_PATH:-/scratch/gpfs/ZHUANGL/el5267/CLEVR/CLEVR_CoGenT_v1.0/questions/CLEVR_valA_questions.json}}"

MODEL_INSTRUCT="/scratch/gpfs/ZHUANGL/el5267/cache/huggingface/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MODEL_THINKING="/scratch/gpfs/ZHUANGL/el5267/cache/huggingface/models--Qwen--Qwen3-VL-8B-Thinking/snapshots/41ea130ce6eaaf7829c72dfc0e4597d49741ed18"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

cd vis_attention_new

# NON-THINKING
python eval_attention_clevr.py \
    --dataset_path "$DATASET_PATH" \
    --save_dir "../eval_clevr_results/eval_attention_clevr_${TIMESTAMP}" \
    --model_thinking "$MODEL_THINKING" \
    --model_instruct "$MODEL_INSTRUCT" \
    --data_range "0,5" \
    --mode both \
    --max_new_tokens 1024 \
    --local_files_only \
    --no_plots

# # THINKING
# python eval_attention_clevr.py \
#     --dataset_path "$DATASET_PATH" \
#     --save_dir ../new_vis_results/eval_attention_clevr_thinking \
#     --model_thinking "$MODEL_THINKING" \
#     --thinking_only \
#     --data_range "0,5" \
#     --mode both \
#     --max_new_tokens 512 \
#     --local_files_only \
#     --image_attn_ratio_by_layer

# # BOTH MODELS
# python eval_attention_clevr.py \
#     --dataset_path "$DATASET_PATH" \
#     --save_dir ../new_vis_results/eval_attention_clevr_both \
#     --model_instruct "$MODEL_INSTRUCT" \
#     --model_thinking "$MODEL_THINKING" \
#     --data_range "0,5" \
#     --mode both \
#     --max_new_tokens 512 \
#     --local_files_only \
#     --image_attn_ratio_by_layer

# # SUBSET OF LAYERS (reduce memory)
# python eval_attention_clevr.py \
#     --dataset_path "$DATASET_PATH" \
#     --save_dir ../new_vis_results/eval_attention_clevr_layers \
#     --model_instruct "$MODEL_INSTRUCT" \
#     --instruct_only \
#     --data_range "0,5" \
#     --layers 0,8,16,24,35 \
#     --mode both \
#     --max_new_tokens 512 \
#     --local_files_only

# # METRICS ONLY (no plots, no per-item folders)
# python eval_attention_clevr.py \
#     --dataset_path "$DATASET_PATH" \
#     --save_dir ../eval_clevr_results/eval_attention_clevr \
#     --model_instruct "$MODEL_INSTRUCT" \
#     --model_thinking "$MODEL_THINKING" \
#     --data_range "0,5" \
#     --mode both \
#     --max_new_tokens 1024 \
#     --local_files_only \
#     --no_plots
