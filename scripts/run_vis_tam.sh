#!/bin/bash
#
# Run TAM (Token Activation Map) analysis with both Instruct and Thinking models.
# Mirrors run_vis_attention.sh but uses hidden-state activations instead of attention.
#
# Usage:
#   bash scripts/run_vis_tam.sh

# FILE_NAME="graph"
# PROMPT="What is the difference between two consecutive major ticks on the Y-axis? Please respond with one integer"

FILE_NAME="clevr"
PROMPT="What number of matte things have the same color as the large matte block? Please respond with ONE word or number. Wrap your answer with <answer> and </answer> tags."


MODEL_INSTRUCT="/scratch/gpfs/ZHUANGL/el5267/cache/huggingface/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MODEL_THINKING="/scratch/gpfs/ZHUANGL/el5267/cache/huggingface/models--Qwen--Qwen3-VL-8B-Thinking/snapshots/41ea130ce6eaaf7829c72dfc0e4597d49741ed18"

cd vis_tam

# Both models, both modes (full comparative run)
python demo_tam.py \
    --img_path ../sample_data/$FILE_NAME.png \
    --prompt "$PROMPT" \
    --model_instruct "$MODEL_INSTRUCT" \
    --model_thinking "$MODEL_THINKING" \
    --output_dir ../new_vis_results/tam_results_2_$FILE_NAME \
    --mode both \
    --max_new_tokens 1024 \
    --local_files_only \
    --image_heatmap_gif

# # Instruct only
# python demo_tam.py \
#     --img_path ../sample_data/$FILE_NAME.jpg \
#     --prompt "$PROMPT" \
#     --model_instruct "$MODEL_INSTRUCT" \
#     --output_dir ../sample_data/tam_results_instruct_$FILE_NAME \
#     --mode both \
#     --max_new_tokens 256 \
#     --local_files_only

# # Thinking only
# python demo_tam.py \
#     --img_path ../sample_data/$FILE_NAME.jpg \
#     --prompt "$PROMPT" \
#     --model_instruct "" \
#     --model_thinking "$MODEL_THINKING" \
#     --output_dir ../sample_data/tam_results_thinking_$FILE_NAME \
#     --mode both \
#     --max_new_tokens 256 \
#     --local_files_only
