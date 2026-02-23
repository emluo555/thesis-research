#!/bin/bash

FILE_NAME="graph" # circle_radius
PROMPT="What is the difference between two consecutive major ticks on the Y-axis? Please respond with one integer"

cd vis_tam

# NON-THINKING
python run_tam.py \
    --img_path ../sample_data/$FILE_NAME.jpg \
    --prompt "$PROMPT" \
    --img_save_dir ../new_vis_results/vis_thinking_2_$FILE_NAME \
    --model_path "/scratch/gpfs/ZHUANGL/el5267/cache/huggingface/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b" \
    --model_name "Qwen3-VL-8B-Instruct"

# # THINKING
# python run_tam.py \
#     --img_path ../sample_data/$FILE_NAME.jpg \
#     --prompt "$PROMPT" \
#     --img_save_dir ../sample_data/vis_thinking_cot_2_$FILE_NAME \
#     --model_path "/scratch/gpfs/ZHUANGL/el5267/cache/huggingface/models--Qwen--Qwen3-VL-8B-Thinking/snapshots/41ea130ce6eaaf7829c72dfc0e4597d49741ed18" \
#     --model_name "Qwen3-VL-8B-Thinking"