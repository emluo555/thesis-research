#!/bin/bash

FILE_NAME="clevr" # circle_radius
PROMPT="What number of matte things have the same color as the large matte block? Please respond with ONE word or number. Wrap your answer with <answer> and </answer> tags."

MODEL_INSTRUCT="/scratch/gpfs/ZHUANGL/el5267/cache/huggingface/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MODEL_THINKING="/scratch/gpfs/ZHUANGL/el5267/cache/huggingface/models--Qwen--Qwen3-VL-8B-Thinking/snapshots/41ea130ce6eaaf7829c72dfc0e4597d49741ed18"

cd vis_attention_new

# Instruct only, just the attention matrix
# python demo_attention.py \
#     --img_path ../sample_data/$FILE_NAME.png \
#     --prompt "$PROMPT" \
#     --model_thinking "$MODEL_THINKING" \
#     --model_instruct "$MODEL_INSTRUCT" \
#     --output_dir ../new_vis_results/attention_results_1_$FILE_NAME \
#     --mode both \
#     --max_new_tokens 256 \
#     --local_files_only

python demo_attention.py \
    --img_path ../sample_data/clevr.png \
    --prompt "$PROMPT" \
    --model_instruct "$MODEL_INSTRUCT" \
    --model_thinking "$MODEL_THINKING" \
    --output_dir ../new_vis_results/attention_results_2_$FILE_NAME \
    --mode both \
    --max_new_tokens 1024 \
    --local_files_only \
    --image_heatmap_gif

# # Instruct only
# python demo_attention.py \
#     --img_path ../sample_data/$FILE_NAME.jpg \
#     --prompt "$PROMPT" \
#     --model_instruct "$MODEL_INSTRUCT" \
#     --output_dir ../sample_data/attention_results_instruct_$FILE_NAME \
#     --mode both \
#     --max_new_tokens 256 \
#     --local_files_only

# # Thinking only
# python demo_attention.py \
#     --img_path ../sample_data/$FILE_NAME.jpg \
#     --prompt "$PROMPT" \
#     --model_instruct "" \
#     --model_thinking "$MODEL_THINKING" \
#     --output_dir ../sample_data/attention_results_thinking_$FILE_NAME \
#     --mode both \
#     --max_new_tokens 256 \
#     --local_files_only

# # Subset of layers (to reduce memory)
# python demo_attention.py \
#     --img_path ../sample_data/$FILE_NAME.jpg \
#     --prompt "$PROMPT" \
#     --model_instruct "$MODEL_INSTRUCT" \
#     --output_dir ../sample_data/attention_results_layers_$FILE_NAME \
#     --mode both \
#     --layers 0,8,16,24,35 \
#     --max_new_tokens 256 \
#     --local_files_only
