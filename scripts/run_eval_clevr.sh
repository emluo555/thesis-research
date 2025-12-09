#!/bin/bash


cd vis_tam

# NON-THINKING
python eval_clevr.py \
    --save_dir ../sample_data/eval_clvr_test \
    --model_path "/scratch/gpfs/ZHUANGL/el5267/cache/huggingface/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b" \
    --model_name "Qwen3-VL-8B-Instruct" \
    --dataset_path "/scratch/gpfs/ZHUANGL/el5267/CLEVR/CLEVR_CoGenT_v1.0/questions/CLEVR_valA_questions.json" \
    --data_range "0,1"

 # THINKING
# python eval_clevr.py \
#     --save_dir ../sample_data/eval_clvr_cot_2 \
#     --model_path "/scratch/gpfs/ZHUANGL/el5267/cache/huggingface/models--Qwen--Qwen3-VL-8B-Thinking/snapshots/41ea130ce6eaaf7829c72dfc0e4597d49741ed18" \
#     --model_name "Qwen3-VL-8B-Thinking" \
#     --dataset_path "/scratch/gpfs/ZHUANGL/el5267/CLEVR/CLEVR_CoGenT_v1.0/questions/CLEVR_valA_questions.json" \
#     --data_range "0,1"