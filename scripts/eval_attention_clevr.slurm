#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=/scratch/gpfs/ZHUANGL/el5267/thesis-research/slrm_log/eval_attn_%j.out
#SBATCH --error=/scratch/gpfs/ZHUANGL/el5267/thesis-research/slrm_log/eval_attn_%j.err
#SBATCH --constraint=a100
#SBATCH --account=zhuangl
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=el5267@princeton.edu

source ~/.bashrc
conda activate tam3
cd /scratch/gpfs/ZHUANGL/el5267/thesis-research/

DATASET_PATH="${1:-${DATASET_PATH:-/scratch/gpfs/ZHUANGL/el5267/CLEVR/CLEVR_CoGenT_v1.0/questions/CLEVR_valA_questions.json}}"

MODEL_INSTRUCT="/scratch/gpfs/ZHUANGL/el5267/cache/huggingface/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MODEL_THINKING="/scratch/gpfs/ZHUANGL/el5267/cache/huggingface/models--Qwen--Qwen3-VL-8B-Thinking/snapshots/41ea130ce6eaaf7829c72dfc0e4597d49741ed18"

cd vis_attention_new

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

python eval_attention_clevr.py \
    --dataset_path "$DATASET_PATH" \
    --save_dir "../eval_clevr_results/eval_attention_clevr_${TIMESTAMP}" \
    --model_thinking "$MODEL_THINKING" \
    --model_instruct "$MODEL_INSTRUCT" \
    --data_range "0,100" \
    --mode both \
    --max_new_tokens 1024 \
    --local_files_only \
    --no_plots