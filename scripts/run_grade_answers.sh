#!/bin/bash
#
# Grade model-generated answers against correct answers using the OpenAI API.
#
# Usage:
#   bash scripts/run_grade_answers.sh [results_dir]
#
# The results directory (containing metrics_instruct.json / metrics_thinking.json)
# can be passed as:
#   1. First positional argument
#   2. RESULTS_DIR environment variable
#   3. Falls back to the default path below
#
# Requires OPENAI_API_KEY to be set in the environment.

RESULTS_DIR="${1:-${RESULTS_DIR:-/scratch/gpfs/ZHUANGL/el5267/thesis-research/eval_clevr_results/eval_attention_clevr_20260219_211727}}"

cd vis_attention_new

# BOTH MODES
# python grade_answers.py \
#     --instruct_path "$RESULTS_DIR/metrics_instruct.json" \
#     --thinking_path "$RESULTS_DIR/metrics_thinking.json" \
#     --output_path "$RESULTS_DIR/correctness_results.json" \
#     --model "gpt-4o-mini" \
#     --concurrency 20

# # ANALYZE: attention over generation for a specific incorrect item (save plot)
python grade_answers.py analyze \
    --csv_path "$RESULTS_DIR/all_token_attn_thinking.csv" \
    --item_index 5 \
    --save_path "$RESULTS_DIR/attn_item_5.png"

# # BOTH MODES, SUBSET OF ITEMS
# python grade_answers.py \
#     --instruct_path "$RESULTS_DIR/metrics_instruct.json" \
#     --thinking_path "$RESULTS_DIR/metrics_thinking.json" \
#     --output_path "$RESULTS_DIR/correctness_results.json" \
#     --model "gpt-4o-mini" \
#     --concurrency 20 \
#     --data_range "0,100"

# # INSTRUCT ONLY
# python grade_answers.py \
#     --instruct_path "$RESULTS_DIR/metrics_instruct.json" \
#     --output_path "$RESULTS_DIR/correctness_results.json" \
#     --model "gpt-4o-mini" \
#     --concurrency 20

# # THINKING ONLY
# python grade_answers.py \
#     --thinking_path "$RESULTS_DIR/metrics_thinking.json" \
#     --output_path "$RESULTS_DIR/correctness_results.json" \
#     --model "gpt-4o-mini" \
#     --concurrency 20



# # ANALYZE: display plot interactively (omit --save_path)
# python grade_answers.py analyze \
#     --csv_path "$RESULTS_DIR/all_token_attn_thinking.csv" \
#     --item_index 42

# REPORT: CSV of attention stats for all incorrect items (both modes)
# python grade_answers.py report \
#     --correctness_path "$RESULTS_DIR/correctness_results.json" \
#     --instruct_metrics "$RESULTS_DIR/metrics_instruct.json" \
#     --thinking_metrics "$RESULTS_DIR/metrics_thinking.json" \
#     --output_csv "$RESULTS_DIR/incorrect_attention.csv"
