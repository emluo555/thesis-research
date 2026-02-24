"""
eval_clevr.py - Batch CLEVR evaluation extracting both attention and TAM values.

Loads a CLEVR questions JSON, iterates over each item, runs a single-pass
combined extraction (attention + TAM) per item, and stores results as:
    - metrics_{model_label}.json  (per-item records with both attention and TAM ratios)
    - all_token_data_{model_label}.csv  (long-format per-token, per-layer data)

Image directory is derived from the JSON path following CLEVR conventions.

Usage:
    python eval_clevr.py \\
        --dataset_path /path/to/CLEVR_valA_questions.json \\
        --save_dir ../eval_clevr_results/combined \\
        --model_instruct /path/to/Qwen3-VL-8B-Instruct \\
        --data_range "0,5" \\
        --local_files_only
"""

import csv
import gc
import os
import json
import random
import argparse
import traceback
import torch
import tqdm

from combined_analyzer import (
    CombinedAnalyzer,
    CombinedResult,
    extract_generated_answer,
)


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------

def load_model_and_processor(model_name: str, local_files_only: bool = False):
    """Load a Qwen3-VL model with eager attention for combined extraction."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    print(f"[Loading] {model_name} (attn_implementation='eager')...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        local_files_only=local_files_only,
    )
    processor = AutoProcessor.from_pretrained(
        model_name, local_files_only=local_files_only,
    )
    model.eval()
    print(f"[Loaded] {model_name} "
          f"({model.config.text_config.num_hidden_layers} layers, "
          f"{model.config.text_config.num_attention_heads} heads)")
    return model, processor


# ------------------------------------------------------------------
# Input preparation
# ------------------------------------------------------------------

def prepare_inputs(
    processor, image_path: str, prompt: str,
    device: str = "cuda", enable_thinking: bool = False,
):
    from PIL import Image

    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        **({"chat_template_kwargs": {"enable_thinking": True}}
           if enable_thinking else {}),
    )
    inputs.pop("token_type_ids", None)
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    return inputs, image


# ------------------------------------------------------------------
# Dataset helpers
# ------------------------------------------------------------------

def load_dataset(dataset_path, random_n=None, data_range=None):
    with open(dataset_path, "r") as f:
        data_json = json.load(f)

    dataset = data_json["questions"]
    if random_n is not None:
        return random.sample(dataset, random_n)
    if data_range is not None:
        start, end = data_range
        return dataset[start:end]
    return dataset


def derive_image_dir(dataset_path):
    dataset_root = os.path.abspath(
        os.path.join(os.path.dirname(dataset_path), "..")
    )
    dataset_type = os.path.basename(dataset_path).split("_")[1]
    return os.path.join(dataset_root, "images", dataset_type)


# ------------------------------------------------------------------
# Memory helpers
# ------------------------------------------------------------------

def _flush_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _save_json_checkpoint(save_dir, model_label, results_dict):
    """Write an intermediate JSON checkpoint so progress survives crashes."""
    out_path = os.path.join(save_dir, f"metrics_{model_label}.json")
    with open(out_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"[Checkpoint] {out_path}  ({len(results_dict)} items)")


def _load_previous_keys(save_dir, model_label):
    path = os.path.join(save_dir, f"metrics_{model_label}.json")
    if not os.path.isfile(path):
        return set()
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return {k for k, v in data.items() if "error" not in v}
    except (json.JSONDecodeError, KeyError):
        return set()


# ------------------------------------------------------------------
# Per-item record helpers
# ------------------------------------------------------------------

def _build_item_record(
    item, prompt,
    generated_answer="", thinking_trace="",
    error=None,
):
    record = {
        "prompt": prompt,
        "correct_answer": item.get("answer", ""),
        "generated_answer": generated_answer,
        "thinking_trace": thinking_trace,
    }
    if error is not None:
        record["error"] = error
    return record


CSV_HEADER = [
    "item", "token_idx", "token", "is_answer", "layer",
    "attn_system", "attn_image", "attn_prompt", "attn_generated",
    "tam_system", "tam_image", "tam_prompt", "tam_generated",
]


def _per_token_to_csv_rows(item_key, attn_data, tam_data):
    """Yield long-format CSV rows combining attention and TAM per-token data.

    Each row: (item, token_idx, token, is_answer, layer,
               attn_system, attn_image, attn_prompt, attn_generated,
               tam_system, tam_image, tam_prompt, tam_generated)
    """
    attn_m = attn_data["all_per_token_metrics"]
    tam_m = tam_data["all_per_token_metrics"]

    tokens = attn_m["generated_tokens"]
    attn_sys = attn_m["system_attn_per_token"]
    attn_img = attn_m["image_attn_per_token"]
    attn_prompt = attn_m["prompt_attn_per_token"]
    attn_gen = attn_m["generated_attn_per_token"]

    tam_sys = tam_m["system_tam_per_token"]
    tam_img = tam_m["image_tam_per_token"]
    tam_prompt = tam_m["prompt_tam_per_token"]
    tam_gen = tam_m["generated_tam_per_token"]

    ans_off = attn_data["answer_token_offset"]
    ans_cnt = attn_data["answer_token_count"]

    for t_idx, token_str in enumerate(tokens):
        if t_idx >= len(attn_img) or t_idx >= len(tam_img):
            break
        is_answer = 1 if ans_off <= t_idx < ans_off + ans_cnt else 0
        num_layers = len(attn_img[t_idx])
        for layer in range(num_layers):
            tam_layer = min(layer, len(tam_img[t_idx]) - 1)
            yield (
                item_key,
                t_idx,
                token_str,
                is_answer,
                layer,
                f"{attn_sys[t_idx][layer]:.6f}",
                f"{attn_img[t_idx][layer]:.6f}",
                f"{attn_prompt[t_idx][layer]:.6f}",
                f"{attn_gen[t_idx][layer]:.6f}",
                f"{tam_sys[t_idx][tam_layer]:.6f}",
                f"{tam_img[t_idx][tam_layer]:.6f}",
                f"{tam_prompt[t_idx][tam_layer]:.6f}",
                f"{tam_gen[t_idx][tam_layer]:.6f}",
            )


# ------------------------------------------------------------------
# Prompt template
# ------------------------------------------------------------------

DEFAULT_PROMPT_TEMPLATE = (
    "{question} Please respond with ONE word or number, you must wrap "
    "your final answer with <answer> and </answer> tags."
)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Combined attention + TAM evaluation over CLEVR.",
    )

    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--data_range", type=str, default=None,
                        help='Subset range as "start,end".')
    parser.add_argument("--random_n", type=int, default=None)
    parser.add_argument("--prompt_template", type=str,
                        default=DEFAULT_PROMPT_TEMPLATE)

    parser.add_argument("--model_instruct", type=str,
                        default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--model_thinking", type=str, default="")

    parser.add_argument("--mode", type=str, default="both",
                        choices=["instruct", "thinking", "both"],
                        help="Which model(s) to run.")
    parser.add_argument("--layers", type=str, default="",
                        help="Comma-separated layer indices (empty = all).")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--checkpoint_every", type=int, default=100,
                        help="Save JSON checkpoint every N items.")

    args = parser.parse_args()

    # ---- Parse subset ----
    data_range = None
    if args.data_range:
        start, end = map(int, args.data_range.split(","))
        data_range = (start, end)

    data = load_dataset(args.dataset_path, random_n=args.random_n,
                        data_range=data_range)
    img_dir = derive_image_dir(args.dataset_path)
    data_start = data_range[0] if data_range is not None else 0

    print(f"[Dataset] {len(data)} items from {args.dataset_path}")
    print(f"[Dataset] Image dir: {img_dir}")
    if data_start:
        print(f"[Dataset] Index offset: {data_start}")

    selected_layers = None
    if args.layers:
        selected_layers = [int(x.strip()) for x in args.layers.split(",")]
        print(f"[Config] Selected layers: {selected_layers}")

    run_instruct = args.mode in ("instruct", "both") and bool(args.model_instruct)
    run_thinking = args.mode in ("thinking", "both") and bool(args.model_thinking)

    os.makedirs(args.save_dir, exist_ok=True)

    failed_items: list[tuple[int, str]] = []
    consolidated: dict[str, dict] = {}

    # ---- Run Instruct model ----
    if run_instruct:
        print(f"\n[Instruct] Loading model: {args.model_instruct}")
        model, processor = load_model_and_processor(
            args.model_instruct, args.local_files_only,
        )
        analyzer = CombinedAnalyzer(model, processor)

        prev_keys = _load_previous_keys(args.save_dir, "instruct")
        instruct_results = {}

        csv_path = os.path.join(args.save_dir, "all_token_data_instruct.csv")
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(CSV_HEADER)

        for idx, item in enumerate(tqdm.tqdm(data, desc="Instruct")):
            real_idx = data_start + idx
            item_key = f"item_{real_idx}"
            prompt = args.prompt_template.format(question=item["question"])

            if item_key in prev_keys:
                print(f"[Instruct] Skipping {item_key} (already processed)")
                instruct_results[item_key] = _build_item_record(item, prompt)
                continue

            result = inputs = image = attn_data = tam_data = None
            try:
                image_path = os.path.join(img_dir, item["image_filename"])
                inputs, image = prepare_inputs(processor, image_path, prompt)

                result = analyzer.extract(
                    inputs,
                    max_new_tokens=args.max_new_tokens,
                    selected_layers=selected_layers,
                )

                attn_data = CombinedAnalyzer.compute_attention_ratios(result)
                tam_data = CombinedAnalyzer.compute_tam_ratios(result)

                gen_answer, think_trace = extract_generated_answer(
                    result.full_token_ids, result.token_regions,
                    processor.tokenizer,
                )

                instruct_results[item_key] = _build_item_record(
                    item, prompt, gen_answer, think_trace,
                )

                csv_writer.writerows(
                    _per_token_to_csv_rows(item_key, attn_data, tam_data)
                )

            except Exception as exc:
                print(f"[Instruct] ERROR on {item_key}: {exc}")
                traceback.print_exc()
                failed_items.append((real_idx, f"instruct: {exc}"))
                instruct_results[item_key] = _build_item_record(
                    item, prompt, error=str(exc),
                )
            finally:
                del result, inputs, image, attn_data, tam_data
                _flush_memory()

            if (idx + 1) % args.checkpoint_every == 0:
                _save_json_checkpoint(
                    args.save_dir, "instruct", instruct_results,
                )

        csv_file.close()
        print(f"[Saved] {csv_path}")
        consolidated["instruct"] = instruct_results

        del model, processor, analyzer
        _flush_memory()
        print("[Memory] Instruct model unloaded, CUDA cache cleared.\n")

    # ---- Run Thinking model ----
    if run_thinking:
        print(f"\n[Thinking] Loading model: {args.model_thinking}")
        model, processor = load_model_and_processor(
            args.model_thinking, args.local_files_only,
        )
        analyzer = CombinedAnalyzer(model, processor)

        prev_keys = _load_previous_keys(args.save_dir, "thinking")
        thinking_results = {}

        csv_path = os.path.join(args.save_dir, "all_token_data_thinking.csv")
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(CSV_HEADER)

        for idx, item in enumerate(tqdm.tqdm(data, desc="Thinking")):
            real_idx = data_start + idx
            item_key = f"item_{real_idx}"
            prompt = args.prompt_template.format(question=item["question"])

            if item_key in prev_keys:
                print(f"[Thinking] Skipping {item_key} (already processed)")
                thinking_results[item_key] = _build_item_record(item, prompt)
                continue

            result = inputs = image = attn_data = tam_data = None
            try:
                image_path = os.path.join(img_dir, item["image_filename"])
                inputs, image = prepare_inputs(
                    processor, image_path, prompt, enable_thinking=True,
                )

                result = analyzer.extract(
                    inputs,
                    max_new_tokens=args.max_new_tokens,
                    selected_layers=selected_layers,
                )

                attn_data = CombinedAnalyzer.compute_attention_ratios(result)
                tam_data = CombinedAnalyzer.compute_tam_ratios(result)

                gen_answer, think_trace = extract_generated_answer(
                    result.full_token_ids, result.token_regions,
                    processor.tokenizer,
                )

                thinking_results[item_key] = _build_item_record(
                    item, prompt, gen_answer, think_trace,
                )

                csv_writer.writerows(
                    _per_token_to_csv_rows(item_key, attn_data, tam_data)
                )

            except Exception as exc:
                print(f"[Thinking] ERROR on {item_key}: {exc}")
                traceback.print_exc()
                failed_items.append((real_idx, f"thinking: {exc}"))
                thinking_results[item_key] = _build_item_record(
                    item, prompt, error=str(exc),
                )
            finally:
                del result, inputs, image, attn_data, tam_data
                _flush_memory()

            if (idx + 1) % args.checkpoint_every == 0:
                _save_json_checkpoint(
                    args.save_dir, "thinking", thinking_results,
                )

        csv_file.close()
        print(f"[Saved] {csv_path}")
        consolidated["thinking"] = thinking_results

        del model, processor, analyzer
        _flush_memory()
        print("[Memory] Thinking model unloaded, CUDA cache cleared.\n")

    # ---- Write consolidated JSON ----
    for model_label, results_dict in consolidated.items():
        out_path = os.path.join(args.save_dir, f"metrics_{model_label}.json")
        with open(out_path, "w") as f:
            json.dump(results_dict, f, indent=2)
        print(f"[Saved] {out_path}  ({len(results_dict)} items)")

    # ---- Summary ----
    if failed_items:
        print(f"\n[Warning] {len(failed_items)} item(s) failed:")
        for item_idx, reason in failed_items:
            print(f"  item {item_idx}: {reason}")

    print(f"\n[Done] Results for {len(data)} items saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
