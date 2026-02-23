"""
demo_tam.py - Entry point for TAM (Token Activation Map) analysis in VLMs.

Compares hidden-state activation patterns between reasoning (Thinking) and
non-reasoning (Instruct) Qwen3-VL models, mirroring demo_attention.py.

Two analysis modes:
    Mode A (deep):    Deep inspection of a single token's activation.
    Mode B (summary): Generation-wide overview of activation evolution.
    Mode "both":      Both A and B (default).

Model-type flags:
    --instruct_only   Run only the Instruct model (skip Thinking & comparative).
    --thinking_only   Run only the Thinking model (skip Instruct & comparative).

Plot toggle flags (opt-in):
    If NONE of these flags are given, ALL plots are generated (default).
    If ANY flag is given, ONLY those plots are produced.

    Per-model (deep):   --activation_bar, --image_heatmap, --layer_grid
    Per-model (summary): --activation_ratios, --top_activated, --reasoning_phase
    Per-model (structural): --activation_matrix, --entropy_by_layer,
                            --image_activation_ratio_by_layer
    Comparative:        --entropy_by_layer_comparison,
                        --image_activation_ratio_comparison,
                        --entropy_over_generation, --image_comparison, --report

Usage examples:

    # Full comparative run (both models, both modes, all plots):
    python demo_tam.py \\
        --img_path image.jpg --prompt "Describe this image." \\
        --model_instruct Qwen/Qwen3-VL-8B-Instruct \\
        --model_thinking Qwen/Qwen3-VL-8B-Thinking \\
        --output_dir tam_results/

    # Instruct only, generate only the image activation ratio plot:
    python demo_tam.py \\
        --img_path image.jpg --prompt "Describe this image." \\
        --model_instruct Qwen/Qwen3-VL-8B-Instruct \\
        --instruct_only --image_activation_ratio_by_layer

    # Both models, only entropy plots:
    python demo_tam.py \\
        --img_path image.jpg --prompt "Describe this image." \\
        --model_instruct Qwen/Qwen3-VL-8B-Instruct \\
        --model_thinking Qwen/Qwen3-VL-8B-Thinking \\
        --entropy_by_layer --entropy_by_layer_comparison
"""

import os
import json
import argparse
import torch
import numpy as np
from PIL import Image

from tam_analyzer import TAMAnalyzer, TAMResult
import tam_vis as tvis


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------

def load_model_and_processor(model_name: str, local_files_only: bool = False):
    """Load a Qwen3-VL model for TAM extraction.

    Unlike the attention pipeline, TAM does NOT require
    attn_implementation='eager' -- it uses hidden states, not attention
    weights.  This means Flash Attention or SDPA can be used, which is
    faster and more memory-efficient.
    """
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    print(f"[Loading] {model_name}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=local_files_only,
    )
    processor = AutoProcessor.from_pretrained(
        model_name, local_files_only=local_files_only
    )
    model.eval()
    print(f"[Loaded] {model_name} "
          f"({model.config.text_config.num_hidden_layers} layers)")
    return model, processor


# ------------------------------------------------------------------
# Input preparation
# ------------------------------------------------------------------

def prepare_inputs(
    processor, image_path: str, prompt: str,
    device: str = "cuda", enable_thinking: bool = False,
):
    """Prepare model inputs from an image path and text prompt.

    Uses processor.apply_chat_template() for image loading, resizing,
    and tokenization in one step.

    Args:
        enable_thinking: When True, passes enable_thinking=True to the chat
            template so that thinking models generate explicit <think>...</think>
            tags around their reasoning.
    """
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
        **( {"chat_template_kwargs": {"enable_thinking": True}}
            if enable_thinking else {} ),
    )
    inputs.pop("token_type_ids", None)
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    return inputs, image


# ------------------------------------------------------------------
# Single-model analysis
# ------------------------------------------------------------------

def run_analysis(
    model,
    processor,
    inputs,
    image: Image.Image,
    model_label: str,
    output_dir: str,
    mode: str = "both",
    token_idx: int = -1,
    max_new_tokens: int = 256,
    enabled_plots: set = None,
):
    """Run full TAM analysis for a single model.

    Args:
        enabled_plots: Set of plot names to generate.  ``None`` means all
            plots are enabled.  When a non-empty set is passed only those
            plots are produced.

    Returns:
        result: TAMResult
        metrics: dict of quantitative metrics
        summary: list of per-token summary dicts (Mode B)
    """
    def _want(name: str) -> bool:
        """Return True if the plot *name* should be generated."""
        return enabled_plots is None or name in enabled_plots

    model_dir = os.path.join(output_dir, model_label)
    os.makedirs(model_dir, exist_ok=True)

    image_np = np.array(image)

    # --- Extract TAM scores ---
    analyzer = TAMAnalyzer(model, processor)
    result = analyzer.extract(inputs, max_new_tokens=max_new_tokens)

    # Resolve token_idx to gen_idx — default to last generated answer
    # token that isn't <|im_end|> or another special/EOS token.
    if token_idx == -1:
        r = result.token_regions
        tokenizer = processor.tokenizer
        eos_ids = set()
        if tokenizer.eos_token_id is not None:
            eos_ids.add(tokenizer.eos_token_id)
        im_end_id = analyzer._get_token_id("<|im_end|>")
        if im_end_id is not None:
            eos_ids.add(im_end_id)

        # Start from the answer region if available, otherwise generation start
        answer_offset = (
            (r.answer_start - result.input_len)
            if r.answer_start is not None else 0
        )
        # Walk backward from the last generated token to find the last
        # non-special token in the answer
        gen_idx = result.num_generated - 1
        while gen_idx >= answer_offset:
            abs_idx = result.input_len + gen_idx
            if result.full_token_ids[abs_idx] not in eos_ids:
                break
            gen_idx -= 1
        gen_idx = max(gen_idx, answer_offset)
    elif token_idx < 0:
        gen_idx = result.num_generated + token_idx
    elif token_idx >= result.input_len:
        gen_idx = token_idx - result.input_len
    else:
        gen_idx = 0
    gen_idx = max(0, min(gen_idx, result.num_generated - 1))
    abs_token_idx = result.input_len + gen_idx

    print(f"\n[{model_label}] Generated text: "
          + processor.tokenizer.decode(
              result.full_token_ids[result.token_regions.generation_start:],
              skip_special_tokens=True
          )[:200] + "...")

    # --- Compute metrics ---
    print(f"[{model_label}] Computing metrics...")
    metrics = analyzer.compute_metrics(result)

    # --- Mode A: Deep single-token inspection ---
    summary = None
    if mode in ("deep", "both"):
        print(f"[{model_label}] Mode A: Deep inspection of generated token "
              f"{gen_idx} (absolute {abs_token_idx}, "
              f"'{result.token_strings[abs_token_idx]}')")

        activation = analyzer.get_activation_for_token(result, gen_idx)
        avg_full = activation["avg_full"]
        per_layer = activation["per_layer"]
        r = result.token_regions

        # Bar chart
        if _want("activation_bar"):
            tvis.plot_activation_bar(
                avg_full,
                result.token_strings,
                result.token_regions,
                abs_token_idx,
                title=model_label,
                save_path=os.path.join(model_dir, "deep_activation_bar.png"),
            )

        # Image heatmap
        if _want("image_heatmap") and result.vision_shape is not None:
            img_attn = avg_full[r.image_start:r.image_end]
            tvis.plot_activation_image_heatmap(
                image_np, img_attn, result.vision_shape,
                abs_token_idx,
                result.token_strings[abs_token_idx],
                title=model_label,
                save_path=os.path.join(model_dir, "deep_image_heatmap.png"),
            )

        # Image heatmap GIF over generated tokens
        if _want("image_heatmap_gif") and result.vision_shape is not None:
            tvis.generate_image_heatmap_gif(
                image_np,
                result.raw_scores,
                result.num_layers,
                result.token_regions,
                result.token_strings,
                result.vision_shape,
                result.input_len,
                result.num_generated,
                title=model_label,
                save_path=os.path.join(
                    model_dir, "image_heatmap_evolution.gif"
                ),
            )

        # Layer grid
        if _want("layer_grid"):
            tvis.plot_activation_layer_grid(
                per_layer,
                abs_token_idx,
                result.token_regions,
                result.token_strings[abs_token_idx],
                save_path=os.path.join(model_dir, "deep_layer_grid.png"),
            )

    # --- Mode B: Generation summary ---
    if mode in ("summary", "both"):
        print(f"[{model_label}] Mode B: Generation summary...")
        summary = analyzer.compute_generation_summary(result)

        if _want("activation_ratios"):
            tvis.plot_activation_ratio_over_generation(
                summary, title=model_label,
                save_path=os.path.join(
                    model_dir, "summary_activation_ratios.png"
                ),
            )

        if _want("top_activated"):
            tvis.plot_top_activated_tokens_heatmap(
                summary,
                token_regions=result.token_regions,
                title=model_label,
                save_path=os.path.join(
                    model_dir, "summary_top_activated.png"
                ),
            )

        if (_want("reasoning_phase")
                and result.token_regions.think_start is not None):
            tvis.plot_reasoning_phase_transition(
                summary, title=model_label,
                save_path=os.path.join(
                    model_dir, "summary_reasoning_phase.png"
                ),
            )

    # --- Structural plots ---
    # Activation matrix
    if _want("activation_matrix"):
        tvis.plot_activation_matrix(
            result.raw_scores,
            result.num_layers,
            result.input_len,
            result.num_generated,
            token_labels=result.token_strings,
            token_regions=result.token_regions,
            title=model_label,
            save_path=os.path.join(model_dir, "activation_matrix.png"),
        )

    # Entropy by layer
    if _want("entropy_by_layer"):
        tvis.plot_activation_entropy_by_layer(
            metrics, title=model_label,
            save_path=os.path.join(
                model_dir, "activation_entropy_by_layer.png"
            ),
        )

    # Image activation ratio by layer
    if _want("image_activation_ratio_by_layer"):
        tvis.plot_image_activation_ratio_by_layer(
            metrics, title=model_label,
            save_path=os.path.join(
                model_dir, "image_activation_ratio_by_layer.png"
            ),
            token_idx=abs_token_idx,
            token_string=result.token_strings[abs_token_idx],
        )

    # Save metrics as JSON
    json_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            json_metrics[k] = {
                "mean": float(v.mean()), "values": v.tolist()
            }
        else:
            json_metrics[k] = v

    # Add generated text fields
    r = result.token_regions
    decode = processor.tokenizer.decode

    full_generated = decode(
        result.full_token_ids[r.generation_start:],
        skip_special_tokens=False,
    ).strip()

    import re
    answer_match = re.search(r"<answer>(.*?)</answer>", full_generated, re.DOTALL)
    if answer_match:
        json_metrics["generated_answer"] = answer_match.group(1).strip()
        json_metrics["thinking_trace"] = full_generated[:answer_match.start()]
    else:
        think_match = re.search(r"</think>", full_generated)
        if think_match:
            json_metrics["thinking_trace"] = full_generated[:think_match.start()]
            json_metrics["generated_answer"] = full_generated[think_match.end():]
        else:
            json_metrics["thinking_trace"] = ""
            json_metrics["generated_answer"] = full_generated

    for tag in ["<think>", "</think>", "<answer>", "</answer>", "<|im_end|>"]:
        json_metrics["thinking_trace"] = json_metrics["thinking_trace"].replace(tag, "")
        json_metrics["generated_answer"] = json_metrics["generated_answer"].replace(tag, "")
    json_metrics["thinking_trace"] = json_metrics["thinking_trace"].strip()
    json_metrics["generated_answer"] = json_metrics["generated_answer"].strip()

    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(json_metrics, f, indent=2)
    print(f"[{model_label}] Metrics saved: {metrics_path}")

    return result, metrics, summary


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _get_avg_image_activation(result: TAMResult, gen_idx: int) -> np.ndarray:
    """Extract layer-averaged, normalized image activation for one token."""
    r = result.token_regions
    layer_scores = [
        result.raw_scores[gen_idx][l]
        for l in range(result.num_layers)
    ]
    avg = np.mean(layer_scores, axis=0)
    s_min, s_max = avg.min(), avg.max()
    if s_max - s_min > 1e-8:
        avg = (avg - s_min) / (s_max - s_min)
    else:
        avg = np.zeros_like(avg)
    return avg[r.image_start:r.image_end]


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def _build_enabled_plots(args) -> set | None:
    """Build the set of enabled plot names from CLI flags.

    If no individual plot flags are given, returns ``None`` which means
    "generate everything".  Otherwise returns the set of names that were
    explicitly turned on.
    """
    PER_MODEL_PLOTS = [
        "activation_bar",
        "image_heatmap",
        "image_heatmap_gif",
        "layer_grid",
        "activation_ratios",
        "top_activated",
        "reasoning_phase",
        "activation_matrix",
        "entropy_by_layer",
        "image_activation_ratio_by_layer",
    ]
    COMPARATIVE_PLOTS = [
        "entropy_by_layer_comparison",
        "image_activation_ratio_comparison",
        "entropy_over_generation",
        "image_comparison",
        "report",
    ]

    all_names = PER_MODEL_PLOTS + COMPARATIVE_PLOTS
    selected = {name for name in all_names if getattr(args, name, False)}
    if not selected:
        return None
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="TAM analysis for reasoning vs non-reasoning VLMs."
    )

    # ---- General arguments ----
    parser.add_argument(
        "--img_path", type=str, required=True,
        help="Path to input image.",
    )
    parser.add_argument(
        "--prompt", type=str, required=True,
        help="Text prompt for the model.",
    )
    parser.add_argument(
        "--model_instruct", type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Instruct (non-reasoning) model name.",
    )
    parser.add_argument(
        "--model_thinking", type=str, default="",
        help="Thinking (reasoning) model name. Empty to skip.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="tam_results",
        help="Output directory for results.",
    )
    parser.add_argument(
        "--mode", type=str, default="both",
        choices=["deep", "summary", "both"],
        help="Analysis mode: deep, summary, or both.",
    )
    parser.add_argument(
        "--token_idx", type=int, default=-1,
        help="Token index for deep inspection (-1 = first answer token).",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=256,
        help="Maximum tokens to generate.",
    )
    parser.add_argument(
        "--local_files_only", action="store_true",
        help="Only use locally cached models.",
    )

    # ---- Model-type selection flags ----
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--instruct_only", action="store_true",
        help="Run only the Instruct model (skip Thinking and comparative).",
    )
    model_group.add_argument(
        "--thinking_only", action="store_true",
        help="Run only the Thinking model (skip Instruct and comparative).",
    )

    # ---- Per-plot toggle flags ----
    plot_group = parser.add_argument_group(
        "Plot toggles",
        "Toggle individual plots on. If NONE of these flags are given, ALL "
        "plots are generated (default). If ANY flag is given, ONLY those "
        "plots are produced.",
    )
    # Deep (Mode A) plots
    plot_group.add_argument("--activation_bar", action="store_true",
                            help="Deep activation bar chart.")
    plot_group.add_argument("--image_heatmap", action="store_true",
                            help="Deep image activation heatmap overlay.")
    plot_group.add_argument("--image_heatmap_gif", action="store_true",
                            help="GIF of image heatmap over generated tokens.")
    plot_group.add_argument("--layer_grid", action="store_true",
                            help="Deep per-layer activation grid.")
    # Summary (Mode B) plots
    plot_group.add_argument("--activation_ratios", action="store_true",
                            help="Activation ratio stacked area chart.")
    plot_group.add_argument("--top_activated", action="store_true",
                            help="Top-activated tokens heatmap.")
    plot_group.add_argument("--reasoning_phase", action="store_true",
                            help="Reasoning phase transition chart.")
    # Structural plots
    plot_group.add_argument("--activation_matrix", action="store_true",
                            help="Activation matrix heatmap.")
    plot_group.add_argument("--entropy_by_layer", action="store_true",
                            help="Activation entropy by layer line plot.")
    plot_group.add_argument("--image_activation_ratio_by_layer",
                            action="store_true",
                            help="Image activation ratio by layer plot.")
    # Comparative plots
    plot_group.add_argument("--entropy_by_layer_comparison",
                            action="store_true",
                            help="Comparative entropy by layer.")
    plot_group.add_argument("--image_activation_ratio_comparison",
                            action="store_true",
                            help="Comparative image activation ratio by layer.")
    plot_group.add_argument("--entropy_over_generation", action="store_true",
                            help="Comparative entropy over generation.")
    plot_group.add_argument("--image_comparison", action="store_true",
                            help="Comparative side-by-side image heatmap.")
    plot_group.add_argument("--report", action="store_true",
                            help="HTML comparative report.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine which plots to generate
    enabled_plots = _build_enabled_plots(args)
    if enabled_plots is not None:
        print(f"[Config] Enabled plots: {sorted(enabled_plots)}")
    else:
        print("[Config] All plots enabled (no individual plot flags given)")

    # Helper for comparative plot checks
    def _want_comp(name: str) -> bool:
        return enabled_plots is None or name in enabled_plots

    # Resolve model-type flags
    run_instruct = bool(args.model_instruct) and not args.thinking_only
    run_thinking = bool(args.model_thinking) and not args.instruct_only

    print(f"[Config] Mode: {args.mode}")
    print(f"[Config] Image: {args.img_path}")
    print(f"[Config] Prompt: {args.prompt}")
    if args.instruct_only:
        print("[Config] --instruct_only: skipping Thinking model and comparative")
    elif args.thinking_only:
        print("[Config] --thinking_only: skipping Instruct model and comparative")

    # ---- Run Instruct model ----
    results_instruct = None
    if run_instruct:
        model_i, proc_i = load_model_and_processor(
            args.model_instruct, args.local_files_only
        )
        inputs_i, image = prepare_inputs(proc_i, args.img_path, args.prompt)

        results_instruct = run_analysis(
            model_i, proc_i, inputs_i, image,
            model_label="instruct",
            output_dir=args.output_dir,
            mode=args.mode,
            token_idx=args.token_idx,
            max_new_tokens=args.max_new_tokens,
            enabled_plots=enabled_plots,
        )

        del model_i, proc_i, inputs_i
        torch.cuda.empty_cache()
        print("[Memory] Instruct model unloaded, CUDA cache cleared.\n")

    # ---- Run Thinking model ----
    results_thinking = None
    if run_thinking:
        model_t, proc_t = load_model_and_processor(
            args.model_thinking, args.local_files_only
        )
        inputs_t, image = prepare_inputs(
            proc_t, args.img_path, args.prompt, enable_thinking=True,
        )

        results_thinking = run_analysis(
            model_t, proc_t, inputs_t, image,
            model_label="thinking",
            output_dir=args.output_dir,
            mode=args.mode,
            token_idx=args.token_idx,
            max_new_tokens=args.max_new_tokens,
            enabled_plots=enabled_plots,
        )

        del model_t, proc_t, inputs_t
        torch.cuda.empty_cache()
        print("[Memory] Thinking model unloaded, CUDA cache cleared.\n")

    # ---- Comparative plots ----
    run_comparative = (
        results_instruct and results_thinking
        and not args.instruct_only and not args.thinking_only
    )

    if run_comparative:
        print("[Comparative] Generating comparison plots...")
        image_np = np.array(image)
        comp_dir = os.path.join(args.output_dir, "comparative")
        os.makedirs(comp_dir, exist_ok=True)

        result_i, metrics_i, summary_i = results_instruct
        result_t, metrics_t, summary_t = results_thinking

        # Entropy by layer comparison
        if _want_comp("entropy_by_layer_comparison"):
            tvis.plot_activation_entropy_by_layer(
                metrics_i, metrics_t,
                title="Instruct vs Thinking",
                save_path=os.path.join(
                    comp_dir, "activation_entropy_comparison.png"
                ),
            )

        # Image activation ratio comparison
        if _want_comp("image_activation_ratio_comparison"):
            tvis.plot_image_activation_ratio_by_layer(
                metrics_i, metrics_t,
                title="Instruct vs Thinking",
                save_path=os.path.join(
                    comp_dir, "image_activation_ratio_comparison.png"
                ),
            )

        # Entropy over generation comparison
        if _want_comp("entropy_over_generation") and summary_i and summary_t:
            tvis.plot_activation_entropy_over_generation(
                summary_i, summary_t,
                title="Instruct vs Thinking",
                save_path=os.path.join(
                    comp_dir, "activation_entropy_over_generation.png"
                ),
            )

        # Side-by-side image heatmap comparison
        if _want_comp("image_comparison") and result_i.vision_shape:
            gen_idx_i = min(
                result_i.num_generated - 1,
                args.token_idx if args.token_idx >= 0
                else result_i.num_generated - 1,
            )
            gen_idx_t = min(
                result_t.num_generated - 1,
                args.token_idx if args.token_idx >= 0
                else result_t.num_generated - 1,
            )

            img_act_i = _get_avg_image_activation(result_i, gen_idx_i)
            img_act_t = _get_avg_image_activation(result_t, gen_idx_t)

            tvis.plot_activation_image_comparison(
                img_act_i, img_act_t,
                image_np, result_i.vision_shape,
                title="Instruct vs Thinking",
                save_path=os.path.join(
                    comp_dir, "activation_image_comparison.png"
                ),
            )

        # HTML report
        if _want_comp("report"):
            tvis.save_comparative_report(
                args.output_dir,
                image_path=args.img_path,
                prompt=args.prompt,
                model_instruct_name=args.model_instruct,
                model_thinking_name=args.model_thinking,
                metrics_instruct=metrics_i,
                metrics_thinking=metrics_t,
                summary_instruct=summary_i,
                summary_thinking=summary_t,
            )

    elif results_instruct or results_thinking:
        if _want_comp("report"):
            result, metrics, summary = results_instruct or results_thinking
            label = "instruct" if results_instruct else "thinking"
            model_name = (
                args.model_instruct if results_instruct
                else args.model_thinking
            )

            tvis.save_comparative_report(
                args.output_dir,
                image_path=args.img_path,
                prompt=args.prompt,
                model_instruct_name=model_name if results_instruct else "",
                model_thinking_name=model_name if results_thinking else "",
                metrics_instruct=metrics if results_instruct else None,
                metrics_thinking=metrics if results_thinking else None,
                summary_instruct=summary if results_instruct else None,
                summary_thinking=summary if results_thinking else None,
            )

    print(f"\n[Done] Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
