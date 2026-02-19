"""
demo_attention.py - Entry point for causal attention analysis in VLMs.

Compares attention patterns between reasoning (Thinking) and non-reasoning
(Instruct) Qwen3-VL models, with two analysis modes:

    Mode A (deep):    Deep inspection of a single token's causal attention.
    Mode B (summary): Generation-wide overview of attention evolution.
    Mode "both":      Both A and B (default).

Model-type flags:
    --instruct_only   Run only the Instruct model (skip Thinking & comparative).
    --thinking_only   Run only the Thinking model (skip Instruct & comparative).

Plot toggle flags (opt-in):
    If NONE of these flags are given, ALL plots are generated (default).
    If ANY flag is given, ONLY those plots are produced.

    Per-model (deep):   --causal_bar, --image_heatmap, --layer_head_grid,
                        --rollout_to_input
    Per-model (summary): --attn_ratios, --top_attended, --reasoning_phase
    Per-model (structural): --attention_matrix, --entropy_by_layer,
                            --head_specialization, --image_attn_ratio_by_layer
    Comparative:        --entropy_by_layer_comparison, --image_attn_ratio_comparison,
                        --entropy_over_generation, --rollout_comparison, --report

Usage examples:

    # Full comparative run (both models, both modes, all plots):
    python demo_attention.py \\
        --img_path image.jpg --prompt "Describe this image." \\
        --model_instruct Qwen/Qwen3-VL-8B-Instruct \\
        --model_thinking Qwen/Qwen3-VL-8B-Thinking \\
        --output_dir attention_results/

    # Instruct only, generate only the attention matrix:
    python demo_attention.py \\
        --img_path image.jpg --prompt "Describe this image." \\
        --model_instruct Qwen/Qwen3-VL-8B-Instruct \\
        --instruct_only --attention_matrix

    # Both models, only entropy and image ratio plots:
    python demo_attention.py \\
        --img_path image.jpg --prompt "Describe this image." \\
        --model_instruct Qwen/Qwen3-VL-8B-Instruct \\
        --model_thinking Qwen/Qwen3-VL-8B-Thinking \\
        --entropy_by_layer --image_attn_ratio_by_layer \\
        --entropy_by_layer_comparison --image_attn_ratio_comparison

    # Select specific layers to reduce memory:
    python demo_attention.py \\
        --img_path image.jpg --prompt "Describe this image." \\
        --model_instruct Qwen/Qwen3-VL-8B-Instruct \\
        --layers 0,8,16,24,35
"""

import os
import json
import argparse
import torch
import numpy as np
from PIL import Image

from attention_map import AttentionAnalyzer, AttentionResult
import attention_vis as avis


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------

def load_model_and_processor(model_name: str, local_files_only: bool = False):
    """Load a Qwen3-VL model with eager attention for weight extraction.

    The model MUST use attn_implementation='eager' because Flash Attention
    and SDPA do not return attention weights.
    """
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
        model_name, local_files_only=local_files_only
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
    """Prepare model inputs from an image path and text prompt.

    Uses processor.apply_chat_template() which handles image loading,
    resizing, and tokenization in one step.

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
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

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
    selected_layers=None,
    max_new_tokens: int = 256,
    enabled_plots: set = None,
    save_metrics_json: bool = True,
):
    """Run full attention analysis for a single model.

    Args:
        enabled_plots: Set of plot names to generate.  ``None`` means all
            plots are enabled.  When a non-empty set is passed only those
            plots are produced.
        save_metrics_json: If False, skip writing the per-item metrics.json
            file to disk (the dict is still computed and returned).

    Returns:
        result: AttentionResult
        metrics: dict of quantitative metrics
        summary: list of per-token summary dicts (Mode B)
        rollout: (S, S) attention rollout matrix
        json_metrics: dict with serializable metrics + generated_answer / thinking_trace
    """
    def _want(name: str) -> bool:
        """Return True if the plot *name* should be generated."""
        return enabled_plots is None or name in enabled_plots

    model_dir = os.path.join(output_dir, model_label)
    os.makedirs(model_dir, exist_ok=True)

    # Convert PIL image to numpy for visualization
    image_np = np.array(image)

    # --- Extract attention ---
    analyzer = AttentionAnalyzer(model, processor)
    result = analyzer.extract(
        inputs,
        max_new_tokens=max_new_tokens,
        selected_layers=selected_layers,
    )

    # Resolve token_idx against the attention matrix size (the last
    # generated token has no attention row because it was never fed back
    # through the model, so total_len can exceed the matrix dimension by 1).
    seq_len = result.attention_matrices.shape[2]
    if token_idx == -1:
        r = result.token_regions
        gen_start = r.generation_start

        # Scan generated tokens for <answer>...</answer> boundaries using
        # running text, since these tags are multi-token (not special tokens).
        answer_tag_start = None  # first content token AFTER <answer>
        answer_tag_end = None    # first token OF </answer>
        running = ""
        for i in range(gen_start, min(len(result.token_strings), seq_len)):
            running += result.token_strings[i]
            if answer_tag_start is None and "<answer>" in running:
                answer_tag_start = i + 1
            elif answer_tag_start is not None and "</answer>" in running:
                # Walk back to find the first token that contributed </answer>
                tag_text = ""
                for j in range(i, answer_tag_start - 1, -1):
                    tag_text = result.token_strings[j] + tag_text
                    if "</answer>" in tag_text:
                        answer_tag_end = j
                        break
                if answer_tag_end is None:
                    answer_tag_end = i
                break

        if answer_tag_start is not None and answer_tag_end is not None:
            # Pick the last content token inside <answer>...</answer>
            token_idx = min(answer_tag_end - 1, seq_len - 1)
            token_idx = max(token_idx, answer_tag_start)
        else:
            # Fallback: walk backward from end, skipping EOS / <|im_end|>
            tokenizer = processor.tokenizer
            eos_ids = set()
            if tokenizer.eos_token_id is not None:
                eos_ids.add(tokenizer.eos_token_id)
            im_end_id = analyzer._get_token_id("<|im_end|>")
            if im_end_id is not None:
                eos_ids.add(im_end_id)

            answer_start = r.answer_start if r.answer_start is not None else gen_start
            token_idx = seq_len - 1
            while token_idx >= answer_start:
                if result.full_token_ids[token_idx] not in eos_ids:
                    break
                token_idx -= 1
            token_idx = max(token_idx, answer_start)
    elif token_idx < 0:
        token_idx = seq_len + token_idx
    token_idx = min(token_idx, seq_len - 1)

    print(f"\n[{model_label}] Generated text: "
          + processor.tokenizer.decode(
              result.full_token_ids[result.token_regions.generation_start:],
              skip_special_tokens=True
          )[:200] + "...")

    # --- Compute metrics ---
    print(f"[{model_label}] Computing metrics...")
    metrics = analyzer.compute_metrics(result)

    # --- Compute rollout ---
    print(f"[{model_label}] Computing attention rollout...")
    rollout = analyzer.attention_rollout(result.attention_matrices)
    rollout_np = rollout.numpy()

    # --- Mode A: Deep single-token inspection ---
    summary = None
    if mode in ("deep", "both"):
        print(f"[{model_label}] Mode A: Deep inspection of token {token_idx} "
              f"('{result.token_strings[token_idx]}')")

        causal = analyzer.get_causal_attention_for_token(result, token_idx)
        contribution = analyzer.compute_causal_contribution(result, token_idx)

        # Average across layers for the bar chart
        avg_row = causal["per_layer"].mean(axis=0)

        if _want("causal_bar"):
            avis.plot_causal_attention_bar(
                avg_row,
                result.token_strings,
                result.token_regions,
                token_idx,
                title=model_label,
                save_path=os.path.join(model_dir, "deep_causal_bar.png"),
            )

        # Image heatmap
        if _want("image_heatmap") and result.vision_shape is not None:
            r = result.token_regions
            img_attn = avg_row[r.image_start:r.image_end]
            avis.plot_causal_image_heatmap(
                image_np, img_attn, result.vision_shape,
                token_idx, result.token_strings[token_idx],
                title=model_label,
                save_path=os.path.join(model_dir, "deep_image_heatmap.png"),
            )

        # Image heatmap GIF over generated tokens
        if _want("image_heatmap_gif") and result.vision_shape is not None:
            avis.generate_image_heatmap_gif(
                image_np,
                result.attention_matrices,
                result,
                title=model_label,
                save_path=os.path.join(model_dir, "image_heatmap_evolution.gif"),
            )

        # Layer x head grid
        if _want("layer_head_grid"):
            avis.plot_causal_layer_head_grid(
                causal["per_layer_head"],
                token_idx,
                result.token_regions,
                result.token_strings[token_idx],
                selected_layers=selected_layers,
                save_path=os.path.join(model_dir, "deep_layer_head_grid.png"),
            )

        # Rollout to input
        if _want("rollout_to_input"):
            avis.plot_causal_rollout_to_input(
                contribution, image_np, result.vision_shape,
                result.token_strings, result.token_regions,
                token_idx,
                title=f"{model_label} (Rollout)",
                save_path=os.path.join(model_dir, "deep_rollout_to_input.png"),
            )

    # --- Mode B: Generation summary ---
    if mode in ("summary", "both"):
        print(f"[{model_label}] Mode B: Generation summary...")
        summary = analyzer.compute_generation_summary(result)

        if _want("attn_ratios"):
            avis.plot_attention_ratio_over_generation(
                summary, title=model_label,
                save_path=os.path.join(model_dir, "summary_attn_ratios.png"),
            )

        if _want("top_attended"):
            avis.plot_top_attended_tokens_heatmap(
                summary,
                token_regions=result.token_regions,
                title=model_label,
                save_path=os.path.join(model_dir, "summary_top_attended.png"),
            )

        # Reasoning phase plot (if thinking model)
        if _want("reasoning_phase") and result.token_regions.think_start is not None:
            avis.plot_reasoning_phase_transition(
                summary, title=model_label,
                save_path=os.path.join(model_dir, "summary_reasoning_phase.png"),
            )

    # --- Structural plots ---
    # Attention matrix (averaged across layers and heads)
    if _want("attention_matrix"):
        attn_avg = result.attention_matrices.float().mean(dim=(0, 1)).numpy()
        avis.plot_attention_matrix(
            attn_avg, result.token_strings,
            token_regions=result.token_regions,
            title=model_label,
            save_path=os.path.join(model_dir, "attention_matrix.png"),
        )

    # Entropy by layer
    if _want("entropy_by_layer"):
        avis.plot_entropy_by_layer(
            metrics, title=model_label,
            save_path=os.path.join(model_dir, "entropy_by_layer.png"),
        )

    # Head specialization
    if _want("head_specialization"):
        avis.plot_head_specialization(
            metrics, model_label=model_label,
            save_path=os.path.join(model_dir, "head_specialization.png"),
        )

    # Image attention ratio by layer
    if _want("image_attn_ratio_by_layer"):
        avis.plot_image_attention_ratio_by_layer(
            metrics, title=model_label,
            save_path=os.path.join(model_dir, "image_attn_ratio_by_layer.png"),
            token_idx=token_idx,
            token_string=result.token_strings[token_idx],
        )

    # Save metrics as JSON
    json_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            json_metrics[k] = {"mean": float(v.mean()), "values": v.tolist()}
        else:
            json_metrics[k] = v

    # Add generated text fields
    r = result.token_regions
    decode = processor.tokenizer.decode

    full_generated = decode(
        result.full_token_ids[r.generation_start:],
        skip_special_tokens=False,
    ).strip()

    # Extract <answer>...</answer> content; everything before it is thinking
    import re
    answer_match = re.search(r"<answer>(.*?)</answer>", full_generated, re.DOTALL)
    if answer_match:
        json_metrics["generated_answer"] = answer_match.group(1).strip()
        json_metrics["thinking_trace"] = full_generated[:answer_match.start()]
    else:
        # Fallback: try <think>...</think> boundary
        think_match = re.search(r"</think>", full_generated)
        if think_match:
            json_metrics["thinking_trace"] = full_generated[:think_match.start()]
            json_metrics["generated_answer"] = full_generated[think_match.end():]
        else:
            json_metrics["thinking_trace"] = ""
            json_metrics["generated_answer"] = full_generated

    # Clean tags from both fields
    for tag in ["<think>", "</think>", "<answer>", "</answer>", "<|im_end|>"]:
        json_metrics["thinking_trace"] = json_metrics["thinking_trace"].replace(tag, "")
        json_metrics["generated_answer"] = json_metrics["generated_answer"].replace(tag, "")
    json_metrics["thinking_trace"] = json_metrics["thinking_trace"].strip()
    json_metrics["generated_answer"] = json_metrics["generated_answer"].strip()

    if save_metrics_json:
        metrics_path = os.path.join(model_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(json_metrics, f, indent=2)
        print(f"[{model_label}] Metrics saved: {metrics_path}")

    return result, metrics, summary, rollout_np, json_metrics


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def _build_enabled_plots(args) -> set | None:
    """Build the set of enabled plot names from CLI flags.

    If no individual plot flags are given, returns ``None`` which means
    "generate everything".  Otherwise returns the set of names that were
    explicitly turned on.
    """
    # All per-model plot flag names (must match argparse dest names)
    PER_MODEL_PLOTS = [
        "causal_bar",
        "image_heatmap",
        "image_heatmap_gif",
        "layer_head_grid",
        "rollout_to_input",
        "attn_ratios",
        "top_attended",
        "reasoning_phase",
        "attention_matrix",
        "entropy_by_layer",
        "head_specialization",
        "image_attn_ratio_by_layer",
    ]
    # Comparative plot flag names
    COMPARATIVE_PLOTS = [
        "entropy_by_layer_comparison",
        "image_attn_ratio_comparison",
        "entropy_over_generation",
        "rollout_comparison",
        "report",
    ]

    all_names = PER_MODEL_PLOTS + COMPARATIVE_PLOTS

    selected = {name for name in all_names if getattr(args, name, False)}
    if not selected:
        return None  # no flags given → generate everything
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Causal attention analysis for reasoning vs non-reasoning VLMs."
    )

    # ---- General arguments ----
    parser.add_argument("--img_path", type=str, required=True,
                        help="Path to input image.")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt for the model.")
    parser.add_argument("--model_instruct", type=str,
                        default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Instruct (non-reasoning) model name.")
    parser.add_argument("--model_thinking", type=str, default="",
                        help="Thinking (reasoning) model name. Empty to skip.")
    parser.add_argument("--output_dir", type=str, default="attention_results",
                        help="Output directory for results.")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["deep", "summary", "both"],
                        help="Analysis mode: deep, summary, or both.")
    parser.add_argument("--token_idx", type=int, default=-1,
                        help="Token index for deep inspection (-1 = last).")
    parser.add_argument("--layers", type=str, default="",
                        help="Comma-separated layer indices (e.g. 0,8,16,24,35). "
                             "Empty = all layers.")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum tokens to generate.")
    parser.add_argument("--local_files_only", action="store_true",
                        help="Only use locally cached models.")

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
    # Deep (Mode A) plots
    plot_group = parser.add_argument_group(
        "Plot toggles",
        "Toggle individual plots on. If NONE of these flags are given, ALL "
        "plots are generated (default). If ANY flag is given, ONLY those "
        "plots are produced.",
    )
    plot_group.add_argument("--causal_bar", action="store_true",
                            help="Deep causal attention bar chart.")
    plot_group.add_argument("--image_heatmap", action="store_true",
                            help="Deep image attention heatmap overlay.")
    plot_group.add_argument("--image_heatmap_gif", action="store_true",
                            help="GIF of image heatmap over generated tokens.")
    plot_group.add_argument("--layer_head_grid", action="store_true",
                            help="Deep layer x head attention grid.")
    plot_group.add_argument("--rollout_to_input", action="store_true",
                            help="Deep rollout-to-input decomposition.")
    # Summary (Mode B) plots
    plot_group.add_argument("--attn_ratios", action="store_true",
                            help="Attention ratio stacked area chart.")
    plot_group.add_argument("--top_attended", action="store_true",
                            help="Top-attended tokens heatmap.")
    plot_group.add_argument("--reasoning_phase", action="store_true",
                            help="Reasoning phase transition chart.")
    # Structural plots
    plot_group.add_argument("--attention_matrix", action="store_true",
                            help="Full attention matrix heatmap.")
    plot_group.add_argument("--entropy_by_layer", action="store_true",
                            help="Entropy by layer line plot.")
    plot_group.add_argument("--head_specialization", action="store_true",
                            help="Head specialization bar chart.")
    plot_group.add_argument("--image_attn_ratio_by_layer", action="store_true",
                            help="Image attention ratio by layer line plot.")
    # Comparative plots
    plot_group.add_argument("--entropy_by_layer_comparison", action="store_true",
                            help="Comparative entropy by layer.")
    plot_group.add_argument("--image_attn_ratio_comparison", action="store_true",
                            help="Comparative image attention ratio by layer.")
    plot_group.add_argument("--entropy_over_generation", action="store_true",
                            help="Comparative entropy over generation.")
    plot_group.add_argument("--rollout_comparison", action="store_true",
                            help="Comparative rollout image overlay.")
    plot_group.add_argument("--report", action="store_true",
                            help="HTML comparative report.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Parse selected layers
    selected_layers = None
    if args.layers:
        selected_layers = [int(x.strip()) for x in args.layers.split(",")]
        print(f"[Config] Selected layers: {selected_layers}")

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
            selected_layers=selected_layers,
            max_new_tokens=args.max_new_tokens,
            enabled_plots=enabled_plots,
        )

        # Free GPU memory before loading next model
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
            selected_layers=selected_layers,
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

        result_i, metrics_i, summary_i, rollout_i, _ = results_instruct
        result_t, metrics_t, summary_t, rollout_t, _ = results_thinking

        # Entropy by layer comparison
        if _want_comp("entropy_by_layer_comparison"):
            avis.plot_entropy_by_layer(
                metrics_i, metrics_t,
                title="Instruct vs Thinking",
                save_path=os.path.join(comp_dir, "entropy_by_layer_comparison.png"),
            )

        # Image attention ratio comparison
        if _want_comp("image_attn_ratio_comparison"):
            avis.plot_image_attention_ratio_by_layer(
                metrics_i, metrics_t,
                title="Instruct vs Thinking",
                save_path=os.path.join(comp_dir, "image_attn_ratio_comparison.png"),
            )

        # Entropy over generation comparison
        if _want_comp("entropy_over_generation") and summary_i and summary_t:
            avis.plot_entropy_over_generation(
                summary_i, summary_t,
                title="Instruct vs Thinking",
                save_path=os.path.join(comp_dir, "entropy_over_generation.png"),
            )

        # Rollout comparison on image
        if _want_comp("rollout_comparison") and result_i.vision_shape:
            tok_i = args.token_idx if args.token_idx >= 0 else None
            avis.plot_rollout_comparison(
                rollout_i, rollout_t,
                image_np, result_i.vision_shape,
                result_i.token_regions, result_t.token_regions,
                tok_i, tok_i,
                title="Instruct vs Thinking",
                save_path=os.path.join(comp_dir, "rollout_comparison.png"),
            )

        # HTML report
        if _want_comp("report"):
            avis.save_comparative_report(
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
        # Single-model report
        if _want_comp("report"):
            result, metrics, summary, _, _ = results_instruct or results_thinking
            label = "instruct" if results_instruct else "thinking"
            model_name = args.model_instruct if results_instruct else args.model_thinking

            avis.save_comparative_report(
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
