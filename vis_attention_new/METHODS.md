# Attention Analysis Methods

## Overview

This document describes the implementation and rationale behind the causal attention analysis pipeline used to study and compare attention patterns in reasoning versus non-reasoning vision-language models (VLMs). The primary research question is whether a model that engages in explicit chain-of-thought reasoning (Qwen3-VL-8B-Thinking) exhibits qualitatively or quantitatively different attention behavior — particularly with respect to visual grounding — compared to a standard instruction-following variant (Qwen3-VL-8B-Instruct) of the same architecture.

All code lives in `vis_attention_new/` and is organized into three layers:

| Module | Role |
|---|---|
| `attention_map.py` | Core extraction, region identification, and quantitative metrics |
| `attention_vis.py` | Visualization functions (all plots) |
| `demo_attention.py` | Single-image entry point; orchestrates the full pipeline |
| `eval_attention_clevr.py` | Batch evaluation over CLEVR questions dataset |

---

## 1. Models and Experimental Setup

Both models are members of the Qwen3-VL family and share the same underlying transformer architecture: a vision encoder followed by a causal language model with cross-modal token fusion. The key difference is in the generation protocol:

- **Instruct model** (`Qwen3-VL-8B-Instruct`): generates a direct answer given the image and text prompt.
- **Thinking model** (`Qwen3-VL-8B-Thinking`): when `enable_thinking=True` is passed to the chat template, the model first generates an explicit reasoning trace wrapped in `<think>...</think>` tags before producing its final answer.

Both models must be loaded with `attn_implementation="eager"` (i.e., the standard scaled dot-product implementation). Flash Attention 2 and PyTorch SDPA do not return attention weight tensors, making them incompatible with attention extraction.

Greedy decoding (`do_sample=False`) is used throughout to ensure reproducibility.

---

## 2. Token Region Segmentation

A key design decision is to treat the full token sequence — which consists of system prompt tokens, image tokens, user prompt tokens, and generated tokens — as a set of semantically distinct **regions**. Every subsequent analysis is expressed in terms of attention mass flowing between these regions.

The `TokenRegions` dataclass (`attention_map.py`) stores Python-style `[start, end)` index pairs for each region within the combined input+output token sequence:

| Region | Description | Color |
|---|---|---|
| **System** | Chat-template tokens before the image (`<\|im_start\|>`, `user`, etc.) | Grey `#888888` |
| **Image** | Vision tokens between `<\|vision_start\|>` and `<\|vision_end\|>` | Blue `#4A90D9` |
| **Prompt** | User text tokens following the image | Green `#50B848` |
| **Thinking** | Tokens inside `<think>...</think>` (reasoning model only) | Red `#E74C3C` |
| **Answer** | Generated tokens after `</think>` (reasoning model only) | Purple `#9B59B6` |
| **Generated** | All generated tokens when there is no thinking phase (instruct model) | Orange `#F39C12` |

Region boundaries are identified in `AttentionAnalyzer.get_token_regions()` by scanning the token-ID sequence for special token IDs (`vision_start_token_id`, `vision_end_token_id`) and the `<think>` / `</think>` markers. A two-strategy fallback is used for the thinking markers: first a fast token-ID lookup (valid when each marker is a single registered special token), then a slower string-based scan of the running decoded text (for models where these markers are tokenized as multi-piece sequences).

The vision token grid shape — required for spatial heatmaps — is inferred from the `image_grid_thw` tensor produced by the processor, accounting for the model's spatial merge factor (default 2).

---

## 3. Attention Extraction

Attention weights are extracted by calling `model.generate()` with `output_attentions=True` and `return_dict_in_generate=True`. This returns a nested tuple `outputs.attentions` with the following structure when `use_cache=True`:

- `attentions[0]`: the **prefill** step — a tuple of `L` tensors, each of shape `(B, H, input_len, input_len)`.
- `attentions[k]` for `k ≥ 1`: the `k`-th **decode** step — a tuple of `L` tensors, each of shape `(B, H, 1, input_len + k)`.

The `_reconstruct_attention()` method assembles these step-wise outputs into a single full causal attention tensor of shape `(L, H, total_len, total_len)` stored in fp16. Specifically:

1. A zero tensor of the target shape is allocated.
2. The prefill attention fills the upper-left `(input_len × input_len)` block (rows 0 to `input_len − 1`).
3. Each decode step contributes one row: row `input_len + k − 1` receives the `kv_len`-length attention vector from step `k`.

The result is a lower-triangular matrix (due to the causal mask) where entry `[l, h, i, j]` is the attention weight that token `i` places on token `j` at layer `l`, head `h`. To reduce memory consumption for long sequences, an optional `selected_layers` argument allows extracting only a subset of layers (e.g., every 4th layer).

---

## 4. Analysis Modes

The pipeline supports two complementary analysis modes, designed to answer different questions about model behavior.

### Mode A — Deep Single-Token Inspection

**Research question:** *When the model generates a specific token, which prior tokens does it attend to, and how does this vary across layers and heads?*

The focal token is selected automatically: when `--token_idx -1` (the default), the pipeline identifies the last content token inside the `<answer>...</answer>` span (for constrained-format prompts) or the last non-EOS generated token otherwise. This choice ensures that the inspected token is semantically meaningful — the final token of the committed answer — rather than an end-of-sequence marker.

**`get_causal_attention_for_token(result, token_idx)`** extracts row `token_idx` of the attention matrix from every layer and head, yielding:
- `per_layer_head`: shape `(L, H, token_idx + 1)` — the full per-head decomposition.
- `per_layer`: shape `(L, token_idx + 1)` — head-averaged.
- `region_breakdown`: `{region_name → (L,)}` — total attention mass directed at each region, per layer.

**Visualizations produced:**

1. **Causal attention bar chart** (`deep_causal_bar.png`): The causal row averaged across all layers, with bars colored by token region. Image tokens are aggregated into a single bar (summing their individual weights) to prevent the typically large image token count from dominating the x-axis. Individual text tokens are shown up to a maximum of 80, subsampled uniformly if necessary.

2. **Image attention heatmap** (`deep_image_heatmap.png`): The image-region slice of the causal row is reshaped to the vision token grid `(H_tok, W_tok)`, then bilinearly upsampled to the original image resolution via `cv2.resize`. The result is normalized to `[0, 1]` and rendered with the `jet` colormap (blue = low, red = high). Three panels are shown: original image, heatmap alone, and a 50/50 alpha blend overlay. This visualization reveals *where in the image* the model is looking when generating the answer token.

3. **Image heatmap evolution GIF** (`image_heatmap_evolution.gif`): The same heatmap overlay is computed for every token in the answer phase and assembled into an animated GIF, showing how spatial attention moves across the image as generation progresses.

4. **Layer × head grid** (`deep_layer_head_grid.png`): A grid of small bar charts, one cell per `(layer, head)` pair, showing the causal attention distribution without any averaging. This reveals head specialization: heads that consistently attend to image tokens, specific prompt tokens, or previously generated tokens. Layers and heads are subsampled to a maximum of 12 and 16, respectively, for readability.

5. **Attention rollout** (`deep_rollout_to_input.png`): The effective causal influence from all input tokens to the focal token, computed via multi-layer propagation (see Section 5). Shown as an image overlay and a bar chart.

### Mode B — Generation Summary

**Research question:** *How do attention patterns evolve over the course of generation? Does the model look back at the image, the prompt, or its own prior tokens as it generates?*

**`compute_generation_summary(result)`** iterates over every generated token `i` and computes a compact summary from the layer-and-head-averaged attention row:
- `image_attn_ratio`, `prompt_attn_ratio`, `generated_attn_ratio`: fraction of attention mass directed at each region.
- For the thinking model: `thinking_attn_ratio` (attention to the `<think>` content) and `answer_attn_ratio` (attention to prior answer tokens).
- `entropy`: Shannon entropy of the full attention row in nats — a measure of how concentrated or diffuse the attention is.
- `top_k_tokens`: the top-5 most-attended source tokens `(index, weight, string)`.
- `phase`: whether the token is in the thinking or answer phase.

**Visualizations produced:**

6. **Attention ratio stacked area chart** (`summary_attn_ratios.png`): A stacked area chart where the three source-region ratios sum to 1.0 at each generation step. The stacked format makes the part-of-whole composition immediately visible while preserving the temporal dimension that a pie chart would lose. For very short generations (< 4 tokens), a stacked bar chart is used instead to avoid degenerate area plots.

7. **Top-attended tokens sparse heatmap** (`summary_top_attended.png`): A `(num_generated_tokens × max_source_index)` matrix where only the top-5 attended source positions per generated token are colored, and all other cells are white (set to `NaN`). Each row is independently normalized by its maximum so the color gradient encodes relative importance within each token's top-5 rather than absolute attention weight. This visualization reveals recurrent attention targets — source positions that many generated tokens attend to consistently.

8. **Reasoning phase transition** (`summary_reasoning_phase.png`, thinking model only): A stacked area chart with four components (image, prompt, thinking-self, answer-self), with dashed vertical lines marking the transitions between the thinking and answer phases. This is a primary visualization for testing the hypothesis that the model's attention distribution shifts meaningfully at the `</think>` boundary.

---

## 5. Attention Rollout

Raw single-layer attention shows only *direct* contributions from one token to another. In a deep transformer, token A can influence token C through an intermediate token B across multiple layers. **Attention rollout** \citep{abnar2020quantifying} captures these indirect multi-hop paths by propagating attention across layers via matrix multiplication.

The implementation in `AttentionAnalyzer.attention_rollout()` follows the standard formulation:

1. Aggregate heads per layer by mean (or max) to obtain `(L, S, S)`.
2. Model the residual connection by blending each layer's attention with the identity matrix: $\tilde{A}_l = (1 - \alpha) A_l + \alpha I$, where $\alpha = 0.5$ by default. This reflects that each token's representation after a transformer layer is a weighted sum of the attention-mixed values and its own previous representation.
3. Re-normalize each row of $\tilde{A}_l$ to sum to 1.
4. Compute the rollout matrix as $R = \prod_{l=0}^{L-1} \tilde{A}_l$.

Row `i` of the resulting `(S, S)` matrix gives the effective influence of each input token on token `i` after information has propagated through all layers.

---

## 6. Quantitative Metrics

`AttentionAnalyzer.compute_metrics(result)` computes a set of scalar and per-layer metrics from the attention matrices of generated tokens. All metrics operate on `gen_attn = attn[:, :, gen_start:, :]` — the sub-tensor containing only rows corresponding to generated tokens — to isolate generative behavior from the prefill phase.

| Metric | Shape | Definition |
|---|---|---|
| `entropy_per_layer` | `(L,)` | Mean Shannon entropy (nats) of the attention distribution per generated token, averaged across heads and tokens. $H = -\sum_j p_j \log p_j$ |
| `entropy_per_layer_head` | `(L, H)` | Same, not averaged across heads |
| `image_attn_ratio_per_layer` | `(L,)` | Mean fraction of attention mass directed at image tokens, averaged across heads and generated tokens |
| `image_attn_ratio_per_layer_head` | `(L, H)` | Same, per head |
| `sparsity_per_layer` | `(L,)` | Fraction of attention weights exceeding `1/S` (the uniform baseline) — a measure of how many tokens receive above-average attention |
| `gini_image_per_layer` | `(L,)` | Gini coefficient of the image attention distribution — measures concentration; 0 = uniform across image patches, 1 = all mass on one patch |
| `head_specialization` | `(H,)` | Standard deviation of `image_attn_ratio_per_layer_head` across layers for each head — heads with consistent image attention across all layers have low specialization; heads whose behavior varies by depth have high specialization |

The image attention ratio is a central metric for the thesis because it measures **visual grounding** as a function of depth. A systematic difference between the instruct and thinking models in this curve would indicate that the reasoning process alters how the model integrates visual information across layers.

The Gini coefficient for image attention measures **spatial selectivity**: whether the model attends uniformly to all image patches or concentrates on a small region. It is computed as:

$$G = \frac{2 \sum_{k=1}^{n} k \cdot x_{(k)}}{n \sum_{k=1}^{n} x_{(k)}} - \frac{n+1}{n}$$

where $x_{(1)} \leq x_{(2)} \leq \cdots \leq x_{(n)}$ are the sorted image attention weights.

---

## 7. Per-Token Per-Layer Attention Ratios

The function `compute_per_token_attention_ratios(result)` (`attention_map.py`) produces a fine-grained decomposition for downstream statistical analysis. For each generated token `i` and each layer `l`, it computes the head-averaged attention mass directed at the image, prompt, and reasoning (thinking) regions. This yields matrices of shape `(gen_len, L)` that are written to long-format CSV files (`all_token_attn_thinking.csv`) during the CLEVR evaluation, enabling aggregation and significance testing across multiple question items.

Two sub-dictionaries are returned:
- `all_per_token_metrics`: covers all generated tokens (both thinking and answer phases for the thinking model).
- `per_answer_token_metrics`: restricted to tokens inside the `<answer>...</answer>` span, identified by scanning the decoded text for these tags. This isolates the final committed answer from the reasoning trace, allowing a direct comparison of visual attention during the answer itself.

---

## 8. Dataset Evaluation (CLEVR)

`eval_attention_clevr.py` extends the single-image pipeline to batch evaluation over a CLEVR question dataset. CLEVR is used because it provides compositional visual questions with unambiguous ground-truth answers, allowing attention patterns to be correlated with task performance.

For each dataset item:
1. The image filename is resolved from the dataset JSON following CLEVR directory conventions.
2. The prompt template wraps the question with an explicit instruction to use `<answer>...</answer>` tags, enabling reliable answer extraction.
3. Both models (instruct and thinking) are run sequentially. To avoid GPU memory overflow, each model is unloaded and `torch.cuda.empty_cache()` is called before loading the next.
4. A resume mechanism checks whether a per-item output directory already contains PNG files before re-running, supporting interrupted runs.
5. Consolidated results are written to `metrics_instruct.json` and `metrics_thinking.json`, along with a long-format CSV of per-token attention ratios for the thinking model.

The CLEVR prompt template used is:

> *{question} Please respond with ONE word or number, you must wrap your final answer with \<answer\> and \</answer\> tags.*

This format was chosen to make answer extraction deterministic and to ensure that the token selected for deep inspection (the final token before `</answer>`) is semantically meaningful.

---

## 9. Implementation Decisions

**Why `attn_implementation="eager"`:** Flash Attention 2 and PyTorch SDPA fuse the QK softmax and V matmul into a single kernel that does not materialize the attention weight matrix. Extracting weights therefore requires the standard eager implementation, at the cost of higher memory usage and slower inference.

**Why `output_attentions=True` with `use_cache=True`:** An alternative approach would be to use forward hooks to capture attention weights during generation. The `output_attentions` approach is simpler — no model surgery required — but produces the nested tuple structure described in Section 3 that must be reassembled. The choice of `use_cache=True` (KV cache enabled) means that each decode step only attends over the cached key-value states, so the attention row for step `k` has length `input_len + k`, not `total_len`; this is handled correctly in `_reconstruct_attention()`.

**Why fp16 storage:** The full attention tensor `(L, H, S, S)` at fp32 for a 28-layer, 16-head model with a 500-token sequence would be 28 × 16 × 500 × 500 × 4 bytes ≈ 1.4 GB. Storing in fp16 halves this to ~700 MB. Given that attention weights are in `[0, 1]`, the fp16 precision loss is negligible for the analyses performed here.

**PowerNorm for the attention matrix visualization:** The full `(S × S)` attention matrix is highly sparse — most weights are near zero, with a few large values on induction heads or at specific structural positions. A linear colormap would map virtually all cells to the minimum color value, making the plot uninformative. `PowerNorm(gamma=0.3)` applies a cube-root-like compression to the low end of the scale. The colormap range is clamped at the 95th percentile of the lower-triangle values so that a few extreme peaks do not wash out the rest of the structure.

**Independent per-panel normalization for heatmaps:** Spatial attention heatmaps are normalized independently per model in comparative plots. This is intentional: different models may have very different absolute magnitudes for image attention, but the goal is to compare *where* each model looks, not *how much* it looks at the image in absolute terms. Absolute magnitudes are captured separately by the `image_attn_ratio_per_layer` metric.

---

## 10. Output Structure

For a single image/prompt, `demo_attention.py` writes the following directory structure:

```
<output_dir>/
├── instruct/
│   ├── deep_causal_bar.png
│   ├── deep_image_heatmap.png
│   ├── image_heatmap_evolution.gif
│   ├── deep_layer_head_grid.png
│   ├── deep_rollout_to_input.png
│   ├── summary_attn_ratios.png
│   ├── summary_top_attended.png
│   ├── attention_matrix.png
│   ├── entropy_by_layer.png
│   ├── head_specialization.png
│   ├── image_attn_ratio_by_layer.png
│   └── metrics.json
├── thinking/
│   ├── (same files as instruct/)
│   └── summary_reasoning_phase.png
├── comparative/
│   ├── entropy_by_layer_comparison.png
│   ├── image_attn_ratio_comparison.png
│   ├── entropy_over_generation.png
│   └── rollout_comparison.png
└── index.html
```

For the CLEVR batch run, the structure mirrors this per item, with additional CSV and consolidated JSON files at the root of `<save_dir>/`.
