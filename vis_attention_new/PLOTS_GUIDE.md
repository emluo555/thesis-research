# Plots Guide

Reference for every image produced by the attention visualization pipeline
(`demo_attention.py` + `attention_vis.py` + `attention_map.py`).

## Token region color key

Every bar chart and many other plots color-code tokens by the region they
belong to. The mapping is defined in `attention_vis.py` lines 35-42
(`REGION_COLORS`):

| Region | Color | Hex | When present |
|---|---|---|---|
| System | grey | `#888888` | Always (chat-template tokens before the image: `<\|im_start\|>`, `user`, `\n`, etc.) |
| Image | blue | `#4A90D9` | Always (vision tokens between `<\|vision_start\|>` and `<\|vision_end\|>`) |
| Prompt | green | `#50B848` | Always (user text prompt tokens after the image) |
| Thinking | red | `#E74C3C` | Thinking model only (tokens inside `<think>...</think>`) |
| Answer | purple | `#9B59B6` | Thinking model only (tokens after `</think>`) |
| Generated | orange | `#F39C12` | Instruct model only (all generated tokens when there is no thinking phase) |

Region boundaries are identified in `attention_map.py` lines 300-350
(`get_token_regions`), which scans the full token-ID sequence for
`vision_start_token_id`, `vision_end_token_id`, `<think>`, and `</think>`.

The legend shown on bar charts is context-aware: `_add_region_legend`
(`attention_vis.py` line 983) accepts an optional `regions_present` set,
computed by `_collect_regions_from_token_regions` (line 957), so that only
regions that actually appear in the data are shown (e.g. the Instruct model
will never show "Thinking" or "Answer").

---

## Mode A -- Deep Single-Token Inspection

These plots are generated when `--mode deep` or `--mode both` (default).
They examine how one specific token (controlled by `--token_idx`, default -1
= last token with an attention row) attends to every preceding token. The
underlying data comes from `get_causal_attention_for_token`
(`attention_map.py` line 427), which extracts row `token_idx` of the causal
attention matrix from every layer and head.

### 1. `deep_causal_bar.png`

**Function:** `plot_causal_attention_bar` (`attention_vis.py` line 78)
**Called from:** `demo_attention.py` line 191

**What it shows:**
The causal attention distribution for a single token *i*, averaged across
all layers. For each prior token *j < i*, the bar height is the mean
attention weight that token *i* placed on token *j* across all layers and
heads.

**Data flow:**
1. `get_causal_attention_for_token` (line 427) extracts
   `attn[:, :, token_idx, :token_idx+1]` -- the full causal row for
   every layer and head, yielding `per_layer` of shape `(L, token_idx+1)`.
2. `demo_attention.py` line 189 averages across layers:
   `avg_row = causal["per_layer"].mean(axis=0)`.
3. The function aggregates all image-region tokens into a single bar
   (line 109: `img_agg = row[img_s:img_e].sum()`) to prevent hundreds of
   tiny image-token bars from dominating the x-axis.
4. Text tokens (prompt + generated) are shown individually, subsampled to
   at most `max_text_tokens=80` (line 134).

**Axes:**
- **X-axis:** Source tokens, labeled `[index] token_string`. The first
  entries are system tokens, then one aggregated "[Image]" bar, then
  individual prompt and generated tokens.
- **Y-axis:** Attention weight (unitless, sums to ~1.0 across the full
  row before aggregation; the image aggregate may be large because it is
  the sum over many individual image tokens).

**Colors:**
Each bar is colored by the region of the source token it represents (see
region color key above). The legend is context-aware and only shows
regions present in the data.

**Why a bar chart:**
A bar chart makes it easy to compare absolute attention weight across
individual source tokens. A heatmap would not resolve individual tokens
well at this scale, and a line plot would imply continuity between
discrete token positions.

---

### 2. `deep_image_heatmap.png`

**Function:** `plot_causal_image_heatmap` (`attention_vis.py` line 164)
**Called from:** `demo_attention.py` line 204

**What it shows:**
A spatial heatmap of how much attention token *i* pays to each image patch,
overlaid on the original image. This reveals *where* in the image the model
is looking when producing token *i*.

**Data flow:**
1. The image attention slice is `avg_row[r.image_start:r.image_end]`
   (demo line 203), where `avg_row` is the layer-averaged causal row.
2. These values are reshaped to the vision token grid
   `(h_tok, w_tok)` -- the spatial resolution after Qwen3-VL's spatial
   merge (line 203 of `attention_vis.py`). The grid shape is inferred
   from `image_grid_thw` in `_infer_vision_shape`
   (`attention_map.py` line 234).
3. The grid is bilinearly upsampled to the original image resolution via
   `cv2.resize` (line 208).
4. The heatmap is normalized to [0, 1] by dividing by its max (line 215).

**Panels (left to right):**
1. **Original Image** -- unmodified RGB.
2. **Attention Heatmap** -- the normalized attention grid rendered with the
   `jet` colormap, `vmin=0, vmax=1` (line 230).
3. **Overlay** -- 50/50 alpha blend of the original image and the
   jet-colored heatmap (line 223).

**Colors:**
`jet` colormap: blue = low attention, red = high attention. Normalization
is per-image (the brightest patch is always red). This is a standard
choice for spatial attention overlays because it has high perceptual
contrast across the full range.

**Why this visualization:**
Spatial attention heatmaps on images are the standard way to show where a
VLM is "looking." Showing the raw heatmap alongside the overlay lets the
reader judge both the absolute spatial pattern and how it relates to image
content.

---

### 3. `deep_layer_head_grid.png`

**Function:** `plot_causal_layer_head_grid` (`attention_vis.py` line 244)
**Called from:** `demo_attention.py` line 212

**What it shows:**
A grid of tiny bar charts (one per layer-head combination) showing token
*i*'s causal attention distribution. This reveals how different layers and
heads specialize -- e.g., some heads may attend mostly to image tokens while
others focus on the prompt.

**Data flow:**
1. Input is `causal["per_layer_head"]` of shape `(L, H, token_idx+1)` --
   the raw causal row without any averaging.
2. Layers are subsampled to at most `max_layers=12`, heads to at most 16
   (lines 268-285).
3. Each subplot is a bar chart of one head's attention row for token *i*,
   with bars colored by region (lines 299-303).

**Axes:**
- **Grid rows:** Layers (labeled `L0`, `L1`, ..., or the actual layer
  index if `--layers` was used).
- **Grid columns:** Heads (labeled `H0`, `H2`, ...).
- **Within each cell:** X = source token position, Y = attention weight.
  Tick labels are suppressed for compactness.

**Colors:**
Bars within each cell are colored by region (same color key). There is no
cell-level legend; the color coding is shared with the other Mode A plots.

**Why a grid of bar charts:**
This is the only way to show the full layer x head decomposition without
averaging away the head dimension. A single heatmap would obscure per-head
patterns, and averaging would hide head specialization. The small-multiple
layout lets the eye scan for outlier heads (e.g., a head that puts all
mass on image tokens).

---

### 4. `deep_rollout_to_input.png`

**Function:** `plot_causal_rollout_to_input` (`attention_vis.py` line 325)
**Called from:** `demo_attention.py` line 222

**What it shows:**
The *effective* causal influence of every input token on token *i*, computed
via attention rollout. Unlike the single-layer average in plot 1, this
propagates attention through all layers to account for multi-hop paths.

**Data flow:**
1. `compute_causal_contribution` (`attention_map.py` line 515) calls
   `attention_rollout` (line 376), which:
   - Averages heads per layer (line 406).
   - At each layer, blends the attention matrix with an identity matrix
     using `residual_weight=0.5` (line 417) to model the residual
     connection.
   - Re-normalizes each row (line 418).
   - Multiplies the blended matrices across all layers (line 419).
2. Row `token_idx` of the resulting `(S, S)` rollout matrix is extracted
   (line 532).

**Panels:**
- **Left (if image present):** Rollout attention to image patches, shown
  as a jet-colormap overlay on the original image (same technique as
  `deep_image_heatmap`).
- **Right:** Bar chart of rollout attention to text tokens (prompt +
  generated), colored by region. Image tokens are excluded (they are
  shown in the left panel).

**Axes (bar chart panel):**
- **X-axis:** Text tokens `[index] token_string`.
- **Y-axis:** Rollout attention weight (unitless; the full rollout row
  sums to ~1.0 across all tokens).

**Colors:**
Bar colors follow the region color key. The image panel uses the `jet`
colormap normalized to [0, 1].

**Why rollout vs. raw attention:**
Raw single-layer attention only shows direct contributions. In deep
networks, token A can influence token C through an intermediate token B
across multiple layers. Attention rollout captures these indirect paths by
chaining attention matrices. The residual blending (0.5) accounts for the
skip connection that preserves a token's own representation.

---

## Mode B -- Generation Summary

These plots are generated when `--mode summary` or `--mode both` (default).
They show how attention patterns evolve across the full generation, rather
than focusing on a single token. The underlying data comes from
`compute_generation_summary` (`attention_map.py` line 538), which iterates
over every generated token and computes per-token statistics from the
layer-and-head-averaged attention matrix.

### 5. `summary_attn_ratios.png`

**Function:** `plot_attention_ratio_over_generation` (`attention_vis.py` line 423)
**Called from:** `demo_attention.py` line 235

**What it shows:**
A stacked area chart showing, for each generated token, what fraction of
its attention mass goes to image tokens vs. prompt tokens vs. previously
generated tokens.

**Data flow:**
For each generated token *i*, `compute_generation_summary` (line 558)
computes:
- `img_mass`: sum of `attn_avg[i, image_start:image_end]` (line 562).
- `prompt_mass`: sum over the prompt region (line 563).
- `gen_mass`: sum over prior generated tokens (line 566).
- Each is divided by the total to get a ratio (lines 587-589).

**Axes:**
- **X-axis:** Generation step (0-indexed from the first generated token).
  Subsampled token labels are shown along the axis (line 456,
  `_add_token_xticks`).
- **Y-axis:** Attention ratio, stacked to 1.0.

**Colors:**
- Blue (`#4A90D9`): Image attention ratio.
- Green (`#50B848`): Prompt attention ratio.
- Orange (`#F39C12`): Prior-generated attention ratio.

**Why a stacked area chart:**
The three ratios sum to 1.0 by construction, making stacked areas the
natural representation for part-of-whole compositions that change over a
sequential axis. A line chart would make it harder to see that the three
components are exhaustive. A pie chart would lose the temporal dimension.

---

### 6. `summary_top_attended.png`

**Function:** `plot_top_attended_tokens_heatmap` (`attention_vis.py` line 492)
**Called from:** `demo_attention.py` line 240

**What it shows:**
A sparse heatmap where each row is one generated token and each column is a
source-token position. Only the top-5 most-attended source tokens per
generated token are colored; all other cells are white. This reveals
recurrent attention targets (e.g., if many generated tokens attend to the
same system token).

**Data flow:**
1. `compute_generation_summary` stores `top_k_tokens` for each generated
   token: the top-5 `(index, weight, string)` tuples from `row.topk(5)`
   (line 575).
2. These are placed into a `(num_display, max_source_idx)` matrix
   (lines 511-515).
3. Each row is **normalized by its max** (lines 518-520) so the color
   gradient shows relative importance within each token's top-5, not
   absolute weight.
4. Zeros are set to `np.nan` (line 523) so they render as white
   background rather than the lowest colormap value.

**Axes:**
- **X-axis:** Source token index (0 to max sequence position). Dashed
  vertical lines mark region boundaries (image start/end, prompt start,
  generation start) when `token_regions` is provided (lines 543-558).
- **Y-axis:** Generated token (labeled with the decoded token string,
  subsampled to ~30 labels).

**Colors:**
`YlOrRd` (Yellow-Orange-Red) colormap with `vmin=0, vmax=1` (line 528-529).
After per-row normalization, 1.0 (dark red) = the strongest-attended source
for that generated token, 0.0 = not in top-5 (rendered white via NaN).

**Why sparse heatmap with NaN masking:**
A dense heatmap of the full attention matrix would be almost entirely
near-zero (attention is concentrated on a few tokens). Showing only the
top-5 and masking everything else as white makes the sparse structure
visible. Per-row normalization ensures each row uses the full color range
regardless of whether the top weight is 0.05 or 0.5.

---

### 7. `summary_reasoning_phase.png` (Thinking model only)

**Function:** `plot_reasoning_phase_transition` (`attention_vis.py` line 566)
**Called from:** `demo_attention.py` line 249

**What it shows:**
A stacked area chart similar to plot 5, but with four components instead of
three: image, prompt, thinking-self, and answer-self. This shows how the
model's attention shifts between the reasoning phase (`<think>...</think>`)
and the answer phase.

**Data flow:**
For thinking models, `compute_generation_summary` additionally computes
(lines 596-609):
- `thinking_attn_ratio`: attention to tokens within the `<think>` region.
- `answer_attn_ratio`: attention to tokens after `</think>`.

**Axes:**
- **X-axis:** Generation step (same as plot 5).
- **Y-axis:** Attention ratio, stacked to 1.0.

**Colors:**
- Blue (`#4A90D9`): Image.
- Green (`#50B848`): Prompt.
- Red (`#E74C3C`): Thinking (self-attention to reasoning tokens).
- Purple (`#9B59B6`): Answer (self-attention to answer tokens).

**Phase markers:**
Dashed vertical lines are drawn at phase transitions
(`_add_phase_markers`, line 1021), marking where generation switches from
"thinking" to "answer" phase. The `phase` label for each token is
determined by `_get_phase` (`attention_map.py` line 622).

**Why this plot exists:**
This is the key visualization for comparing reasoning vs. non-reasoning
models. It directly shows whether the model "looks back" at its own
thinking during the answer phase -- a hypothesis central to the thesis.
The stacked area format makes the relative shift between phases immediately
visible.

---

## Structural Plots

These plots are always generated for every model, regardless of `--mode`.

### 8. `attention_matrix.png`

**Function:** `plot_attention_matrix` (`attention_vis.py` line 612)
**Called from:** `demo_attention.py` line 257

**What it shows:**
The full causal attention matrix, averaged across all layers and heads.
Entry (i, j) is the mean attention weight that token i places on token j.
Because the model is decoder-only with a causal mask, only the lower
triangle (j <= i) is non-zero.

**Data flow:**
1. `demo_attention.py` line 256 computes
   `attn_avg = result.attention_matrices.float().mean(dim=(0, 1)).numpy()`
   -- averaging the `(L, H, S, S)` tensor across layers and heads.
2. The upper triangle is set to `np.nan` (line 644) so it renders as
   white background.
3. If the sequence is longer than `max_size=200`, it is downsampled by
   strided indexing (lines 632-636).

**Axes:**
- **X-axis:** Key position (source token index).
- **Y-axis:** Query position (target token index).
- If the sequence is short enough (<= 80 tokens after downsampling),
  token strings are shown as tick labels (lines 667-672).

**Colors:**
`viridis` colormap with `PowerNorm(gamma=0.3, vmin=0, vmax=p95)`
(line 661), where `p95` is the 95th percentile of the lower-triangle
values (line 651). The power normalization with gamma < 1 compresses the
dynamic range: small but non-zero attention values are pushed toward the
middle of the color scale rather than being crushed to the minimum.

Dashed lines in region colors mark the boundaries between image, prompt,
and generation regions (lines 675-687).

**Why PowerNorm and 95th-percentile clipping:**
Raw attention matrices are extremely sparse (a few values near 1.0 on the
diagonal, most near zero). Without normalization, >95% of pixels map to
the colormap minimum (dark purple), producing a visually uninformative
plot. `PowerNorm(gamma=0.3)` acts like a cube-root transform, expanding
the low end of the scale. Clipping at the 95th percentile prevents a few
extreme self-attention peaks from washing out all other structure. The
upper triangle is masked as NaN rather than shown as zero to avoid
wasting half the color range on known-zero cells.

---

### 9. `entropy_by_layer.png`

**Function:** `plot_entropy_by_layer` (`attention_vis.py` line 761)
**Called from:** `demo_attention.py` line 265 (single model) and line 399
(comparative)

**What it shows:**
Mean attention entropy per layer. Higher entropy means the attention
distribution is more spread out (less concentrated on specific tokens);
lower entropy means more peaked/focused attention.

**Data flow:**
`compute_metrics` (`attention_map.py` line 634) computes:
1. `gen_attn = attn[:, :, gen_start:, :]` -- attention from generated
   tokens only (line 656).
2. Entropy: `-(p * p.log()).sum(dim=-1)` where p is clamped to 1e-12
   (lines 659-660). This is Shannon entropy in nats (natural log).
3. `entropy_per_layer` is the mean over heads and generated tokens
   (line 662).

**Axes:**
- **X-axis:** Layer index (0 to L-1).
- **Y-axis:** Mean attention entropy in nats.

**Colors:**
- Blue (`#2196F3`) with circle markers: Instruct model.
- Red (`#E74C3C`) with square markers: Thinking model.
- Grid lines at alpha 0.3.

**Why a line chart:**
Entropy across layers is a continuous profile -- it makes sense to connect
the points because adjacent layers have related representations. Markers at
each layer make individual readings easy to identify. A bar chart would
also work but would not convey the layer-to-layer trend as clearly.

---

### 10. `head_specialization.png`

**Function:** `plot_head_specialization` (`attention_vis.py` line 789)
**Called from:** `demo_attention.py` line 271

**What it shows:**
How specialized each attention head is with respect to image tokens. A head
with high specialization consistently attends to image tokens more (or
less) than the average head -- it has a dedicated role.

**Data flow:**
`compute_metrics` (`attention_map.py` line 684):
1. `image_attn_ratio_per_layer_head` is `(L, H)` -- the fraction of
   attention going to image tokens, per layer and head (lines 667-670).
2. `head_specialization = .std(axis=0)` -- the standard deviation across
   layers for each head (line 687). A head that has a consistent image
   attention ratio across all layers will have low std; a head whose
   image attention varies wildly across layers will have high std.

**Axes:**
- **X-axis:** Head index (0 to H-1).
- **Y-axis:** Specialization score (standard deviation of image attention
  ratio across layers, unitless, range ~0 to ~0.5).

**Colors:**
All bars are blue (`#4A90D9`), alpha 0.8. This is a single-variable chart
so color does not encode additional information.

**Why a bar chart:**
Each head is a discrete entity, so a bar chart is the natural choice. The
std metric captures cross-layer consistency: a high bar means the head
behaves very differently at different depths with respect to image tokens,
which is itself an interesting signal for understanding model internals.

---

### 11. `image_attn_ratio_by_layer.png`

**Function:** `plot_image_attention_ratio_by_layer` (`attention_vis.py` line 809)
**Called from:** `demo_attention.py` line 277 (single model) and line 406
(comparative)

**What it shows:**
The fraction of total attention mass directed at image tokens, per layer.
This reveals whether deeper layers attend more or less to the image.

**Data flow:**
`compute_metrics` (`attention_map.py` line 664):
1. `img_mass = gen_attn[:, :, :, img_s:img_e].sum(dim=-1)` (line 667).
2. `ratio = img_mass / total_mass` (line 669).
3. `image_attn_ratio_per_layer = ratio.mean(dim=(1,2))` (line 671) --
   averaged across heads and generated tokens.

**Axes:**
- **X-axis:** Layer index (0 to L-1).
- **Y-axis:** Image attention ratio (unitless, 0 to 1).

**Colors:**
Same as entropy plot: blue for Instruct, red for Thinking, with
circle/square markers.

**Why this metric matters:**
The image attention ratio directly measures how much the model relies on
visual information at each depth. For thesis comparisons between reasoning
and non-reasoning models, a divergence in this curve (e.g., the thinking
model sustaining higher image attention in deeper layers) would be a
significant finding.

---

## Comparative Plots

Generated only when both `--model_instruct` and `--model_thinking` are
provided. Saved to `<output_dir>/comparative/`.

### 12. `entropy_by_layer_comparison.png`

**Function:** `plot_entropy_by_layer` (`attention_vis.py` line 761) -- same
function as plot 9, but called with both `metrics_instruct` and
`metrics_thinking`.
**Called from:** `demo_attention.py` line 399

Identical to plot 9, but both models are overlaid on the same axes. See
plot 9 for full details.

---

### 13. `image_attn_ratio_comparison.png`

**Function:** `plot_image_attention_ratio_by_layer` (`attention_vis.py`
line 809) -- same function as plot 11, called with both metrics dicts.
**Called from:** `demo_attention.py` line 406

Identical to plot 11, but both models are overlaid. See plot 11 for full
details.

---

### 14. `entropy_over_generation.png`

**Function:** `plot_entropy_over_generation` (`attention_vis.py` line 462)
**Called from:** `demo_attention.py` line 414

**What it shows:**
Attention entropy for each generated token (a time series), comparing both
models. This reveals whether the reasoning model has more or less focused
attention during generation.

**Data flow:**
The `entropy` field of each entry in the generation summary
(`compute_generation_summary`, `attention_map.py` line 571):
`-(p * p.log()).sum()` where p is the full attention row for token *i*
averaged across layers and heads.

**Axes:**
- **X-axis:** Generation step (0-indexed).
- **Y-axis:** Attention entropy in nats.

**Colors:**
- Blue (`#2196F3`): Instruct.
- Red (`#E74C3C`): Thinking.

**Phase markers:**
For the thinking model, dashed vertical lines mark the transition from
"thinking" to "answer" phase (`_add_phase_markers`, line 1021), with the
phase name annotated.

**Why overlay both models:**
Direct overlay on the same axes makes it visually immediate where the
models diverge in attention concentration. Phase markers on the thinking
model's timeline let the reader see whether entropy changes at the
think/answer boundary.

---

### 15. `rollout_comparison.png`

**Function:** `plot_rollout_comparison` (`attention_vis.py` line 694)
**Called from:** `demo_attention.py` line 423

**What it shows:**
Side-by-side attention rollout heatmaps on the original image for both
models. This directly compares *where* each model looks after multi-layer
propagation.

**Data flow:**
1. The full `(S, S)` rollout matrix is computed per model via
   `attention_rollout` (`attention_map.py` line 376).
2. For each model, the image-region slice of the rollout row for
   `token_idx` is extracted (line 732), reshaped to the vision grid,
   upsampled, normalized to [0, 1], and blended with the original image
   using the `jet` colormap (lines 742-749).

**Panels (left to right):**
1. **Original** -- unmodified image.
2. **Instruct Rollout** -- jet-colormap overlay for the Instruct model.
3. **Thinking Rollout** -- jet-colormap overlay for the Thinking model.

**Colors:**
`jet` colormap, each panel independently normalized to [0, 1] by its own
max (line 746). Blue = low rollout attention, red = high.

**Why independent normalization:**
Each model may have very different absolute rollout magnitudes. Normalizing
each panel independently ensures the full color range is used for both,
making spatial patterns comparable even if absolute scales differ.

---

## Non-image outputs

### `metrics.json`

Saved per model at `<output_dir>/<model_label>/metrics.json`
(`demo_attention.py` lines 283-293). Contains all metrics from
`compute_metrics` serialized as JSON. Array metrics are stored as
`{"mean": float, "values": list}`.

### `index.html`

Generated by `save_comparative_report` (`attention_vis.py` line 841).
An HTML page that embeds all PNG plots from the output directory and
displays a metrics summary table. It is an index for browsing results in a
browser.
