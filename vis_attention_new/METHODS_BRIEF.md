# Approach and Implementation

## Models

We compare two variants of the Qwen3-VL-8B vision-language model: a standard instruction-following model (**Instruct**) and a chain-of-thought reasoning model (**Thinking**). The two share the same architecture — a vision encoder feeding into a causal language model — but differ in their generation protocol. When prompted with `enable_thinking=True`, the Thinking model first produces an explicit reasoning trace enclosed in `<think>...</think>` tags before committing to a final answer. This allows us to isolate and compare attention behavior during reasoning versus direct answering, while controlling for model size and architecture.

All experiments use greedy decoding for reproducibility, and models are loaded with standard (eager) attention to enable direct extraction of attention weight tensors.

## Attention Extraction

We extract the full causal attention tensor by running generation with `output_attentions=True`, capturing attention weights at every layer and decode step. The prefill and per-step decode outputs are reassembled into a single `(L, H, S, S)` matrix — where `L` is the number of layers, `H` is the number of heads, and `S` is the total sequence length — stored in fp16 to manage memory. To reduce cost when needed, a subset of layers can be selected for extraction.

## Token Region Segmentation

A central contribution of our analysis framework is the decomposition of the token sequence into semantically distinct **regions**: system tokens, image tokens, user prompt tokens, and generated tokens (further split into *thinking* and *answer* sub-regions for the Thinking model). Boundaries are identified by scanning for the model's special vision delimiters and, for the Thinking model, the `<think>` / `</think>` markers. Every metric and visualization is expressed in terms of attention mass flowing between these regions, enabling direct comparison across models.

## Analysis

We conduct analysis at two levels of granularity.

**Single-token causal inspection.** For each model and input, we identify the final answer token — specifically, the last content token inside the model's `<answer>...</answer>` span — and examine which prior tokens it attends to. Attention weights are extracted from every layer and head, then averaged to produce a compact source distribution. We additionally compute **attention rollout** to capture multi-hop causal influence: at each layer the attention matrix is blended with the identity (weight 0.5) to model the residual connection, and the blended matrices are multiplied across all layers. This yields an effective influence score from every input token to the focal output token, accounting for indirect paths through intermediate positions.

**Generation-level summary.** For every generated token we record the fraction of attention mass directed at each region (image, prompt, prior generated), the Shannon entropy of the attention distribution, and the top-5 most-attended source positions. For the Thinking model, we separately track attention to the reasoning trace (`<think>` content) and to prior answer tokens, enabling direct measurement of how the model uses its own chain-of-thought during answer generation.

## Quantitative Metrics

From the full attention tensor we compute the following per-layer metrics, averaged over heads and generated tokens:

- **Image attention ratio** — fraction of total attention mass directed at image tokens, measuring visual grounding as a function of depth.
- **Attention entropy** — Shannon entropy of the attention distribution in nats, measuring how concentrated or diffuse attention is at each layer.
- **Gini coefficient** — spatial concentration of image attention across vision-token patches; values near 1 indicate the model attends to a small region of the image.
- **Head specialization** — standard deviation of image attention ratio across layers for each head, identifying heads with depth-consistent visual roles.

## Dataset Evaluation

We evaluate on the CLEVR visual question answering benchmark, which provides compositional questions with unambiguous ground-truth answers. Each item is processed independently: the question is formatted with a prompt template requiring a `<answer>...</answer>`-tagged response, and both models are run with the same image and prompt. Per-token, per-layer attention ratios are written to long-format CSV files for downstream statistical analysis across the dataset. The two models are run sequentially with explicit GPU memory clearing between them to fit within a single 80 GB A100.
