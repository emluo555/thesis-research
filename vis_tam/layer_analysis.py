import math
import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import argparse
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import seaborn as sns


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def attention_compute(attention, sys_len, img_len):
    attention = torch.mean(attention, dim=1) # average over heads
    attention = attention.squeeze(0).numpy()
    props_sys = attention[-1][:sys_len].sum()
    props_img = attention[-1][sys_len:sys_len+img_len].sum()
    props_txt = attention[-1][sys_len+img_len:].sum()
    return [props_sys, props_img, props_txt]


def run_attention_analysis(
    model_path, image_folder, question_file, output_file,
    num_chunks=1, chunk_idx=0, max_samples=5, device="cuda:0"
):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        attn_implementation="eager", #hmm
        # load_in_4bit=True,
        device_map=device,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, padding_side='left', use_fast=True)

    questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
    questions = get_chunk(questions, num_chunks, chunk_idx)[:max_samples]
 
    results = []

    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]

        # Base Model 
        # messages_query = [{
        #     "role": "user",
        #     "content": [
        #         {"type": "image", "image": os.path.join(image_folder, image_file)},
        #         {"type": "text", "text": qs},
        #     ],
        # }]

        # Reasoning Model 
        messages_query = [{
            "role": "user",
            "content": [
                {"type": "image", "image": os.path.join(image_folder, image_file)},
                {"type": "text", "text": f"{qs} You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE in <answer> </answer> tags."},
            ],
        }]

        image_inputs, _ = process_vision_info(messages_query)
        text_query = processor.apply_chat_template(messages_query, tokenize=False, add_generation_prompt=False)
        inputs = processor(
            text=[text_query],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        input_ids = inputs['input_ids'][0].tolist()
        input_ids_tensor = inputs['input_ids'][0]
        

        
        try:
            vision_start_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
            vision_end_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
            pos = input_ids.index(vision_start_token_id) + 1
            pos_end = input_ids.index(vision_end_token_id)
        except ValueError:
            print(f"{idx} skipped: vision tokens not found.")
            continue

        sys_len = pos
        img_len = pos_end - pos
        txt_len = len(input_ids) - pos_end - 1
        print(f"System tokens: [0,{pos}) {sys_len}; Image tokens: [{pos},{pos_end}) {img_len}, Text tokens: [{pos_end}, {len(input_ids)-1}) {txt_len}")


        seq_decoded = processor.tokenizer.batch_decode(input_ids_tensor, skip_special_tokens=False)
        txt_start = pos_end + 1  
        text_seq_decoded = seq_decoded[txt_start:-2]
        print("text seq decoded: ", text_seq_decoded) 

        # only compute loss for text generation part, ignore sys, img, and padding
        labels = inputs['input_ids'].clone()
        labels[:, :pos_end + 1] = -100
        labels[labels == processor.tokenizer.pad_token_id] = -100
        

        model.zero_grad()
        with torch.set_grad_enabled(False): # disable gradients for memory efficiency
            outputs = model(
                **inputs,
                labels=labels,
                output_attentions=True,
                return_dict=True,
            )
        for name, module in model.named_modules():
            print(f"Layer Name: {name}, Layer Type: {type(module)}")
        print("layers: ", model.language_model.layers)
        print("first layer: ", model.language_model.layers[0])
        
        print("features: ", model.language_model.layers[0].feat)
        print("dimensions (batch, num_heads, seq_len, seq_len):", outputs.attentions[0].shape)

        # pick layer of choice. TODO: make this more dynamic
        layer_idx = 35
        # plot 3d feature map of hidden states
        stats = {}
        stats["seq"] = text_seq_decoded
        feat_abs = model.language_model.layers[layer_idx].feat.abs()
        stats[f"{layer_idx}"] = feat_abs 


        if outputs.attentions is None:
            print(f"No attentions found for {idx}")
            continue
        
        attn_from_output = outputs.attentions[0]
        attn_from_manual = attn_from_output.softmax(dim=-1)
        print(torch.allclose(attn_from_output, attn_from_manual, atol=1e-4))
        print(attn_from_output.sum(dim=-1).mean())  # large, not ~1
        print(attn_from_output.softmax(dim=-1).sum(dim=-1).mean()) 

        layers_attn = []
        for attn in outputs.attentions:
            attn_score = attn.detach().clone().to(torch.float32).cpu()
            attn_props = attention_compute(attn_score, sys_len, img_len)
            layers_attn.append(attn_props)

        results.append(layers_attn)
        layer_attn_score = outputs.attentions[layer_idx].detach().clone().to(torch.float32).cpu()
        attn_avg = layer_attn_score.numpy()[0].mean(0) # average over heads

        plot_layer_attention(
            attn_avg=attn_avg,
            layer_idx=layer_idx,
            output_dir="./",
            model_name="Qwen3VL-8B-Thinking",
            sys_len=sys_len,
            txt_len=txt_len,
            img_len=img_len,
        )
        

    torch.save(results, output_file)

def plot_layer_attention(attn_avg, layer_idx, output_dir, model_name,
                         sys_len=0, txt_len=0, img_len=0):
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4.75))
    plt.tight_layout() 
    plt.subplots_adjust(wspace=0.15)

    attn_np = attn_avg
    mask = np.zeros_like(attn_np)
    mask[np.triu_indices_from(mask, k=1)] = True
    sns.heatmap(attn_np, mask=mask, square=True, ax=ax,
                      cmap="YlGnBu",
                      vmin=-6, vmax=13, # for normalizing the scales
                      cbar_kws={"shrink": 1.0, "pad": 0.01, "aspect":50})

    ax.set_facecolor("whitesmoke") 
    cax = ax.figure.axes[-1]
    cax.tick_params(labelsize=18)

    ax.tick_params(axis='x', which='major')
    ax.set(xticklabels=[])
    ax.set(yticklabels=[])
    ax.tick_params(left=False, bottom=False)
    ax.set_title(f"{model_name}, Layer {layer_idx+1}", fontsize=18, fontweight="bold")

    plt.savefig(f"{output_dir}/attention_layer_{layer_idx+1}.png", bbox_inches="tight", dpi=200)
    plt.close()


def plot_attn_alloc(data, path, model_name):

    normalized_data = data / data.sum(axis=0)
    x = np.arange(normalized_data.shape[1])

    colors = ['#eaa1a6', '#afd3e6', '#fcc280']  # Visual / Text / System


    fig, ax = plt.subplots(figsize=(7.5, 5))

    ax.bar(x, normalized_data[1], color=colors[0], edgecolor='gray', linewidth=0.5, label='Visual Features')
    ax.bar(x, normalized_data[2], color=colors[1], bottom=normalized_data[1], edgecolor='gray', linewidth=0.5,
           label='User Instructions')
    ax.bar(x, normalized_data[0], color=colors[2], bottom=normalized_data[1] + normalized_data[2], edgecolor='gray',
           linewidth=0.5, label='System Prompts')


    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attention', fontsize=12, fontweight='bold')


    ax.set_xticks(x[::3])
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_ylim(0, 1)

    ax.tick_params(axis='both', labelsize=16, width=0.5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    plt.title(f'Attention Allocation Across Layers in {model_name}', fontsize=10, fontweight='bold', pad=25)

    legend = ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.1),
        ncol=3,
        frameon=True,
        edgecolor='black',
        fontsize=10
    )


    for spine in ax.spines.values():
        spine.set_linewidth(1)

    plt.tight_layout()
    plt.savefig(path, dpi=600)
    plt.close()

def prepare_attention_data(path):
    results = torch.load(path, weights_only=False)
    attn_allocs = [torch.tensor(r, dtype=torch.float32) for r in results]
    attn_allocs = torch.stack(attn_allocs, dim=0).mean(dim=0).numpy().T
    return attn_allocs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the pretrained model")
    parser.add_argument("--image-folder", type=str, required=True, help="Path to folder containing images")
    parser.add_argument("--question-file", type=str, required=True, help="Path to the JSONL question file")
    parser.add_argument("--answers-file", type=str, required=True, help="Path to save .pt result file")
    parser.add_argument("--plot-path", type=str, required=True, help="Path to save attention plot image")
    parser.add_argument("--model-name", type=str, required=True, help="Name of model")


    args = parser.parse_args()

    # Run attention analysis
    run_attention_analysis(
        model_path=args.model_path,
        image_folder=args.image_folder,
        question_file=args.question_file,
        output_file=args.answers_file,
        device="cuda:1"
    )
    print("load and vis")

    # Load and visualize
    attn_data = prepare_attention_data(args.answers_file)
    # plot_attn_alloc(attn_data, args.plot_path, args.model_name)

