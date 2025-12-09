import os
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_utils import process_vision_info
from PIL import Image
from tam import TAM
import argparse
from layer_analysis import plot_activations_all_layers, append_tam_scores_csv, append_full_layer_scores_csv
import json
import random
import glob
import tqdm
import numpy as np

def load_dataset(dataset_path, random_n=None, data_range=None):
    with open(dataset_path, 'r') as f:
        data_json = json.load(f)

    dataset = data_json["questions"] 
    if random_n is not None:
        return random.sample(dataset, random_n)

    if data_range is not None:
        start, end = data_range
        return dataset[start:end]

    return dataset

def tam_demo_for_qwen3_vl(data, model_path, model_name, save_dir):
    # Load Qwen2-VL model and processor
    # huge note to self: Qwen2-VL-7B does not work
    model_path = model_path
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto", local_files_only=True
    )
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    dataset_root = os.path.abspath(os.path.join(os.path.dirname(dataset_path), '..'))
    dataset_type = os.path.basename(dataset_path).split('_')[1]
    img_data_path = os.path.join(dataset_root, 'images', dataset_type)  

    for item in tqdm.tqdm(data):
        image_path = os.path.join(img_data_path, item['image_filename'])
        prompt_question = item['question']

        # prompt_text = f"{prompt_question} First THINK about the answer and wrap your thinking in <think> </think> tags. Then please respond with ONE word, you must wrap your final answer with <answer> and </answer> tags."
        prompt_text = f"{prompt_question} Please respond with a phrase, you must wrap your final answer with <answer> and </answer> tags."

        messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt_text}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to(model.device)

        # Generate model output with hidden states for visualization
        outputs = model.generate(
            **inputs,
            max_new_tokens=512, # increase from 256
            use_cache=True,
            output_hidden_states=True, 
            return_dict_in_generate=True
        )

        generated_ids = outputs.sequences

        # === TAM code part ====
        print("sequence length: ", len(outputs.hidden_states))

        # Compute logits from last hidden states with vocab classifier for TAM
        logits = [model.lm_head(feats[-1]) for feats in outputs.hidden_states] # logits[0].shape = (1, seq_len, vocab_size)
        
        num_tokens = len(outputs.hidden_states)
        num_layers = len(outputs.hidden_states[0])
        per_layer_logits = [ [] for _ in range(num_layers) ] # num_layers × num_tokens

        # go through each generated token 
        for t in range(num_tokens):
            for layer in range(num_layers):
                layer_hs = outputs.hidden_states[t][layer]  # (1, seq_len_t, hidden_dim)
                logits = model.lm_head(layer_hs)       # (1, seq_len, vocab_size)
                per_layer_logits[layer].append(logits) # store logits[layer][t]

        print('per layer logits', len(per_layer_logits), len(per_layer_logits[0]))  
        print(per_layer_logits[0][0].shape)  # torch.Size([1, 1, 151936])

        # Define special token IDs to separate image/prompt/answer tokens
        # See TAM in tam.py about its usage. See ids from the specific model.
        special_ids = {
            'img_id': [151652, 151653],  
            'prompt_id': [151653, [151645, 198, 151644, 77091]], 
            'answer_id': [[198, 151644, 77091, 198], -1]
        }

        vision_shape = (inputs['image_grid_thw'][0, 1] // 2, inputs['image_grid_thw'][0, 2] // 2)
        vis_inputs = image_inputs

        print('vision shape:', vision_shape)

        # === TAM Visualization ===
        prompt_ids = inputs["input_ids"][0]  # shape: (prompt_seq_len,)
        
        print(np.count_nonzero(prompt_ids.cpu().numpy() == 151655))  # check how many img tokens
        arr = np.array(prompt_ids.cpu().numpy())
        start_idxs = np.where(arr == 151653)[0]
        end_idxs   = np.where(arr == 151645)[0]
        print(start_idxs, end_idxs, end_idxs[0] - start_idxs[0] - 1)

        text_tokens_decoded = [processor.tokenizer.decode([prompt_ids[i].item()]) for i in range(start_idxs[0] + 1, end_idxs[0])]
        prompt_len = len(text_tokens_decoded)
        print("Prompt text tokens:", text_tokens_decoded, prompt_len)
        input_seq_len = inputs['input_ids'].shape[1]

        # decode the generated text tokens
        for token_idx in range(num_tokens):
            token_id = generated_ids[0][input_seq_len + token_idx].item()
            token_text = processor.tokenizer.decode([token_id])
            text_tokens_decoded.append(token_text)
        
        raw_map_records = [[] for _ in range(num_layers)]  # num_layers × list of img maps
        is_answer_token = False 
        decoded_so_far = ""

        for token_idx in range(num_tokens):
            logit_scores_for_token = []
            token_text = text_tokens_decoded[prompt_len + token_idx]
            print('token text ', token_text)
            decoded_so_far += token_text
            if "<answer>" in decoded_so_far: # indicate for next iteration that it is an answer token
                is_answer_token = True

            for layer_idx in range(num_layers):
                run_vis = False # (layer_idx == num_layers - 1) and (is_answer_token)  # only visualize last layer of answer tokens
                img_map, per_layer_scores = TAM(
                    tokens = generated_ids[0].cpu().tolist(),
                    vision_shape = vision_shape,
                    logit_list = per_layer_logits[layer_idx],   # ✔ full layer list
                    special_ids = special_ids,
                    vision_input = vis_inputs,
                    processor = processor,
                    save_fn = os.path.join(save_dir, f"{token_idx}_{layer_idx}.jpg"),
                    target_token = token_idx,                  # ✔ which token inside the layer list
                    img_scores_list = raw_map_records[layer_idx], 
                    eval_only = False,
                    run_vis = run_vis
                )

                logit_scores_for_token.append(per_layer_scores)

            # ---- DONE PROCESSING ALL LAYERS FOR THIS TOKEN ----

            # plot_activations_all_layers(
            #     all_scores = logit_scores_for_token,         # shape: num_layers × scores
            #     path = os.path.join(save_dir, 'all_layer_plots'),
            #     model_name = model_name,
            #     target_token_idx = token_idx,
            #     target_token = token_text
            # )
            # append_tam_scores_csv(
            #     save_dir = save_dir, 
            #     token_idx = token_idx, 
            #     token_text = token_text, 
            #     all_scores = logit_scores_for_token,
            #     csv_name=f'tam_scores_{model_name}.csv'
            # )

            append_full_layer_scores_csv(
                save_dir = save_dir, 
                token_idx = token_idx, 
                token_text = token_text, 
                all_scores = logit_scores_for_token,
                csv_name=f'full_tam_scores_{model_name}.csv',
                header = text_tokens_decoded
            )


if __name__ == "__main__":
    # single img demo (qwen)
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--model_name", type=str, default="Qwen3-VL-8B-Instruct")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--random_n", type=int, default=None) 
    parser.add_argument("--data_range", type=str, default=None) # if both are these are not set, then use full dataset

    args = parser.parse_args()

    save_dir = args.save_dir
    model_path = args.model_path
    model_name = args.model_name
    dataset_path = args.dataset_path
    random_n = args.random_n
    if args.data_range:
        start, end = map(int, args.data_range.split(','))
        data_range = (start, end)

    data = load_dataset(dataset_path, random_n=random_n, data_range=data_range)
    tam_demo_for_qwen3_vl(data, model_path, model_name, save_dir=save_dir)

