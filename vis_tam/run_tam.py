import os
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_utils import process_vision_info
from PIL import Image
from tam import TAM
import argparse
from layer_analysis import plot_activations_all_layers


def tam_demo_for_qwen3_vl(image_path, prompt_text, model_name, save_dir='vis_results'):
    # Load Qwen2-VL model and processor
    # huge note to self: Qwen2-VL-7B does not work
    model_name = model_name
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto", local_files_only=True
    )
    processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)
    # img = Image.open(image_path).convert("RGB")

    # Prepare input message with image/video and prompt
    if isinstance(image_path, list):
        # this might not work if video param doesn't support image_path
        messages = [{"role": "user", "content": [{"type": "video", "video": image_path}, {"type": "text", "text": prompt_text}]}]
    else:
        messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt_text}]}]

    # Process input text and visual info
    # print("messages ", messages)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # print("TEXT ", type(text))
    system_start_seq = processor.tokenizer(
        "<|im_start|>system\n", add_special_tokens=False)["input_ids"]
    system_end_seq = processor.tokenizer(
        "<|im_end|>\n", add_special_tokens=False)["input_ids"]

    print("system_start_seq:", system_start_seq)
    print("system_end_seq:", system_end_seq)

    
    image_inputs, video_inputs = process_vision_info(messages)
    # print("image_inputs and video_inputs: ", image_inputs, video_inputs)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    # print("inputs ", inputs)

    # Generate model output with hidden states for visualization
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        use_cache=True,
        output_hidden_states=True, # ---> TAM needs hidden states
        return_dict_in_generate=True
    )

    generated_ids = outputs.sequences

    # === TAM code part ====
    print("hidden states shape: ", len(outputs.hidden_states))
    print("features shape: ", len(outputs.hidden_states[0]))
    print("layer length: ", len(outputs.hidden_states[0][0]))

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
        'answer_id': [[198, 151644, 77091, 198], -1], 
        'system_id': [system_start_seq, system_end_seq]
    }

    # get shape of vision output
    if isinstance(image_path, list):
        vision_shape = (inputs['video_grid_thw'][0, 0], inputs['video_grid_thw'][0, 1] // 2, inputs['video_grid_thw'][0, 2] // 2)
    else:
        vision_shape = (inputs['image_grid_thw'][0, 1] // 2, inputs['image_grid_thw'][0, 2] // 2)

    # get img or video inputs for next vis
    vis_inputs = [[video_inputs[0][i] for i in range(0, len(video_inputs[0]))]] if isinstance(image_path, list) else image_inputs

    # === TAM Visualization ===
    # Call TAM() to generate token activation map for each generation round
    # Arguments:
    # - token ids (inputs and generations)
    # - shape of vision token
    # - logits for each round
    # - special token identifiers for localization
    # - image / video inputs for visualization
    # - processor for decoding
    # - output image path to save the visualization
    # - round index (0 here)
    # - raw_vis_records: list to collect intermediate visualization data
    # - eval only, False to vis
    # return TAM vision map for eval, saving multimodal TAM in the function
    
    input_seq_len = inputs['input_ids'].shape[1]
    raw_map_records = [[] for _ in range(num_layers)]  # num_layers × list of img maps

    for token_idx in range(num_tokens):
        logit_scores_for_token = []
        for layer_idx in range(num_layers):

            img_map, per_layer_scores = TAM(
                tokens = generated_ids[0].cpu().tolist(),
                vision_shape = vision_shape,
                logit_list = per_layer_logits[layer_idx],   # ✔ full layer list
                special_ids = special_ids,
                vision_input = vis_inputs,
                processor = processor,
                save_fn = os.path.join(save_dir, f"{layer_idx}_{token_idx}.jpg"),
                target_token = token_idx,                  # ✔ which token inside the layer list
                img_scores_list = raw_map_records[layer_idx], 
                eval_only = False
            )

            logit_scores_for_token.append(per_layer_scores)

        # ---- DONE PROCESSING ALL LAYERS FOR THIS TOKEN ----

        token_id = generated_ids[0][input_seq_len + token_idx].item()
        token_text = processor.tokenizer.decode([token_id])

        plot_activations_all_layers(
            all_scores = logit_scores_for_token,         # shape: num_layers × scores
            path = '.',
            model_name = "Qwen3-VL",
            target_token_idx = token_idx,
            target_token = token_text
        )


if __name__ == "__main__":
    # single img demo (qwen)
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--img_save_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    args = parser.parse_args()

    img_path = args.img_path
    prompt = args.prompt
    img_save_dir = args.img_save_dir
    model_name = args.model_name

    tam_demo_for_qwen3_vl(img_path, prompt, model_name, save_dir=img_save_dir)

