import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
from PIL import Image
from vis_attention import vis_attention

MODEL_DIR = "/scratch/gpfs/ZHUANGL/el5267/thesis-research/models/models--meta-llama-Llama-3.2-11B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    MODEL_DIR, 
    torch_dtype=torch.float16,
    attn_implementation="eager"
).to("cuda")
processor = AutoProcessor.from_pretrained(MODEL_DIR)

image_path = "pastry.png"
image = Image.open(image_path)

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "How to make this pastry?"}
    ]}
]

input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(
    image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt"
).to(model.device)

# ===== DEBUG: Figure out number of image tokens =====
print("\n" + "="*60)
print("TOKEN STRUCTURE ANALYSIS")
print("="*60)

# Method 1: Check input_ids for image token markers
input_ids = inputs["input_ids"][0]
print(f"Total input tokens: {len(input_ids)}")

# Mllama uses special image tokens (usually <|image|> token ID: 128256)
image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image|>")
print(f"Image token ID: {image_token_id}")

num_image_tokens_in_input = (input_ids == image_token_id).sum().item()
print(f"Number of <|image|> placeholder tokens: {num_image_tokens_in_input}")

# Method 2: Check pixel_values shape
if "pixel_values" in inputs:
    pixel_values = inputs["pixel_values"]
    print(f"\nPixel values shape: {pixel_values.shape}")
    # Shape is typically [batch, channels, height, width]
    
# Method 3: Check aspect_ratio_ids and aspect_ratio_mask (Mllama specific)
if "aspect_ratio_ids" in inputs:
    print(f"Aspect ratio IDs: {inputs['aspect_ratio_ids']}")
if "aspect_ratio_mask" in inputs:
    print(f"Aspect ratio mask shape: {inputs['aspect_ratio_mask'].shape}")

# Method 4: Check cross_attention_mask (most reliable!)
if "cross_attention_mask" in inputs:
    cross_attn_mask = inputs["cross_attention_mask"][0]  # [seq_len, num_image_tokens]
    num_image_tokens = cross_attn_mask.shape[-1]
    print(f"\nFOUND: Number of image tokens = {num_image_tokens}")
    print(f"   Cross attention mask shape: {cross_attn_mask.shape}")
    
    # Calculate grid dimensions
    grid_size = int(num_image_tokens ** 0.5)
    print(f"   Image token grid: {grid_size}x{grid_size}")
else:
    # Fallback: infer from image dimensions
    print("\ncross_attention_mask not found, using fallback calculation")
    patch_size = 14  # Default for Mllama
    num_image_tokens = (image.height // patch_size) * (image.width // patch_size)
    grid_size = int(num_image_tokens ** 0.5)
    print(f"   Estimated image tokens: {num_image_tokens} ({grid_size}x{grid_size})")

print("="*60 + "\n")

# ===== Run Inference =====
print("Running inference...")
model.config.output_attentions = True

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        output_attentions=True,
        return_dict_in_generate=True
    )

if "attentions" in outputs:
    attentions = outputs.attentions
    print(f"Inference completed! Extracted {len(attentions)} generation steps.")
    print(f"   Each step has {len(attentions[0])} layers.")
    
    # Important: For generation, attentions is a tuple of tuples
    # attentions[generation_step][layer_idx] -> attention tensor
    # We need to handle this differently than encoder attentions
    
    print("\nAttention structure:")
    if len(attentions) > 0 and len(attentions[0]) > 0:
        first_attn = attentions[0][0]
        if isinstance(first_attn, tuple):
            first_attn = first_attn[0]
        print(f"   First attention shape: {first_attn.shape}")
        print(f"   Format: (batch_size, num_heads, seq_len, seq_len)")
else:
    raise ValueError("Attention extraction failed.")

# ===== Visualize Attention =====
print("\nVisualizing attention maps...")

# For generation, we typically want to look at the first generation step
# when the model has seen the full input (image + text)
first_step_attentions = attentions[0]  # First generation step

vis_attention(
    image=image, 
    attentions=first_step_attentions,
    num_image_tokens=num_image_tokens  # Pass the calculated value
)

# ===== Decode Output =====
generated_text = processor.decode(
    outputs.sequences[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True
)
print("\n" + "="*60)
print("GENERATED TEXT")
print("="*60)
print(generated_text)
print("="*60)

# import torch
# from transformers import MllamaForConditionalGeneration, AutoProcessor
# from PIL import Image
# from vis_attention import vis_attention

# MODEL_DIR = "/scratch/gpfs/ZHUANGL/el5267/thesis-research/models/models--meta-llama-Llama-3.2-11B-Vision-Instruct"

# model = MllamaForConditionalGeneration.from_pretrained(
#     MODEL_DIR, 
#     torch_dtype=torch.float16,
#     attn_implementation = "eager").to("cuda")
# processor = AutoProcessor.from_pretrained(MODEL_DIR)
# image_path = "pastry.png"
# image = Image.open(image_path)

# messages = [
#     {"role": "user", "content": [
#         {"type": "image"},
#         {"type": "text", "text": "How to make this pastry?"}
#     ]}
# ]

# input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
# inputs = processor(
#     image,
#     input_text,
#     add_special_tokens=False,
#     return_tensors="pt"
# ).to(model.device)

# print("running inference")
# model.config.output_attentions = True
# with torch.no_grad():
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=2048,
#         output_attentions=True,  # Ensure attention extraction
#         return_dict_in_generate=True  # Return a dictionary for easy extraction
#     )
# if "attentions" in outputs:
#     attentions = outputs.attentions  # Extract all decoder layer attentions
#     print(f"✅ Inference completed! Extracted {len(attentions)} attention layers.")
# else:
#     raise ValueError("Attention extraction failed. The model did not return attention weights.")

# vis_attention(image, attentions)

# print(processor.decode(outputs.sequences[0][inputs["input_ids"].shape[-1]:]))
