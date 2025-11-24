import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
import os 

def vis_attention(image, attentions, num_image_tokens):

    output_dir = "./attention_maps"
    os.makedirs(output_dir, exist_ok=True)

    grid_size = int(num_image_tokens ** 0.5)
    if grid_size ** 2 != num_image_tokens:
        print(f"Warning: num_image_tokens ({num_image_tokens}) is not a perfect square")
        # Try to find closest factors
        grid_h = int(np.sqrt(num_image_tokens))
        grid_w = num_image_tokens // grid_h
        print(f"Using grid dimensions: {grid_h}x{grid_w}")
    else:
        grid_h = grid_w = grid_size
   
    for layer_idx, layer_attn in enumerate(attentions):
        print(f"Processing attention map for Layer {layer_idx + 1}")

        if isinstance(layer_attn, tuple):  
            layer_attn = layer_attn[0]

        if layer_attn.ndim == 4:  # Expected (batch_size, num_heads, seq_len, seq_len)
            attn_map = layer_attn.mean(dim=1).squeeze().to(torch.float32).cpu().detach().numpy() # convert from bfloat16 to float32
        else:
            print(f"Unexpected shape {layer_attn.shape}. Skipping Layer {layer_idx + 1}.")
            continue

        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-9)

        if attn_map.shape[0] < num_image_tokens or attn_map.shape[1] < num_image_tokens:
            print(f"  Skipping: sequence too short ({attn_map.shape})")
            continue

        img_to_img_attn = attn_map[:num_image_tokens, :num_image_tokens]
        # This shows "where the image is looking on average"
        attn_weights = img_to_img_attn.mean(axis=0)
        
        # Normalize
        attn_weights = (attn_weights - attn_weights.min()) / (attn_weights.max() - attn_weights.min() + 1e-9)
        
        # Reshape to spatial grid
        try:
            attn_map_spatial = attn_weights.reshape(grid_h, grid_w)
        except ValueError as e:
            print(f"  Skipping: cannot reshape {attn_weights.shape} to ({grid_h}, {grid_w}): {e}")
            continue

        resize_transform = transforms.Resize((image.height, image.width), antialias=True)
        attn_map_resized = resize_transform(torch.tensor(attn_map_spatial).unsqueeze(0).unsqueeze(0)).squeeze().numpy()

        plt.figure(figsize=(12, 6))

        # Original Image
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.axis("off")
        plt.title("Original Image")

        plt.subplot(1, 2, 2)
        plt.imshow(image)
        plt.imshow(attn_map_resized, cmap="jet", alpha=0.5)
        plt.axis("off")
        plt.title(f"Attention Heatmap - Layer {layer_idx+1}")

        heatmap_filename = os.path.join(output_dir, f"attention_layer_original_{layer_idx+1}.png")
        plt.savefig(heatmap_filename, dpi=300)
        print(f"Saved heatmap: {heatmap_filename}")
        plt.close()