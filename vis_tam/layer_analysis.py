import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy.special import softmax
from pathlib import Path


def plot_activations(img_scores, txt_scores, path, model_name, target_token_idx, target_token):
    img_mean = img_scores.mean()
    txt_mean = txt_scores.mean() 
    # print(f"image stats: img_min: {img_scores.min()} img_max: {img_scores.max()} img_mean: {img_mean}")
    # print(f"text stats: txt min: {txt_scores.min()} txt max: {txt_scores.max()} txt_mean: {txt_mean}")

    values = [img_mean, txt_mean]
    softmax_values = softmax(values)
    # print(f"Softmax values: {softmax_values}")

    labels = ['Image Tokens', 'Text Tokens']
    x = np.arange(len(values))

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1: Average Token Activation
    ax1.bar(x, values)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Average Token Activation")
    ax1.set_title(f"Token Activation")
    ax1.set_ylim(0, 1)
    ax1.set_yticks(np.arange(0, 1.1, 0.1))
    
    # Plot 2: Softmax Values
    colors = ['#eaa1a6', '#afd3e6']  # Visual / Text

    ax2.bar(0, softmax_values[0], color=colors[0], edgecolor='gray', linewidth=0.5, 
            label='Visual Features')
    ax2.bar(0, softmax_values[1], color=colors[1], bottom=softmax_values[0], 
            edgecolor='gray', linewidth=0.5, label='User Instructions')
    
    ax2.set_xticks([])
    ax2.set_ylabel("Softmax Probability", fontsize=12, fontweight='bold')
    ax2.set_title(f"Softmax Distribution", fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='y', labelsize=12, width=0.5)
    for label in ax2.get_yticklabels():
        label.set_fontweight('bold')
    for spine in ax2.spines.values():
        spine.set_linewidth(1)

    legend = ax2.legend(
        loc='upper right',
        bbox_to_anchor=(1.0, 1.0),
        ncol=1,
        frameon=True,
        edgecolor='black',
        fontsize=10
    )
    
    # Add overall title
    fig.suptitle(f"{model_name} - Token #{target_token_idx}: {target_token}", 
                 fontsize=12, y=1.02)
    
    plt.tight_layout()
    
    # Create dynamic path
    save_path = Path(path) / f"{model_name}_token_{target_token_idx}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {save_path}")
    return save_path


def plot_activations_all_layers(img_scores_per_layer, txt_scores_per_layer, 
                                path, model_name, target_token_idx, target_token):
    """
    Plot img_scores and txt_scores across all layers on one graph.
    
    Args:
        img_scores_per_layer: list of np.ndarray, one per layer
        txt_scores_per_layer: list of np.ndarray, one per layer  
        path: Path to save the plot
        model_name: Name of the model
        target_token_idx: Index of target token
        target_token: String representation of target token
    """
    num_layers = len(img_scores_per_layer)
    
    # Compute mean activation per layer
    img_means = [scores.mean() for scores in img_scores_per_layer]
    txt_means = [scores.mean() for scores in txt_scores_per_layer]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    layer_indices = np.arange(1, num_layers + 1)
    
    # Plot 1: Mean activation across layers
    ax1.plot(layer_indices, img_means, 'o-', color='#eaa1a6', linewidth=2, 
             markersize=6, label='Image Tokens', markerfacecolor='white', markeredgewidth=2)
    ax1.plot(layer_indices, txt_means, 's-', color='#afd3e6', linewidth=2, 
             markersize=6, label='Text Tokens', markerfacecolor='white', markeredgewidth=2)
    ax1.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Activation', fontsize=12, fontweight='bold')
    ax1.set_title('Mean Activation Across Layers', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10, frameon=True, edgecolor='black')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(layer_indices[::max(1, num_layers//10)])  # Show every nth layer
    
    # Plot 2: Stacked bar chart showing ratio at each layer
    img_softmax_per_layer = []
    txt_softmax_per_layer = []
    for i in range(num_layers):
        values = [img_means[i], txt_means[i]]
        softmax_vals = softmax(values)
        img_softmax_per_layer.append(softmax_vals[0])
        txt_softmax_per_layer.append(softmax_vals[1])
    
    x = layer_indices
    width = 0.8
    ax2.bar(x, img_softmax_per_layer, width, label='Image Tokens', 
            color='#eaa1a6', edgecolor='gray', linewidth=0.5)
    ax2.bar(x, txt_softmax_per_layer, width, bottom=img_softmax_per_layer,
            label='Text Tokens', color='#afd3e6', edgecolor='gray', linewidth=0.5)
    ax2.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Softmax Probability', fontsize=12, fontweight='bold')
    ax2.set_title('Softmax Distribution Across Layers', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=10, frameon=True, edgecolor='black')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(layer_indices[::max(1, num_layers//10)])
    
    # Overall title
    fig.suptitle(f"{model_name} - Token #{target_token_idx}: {target_token}", 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save
    save_path = Path(path) / f"{model_name}_token_{target_token_idx}_all_layers.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"All-layers plot saved to: {save_path}")
    return save_path


