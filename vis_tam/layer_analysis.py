import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy.special import softmax
from pathlib import Path


def plot_activations(all_scores, img_scores, txt_scores, path, model_name, target_token, time_step = 0):
    img_mean = img_scores.mean()
    txt_mean = txt_scores.mean() 
    print(f"image stats: img_min: {img_scores.min()} img_max: {img_scores.max()} img_mean: {img_mean}")
    print(f"text stats: txt min: {txt_scores.min()} txt max: {txt_scores.max()} txt_mean: {txt_mean}")

    values = [img_mean, txt_mean]
    softmax_values = softmax(values)
    print(softmax_values)

    labels = ['Image Tokens', 'Text Tokens']

    x = np.arange(len(values))

    plt.figure(figsize=(5,4))
    plt.bar(x, values)
    plt.xticks(x, labels)
    plt.ylabel("Average Token Activation")
    plt.title(f"Token Activation in {model_name} for Token {target_token}")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_attn_alloc(all_scores, img_scores, txt_scores, path, model_name, target_token):
    print('all scores shape: ', all_scores.shape)
    x = np.arange(all_scores.shape[1])

    colors = ['#eaa1a6', '#afd3e6']  # Visual / Text


    fig, ax = plt.subplots(figsize=(7.5, 5))

    ax.bar(x, img_scores, color=colors[0], edgecolor='gray', linewidth=0.5, label='Visual Features')
    ax.bar(x, txt_scores, color=colors[1], bottom=img_scores, edgecolor='gray', linewidth=0.5,
           label='User Instructions')


    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Token Activations', fontsize=12, fontweight='bold')


    ax.set_xticks(x[::3])
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_ylim(0, 1)

    ax.tick_params(axis='both', labelsize=16, width=0.5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    plt.title(f'Token Activation Across Layers in {model_name} for Token {target_token}', fontsize=6, fontweight='bold', pad=25)

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


