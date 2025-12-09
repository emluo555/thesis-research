import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np

def plot_token_scores(csv_path, start_idx=None, end_idx=None):
    """
    Read CSV and plot average mean scores across token indices.
    
    Args:
        csv_path: Path to the CSV file
    """
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Group by token_idx and calculate the average of mean scores
    grouped = df.groupby('token_idx').agg({
        'mean_vision_score': 'mean',
        'mean_text_score': 'mean'
    }).reset_index()
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    plt.plot(grouped['token_idx'], grouped['mean_vision_score'], 
             marker='o', label='Vision Score', linewidth=2, markersize=4)
    plt.plot(grouped['token_idx'], grouped['mean_text_score'], 
             marker='s', label='Text Score', linewidth=2, markersize=4)

    if start_idx is not None:
        plt.axvline(x=start_idx, color='red', linestyle='--', linewidth=2, 
                   label=f'Answer Start (idx {start_idx})')
    if end_idx is not None:
        plt.axvline(x=end_idx, color='red', linestyle='--', linewidth=2, 
                   label=f'Answer End (idx {end_idx})')
                   
    plt.xlabel('Token Index', fontsize=12)
    plt.ylabel('Average Mean Score', fontsize=12)
    plt.title('Average Mean Scores Across Token Indices', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    output_path = csv_path.rsplit('.', 1)[0] + '_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    

def plot_last_layer_token_scores(csv_path, start_idx=None, end_idx=None):
    """
    Plot mean scores across all tokens using only the last layer for each token.
    
    Args:
        csv_path: Path to the CSV file
    """
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # For each token, get only the last (maximum) layer
    last_layer_df = df.loc[df.groupby('token_idx')['layer_idx'].idxmax()]
    
    # Sort by token_idx to ensure proper ordering
    last_layer_df = last_layer_df.sort_values('token_idx')
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    plt.plot(last_layer_df['token_idx'], last_layer_df['mean_vision_score'], 
             marker='o', label='Vision Score', linewidth=2, markersize=4)
    plt.plot(last_layer_df['token_idx'], last_layer_df['mean_text_score'], 
             marker='s', label='Text Score', linewidth=2, markersize=4)
    
    if start_idx is not None:
        plt.axvline(x=start_idx, color='red', linestyle='--', linewidth=2, 
                   label=f'Answer Start (idx {start_idx})')
    if end_idx is not None:
        plt.axvline(x=end_idx, color='red', linestyle='--', linewidth=2, 
                   label=f'Answer End (idx {end_idx})')
    
    plt.xlabel('Token Index', fontsize=12)
    plt.ylabel('Mean Score', fontsize=12)
    plt.title('Mean Scores Across Tokens (Last Layer Only)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    output_path = csv_path.rsplit('.', 1)[0] + '_last_layer_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

def plot_token_top_activations(
    csv_path,
    token_idx,
    num_top_tokens=3,
    vision_prefix="img_token",
    text_prefix="txt_token"
):
    df = pd.read_csv(csv_path)
    df_token = df[df['token_idx'] == token_idx]

    if df_token.empty:
        print(f"No rows found with token_idx={token_idx}")
        return

    vision_cols = [c for c in df.columns if c.startswith(vision_prefix)]
    text_cols = [c for c in df.columns if c.startswith(text_prefix)]

    vision_means = df_token[vision_cols].mean(axis=0)
    text_means = df_token[text_cols].fillna(0).mean(axis=0)

    top_vision = vision_means.nlargest(num_top_tokens)
    top_text = text_means.nlargest(num_top_tokens)

    print("Top vision tokens:", top_vision.index.tolist())
    print("Top text tokens:", top_text.index.tolist())

    layers = df_token["layer_idx"].values

    plt.figure(figsize=(12, 7))

    for col in top_vision.index:
        plt.plot(layers, df_token[col], label=f"{col} (vision)")

    for col in top_text.index:
        plt.plot(layers, df_token[col].fillna(0), linestyle="--", label=f"{col} (text)")

    plt.xlabel("Layer Index")
    plt.ylabel("Activation Value")
    plt.title(f"Top Vision/Text Token Activations Across Layers (token_idx={token_idx})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = csv_path.rsplit('.', 1)[0] + f'_activations_{token_idx}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

def plot_topk_mean_lines(csv_path, top_k=5):
    df = pd.read_csv(csv_path)
    df = df.fillna(0)

    # Identify token columns
    vision_cols = [c for c in df.columns if c.startswith("img_token")]
    text_cols   = [c for c in df.columns if c.startswith("txt_token")]

    # Average over layers first
    df_avg = df.groupby("token_idx").mean(numeric_only=True).reset_index()

    # Compute mean activation for each vision token
    vision_means = df_avg[vision_cols].mean(axis=0)
    top_vision = vision_means.sort_values(ascending=False).head(top_k).index.tolist()

    # Compute mean activation for each text token
    text_means = df_avg[text_cols].mean(axis=0)
    top_text = text_means.sort_values(ascending=False).head(top_k).index.tolist()

    plt.figure(figsize=(14, 7))
    for col in top_vision:
        plt.plot(df_avg["token_idx"], df_avg[col], label=f"{col} (vision)")

    for col in top_text:
        plt.plot(df_avg["token_idx"], df_avg[col], label=f"{col} (text)", linestyle="--")
    
    plt.xlabel("Output Token Index")
    plt.ylabel("Activation (Avg across Layers)")
    plt.title(f"Top {top_k} Vision/Text Token Activations Over Generated Tokens")
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.0))  # move legend outside
    plt.tight_layout()
    output_path = csv_path.rsplit('.', 1)[0] + f'_topk_{top_k}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_csv>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    start_idx = int(sys.argv[2]) if len(sys.argv) > 2 else None
    end_idx = int(sys.argv[3]) if len(sys.argv) > 3 else None

    # plot_token_scores(csv_file, start_idx, end_idx)
    # plot_last_layer_token_scores(csv_file, start_idx, end_idx)
    plot_token_top_activations(csv_file, 3)
    plot_topk_mean_lines(csv_file, top_k=5)