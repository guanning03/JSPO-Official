import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd
import os

def plot_grad_norms(file1, file2, save_dir):
    """
    Plot gradient norms from two JSONL files with specified formatting.
    
    Args:
        file1 (str): Path to first JSONL file
        file2 (str): Path to second JSONL file
        save_dir (str): Directory to save the plot
    """

    font = 'serif'
    label_fontsize = 19
    tick_fontsize = 15
    legend_fontsize = 19
    linewidth = 2.5
    ema_alpha = 0.5
    colors = ['#2E86C1', '#E74C3C']  # Blue and Red colors

    # Read and process data from files
    def process_jsonl(filename):
        steps, grad_norms = [], []
        with open(filename, 'r') as f:
            for line in f:
                if 'actor/grad_norm' in line:
                    data = json.loads(line)
                    if data['step'] <= 800:
                        steps.append(data['step'])
                        grad_norms.append(data['actor/grad_norm'])
        return np.array(steps), np.array(grad_norms)
    
    # Get data from both files
    steps1, grad_norms1 = process_jsonl(file1)
    steps2, grad_norms2 = process_jsonl(file2)
    
    # Calculate EMA for smoothing
    def calculate_ema(data, alpha):
        return pd.Series(data).ewm(alpha=alpha, adjust=False).mean()
    
    smooth_grad_norms1 = calculate_ema(grad_norms1, ema_alpha)
    smooth_grad_norms2 = calculate_ema(grad_norms2, ema_alpha)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Add horizontal grid only
    ax.yaxis.grid(True, linestyle='--', color='gray', alpha=0.2, zorder=0)
    ax.xaxis.grid(False)
    
    # Plot original data (light) and smoothed data (bold)
    ax.plot(steps1, grad_norms1, alpha=0.3, color=colors[0], linewidth=linewidth/2)
    ax.plot(steps1, smooth_grad_norms1, color=colors[0], label=f'SPEED-RLOO', linewidth=linewidth)
    ax.plot(steps2, grad_norms2, alpha=0.3, color=colors[1], linewidth=linewidth/2)
    ax.plot(steps2, smooth_grad_norms2, color=colors[1], label=f'RLOO', linewidth=linewidth)
    
    # Format the plot
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    
    # Set y-axis limits
    ax.set_ylim(-0.1, 1.19)
    
    # Remove ticks but set font size for tick labels
    ax.tick_params(length=0)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    for tick in ax.get_xticklabels():
        tick.set_fontfamily(font)
    for tick in ax.get_yticklabels():
        tick.set_fontfamily(font)
    
    # Set labels with bold font
    ax.set_xlabel('Steps', fontweight='bold', fontsize=label_fontsize, fontfamily=font)
    ax.set_ylabel('Gradient Norm', fontweight='bold', fontsize=label_fontsize, fontfamily=font)
    
    # Format legend
    legend = ax.legend(prop={'family': font, 'size': legend_fontsize, 'weight': 'bold'})
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'grad_norm.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

# Example usage:
if __name__ == "__main__":
    # Replace these with your actual file paths
    file1 = "/u/rzhang15/projects/DAPO/metrics/7B-Math-FAST-RLOO-DeepScaleR-N4+20-offload_20250511_204038.jsonl"
    file2 = "/u/rzhang15/projects/DAPO/metrics/7B-Math-RLOO-DeepScaleR-N24-offload_20250511_203242.jsonl"
    save_dir = "./fig"
    plot_grad_norms(file1, file2, save_dir)
