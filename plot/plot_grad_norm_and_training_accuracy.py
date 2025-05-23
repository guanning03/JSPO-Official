import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd
import os

def process_jsonl(filename, metric_key):
    """
    Process JSONL file and extract specified metric.
    
    Args:
        filename (str): Path to JSONL file
        metric_key (str): Key to extract from JSON data
    """
    steps, values = [], []
    with open(filename, 'r') as f:
        for line in f:
            if metric_key in line:
                data = json.loads(line)
                if data['step'] <= 800:
                    steps.append(data['step'])
                    values.append(data[metric_key])
    return np.array(steps), np.array(values)

def calculate_ema(data, alpha):
    """Calculate exponential moving average."""
    return pd.Series(data).ewm(alpha=alpha, adjust=False).mean()

def plot_metrics(file1, file2, save_dir):
    """
    Plot rewards and gradient norms from two JSONL files with specified formatting.
    
    Args:
        file1 (str): Path to first JSONL file
        file2 (str): Path to second JSONL file
        save_dir (str): Directory to save the plot
    """
    # Styling parameters
    font = 'serif'
    label_fontsize = 19
    tick_fontsize = 15
    legend_fontsize = 19
    linewidth = 2.5
    colors = ['#E74C3C', '#2E86C1']  # Blue and Red colors
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    
    # Plot 1: Mean Rewards
    steps1_rewards, values1_rewards = process_jsonl(file1, 'critic/rewards/mean')
    steps2_rewards, values2_rewards = process_jsonl(file2, 'critic/rewards/mean')
    
    # Different EMA alpha for rewards
    ema_alpha_rewards = 0.01
    smooth_values1_rewards = calculate_ema(values1_rewards, ema_alpha_rewards)
    smooth_values2_rewards = calculate_ema(values2_rewards, ema_alpha_rewards)
    
    # Configure rewards plot
    ax1.yaxis.grid(True, linestyle='--', color='gray', alpha=0.4, zorder=0)
    ax1.xaxis.grid(False)
    
    # Add horizontal line at y=0.5
    ax1.axhline(y=0.5, color='black', linestyle='-', linewidth=linewidth, zorder=1)
    
    # Plot data with higher zorder to appear above the horizontal line
    line1 = ax1.plot(steps1_rewards, values1_rewards, alpha=0.3, color=colors[0], linewidth=linewidth/2, zorder=2)[0]
    line2 = ax1.plot(steps1_rewards, smooth_values1_rewards, color=colors[0], label='SPEED-RLOO', linewidth=linewidth*2, zorder=2)[0]
    line3 = ax1.plot(steps2_rewards, values2_rewards, alpha=0.3, color=colors[1], linewidth=linewidth/2, zorder=2)[0]
    line4 = ax1.plot(steps2_rewards, smooth_values2_rewards, color=colors[1], label='RLOO', linewidth=linewidth*2, zorder=2)[0]
    
    # Format rewards plot
    ax1.spines['top'].set_visible(True)
    ax1.spines['right'].set_visible(True)
    ax1.spines['bottom'].set_visible(True)
    ax1.spines['left'].set_visible(True)
    
    ax1.tick_params(length=0)
    ax1.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    for tick in ax1.get_xticklabels():
        tick.set_fontfamily(font)
    for tick in ax1.get_yticklabels():
        tick.set_fontfamily(font)
    
    ax1.set_ylabel('Average Training Accuracy', fontweight='bold', fontsize=label_fontsize, fontfamily=font)
    
    # Plot 2: Gradient Norm
    steps1_grad, values1_grad = process_jsonl(file1, 'actor/grad_norm')
    steps2_grad, values2_grad = process_jsonl(file2, 'actor/grad_norm')
    
    # Different EMA alpha for gradient norm
    ema_alpha_grad = 0.3
    smooth_values1_grad = calculate_ema(values1_grad, ema_alpha_grad)
    smooth_values2_grad = calculate_ema(values2_grad, ema_alpha_grad)
    
    # Configure gradient norm plot
    ax2.yaxis.grid(True, linestyle='--', color='gray', alpha=0.4, zorder=0)
    ax2.xaxis.grid(False)
    
    ax2.plot(steps1_grad, values1_grad, alpha=0.4, color=colors[0], linewidth=linewidth/2)
    ax2.plot(steps1_grad, smooth_values1_grad, color=colors[0], label='SPEED-RLOO', linewidth=linewidth)
    ax2.plot(steps2_grad, values2_grad, alpha=0.4, color=colors[1], linewidth=linewidth/2)
    ax2.plot(steps2_grad, smooth_values2_grad, color=colors[1], label='RLOO', linewidth=linewidth)
    
    # Format gradient norm plot
    ax2.spines['top'].set_visible(True)
    ax2.spines['right'].set_visible(True)
    ax2.spines['bottom'].set_visible(True)
    ax2.spines['left'].set_visible(True)
    
    ax2.set_ylim(-0.1, 1.19)
    
    ax2.tick_params(length=0)
    ax2.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    for tick in ax2.get_xticklabels():
        tick.set_fontfamily(font)
    for tick in ax2.get_yticklabels():
        tick.set_fontfamily(font)
    
    ax2.set_ylabel('Average Gradient Norm', fontweight='bold', fontsize=label_fontsize, fontfamily=font)
    
    # Add common x-label and legend
    fig.text(0.45, 0.1, 'Steps', fontweight='bold', fontsize=label_fontsize, fontfamily=font, ha='center')
    
    # Create common legend
    lines = [line2, line4]  # Use only the smooth lines for legend
    labels = ['SPEED-RLOO', 'RLOO']
    fig.legend(lines, labels, 
              loc='center', 
              bbox_to_anchor=(0.65, 0.11),
              ncol=2,
              prop={'family': font, 'size': legend_fontsize, 'weight': 'bold'})
    
    plt.tight_layout()
    # Adjust bottom margin to make room for x-label and legend
    plt.subplots_adjust(bottom=0.2)
    
    save_path = os.path.join(save_dir, 'training_rewards_and_grad_norm.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

# Example usage:
if __name__ == "__main__":
    # Replace these with your actual file paths
    file1 = "/u/rzhang15/projects/DAPO/metrics/7B-Math-FAST-RLOO-DeepScaleR-N4+20-offload_20250511_204038.jsonl"
    file2 = "/u/rzhang15/projects/DAPO/metrics/7B-Math-RLOO-DeepScaleR-N24-offload_20250511_203242.jsonl"
    save_dir = "./fig"
    plot_metrics(file1, file2, save_dir)
