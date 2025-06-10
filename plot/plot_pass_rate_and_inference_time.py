import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import json
import pandas as pd
import os

def create_combined_plot(data1, data2, jsonl_file, save_dir='./fig'):
    """
    Create a combined figure with pass rate distributions and timing comparison.
    
    Args:
        data1: Pass rate data for Qwen2.5-Math-1.5B
        data2: Pass rate data for Qwen2.5-Math-7B
        jsonl_file: Path to timing data file
        save_dir: Directory to save the plot
    """
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6), gridspec_kw={'width_ratios': [1.2, 1.2, 0.8]})
    
    # Common settings
    bins = np.arange(0, 1.05, 0.05)
    fontfamily = 'serif'
    fontsize_title = 15
    fontsize_label = 15
    line_alpha = 0.3
    grid_color = '#E0E0E0'
    
    # Style settings for all axes
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(1.5)
        ax.yaxis.grid(True, color=grid_color, linestyle='-', linewidth=2, alpha=0.5)
        ax.set_axisbelow(True)
        # Increase y tick font size for all figures
        ax.tick_params(axis='y', labelsize=13)
    
    # Increase x tick font size for pass rate figures
    ax1.tick_params(axis='x', labelsize=13)
    ax2.tick_params(axis='x', labelsize=13)
    
    # Plot first pass rate histogram
    counts1, bins1, _ = ax1.hist(data1, bins=bins, density=False, alpha=0.7, color='skyblue', edgecolor='black')
    for i in range(len(counts1)):
        if counts1[i] > 0:
            ax1.text(bins1[i] + 0.025, counts1[i], int(counts1[i]), 
                    ha='center', va='bottom', fontsize=11, fontfamily=fontfamily)
    
    ax1.set_title('DAPO-1k', fontsize=fontsize_title, fontfamily=fontfamily, pad=0, weight='bold')
    ax1.set_xlabel('')
    ax1.set_ylabel('Number of prompts', fontsize=fontsize_label, fontfamily=fontfamily, weight='bold')
    
    # Plot second pass rate histogram
    counts2, bins2, _ = ax2.hist(data2, bins=bins, density=False, alpha=0.7, color='lightgreen', edgecolor='black')
    for i in range(len(counts2)):
        if counts2[i] > 0:
            ax2.text(bins2[i] + 0.025, counts2[i], int(counts2[i]), 
                    ha='center', va='bottom', fontsize=11, fontfamily=fontfamily)
    
    ax2.set_title('Math500', fontsize=fontsize_title, fontfamily=fontfamily, pad=0, weight='bold')
    ax2.set_xlabel('')
    
    # Add common x-label for the first two subplots
    fig.text(0.35, -0.01, 'Pass@1 Rate', fontsize=fontsize_label, fontfamily=fontfamily, weight='bold', ha='center')
    fig.text(0.88, -0.01, 'Components within one step', fontsize=fontsize_label, fontfamily=fontfamily, weight='bold', ha='center')
    
    # Plot timing information
    data = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    timing_data = []
    for entry in data:
        timing_info = {}
        timing_info['step'] = entry.get('step', None)
        for key, value in entry.items():
            if key.startswith('timing'):
                timing_info[key] = value
        timing_data.append(timing_info)
    
    df = pd.DataFrame(timing_data)
    gen_avg = df['timing_s/gen'].mean()
    combined_avg = (df['timing_s/old_log_prob'] + df['timing_s/update_actor']).mean()
    
    bars = ax3.bar(['Inference', 'Training'], 
                  [gen_avg, combined_avg],
                  color=['#2E86C1', '#E74C3C'])
    
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom',
                fontfamily=fontfamily, fontsize=fontsize_title,
                weight='bold')
    
    ax3.set_ylabel('Average Time (seconds)', fontfamily=fontfamily, weight='bold', fontsize=fontsize_label)
    ax3.set_xticklabels(['Inference', 'Training'], fontfamily=fontfamily, weight='bold', fontsize=fontsize_label)
    ax3.set_title('Qwen2.5-Math-7B', fontsize=fontsize_title, fontfamily=fontfamily, pad=0, weight='bold')
    ax3.set_ylim(0, 80)
    
    # Adjust layout and save
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'passrate_distribution_inference_time.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    passrate_file = "/u/rzhang15/projects/DAPO/metrics_summary/passrate_distribution.jsonl"
    with open(passrate_file, 'r') as f:
        data = [json.loads(line) for line in f][0]
    data1 = data['Qwen2.5-Math-1.5B']
    data2 = data['Qwen2.5-Math-1.5B-Math500']
    
    
    # Timing data file
    jsonl_file = "/u/rzhang15/projects/DAPO/metrics/7B-Math-FAST-RLOO-DeepScaleR-N4+20-offload_20250511_204038.jsonl"
    
    create_combined_plot(data1, data2, jsonl_file) 