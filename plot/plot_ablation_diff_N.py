import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

def compute_ema(values, times, alpha=0.9):
    """
    Compute exponential moving average for a list of values.
    
    Args:
        values (list): List of values to compute EMA for
        times (list): List of corresponding times/steps
        alpha (float): Smoothing factor between 0 and 1
                      Lower alpha means more smoothing
    
    Returns:
        list: EMA-smoothed values
    """
    if len(values) != len(times):
        raise ValueError("values and times must have the same length")
    
    ema = [values[0]]  # Initialize with first value
    
    for i in range(1, len(values)):
        # Compute time-aware alpha
        time_diff = times[i] - times[i-1]
        adjusted_alpha = 1 - (1 - alpha) ** time_diff
        
        # Update EMA
        ema.append(adjusted_alpha * values[i] + (1 - adjusted_alpha) * ema[-1])
    
    return ema

def compute_ema_steps(values, alpha=0.9):
    return pd.Series(values).ewm(alpha=alpha, adjust=False).mean()

def process_jsonl_data(filename, metric_key, x_axis_type='step', x_lim=800):
    """
    Process JSONL file and extract specified metric.
    
    Args:
        filename (str): Path to JSONL file
        metric_key (str): Key to extract from JSON data
        x_axis_type (str): Either 'step' for global steps or 'time' for accumulated training time
        x_lim (int): Maximum x value to include
    """
    x_values, metric_values = [], []
    accumulated_time = 0  # Track accumulated training time
    
    with open(filename, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            # Get x value based on type
            if x_axis_type == 'step':
                x_val = data['train_global_steps'] if 'train_global_steps' in data else data['step']
            else:  # time
                # Add the pure training time to accumulator
                if 'timing_s/time_pure_training' in data:
                    accumulated_time += data['timing_s/time_pure_training']
                x_val = accumulated_time
            
            # Only process if metric exists and x_val is within limit
            if metric_key in data and x_val <= x_lim:
                x_values.append(x_val)
                metric_values.append(data[metric_key])
                
    return np.array(x_values), np.array(metric_values)

def plot_ablation_figure(file_paths, algorithm_names, save_dir,
                        time_limit=3600*7,  # time limit in seconds for first subplot
                        step_limit=500,     # step limit for other subplots
                        line_width=2.5,
                        ema_alpha_acc=0.9,
                        ema_alpha_grad=0.3,
                        ema_alpha_train=0.01):
    """
    Plot three metrics in a row: accuracy vs time, gradient norm, and training accuracy.
    
    Args:
        file_paths (list): List of paths to jsonl files
        algorithm_names (list): Names of algorithms for legend
        save_dir (str): Directory to save the plot
        time_limit (int): Maximum time in seconds to include in first plot
        step_limit (int): Maximum steps to include in other plots
        line_width (float): Width of the plotted lines
        ema_alpha_acc (float): Smoothing factor for accuracy EMA
        ema_alpha_grad (float): Smoothing factor for gradient norm EMA
        ema_alpha_train (float): Smoothing factor for training accuracy EMA
    """
    # Styling parameters
    font = 'serif'
    label_fontsize = 19
    tick_fontsize = 15
    legend_fontsize = 19
    colors = ['#2E86C1', '#E74C3C', '#27AE60', '#8E44AD']  # Blue, Red, Green, Purple
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
    
    # Process each file
    for file_idx, file_path in enumerate(file_paths):
        color = colors[file_idx % len(colors)]
        
        # Plot 1: Accuracy vs Time
        x_vals_acc, acc_values = process_jsonl_data(
            file_path, 
            "val-core/math_dapo-Qwen-instruct/acc/mean@1",
            x_axis_type='time',
            x_lim=time_limit
        )
        x_vals_acc = x_vals_acc / 3600  # Convert to hours
        smooth_acc = compute_ema(acc_values, x_vals_acc, ema_alpha_acc)
        ax1.plot(x_vals_acc, acc_values, alpha=0.2, color=color, linewidth=3.5)
        ax1.plot(x_vals_acc, smooth_acc, color=color, label=algorithm_names[file_idx], linewidth=3.5)
        
        # Plot 2: Gradient Norm vs Steps
        x_vals_grad, grad_values = process_jsonl_data(
            file_path,
            "actor/grad_norm",
            x_axis_type='step',
            x_lim=step_limit
        )
        smooth_grad = compute_ema_steps(grad_values, ema_alpha_grad)
        ax2.plot(x_vals_grad, grad_values, alpha=0.2, color=color, linewidth=line_width/2)
        ax2.plot(x_vals_grad, smooth_grad, color=color, label=algorithm_names[file_idx], linewidth=3)
        
        # Plot 3: Training Accuracy vs Steps
        x_vals_train, train_values = process_jsonl_data(
            file_path,
            "critic/rewards/mean",
            x_axis_type='step',
            x_lim=step_limit
        )
        smooth_train = compute_ema_steps(train_values, ema_alpha_train)
        ax3.plot(x_vals_train, train_values, alpha=0.2, color=color, linewidth=line_width/2)
        ax3.plot(x_vals_train, smooth_train, color=color, label=algorithm_names[file_idx], linewidth=line_width*2)
    
    # Configure subplots
    for ax_idx, (ax, title, x_label) in enumerate([
        (ax1, 'Validation Accuracy', 'Training Time (hours)'),
        (ax2, 'Gradient Norm', 'Steps'),
        (ax3, 'Training Accuracy', 'Steps')
    ]):
        # Add grid
        ax.yaxis.grid(True, linestyle='--', color='gray', alpha=0.2, zorder=0)
        ax.xaxis.grid(False)
        
        # Format plot with black frames
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(2.0)
        
        # Set y-axis limits for grad norm
        if ax_idx == 1:  # Gradient norm plot
            ax.set_ylim(0.0, 0.2)
        elif ax_idx == 2:
            ax.set_ylim(0.1, 0.6)
        
        # Set x-axis limits
        if ax_idx == 0:  # First subplot (time-based)
            ax.set_xlim(0, time_limit/3600)  # Convert seconds to hours
        else:  # Other subplots (step-based)
            ax.set_xlim(0, step_limit)
        
        # Remove ticks but keep labels
        ax.tick_params(length=0)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontfamily(font)
        
        # Set title and labels
        ax.set_title(title, fontweight='bold', fontsize=label_fontsize, fontfamily=font)
        # ax.set_ylabel(title, fontweight='bold', fontsize=label_fontsize, fontfamily=font)
        ax.set_xlabel(x_label, fontweight='bold', fontsize=label_fontsize, fontfamily=font)
    
    # Create common legend at the bottom center
    lines = []
    labels = []
    for ax in [ax1, ax2, ax3]:
        ax_lines, ax_labels = ax.get_legend_handles_labels()
        if not lines:  # Only take the first set of lines and labels
            lines = ax_lines
            labels = ax_labels
        ax.get_legend().remove() if ax.get_legend() is not None else None
    
    # Make legend lines wider
    for line in lines:
        line.set_linewidth(6.0)  # Increase line width in legend
    
    fig.legend(
        lines, labels,
        bbox_to_anchor=(0.5, 0.11),
        loc='center',
        ncol=len(file_paths),
        prop={'family': font, 'size': legend_fontsize, 'weight': 'bold'}
    )
    
    plt.tight_layout()
    # Adjust bottom margin to make room for legend
    plt.subplots_adjust(bottom=0.25)
    
    # Save the plot
    save_path = os.path.join(save_dir, 'ablation_diff_N.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

# Example usage:
if __name__ == "__main__":
    # Replace these with your actual file paths
    file_paths = [
        "/u/rzhang15/projects/DAPO/metrics/1.5B-Math-RLOO-dataDAPO-N24-offload_20250509_052640.jsonl",
        "/u/rzhang15/projects/DAPO/metrics/1.5B-Math-FAST-RLOO-dataDAPO-N4+20-offload_20250509_161243.jsonl",
        "/u/rzhang15/projects/DAPO/metrics/1.5B-Math-FAST-RLOO-dataDAPO-N6+18-offload_20250509_121842.jsonl",
        "/u/rzhang15/projects/DAPO/metrics/1.5B-Math-FAST-RLOO-dataDAPO-N8+16-offload_20250509_161202.jsonl"
    ]
    algorithm_names = [
        "RLOO",
        "SPEED-RLOO, N_init=4",
        "SPEED-RLOO, N_init=6",
        "SPEED-RLOO, N_init=8"
    ]
    save_dir = "./fig"
    
    plot_ablation_figure(
        file_paths=file_paths,
        algorithm_names=algorithm_names,
        save_dir=save_dir,
        time_limit=3600*7,  # 7 hours for first subplot
        step_limit=400,     # 500 steps for other subplots
        line_width=2.5,
        ema_alpha_acc=0.9,
        ema_alpha_grad=0.3,
        ema_alpha_train=0.01
    )
