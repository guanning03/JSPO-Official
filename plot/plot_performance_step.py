import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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

def plot_metrics_comparison(file_paths, 
                          algorithm_names, 
                          line_width=2, 
                          marker_size=3, 
                          marker_style='o', 
                          x_lim=500,  # max steps or seconds to include
                          x_axis_type='step',  # 'step' or 'time'
                          ema_alpha=0.1  # EMA smoothing factor
                          ):
    """
    Plot metrics comparison from multiple jsonl files.
    
    Args:
        file_paths (list): List of paths to jsonl files
        algorithm_names (list): Names of algorithms for legend
        line_width (float): Width of the plotted lines
        marker_size (float): Size of the markers
        marker_style (str): Style of the markers
        x_lim (int): Maximum steps or seconds to include in plot
        x_axis_type (str): Either 'step' for global steps or 'time' for accumulated training time
        ema_alpha (float): Smoothing factor for exponential moving average
    """
    # Input validation
    if x_axis_type not in ['step', 'time']:
        raise ValueError("x_axis_type must be either 'step' or 'time'")
    
    # Set style
    sns.set_palette("husl")
    plt.rcParams['grid.alpha'] = 0.2
    plt.rcParams['grid.color'] = '#666666'
    
    # Metrics to plot (4 metrics now, with AIME metrics merged)
    if '7B' in file_paths[0] or 'dataDAPO' in file_paths[0] or 'DAPO-NuminaMath' in file_paths[0]:
        metrics = [
                "val-core/math_dapo-Qwen-instruct/acc/mean@1",
                "val-core/MATH500-Qwen-instruct/acc/mean@1",
                "val-core/AMC-Qwen-instruct/acc/mean@4",
                ["val-core/AIME2025-Qwen-instruct/acc/mean@16",
                "val-core/AIME-Qwen-instruct/acc/mean@16"]  # AIME metrics to average
        ]
    elif '1.5B' in file_paths[0]:
        metrics = [
            "val-core/math_dapo-Qwen-instruct/acc/mean@1",
            "val-core/MATH500-Qwen-instruct/acc/mean@1",
            "val-core/AMC-Qwen-instruct/acc/mean@1",
            ["val-core/AIME2025-Qwen-instruct/acc/mean@4",
            "val-core/AIME-Qwen-instruct/acc/mean@4"]  # AIME metrics to average
        ]
    else:
        raise ValueError("Unknown model size")
    
    # Shorter names for titles
    metric_titles = [
        "DAPO-1k",
        "MATH500",
        "AMC2023",
        "AIME"  # Combined AIME title
    ]

    # Set the parameters
    fontfamily = 'serif'
    font_size_title = 15
    font_size_label = 15
    font_size_legend = 15
    font_size_tick = 11
    line_width = 2
    marker_size = 3
    marker_style = 'o'
    ema_alpha = 0.6
    
    fontweight_label = 'bold'
    fontweight_title = 'bold'
    fontweight_legend = 'bold'
    
    # Create figure and subplots with extra space for subtitles
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))  # Changed to 4 subplots
    fig.subplots_adjust(bottom=0.25, top=0.85, wspace=0.3)
    
    # Store lines for legend
    legend_lines = []
    legend_labels = []
    
    # Store all values for each metric to compute y limits
    all_values = {i: [] for i in range(len(metric_titles))}
    
    # Process each file
    for file_idx, file_path in enumerate(file_paths):
        # Read data
        x_values = []  # Will store either steps or time
        metrics_data = {i: [] for i in range(len(metric_titles))}
        
        with open(file_path, 'r') as f:
            data_points = [json.loads(line) for line in f]
            
            for data in data_points:
                # Get x value based on type
                if x_axis_type == 'step':
                    x_val = data['train_global_steps']
                else:  # time
                    x_val = data['accumulated_training_time']
                
                # Only include points up to x_lim and handle x=0 for log scale
                if x_val <= x_lim and x_val >= 0:
                    x_values.append(x_val)
                    
                    # Process first three metrics normally
                    for i in range(3):
                        value = data[metrics[i]]
                        metrics_data[i].append(value)
                        all_values[i].append(value)
                    
                    # Average AIME metrics
                    aime_value = np.mean([data[m] for m in metrics[3]])
                    metrics_data[3].append(aime_value)
                    all_values[3].append(aime_value)
        
        # Convert time to hours if needed
        if x_axis_type == 'time':
            x_values = [x/3600 for x in x_values]  # Convert seconds to hours
        
        # Plot each metric in its subplot with EMA smoothing
        for ax_idx in range(4):
            # Plot original data with transparency
            line_orig = axes[ax_idx].plot(x_values, metrics_data[ax_idx], 
                                   marker=marker_style, 
                                   markersize=0,
                                   linewidth=line_width,
                                   alpha=0.2)  # Make original line transparent
            
            # Apply and plot EMA smoothing
            smoothed_values = compute_ema(metrics_data[ax_idx], x_values, alpha=ema_alpha)
            line = axes[ax_idx].plot(x_values, smoothed_values, 
                                   marker=marker_style, 
                                   markersize=0,
                                   linewidth=line_width,
                                   color=line_orig[0].get_color())  # Use same color as original
            
            if ax_idx == 0:  # Only store legend info from first subplot
                legend_lines.extend(line)
                legend_labels.append(f'{algorithm_names[file_idx]}')
    
    # Customize each subplot
    for ax_idx, (ax, title) in enumerate(zip(axes, metric_titles)):
        # Set adaptive y limits
        y_min = min(all_values[ax_idx]) - 0.03
        y_max = max(all_values[ax_idx]) + 0.03
        ax.set_ylim(max(0, y_min), min(1, y_max))
        
        if x_axis_type == 'step':
            ax.set_xlim(1, x_lim + 10)
        else:
            ax.set_xlim(-0.5, (x_lim + 3600)/3600)
        
        # Set title
        ax.set_title(title, pad=0, fontsize=font_size_title, fontweight=fontweight_title, fontfamily=fontfamily)
        ax.grid(True, alpha=0.2, color='#666666')
        
        # Remove ticks
        ax.tick_params(length=0)
        
        if ax_idx == 0:  # Only add y-label to leftmost plot
            ax.set_ylabel('Accuracy', fontsize=font_size_label, fontfamily=fontfamily, fontweight=fontweight_label)
        
        # Make the plot more beautiful with dark frame
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.0)
        
        ax.tick_params(labelsize=font_size_tick, labelcolor='black')
        
        # Set grid for both major and minor ticks in log scale
        ax.grid(True, which='both', alpha=0.2, color='#666666')
        ax.grid(True, which='major', alpha=0.4, color='#666666')
    
    # Add common x-label at the bottom center
    x_label = 'Steps' if x_axis_type == 'step' else 'Training Time (hours)'
    fig.text(0.5, 0.08, x_label, ha='center', fontsize=font_size_label, fontfamily=fontfamily, fontweight=fontweight_label)
    
    # Add common legend below the subplots
    fig.legend(legend_lines, legend_labels, 
              loc='center',
              bbox_to_anchor=(0.5, 0.18),
              ncol=len(file_paths),
              prop={'size': font_size_legend, 'family': fontfamily, 'weight': fontweight_legend},
              frameon=True,
              edgecolor='black',
              columnspacing=1.0)
    
    # Adjust layout to prevent overlapping
    plt.tight_layout(rect=[0, 0.2, 1, 0.95])
    
    return fig, axes

# Example usage:
if __name__ == "__main__":
    # Example file paths

    file_lst = [
        ["/u/rzhang15/projects/DAPO/metrics_summary/7B-Math-FAST-RLOO-dataDAPO-N4+20/processed_metrics.json",
        "/u/rzhang15/projects/DAPO/metrics_summary/7B-Math-RLOO-dataDAPO-N24/processed_metrics.json"],

        ["/u/rzhang15/projects/DAPO/metrics_summary/7B-Math-FAST-GRPO-dataDAPO-N4+20/processed_metrics.json",
        "/u/rzhang15/projects/DAPO/metrics_summary/7B-Math-GRPO-dataDAPO-N24/processed_metrics.json"],

        ["/u/rzhang15/projects/DAPO/metrics_summary/7B-Math-FAST-DAPO-dataDAPO-N4+20/processed_metrics.json",
        "/u/rzhang15/projects/DAPO/metrics_summary/7B-Math-DAPO-dataDAPO-N24/processed_metrics.json"],

        ["/u/rzhang15/projects/DAPO/metrics_summary/7B-Math-FAST-RLOO-DeepScaleR-N4+20/processed_metrics.json",
        "/u/rzhang15/projects/DAPO/metrics_summary/7B-Math-RLOO-DeepScaleR-N24/processed_metrics.json"],

        ["/u/rzhang15/projects/DAPO/metrics_summary/7B-Math-FAST-DAPO-DeepScaleR-N4+20/processed_metrics.json",
        "/u/rzhang15/projects/DAPO/metrics_summary/7B-Math-DAPO-DeepScaleR-N24/processed_metrics.json"],

        ["/u/rzhang15/projects/DAPO/metrics_summary/1.5B-Math-FAST-RLOO-Numina-N4+20/processed_metrics.json",
        "/u/rzhang15/projects/DAPO/metrics_summary/1.5B-Math-RLOO-Numina-N24/processed_metrics.json"],

        ["/u/rzhang15/projects/DAPO/metrics_summary/1.5B-Math-FAST-RLOO-dataDAPO-N4+20/processed_metrics.json",
        "/u/rzhang15/projects/DAPO/metrics_summary/1.5B-Math-RLOO-dataDAPO-N24/processed_metrics.json"],

        ["/u/rzhang15/projects/DAPO/metrics_summary/1.5B-Math-FAST-DAPO-NuminaMath-N4+20/processed_metrics.json",
        "/u/rzhang15/projects/DAPO/metrics_summary/1.5B-Math-DAPO-NuminaMath-N24/processed_metrics.json"]
        ]

    algorithm_names_lst = [
        ["SPEED-RLOO", "RLOO"],
        ["SPEED-GRPO", "GRPO"],
        ["SPEED-DAPO", "DAPO"],
        ["SPEED-RLOO", "RLOO"],
        ["SPEED-DAPO", "DAPO"],
        ["SPEED-RLOO", "RLOO"],
        ["SPEED-RLOO", "RLOO"],
        ["SPEED-DAPO", "DAPO"]
    ]

    training_data_lst = [
        'DAPO',
        'DAPO',
        'DAPO',
        'DeepScaleR',
        'DeepScaleR',
        'NuminaMath',
        'dataDAPO',
        'NuminaMath'
    ]

    model_size_lst = [
        '7B',
        '7B',
        '7B',
        '7B',
        '7B',
        '1.5B',
        '1.5B',
        '1.5B'
    ]

    for files, algorithm_names, training_data, model_size in zip(file_lst, algorithm_names_lst, training_data_lst, model_size_lst):
        save_dir = f"/u/rzhang15/projects/DAPO/fig/{model_size}-Math-{algorithm_names[-1]}-{training_data}"
        os.makedirs(save_dir, exist_ok=True)
        
        for x_type in ['step', 'time']:
            fig, axes = plot_metrics_comparison(
                files,
                algorithm_names,
                x_axis_type=x_type,
                x_lim=1000 if x_type == 'step' else 3600*20  # 500 steps or 24 hours
            )
            
            # Save with appropriate filename
            filename = f'performance_versus_{x_type}.png'
            plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
            plt.close() 