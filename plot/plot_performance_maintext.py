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

def plot_performance_maintext(file_paths_list, algorithm_names_list,
                            line_width=5,
                            marker_size=3,
                            marker_style='o',
                            x_lim=500,  # max steps or seconds to include
                            x_lims_list=None,  # optional list of x_lim for each row
                            x_axis_type='step',  # 'step' or 'time'
                            ema_alpha=0.6,
                            save_dir='./fig',
                            frame_linewidth=2.0,  # thickness of subplot frames
                            y_labels=None):  # optional list of y-labels for each row
    """
    Plot metrics comparison in seven rows with four figures each.
    
    Args:
        file_paths_list (list): List of 7 lists, each containing paths to jsonl files for each row
        algorithm_names_list (list): List of 7 lists, each containing names of algorithms for each row
        line_width (float): Width of the plotted lines
        marker_size (float): Size of the markers
        marker_style (str): Style of the markers
        x_lim (int): Default maximum steps or seconds to include in plot
        x_lims_list (list): Optional list of 7 x_lim values, one for each row
        x_axis_type (str): Either 'step' for global steps or 'time' for accumulated training time
        ema_alpha (float): Smoothing factor for exponential moving average
        frame_linewidth (float): Width of the subplot frames
        y_labels (list): Optional list of 7 y-labels for each row. If not provided, uses default row labels.
    """
    # Input validation
    if x_axis_type not in ['step', 'time']:
        raise ValueError("x_axis_type must be either 'step' or 'time'")
    
    if len(file_paths_list) != 7 or len(algorithm_names_list) != 7:
        raise ValueError("Must provide exactly 7 sets of file paths and algorithm names")
    
    if x_lims_list is None:
        x_lims_list = [x_lim] * 7
    elif len(x_lims_list) != 7:
        raise ValueError("x_lims_list must contain exactly 7 values")

    # Default row labels based on file paths
    default_row_labels = [
        "7B + DeepScaleR + RLOO",
        "7B + DeepScaleR + DAPO",
        "7B + dataDAPO + RLOO",
        "7B + dataDAPO + DAPO",
        "1.5B + dataDAPO + RLOO",
        "1.5B + Numina + RLOO",
        "1.5B + NuminaMath + DAPO"
    ]

    # Use provided y_labels if available, otherwise use default row labels
    if y_labels is None:
        y_labels = [f"{label}\nAccuracy" for label in default_row_labels]
    elif len(y_labels) != 7:
        raise ValueError("y_labels must contain exactly 7 values")

    # Set style
    colors = ['#E74C3C', '#2E86C1']  # Blue and Red colors
    plt.rcParams['grid.alpha'] = 0.2
    plt.rcParams['grid.color'] = '#666666'
    
    # Metrics to plot (4 metrics now, with AIME metrics merged)
    base_metrics = [
        "val-core/math_dapo-Qwen-instruct/acc/mean@1",
        "val-core/MATH500-Qwen-instruct/acc/mean@1",
        ["val-core/AMC-Qwen-instruct/acc/mean@1",
         "val-core/AMC-Qwen-instruct/acc/mean@4"],
        {
            "AIME2025": ["val-core/AIME2025-Qwen-instruct/acc/mean@4",
                        "val-core/AIME2025-Qwen-instruct/acc/mean@16"],
            "AIME": ["val-core/AIME-Qwen-instruct/acc/mean@4",
                    "val-core/AIME-Qwen-instruct/acc/mean@16"]
        }
    ]
    
    # Function to find the first existing metric in data
    def find_existing_metric(data, metric_options):
        if isinstance(metric_options, str):
            return metric_options if metric_options in data else None
        elif isinstance(metric_options, list):
            for metric in metric_options:
                if metric in data:
                    return metric
            return None
        elif isinstance(metric_options, dict):
            # For AIME metrics, find one metric from each group and average them
            values = []
            for metric_group in metric_options.values():
                for metric in metric_group:
                    if metric in data:
                        values.append(data[metric])
                        break
            if len(values) == len(metric_options):  # Found one metric from each group
                return sum(values) / len(values)
            return None
        return None

    # Shorter names for titles
    metric_titles = [
        "DAPO-1k",
        "MATH500",
        "AMC2023",
        "AIME"
    ]

    # Set the parameters
    fontfamily = 'serif'
    font_size_title = 20
    font_size_label = 28
    font_size_legend = 28
    font_size_tick = 18
    fontweight_label = 'bold'
    fontweight_title = 'bold'
    fontweight_legend = 'bold'
    
    # Create figure and subplots
    fig, axes = plt.subplots(7, 4, figsize=(20, 30))  # Increased height for 7 rows
    fig.subplots_adjust(wspace=0.3, hspace=0.4)
    
    # Process each row
    for row_idx in range(7):
        file_paths = file_paths_list[row_idx]
        algorithm_names = algorithm_names_list[row_idx]
        
        # Store all values for each metric to compute y limits
        all_values = {i: [] for i in range(len(metric_titles))}
        legend_lines = []
        legend_labels = []
        
        # Process each file in the row
        for file_idx, file_path in enumerate(file_paths):
            # Read data
            x_values = []
            metrics_data = {i: [] for i in range(len(metric_titles))}
            
            with open(file_path, 'r') as f:
                data_points = [json.loads(line) for line in f]
                
                for data in data_points:
                    # Get x value based on type
                    if x_axis_type == 'step':
                        x_val = data['train_global_steps']
                    else:  # time
                        x_val = data['accumulated_training_time']
                    
                    if x_val <= x_lims_list[row_idx] and x_val >= 0:
                        x_values.append(x_val)
                        
                        # Process each metric by finding the first existing one
                        for i in range(len(base_metrics)):
                            if isinstance(base_metrics[i], dict):  # AIME case
                                value = find_existing_metric(data, base_metrics[i])
                                if value is not None:  # value is already averaged
                                    metrics_data[i].append(value)
                                    all_values[i].append(value)
                                else:
                                    print(f"Warning: No valid AIME metrics found in file {file_path}")
                                    break
                            else:  # Other metrics
                                metric = find_existing_metric(data, base_metrics[i])
                                if metric is not None:
                                    value = data[metric]
                                    metrics_data[i].append(value)
                                    all_values[i].append(value)
                                else:
                                    print(f"Warning: No valid metric found for {base_metrics[i]} in file {file_path}")
                                    break
            
            # Convert time to hours if needed
            if x_axis_type == 'time':
                x_values = [x/3600 for x in x_values]
            
            # Plot each metric in its subplot with EMA smoothing
            for col_idx in range(4):
                ax = axes[row_idx, col_idx]
                
                # Plot original data with transparency
                line_orig = ax.plot(x_values, metrics_data[col_idx],
                                  marker=marker_style,
                                  markersize=0,
                                  linewidth=line_width,
                                  alpha=0.2,
                                  color=colors[file_idx])
                
                # Apply and plot EMA smoothing
                smoothed_values = compute_ema(metrics_data[col_idx], x_values, alpha=ema_alpha)
                line = ax.plot(x_values, smoothed_values,
                             marker=marker_style,
                             markersize=0,
                             linewidth=line_width,
                             color=colors[file_idx])
                
                if col_idx == 0:  # Only store legend info from first subplot
                    legend_lines.extend(line)
                    legend_labels.append(algorithm_names[file_idx])
        
        # Customize each subplot in the row
        for col_idx in range(4):
            ax = axes[row_idx, col_idx]
            
            # Set adaptive y limits
            y_min = min(all_values[col_idx]) - 0.03
            y_max = max(all_values[col_idx]) + 0.03
            ax.set_ylim(max(0, y_min), min(1, y_max))
            
            # Make the plot more beautiful with dark frame
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(frame_linewidth)
            
            # Set grid for both major and minor ticks
            ax.grid(True, which='both', alpha=0.2, color='#666666')
            ax.grid(True, which='major', alpha=0.4, color='#666666')
            
            # Increase tick sizes
            ax.tick_params(axis='both', which='major', labelsize=font_size_tick)
            
            if col_idx == 0:  # Only add y-label to leftmost plot
                ax.set_ylabel(y_labels[row_idx],
                            fontsize=15,
                            fontfamily=fontfamily,
                            fontweight=fontweight_label)
            
            # Set x limits based on row-specific value
            if x_axis_type == 'step':
                ax.set_xlim(1, x_lims_list[row_idx] + 10)
            else:
                ax.set_xlim(-0.5, (x_lims_list[row_idx] + 3600)/3600)
    
    # Add x-label at the bottom of the figure
    fig.text(0.35, 0.07, 'Steps' if x_axis_type == 'step' else 'Training Time (hours)',
             fontsize=font_size_label,
             fontfamily=fontfamily,
             fontweight=fontweight_label,
             ha='center')
    
    # Add a single legend at the bottom center of the figure
    all_legend_lines = []
    all_legend_labels = []
    
    # Use common algorithm names
    common_algorithm_names = ['SPEED', 'Base RL Algorithm']
    legend_line_width = 10  # Increased line width for legend
    for i in range(len(common_algorithm_names)):
        all_legend_lines.append(plt.Line2D([0], [0], color=colors[i], linewidth=legend_line_width))
        all_legend_labels.append(common_algorithm_names[i])
    
    fig.legend(all_legend_lines, all_legend_labels,
              bbox_to_anchor=(0.75, 0.07),  # Position at bottom center
              loc='center',
              ncol=len(all_legend_labels),
              prop={'size': font_size_legend, 'family': fontfamily, 'weight': fontweight_legend},
              frameon=True,
              edgecolor='black')
    
    # Adjust layout to make room for the bottom legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # Make room for the legend at the bottom
    
    # Save in both PNG and PDF formats
    plt.savefig(os.path.join(save_dir, 'performance_summary_appendix.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'performance_summary_appendix.pdf'), bbox_inches='tight')
    plt.close()
    return fig, axes

# Example usage:
if __name__ == "__main__":
    # Example file paths for all seven rows
    base_paths = [
        ["/u/rzhang15/projects/DAPO/metrics_summary/7B-Math-FAST-RLOO-DeepScaleR-N4+20/processed_metrics.json",
         "/u/rzhang15/projects/DAPO/metrics_summary/7B-Math-RLOO-DeepScaleR-N24/processed_metrics.json"],
        
        ["/u/rzhang15/projects/DAPO/metrics_summary/7B-Math-FAST-DAPO-DeepScaleR-N4+20/processed_metrics.json",
        "/u/rzhang15/projects/DAPO/metrics_summary/7B-Math-DAPO-DeepScaleR-N24/processed_metrics.json"],

        ["/u/rzhang15/projects/DAPO/metrics_summary/7B-Math-FAST-RLOO-dataDAPO-N4+20/processed_metrics.json",
        "/u/rzhang15/projects/DAPO/metrics_summary/7B-Math-RLOO-dataDAPO-N24/processed_metrics.json"],

        ["/u/rzhang15/projects/DAPO/metrics_summary/7B-Math-FAST-DAPO-dataDAPO-N4+20/processed_metrics.json",
        "/u/rzhang15/projects/DAPO/metrics_summary/7B-Math-DAPO-dataDAPO-N24/processed_metrics.json"],

        ["/u/rzhang15/projects/DAPO/metrics_summary/1.5B-Math-FAST-RLOO-dataDAPO-N4+20/processed_metrics.json",
        "/u/rzhang15/projects/DAPO/metrics_summary/1.5B-Math-RLOO-dataDAPO-N24/processed_metrics.json"],

        ["/u/rzhang15/projects/DAPO/metrics_summary/1.5B-Math-FAST-RLOO-Numina-N4+20/processed_metrics.json",
        "/u/rzhang15/projects/DAPO/metrics_summary/1.5B-Math-RLOO-Numina-N24/processed_metrics.json"],

        ["/u/rzhang15/projects/DAPO/metrics_summary/1.5B-Math-FAST-DAPO-NuminaMath-N4+20/processed_metrics.json",
        "/u/rzhang15/projects/DAPO/metrics_summary/1.5B-Math-DAPO-NuminaMath-N24/processed_metrics.json"]
    ]
        
        
    
    # Use the same algorithm names for all rows
    algorithm_names = ['SPEED', 'Base RL Algorithm']
    algorithm_names_list = [algorithm_names for _ in range(7)]
    
    # Example of different x_lims for each row
    x_lims = [3600*24,
              3600*24,
              3600*24,
              3600*24,
              3600*10,
              3600*12,
              3600*12]
    # Same limit for all rows, but could be different
    
    # Example of custom y-labels for each row
    custom_labels = [
        "7B + DeepScaleR + RLOO\nAccuracy",
        "7B + DeepScaleR + DAPO\nAccuracy",
        "7B + DAPO17k + RLOO\nAccuracy",
        "7B + DAPO17k + DAPO\nAccuracy",
        "1.5B + DAPO17k + RLOO\nAccuracy",
        "1.5B + NuminaMath + RLOO\nAccuracy",
        "1.5B + NuminaMath + DAPO\nAccuracy"
    ]
    
    fig, axes = plot_performance_maintext(
        file_paths_list=base_paths,
        algorithm_names_list=algorithm_names_list,
        x_axis_type='time',
        x_lim=3600*24,
        x_lims_list=x_lims,
        frame_linewidth=2.0,  # Make frames thicker
        y_labels=custom_labels  # Use custom y-labels
    )
