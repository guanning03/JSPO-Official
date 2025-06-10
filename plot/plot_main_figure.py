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

def plot_main_figure(file_paths_row1, file_paths_row2,
                    algorithm_names_row1, algorithm_names_row2,
                    line_width=5,
                    marker_size=3,
                    marker_style='o',
                    x_lim=500,  # max steps or seconds to include
                    x_axis_type='step',  # 'step' or 'time'
                    ema_alpha=0.6,
                    save_dir='./fig'):
    """
    Plot metrics comparison in two rows with four figures each.
    
    Args:
        file_paths_row1 (list): List of paths to jsonl files for first row
        file_paths_row2 (list): List of paths to jsonl files for second row
        algorithm_names_row1 (list): Names of algorithms for legend in first row
        algorithm_names_row2 (list): Names of algorithms for legend in second row
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
    colors = ['#E74C3C', '#2E86C1']  # Blue and Red colors
    plt.rcParams['grid.alpha'] = 0.2
    plt.rcParams['grid.color'] = '#666666'
    
    # Metrics to plot (4 metrics now, with AIME metrics merged)
    if '7B' in file_paths_row1[0] or 'dataDAPO' in file_paths_row1[0] or 'DAPO-NuminaMath' in file_paths_row1[0]:
        metrics = [
            "val-core/math_dapo-Qwen-instruct/acc/mean@1",
            "val-core/MATH500-Qwen-instruct/acc/mean@1",
            "val-core/AMC-Qwen-instruct/acc/mean@4",
            ["val-core/AIME2025-Qwen-instruct/acc/mean@16",
             "val-core/AIME-Qwen-instruct/acc/mean@16"]
        ]
    else:
        metrics = [
            "val-core/math_dapo-Qwen-instruct/acc/mean@1",
            "val-core/MATH500-Qwen-instruct/acc/mean@1",
            "val-core/AMC-Qwen-instruct/acc/mean@1",
            ["val-core/AIME2025-Qwen-instruct/acc/mean@4",
             "val-core/AIME-Qwen-instruct/acc/mean@4"]
        ]
    
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
    font_size_label = 20
    font_size_legend = 20
    font_size_tick = 18
    fontweight_label = 'bold'
    fontweight_title = 'bold'
    fontweight_legend = 'bold'
    
    # Create figure and subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.subplots_adjust(wspace=0.3, hspace=0.4)
    
    # Process each row
    for row_idx, (file_paths, algorithm_names) in enumerate([(file_paths_row1, algorithm_names_row1),
                                                           (file_paths_row2, algorithm_names_row2)]):
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
            
            if x_axis_type == 'step':
                ax.set_xlim(1, x_lim + 10)
            else:
                ax.set_xlim(-0.5, (x_lim + 3600)/3600)
            
            # Set title for both rows
            if row_idx == 0:
                ax.set_title(metric_titles[col_idx], pad=0,
                        fontsize=font_size_title,
                        fontweight=fontweight_title,
                        fontfamily=fontfamily)
            
            ax.grid(True, alpha=0.2, color='#666666')
            ax.tick_params(length=0, labelsize=font_size_tick, labelcolor='black')
            
            if col_idx == 0:  # Only add y-label to leftmost plot
                ax.set_ylabel('Accuracy',
                            fontsize=font_size_label,
                            fontfamily=fontfamily,
                            fontweight=fontweight_label)
            
            # Make the plot more beautiful with dark frame
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(1.0)
            
            # Set grid for both major and minor ticks
            ax.grid(True, which='both', alpha=0.2, color='#666666')
            ax.grid(True, which='major', alpha=0.4, color='#666666')
            
            # Increase tick sizes
            ax.tick_params(axis='both', which='major', labelsize=font_size_tick)
            
            # Add x-label to all bottom plots
            # if row_idx == 1:
            #     x_label = 'Steps' if x_axis_type == 'step' else 'Training Time (hours)'
            #     ax.set_xlabel(x_label,
            #                 fontsize=font_size_label,
            #                 fontfamily=fontfamily,
            #                 fontweight=fontweight_label)
            fig.text(0.4, 0.13, 'Steps' if x_axis_type == 'step' else 'Training Time (hours)',
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
              bbox_to_anchor=(0.7, 0.14),  # Position at bottom center
              loc='center',
              ncol=len(all_legend_labels),  # Put all labels in one row
              prop={'size': font_size_legend, 'family': fontfamily, 'weight': fontweight_legend},
              frameon=True,
              edgecolor='black')
    
    # Adjust layout to make room for the bottom legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for the legend at the bottom
    
    plt.savefig(os.path.join(save_dir, 'main_figure.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    return fig, axes

# Example usage:
if __name__ == "__main__":
    # Example file paths for both rows
    file_paths_row1 = ["/u/rzhang15/projects/DAPO/metrics_summary/7B-Math-FAST-RLOO-DeepScaleR-N4+20/processed_metrics.json",
        "/u/rzhang15/projects/DAPO/metrics_summary/7B-Math-RLOO-DeepScaleR-N24/processed_metrics.json"]
    
    file_paths_row2 = ["/u/rzhang15/projects/DAPO/metrics_summary/7B-Math-FAST-DAPO-DeepScaleR-N4+20/processed_metrics.json",
        "/u/rzhang15/projects/DAPO/metrics_summary/7B-Math-DAPO-DeepScaleR-N24/processed_metrics.json"]
    
    # Use the same algorithm names for both rows
    algorithm_names_row1 = ['SPEED', 'Base RL Algorithm']
    algorithm_names_row2 = ['SPEED', 'Base RL Algorithm']
    
    fig, axes = plot_main_figure(
        file_paths_row1=file_paths_row1,
        file_paths_row2=file_paths_row2,
        algorithm_names_row1=algorithm_names_row1,
        algorithm_names_row2=algorithm_names_row2,
        x_axis_type='time',
        x_lim=3600*10,
        ema_alpha=1.0
    )
