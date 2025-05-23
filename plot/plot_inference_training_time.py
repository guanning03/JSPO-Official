import json
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_timings(jsonl_file, 
                save_dir):
    """
    Process a JSONL file to extract timing information and create a bar plot.
    
    Args:
        jsonl_file (str): Path to the JSONL file
        save_dir (str): Directory to save the plot
    """
    legend_fontsize = 15
    fontsize = 15
    font = 'serif'

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Read JSONL file
    data = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    # Extract relevant timing information
    timing_data = []
    for entry in data:
        timing_info = {}
        timing_info['step'] = entry.get('step', None)
        
        # Get all timing related fields
        for key, value in entry.items():
            if key.startswith('timing'):
                timing_info[key] = value
        
        timing_data.append(timing_info)
    
    # Convert to DataFrame
    df = pd.DataFrame(timing_data)
    
    # Calculate averages
    gen_avg = df['timing_s/gen'].mean()
    combined_avg = (df['timing_s/old_log_prob'] + df['timing_s/update_actor']).mean()
    
    # Create bar plot
    plt.figure(figsize=(5.5,7))
    
    # Add horizontal grid lines
    ax = plt.gca()
    ax.yaxis.grid(True, color='gray', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)  # This ensures grid lines are drawn behind the bars
    
    # Create bars
    bars = plt.bar(['Inference', 'Training'], 
                   [gen_avg, combined_avg],
                   color=['#2E86C1', '#E74C3C'],
                   label=['Inference Time', 'Training Time'])
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom',
                fontfamily=font, fontsize=fontsize,
                weight='bold')
    
    # Customize the plot
    # plt.xlabel('Timing Components', fontfamily=font, weight='bold', fontsize=fontsize)
    plt.ylabel('Average Time (seconds)', fontfamily=font, weight='bold', fontsize=fontsize)
    
    # Remove ticks
    plt.tick_params(axis='both', length=0)
    
    # Make x-axis labels bold and larger
    ax = plt.gca()
    ax.set_xticklabels(['Inference', 'Training'], 
                       fontfamily=font, 
                       weight='bold', 
                       fontsize=fontsize)
    
    # Add dark frame
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)
        spine.set_visible(True)
    
    # Customize legend
    # plt.legend(prop={'family': font, 'size': legend_fontsize})
    
    # Save the plot
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'inference_training_time.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

# Example usage
if __name__ == "__main__":
    # These parameters can be modified as needed
    jsonl_file = "/u/rzhang15/projects/DAPO/metrics/7B-Math-FAST-RLOO-DeepScaleR-N4+20-offload_20250511_204038.jsonl"
    save_dir = "./fig"
    plot_timings(jsonl_file, save_dir) 