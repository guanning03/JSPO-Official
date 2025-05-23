import json
import numpy as np

def get_first_threshold_time(file_path, thresholds):
    """
    Get the first time each metric reaches its threshold.
    
    Args:
        file_path (str): Path to metrics file
        thresholds (dict): Dictionary of metric names to threshold values
    
    Returns:
        dict: Dictionary of metric names to first time (in hours) threshold was reached
    """
    # Initialize results
    first_times = {metric: None for metric in thresholds.keys()}
    reached = {metric: False for metric in thresholds.keys()}
    
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            time = data['accumulated_training_time'] / 3600  # Convert to hours
            
            # Check regular metrics
            for metric in ["val-core/math_dapo-Qwen-instruct/acc/mean@1",
                         "val-core/MATH500-Qwen-instruct/acc/mean@1",
                         "val-core/AMC-Qwen-instruct/acc/mean@4",
                         "val-core/AMC-Qwen-instruct/acc/mean@1"]:
                if metric in data and metric in thresholds and not reached[metric]:
                    if data[metric] >= thresholds[metric]:
                        first_times[metric] = time
                        reached[metric] = True
            
            # Check AIME average
            aime_metrics = ["val-core/AIME2025-Qwen-instruct/acc/mean@16",
                          "val-core/AIME-Qwen-instruct/acc/mean@16"]
            aime_metrics_1_5b = ["val-core/AIME2025-Qwen-instruct/acc/mean@4",
                               "val-core/AIME-Qwen-instruct/acc/mean@4"]
            
            if all(m in data for m in aime_metrics) and "AIME" in thresholds and not reached["AIME"]:
                aime_avg = np.mean([data[m] for m in aime_metrics])
                if aime_avg >= thresholds["AIME"]:
                    first_times["AIME"] = time
                    reached["AIME"] = True
            elif all(m in data for m in aime_metrics_1_5b) and "AIME" in thresholds and not reached["AIME"]:
                aime_avg = np.mean([data[m] for m in aime_metrics_1_5b])
                if aime_avg >= thresholds["AIME"]:
                    first_times["AIME"] = time
                    reached["AIME"] = True
    
    return first_times

def main():
    # Define thresholds for different model sizes
    thresholds_7b = {
        "val-core/math_dapo-Qwen-instruct/acc/mean@1": 0.45,  # DAPO-1k
        "val-core/MATH500-Qwen-instruct/acc/mean@1": 0.8,    # MATH500
        "val-core/AMC-Qwen-instruct/acc/mean@4": 0.55,       # AMC2023
        "AIME": 0.18                                         # AIME average
    }
    
    thresholds_1_5b = {
        "val-core/math_dapo-Qwen-instruct/acc/mean@1": 0.3,  # DAPO-1k
        "val-core/MATH500-Qwen-instruct/acc/mean@1": 0.7,    # MATH500
        "val-core/AMC-Qwen-instruct/acc/mean@1": 0.4,       # AMC2023
        "AIME": 0.1                                          # AIME average
    }

    thresholds_1_5b_more_validation = {
        "val-core/math_dapo-Qwen-instruct/acc/mean@1": 0.3,  # DAPO-1k
        "val-core/MATH500-Qwen-instruct/acc/mean@1": 0.7,    # MATH500
        "val-core/AMC-Qwen-instruct/acc/mean@4": 0.4,       # AMC2023
        "AIME": 0.1                                          # AIME average
    }
    
    # List of experiments to analyze
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
        'Numina',
        'dataDAPO',
        'NuminaMath',
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

    # Process each experiment
    for files, alg_names, train_data, model_size in zip(file_lst, algorithm_names_lst, training_data_lst, model_size_lst):
        if model_size == '1.5B':
            if '1.5B-Math-DAPO-NuminaMath' in files[-1] or '1.5B-Math-RLOO-dataDAPO-N24' in files[-1]: 
                thresholds = thresholds_1_5b_more_validation
            else:
                thresholds = thresholds_1_5b
        else:
            thresholds = thresholds_7b
        
        print(f"\nModel: {model_size}")
        print(f"Training Data: {train_data}")
        
        for file_path, alg_name in zip(files, alg_names):
            print(f"\nAlgorithm: {alg_name}")
            print(f'Threshold: {thresholds}')
            times = get_first_threshold_time(file_path, thresholds)

            print(times)

if __name__ == "__main__":
    main() 