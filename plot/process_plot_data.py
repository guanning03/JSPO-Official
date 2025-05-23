import json
import os
from typing import List, Dict

def process_jsonl_data(jsonl_path: str, save_dir: str) -> None:
    processed_data = []
    accumulated_training_time = 0.0
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            # Track training time
            if 'timing_s/time_pure_training' in data:
                accumulated_training_time += float(data['timing_s/time_pure_training'])
            
            # Check if this line contains validation metrics
            val_metrics = {k: v for k, v in data.items() if k.startswith('val-core')}
            
            if val_metrics:
                entry = {
                    'train_global_steps': data['step'],
                    'accumulated_training_time': accumulated_training_time
                }
                # Add all validation metrics
                entry.update(val_metrics)
                processed_data.append(entry)
    
    # Save processed data
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    output_path = os.path.join(save_dir, 'processed_metrics.json')
    with open(output_path, 'w') as f:
        for data in processed_data:
            json.dump(data, f)
            f.write('\n')
    
    print(f"Processed data saved to: {output_path}")

if __name__ == "__main__":
    path_lst = [
        (
            "/u/rzhang15/projects/DAPO/metrics/7B-Math-FAST-RLOO-dataDAPO-N4+20-offload_20250510_195941.jsonl",
            "/u/rzhang15/projects/DAPO/metrics_summary/7B-Math-FAST-RLOO-dataDAPO-N4+20"
        ), # 7B SPEED-RLOO DAPO
        (
            "/u/rzhang15/projects/DAPO/metrics/7B-Math-RLOO-dataDAPO-N24-offload_20250510_172958.jsonl",
            "/u/rzhang15/projects/DAPO/metrics_summary/7B-Math-RLOO-dataDAPO-N24"
        ), # 7B RLOO DAPO
        (
            "/u/rzhang15/projects/DAPO/metrics/7B-Math-GRPO-dataDAPO-N24-offload_20250510_173705.jsonl",
            "/u/rzhang15/projects/DAPO/metrics_summary/7B-Math-GRPO-dataDAPO-N24"
        ), # 7B GRPO DAPO
        (
            "/u/rzhang15/projects/DAPO/metrics/7B-Math-FAST-GRPO-dataDAPO-N4+20-offload_20250510_190846.jsonl",
            "/u/rzhang15/projects/DAPO/metrics_summary/7B-Math-FAST-GRPO-dataDAPO-N4+20"
        ), # 7B SPEED-GRPO DAPO
        (
            "/u/rzhang15/projects/DAPO/metrics/7B-Math-FAST-DAPO-dataDAPO-N4+20-offload_20250510_202948.jsonl",
            "/u/rzhang15/projects/DAPO/metrics_summary/7B-Math-FAST-DAPO-dataDAPO-N4+20"
        ), # 7B SPEED-DAPO DAPO
        (
            "/u/rzhang15/projects/DAPO/metrics/7B-Math-DAPO-dataDAPO-N24-offload-corrected_20250511_201737.jsonl",
            "/u/rzhang15/projects/DAPO/metrics_summary/7B-Math-DAPO-dataDAPO-N24"
        ), # 7B DAPO DAPO
        (
            "/u/rzhang15/projects/DAPO/metrics/7B-Math-RLOO-DeepScaleR-N24-offload_20250511_203242.jsonl",
            "/u/rzhang15/projects/DAPO/metrics_summary/7B-Math-RLOO-DeepScaleR-N24"
        ), # 7B RLOO DeepScaleR
        (
            "/u/rzhang15/projects/DAPO/metrics/7B-Math-FAST-RLOO-DeepScaleR-N4+20-offload_20250511_204038.jsonl",
            "/u/rzhang15/projects/DAPO/metrics_summary/7B-Math-FAST-RLOO-DeepScaleR-N4+20"
        ), # 7B SPEED-RLOO DeepScaleR
        (
            "/u/rzhang15/projects/DAPO/metrics/7B-Math-DAPO-DeepScaleR-N24-offload_20250512_002333.jsonl",
            "/u/rzhang15/projects/DAPO/metrics_summary/7B-Math-DAPO-DeepScaleR-N24"
        ), # 7B DAPO DeepScaleR
        (
            "/u/rzhang15/projects/DAPO/metrics/7B-Math-FAST-DAPO-DeepScaleR-N4+20-offload_20250512_002444.jsonl",
            "/u/rzhang15/projects/DAPO/metrics_summary/7B-Math-FAST-DAPO-DeepScaleR-N4+20"
        ), # 7B SPEED-DAPO DeepScaleR
        (
            "/u/rzhang15/projects/DAPO/metrics/1.5B-Math-RLOO-Numina-N24-offload_20250508_042813.jsonl",
            "/u/rzhang15/projects/DAPO/metrics_summary/1.5B-Math-RLOO-Numina-N24"
        ), # 1.5B RLOO Numina
        (
            "/u/rzhang15/projects/DAPO/metrics/1.5B-Math-FAST-RLOO-Numina-N4+20-offload_20250508_095737.jsonl",
            "/u/rzhang15/projects/DAPO/metrics_summary/1.5B-Math-FAST-RLOO-Numina-N4+20"
        ), # 1.5B SPEED-RLOO Numina N_init = 4
        (
            "/u/rzhang15/projects/DAPO/metrics/1.5B-Math-RLOO-dataDAPO-N24-offload_20250509_052640.jsonl",
            "/u/rzhang15/projects/DAPO/metrics_summary/1.5B-Math-RLOO-dataDAPO-N24"
        ), # 1.5B RLOO dataDAPO
        (
            "/u/rzhang15/projects/DAPO/metrics/1.5B-Math-FAST-RLOO-dataDAPO-N4+20-offload_20250509_161243.jsonl",
            "/u/rzhang15/projects/DAPO/metrics_summary/1.5B-Math-FAST-RLOO-dataDAPO-N4+20"
        ), # 1.5B SPEED-RLOO dataDAPO N_init = 4
        (
            "/u/rzhang15/projects/DAPO/metrics/1.5B-Math-FAST-RLOO-dataDAPO-N6+18-offload_20250509_121842.jsonl",
            "/u/rzhang15/projects/DAPO/metrics_summary/1.5B-Math-FAST-RLOO-dataDAPO-N6+18"
        ), # 1.5B SPEED-RLOO dataDAPO N_init = 6
        (
            "/u/rzhang15/projects/DAPO/metrics/1.5B-Math-FAST-RLOO-dataDAPO-N8+16-offload_20250509_161202.jsonl",
            "/u/rzhang15/projects/DAPO/metrics_summary/1.5B-Math-FAST-RLOO-dataDAPO-N8+16"
        ), # 1.5B SPEED-RLOO dataDAPO N_init = 8
        (
            "/u/rzhang15/projects/DAPO/metrics/1.5B-Math-DAPO-NuminaMath-N24-offload_20250513_050317.jsonl",
            "/u/rzhang15/projects/DAPO/metrics_summary/1.5B-Math-DAPO-NuminaMath-N24"
        ), # 1.5B DAPO NuminaMath
        (
            "/u/rzhang15/projects/DAPO/metrics/1.5B-Math-FAST-DAPO-NuminaMath-N4+20-offload_20250513_050525.jsonl",
            "/u/rzhang15/projects/DAPO/metrics_summary/1.5B-Math-FAST-DAPO-NuminaMath-N4+20"
        ), # 1.5B SPEED-DAPO NuminaMath N_init = 4
    ]

    for jsonl_path, save_dir in path_lst:
        process_jsonl_data(
            jsonl_path=jsonl_path,
            save_dir=save_dir
        )
