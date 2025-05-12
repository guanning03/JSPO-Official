# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import pandas as pd
import datasets
from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string

def make_map_fn(split):
    def process_fn(example, idx):
        data_source = example.pop('data_source')
        prompt = example.pop('prompt')
        ability = example.pop('ability')
        reward_model = example.pop('reward_model')
        extra_info = example.pop('extra_info')
        return {
            "data_source":  data_source,
            "prompt":       prompt,
            "ability": ability,
            "reward_model": reward_model,
            "extra_info": extra_info
        }
    return process_fn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/AIME2025-Qwen-base')
    parser.add_argument('--model_type', default='base')
    parser.add_argument('--hdfs_dir', default=None)
    args = parser.parse_args()

    dataset_paths = [
        'rqzhang/DAPO-17k-instruct',
        'rqzhang/gsm8k-instruct',
        'rqzhang/Math500-instruct',
        'rqzhang/NuminaMath-processed-instruct'
    ]

    train_datasets = []
    for dataset_path in dataset_paths:
        print(f"Loading the {dataset_path} dataset from HuggingFace...", flush=True)
        ds = datasets.load_dataset(dataset_path)['train']
        train_datasets.append(ds)
    merged_dataset = datasets.concatenate_datasets(train_datasets)
    print(f"Total number of examples after merging: {len(merged_dataset)}")
        

    local_dir = './data/merged4-instruct'
    hdfs_dir = args.hdfs_dir

    merged_dataset.to_parquet(os.path.join(local_dir, f'train.parquet'))
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)

    print(f"Processed the train split of the merged dataset.", flush=True)

