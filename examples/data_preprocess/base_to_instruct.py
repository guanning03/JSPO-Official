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
        data_source = example.pop('data_source').replace('base', 'instruct')
        prompt = example.pop('prompt')
        ability = example.pop('ability')
        reward_model = example.pop('reward_model')
        extra_info = example.pop('extra_info')
        prompt = [{
            "content":prompt,
            "role": "user"
        }]
        return {
            "data_source":  data_source.replace('-base', '-instruct'),
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
        # 'rqzhang/AIME2024-base',
        # 'rqzhang/AIME2025-base',
        # 'rqzhang/DAPO-17k-base',
        # 'rqzhang/gsm8k-base',
        # 'rqzhang/Math500-base',
        # 'rqzhang/Math-base',
        # 'rqzhang/AMC23-base',
        # 'rqzhang/NuminaMath-processed-base',
        # 'rqzhang/Numinamath-filtered-base'
        'rqzhang/DeepScaleR-base'
    ]

    for dataset_path in dataset_paths:
        print(f"Loading the {dataset_path} dataset from HuggingFace...", flush=True)
        ds = datasets.load_dataset(dataset_path)


        if 'train' in ds and 'test' in ds:
            splits = ['train', 'test']
        elif 'train' in ds:
            splits = ['train']
        elif 'test' in ds:
            splits = ['test']
        else:
            raise ValueError(f"Invalid dataset: {dataset_path}")

        for split in splits:
            dataset = ds[split]

            dataset = dataset.map(function=make_map_fn(split), with_indices=True)

            local_dir = './data/' + dataset_path.split('/')[-1].replace('-base', '-instruct')
            hdfs_dir = args.hdfs_dir

            dataset.to_parquet(os.path.join(local_dir, f'{split}.parquet'))
            if hdfs_dir is not None:
                makedirs(hdfs_dir)
                copy(src=local_dir, dst=hdfs_dir)

            print(f"Processed the {split} split of the {dataset_path} dataset.", flush=True)

