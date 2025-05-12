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
from datasets import load_dataset
from verl.utils.hdfs_io import copy, makedirs
from datasets import Dataset
from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string

def extract_solution(solution_str):
    try:
        return remove_boxed(last_boxed_only_string(solution_str))
    except:
        return None

def is_integer_string(x):
    if x == '' or x is None:
        return False
    try:
        int(x)
        return True
    except ValueError:
        return False

def make_map_fn(split):
    def process_fn(example, idx):
        data_source = 'NuminaMath-' + example.pop('source')
        ability = 'math'
        reward_model = {
            'ground_truth': extract_solution(example.pop('solution')),
            'style': 'rule'
        }
        extra_info = {
            'index': 'NuminaMath-' + str(idx)
        }
        prompt = example.pop('problem')
        # prompt = [{
        #     "content":prompt,
        #     "role": "user"
        # }]
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
    parser.add_argument('--local_dir', default='./data/NumimaMath-allsources-base')
    parser.add_argument('--model_type', default='base')
    parser.add_argument('--hdfs_dir', default=None)
    args = parser.parse_args()

    dataset = load_dataset("AI-MO/NuminaMath-CoT")['train']
    # dataset = dataset.filter(lambda x: x['source'] not in ['amc_aime','gsm8k','math'])
    dataset = dataset.filter(lambda x: 'proof' not in x['solution'] and 'Proof' not in x['solution'])
    dataset = dataset.filter(lambda x: isinstance(x['solution'], str) and is_integer_string(extract_solution(x['solution'])))

    # source_counts = {}
    # for example in dataset:
    #     source = example['source']
    #     if source not in source_counts:
    #         source_counts[source] = 0
    #     source_counts[source] += 1
    
    # print("\nDataset size by source:")
    # for source, count in sorted(source_counts.items()):
    #     print(f"{source}: {count} examples")
    # print(f"Total: {len(dataset)} examples")
    split = 'train'
    dataset = dataset.remove_columns(["messages"]).map(function=make_map_fn(split), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)

    print(f"Processed the train split of the NuminaMath dataset.", flush=True)

