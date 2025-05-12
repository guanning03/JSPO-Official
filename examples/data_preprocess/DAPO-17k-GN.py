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
"""
Preprocess the DAPO-17k dataset to parquet format
"""

import os
import datasets
import random
from datasets import Dataset, concatenate_datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))

def process_datasets(dataset1, dataset2, test_size=1000, seed=42):
    merged_dataset = concatenate_datasets([dataset1, dataset2])
    print(f"Size after merging: {len(merged_dataset)}")
    unique_problems = {}
    unique_indices = []
    
    for idx, example in enumerate(merged_dataset):
        if example['problem'] not in unique_problems:
            unique_problems[example['problem']] = idx
            unique_indices.append(idx)
    
    deduplicated_dataset = merged_dataset.select(unique_indices)
    print(f"Size after deduplication: {len(deduplicated_dataset)}")
    all_indices = list(range(len(deduplicated_dataset)))
    random.seed(seed)
    test_indices = random.sample(all_indices, test_size)
    train_indices = [idx for idx in all_indices if idx not in test_indices]
    
    train_dataset = deduplicated_dataset.select(train_indices)
    test_dataset = deduplicated_dataset.select(test_indices)
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    return train_dataset, test_dataset

def process_single_dataset(dataset, test_size=1000, seed=42):
    print(f"Size before deduplication: {len(dataset)}")
    all_indices = list(range(len(dataset)))
    random.seed(seed)
    # problems = [example['prompt'][0]['content'] for example in dataset]
    unique_indices = []
    unique_problems = {}
    for idx, example in enumerate(dataset):
        if example['prompt'][0]['content'] not in unique_problems:
            unique_problems[example['prompt'][0]['content']] = idx
            unique_indices.append(idx)
    print(f"Size after deduplication: {len(dataset.select(unique_indices))}")
    test_indices = random.sample(unique_indices, test_size)
    train_indices = [idx for idx in unique_indices if idx not in test_indices]
    
    train_dataset = dataset.select(train_indices)
    test_dataset = dataset.select(test_indices)
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    return train_dataset, test_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/DAPO-unique-Qwen-base')
    parser.add_argument('--model_type', default='base')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    # data_source = 'guanning/dapo17k'
    data_source = 'BytedTsinghua-SIA/DAPO-Math-17k'
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    train_dataset = dataset['train']
    import pdb; pdb.set_trace()
    # test_dataset = dataset['test']
    # train_dataset, test_dataset = process_datasets(train_dataset, test_dataset)
    train_dataset, test_dataset = process_single_dataset(train_dataset)
    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = example.pop('prompt')[0]['content']
            question = question.replace('Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n', '')
            question = question.replace('\n\nRemember to put your answer on its own line after "Answer:".', '')
            question = question + ' ' + instruction_following

            if args.model_type == 'base':
                prompt = question
            elif args.model_type == 'instruct':
                prompt = [{
                    "content": question,
                    "role": "user"
                }]
            else:
                raise ValueError(f"Invalid model type: {args.model_type}")

            data = {
                "data_source": 'math_dapo-Qwen-base',
                "prompt": prompt,
                "ability": "math",
                "reward_model": example['reward_model'],
                "extra_info": example['extra_info']
            }
            return data
        return process_fn

    # Map the full dataset first
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
