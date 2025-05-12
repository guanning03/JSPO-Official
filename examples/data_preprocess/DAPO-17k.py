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

from verl.utils.hdfs_io import copy, makedirs
import argparse

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/DAPO-17k')
    parser.add_argument('--model_type', default='instruct')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'BytedTsinghua-SIA/DAPO-Math-17k'
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    train_dataset = dataset['train']
    instruction_following = "Let's think step by step and output the final answer within \\boxed{}." 

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            if args.model_type == 'base':
                raise NotImplementedError("Base model is not supported for DAPO-17k")
            elif args.model_type == 'instruct':
                pass
            
            data_source = example.pop('data_source')
            prompt = example.pop('prompt') # There are instructions in the prompt.
            ability = example.pop('ability')
            reward_model = example.pop('reward_model')
            extra_info = example.pop('extra_info')
            data = {
                "data_source": data_source,
                "prompt": prompt, # the system prompt is different from the one in MATH dataset.
                "ability": ability,
                "reward_model": reward_model,
                "extra_info": extra_info
            }
            return data
        return process_fn

    # Map the full dataset first
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
