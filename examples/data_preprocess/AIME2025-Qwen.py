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
Preprocess the AIME2025 dataset to parquet format

Source: https://huggingface.co/datasets/yentinglin/aime_2025
"""

import os
import argparse
import pandas as pd
import datasets
from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/AIME2025-Qwen-base')
    parser.add_argument('--model_type', default='base')
    parser.add_argument('--hdfs_dir', default=None)
    args = parser.parse_args()

    data_source = 'AIME2025-Qwen-base'
    print(f"Loading the {data_source} dataset from HuggingFace...", flush=True)
    ds = datasets.load_dataset("yentinglin/aime_2025", "default")

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop('problem') # There are instructions in the prompt.
            answer = example.pop('answer')
            question = question_raw + " " + instruction_following
            if args.model_type == 'base':
                prompt = question
            elif args.model_type == 'instruct':
                prompt = [{
                    "content":question,
                    "role": "user"
                }]
            else:
                raise ValueError(f"Invalid model type: {args.model_type}")

            return {
                "data_source":  "AIME2025-Qwen-base",
                "prompt":       prompt,
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'index': str(idx)
                }
            }
        return process_fn

    model_type = args.model_type

    train_dataset = ds['train'].remove_columns(["__index_level_0__", "year",'url','id','solution']).map(function=make_map_fn('train'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)

