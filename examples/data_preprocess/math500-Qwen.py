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
Preprocess the MATH500 dataset to parquet format
"""

import os
import json
import datasets
from datasets import Dataset

from verl.utils.hdfs_io import copy, makedirs
import argparse

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/math500-Qwen-base')
    parser.add_argument('--model_type', default='base')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()
    
    # Read from jsonl file
    data_list = []
    with open('./data/math500/train.jsonl', 'r') as f:
        for line in f:
            data_list.append(json.loads(line))
    train_dataset = Dataset.from_list(data_list)

    data_list = []
    with open('./data/math500/test.jsonl', 'r') as f:
        for line in f:
            data_list.append(json.loads(line))
    test_dataset = Dataset.from_list(data_list)


    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop('problem')
            question = question_raw + " " + instruction_following
            answer = example.pop('answer')
            solution = example.pop('solution')
            subject = example.pop('subject')
            level = example.pop('level')
            unique_id = example.pop('unique_id')

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
                "data_source": "MATH500-Qwen-base",
                "prompt": prompt,
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'index': 'MATH500-'+str(idx)
                }
            }
            return data
        return process_fn


    model_type = args.model_type

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    import pdb; pdb.set_trace()

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
