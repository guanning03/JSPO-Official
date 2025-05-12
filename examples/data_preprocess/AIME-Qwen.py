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
import argparse
import pandas as pd
import datasets
from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/AIME-Qwen-base')
    parser.add_argument('--model_type', default='base')
    parser.add_argument('--hdfs_dir', default=None)
    args = parser.parse_args()

    data_source = 'BytedTsinghua-SIA/AIME-2024'
    print(f"Loading the {data_source} dataset from HuggingFace...", flush=True)
    ds = datasets.load_dataset(data_source, trust_remote_code=True)

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # 1) Take the train split and drop the unwanted auto‐index column
    train_orig = ds['train'].remove_columns("__index_level_0__")
    # Keep track of all columns so we can remove them in map()
    old_cols = train_orig.column_names

    # 2) Build a transformer that only returns the fields you want,
    #    including a new one‐field extra_info struct.
    def make_map_fn(split):
        def process_fn(example, idx):
            prompt = example.pop('prompt') # There are instructions in the prompt.
            question_raw = prompt[0]['content']
            question_raw = question_raw.replace('Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n', '')
            question_raw = question_raw.replace('\n\nRemember to put your answer on its own line after "Answer:".', '')
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
                "data_source":  "AIME-Qwen-base",
                "prompt":       prompt,
                "ability":      example["ability"],
                "reward_model": example["reward_model"],
                # 2) use the `idx` counter, *not* example["extra_info"]
                "extra_info":   {"index": str(idx)},
            }
        return process_fn

    train_ds = (
        ds["train"]
        .remove_columns(["__index_level_0__", "extra_info"])   # drop the old struct
        .map(
            make_map_fn("train"),
            with_indices=True,
            remove_columns=[
                "data_source", "prompt", "ability", "reward_model"
            ],  # or simply remove all old columns
        )
    )
    # The original extra_info is a struct with three keys index, problem_id, and solution. And the dtype for index is a integer. We need to convert it to a string.

    # 4) Deduplicate via pandas
    df = train_ds.to_pandas().reset_index(drop=True)
    if args.model_type == 'base':
        df = df.drop_duplicates(subset=['prompt'])
    else:
        df = df.drop_duplicates(subset=['prompt'], keep='first')
    df = df.reset_index(drop=True)

    print(f"Original dataset size: {len(train_ds)}")
    print(f"Dataset size after deduplication: {len(df)}")

    # 5) Rebuild HF Dataset and write out
    deduped = datasets.Dataset.from_pandas(df, preserve_index=False)
    os.makedirs(args.local_dir, exist_ok=True)
    deduped.to_parquet(os.path.join(args.local_dir, 'train.parquet'))

    if args.hdfs_dir:
        makedirs(args.hdfs_dir)
        copy(src=args.local_dir, dst=args.hdfs_dir)
