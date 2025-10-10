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
Preprocess the DeepScaledR dataset to parquet format
"""

import os
import json
import datasets
from datasets import Dataset

from verl.utils.hdfs_io import copy, makedirs
import argparse

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string
from verl.utils.reward_score.math_verify import compute_score
try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/DeepScaleR-instruct')
    parser.add_argument('--model_type', default='instruct')
    parser.add_argument('--hdfs_dir', default=None)
    args = parser.parse_args()

    ds = datasets.load_dataset("rqzhang/DeepScaleR-instruct")

    def make_map_fn(split):
        def process_fn(example, idx):

            return {
                "data_source":  example.pop('data_source'),
                "prompt":       example.pop('prompt'),
                "ability": example.pop('ability'),
                "reward_model": example.pop('reward_model'),
                "extra_info": example.pop('extra_info'),
            }
        return process_fn

    train_dataset = ds['train'].map(function=make_map_fn('train'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)

