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
# from . import gsm8k, math, prime_math, prime_code

from verl.utils.import_utils import deprecated
from verl.utils.reward_score.maze import compute_score_maze, extract_answer_maze
import re
import string
from verl.utils.reward_score.countdown import compute_score_countdown
from math_verify.errors import TimeoutException
import concurrent.futures

def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    if data_source.startswith("knk") or data_source.startswith("self-label-zanette-lab/knight-knave"):
        """Compute the score for a given solution based on the data source."""
        # 提取所有 \boxed{...} 的内容
        answers = re.findall(r"\\boxed\{(.*?)\}", solution_str)
        if not answers:
            return 0.0
        extracted_answer = answers[-1].strip()
        def normalize(text):
            text = text.strip().lower()
            return ''.join(c for c in text if c not in string.punctuation)
        norm_gt = normalize(ground_truth)
        norm_ans = normalize(extracted_answer)
        if norm_ans and norm_ans in norm_gt and len(norm_ans) >= 0.7 * len(norm_gt):
            return 1.0
        else:
            return 0.0
    elif data_source.startswith("guanning") or data_source == 'default':
        from . import math_verify
        res = math_verify.compute_score(solution_str, ground_truth)
    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(
        data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb
    )


__all__ = ["default_compute_score"]
