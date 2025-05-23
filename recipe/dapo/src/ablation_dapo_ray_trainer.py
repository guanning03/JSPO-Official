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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
import time
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, RayPPOTrainer, _timer, apply_kl_penalty, compute_advantage


class CustomRayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        metrics_dir = self.config.trainer.save_metrics_local_dir
        os.makedirs(metrics_dir, exist_ok=True)
        jl_file = os.path.join(metrics_dir, f"{self.config.trainer.save_metric_path}.jsonl")
        jf = open(jl_file, "a") # RZ: Append to the file. We stream the metrics as json lines.

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            jf.write(json.dumps({'step': self.global_steps, **val_metrics}) + "\n") # Save metrics to JSON file
            jf.flush() # get flushed to disk immediately
            os.fsync(jf.fileno())
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        self.training_start_time = time.time()
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                new_batch: DataProto = DataProto.from_single_dict(batch_dict) # RZ: Generate a new batch.
                num_gen_batches += 1

                assert "multi_modal_inputs" not in new_batch.non_tensor_batch.keys(), "Multi-modal inputs are not supported yet."
                gen_batch = new_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids"], # RZ: Why do we need this?
                ) # DataProto.

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    gen_start_time = time.time()
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch) # DataProto with only batch, and the non_tensor_batch an meta_info are enpty.
                    gen_end_time = time.time()
                    print(f"Time taken for generation: {gen_end_time - gen_start_time} seconds")

                    if self.config.algorithm.adv_estimator== AdvantageEstimator.REMAX:
                        raise NotImplementedError

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    ) # DataProto with non_tenor_batch.keys() = dict_keys(['data_source', 'ability', 'reward_model', 'extra_info', 'index', 'uid'])
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)
                    # gen_batch_output; DataProto. gen_batch_output.batch.keys() = dict_keys(['attention_mask', 'input_ids', 'position_ids', 'prompts', 'responses']). 
                    # gen_batch_output.non_tensor_batch is empty. gen_batch_output.meta_info is empty.
                    # new_batch is a DataProto. new_batch.batch is empty.
                    # new_batch.non_tensor_batch.keys() = dict_keys(['data_source', 'ability', 'reward_model', 'extra_info', 'index', 'uid']).
                    # new_batch.meta_info is empty.

                    with _timer("reward", timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result["reward_extra_info"]
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward: # RZ: By default this is False
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(
                                kl_metrics
                            )  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )

                        ##### Ablation Study. Edit: Filter the prompts with pass rate in an interval. #####
                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list) # RZ: A dictionary that maps prompt_uid to a list of metric values.
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name]
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        threshold = np.std(np.array([1 for _ in range(4)] + [0 for _ in range(20)])) # RZ; The only thing that we edit here.

                        kept_prompt_uids = [
                            uid
                            for uid, std in prompt_uid2metric_std.items()
                            if std > threshold or len(prompt_uid2metric_vals[uid]) == 1 # us the new threshold.
                        ]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = [] # RZ: All indices of the trajectories that are kept.
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)
                        ##### End of Ablation Study #####

                        new_batch = new_batch[kept_traj_idxs] # RZ: Select the trajectories with qualified prompts.
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch]) # RZ: Concatenate the new batch with the old batch. The 'batch' keeps all data that the model is trained on.

                        prompt_bsz = self.config.data.train_batch_size

                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                continue
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        else:
                            # Align the batch
                            print(f"{num_gen_batches=}. We have enought prompts for training. We have {num_prompt_in_batch=} prompts in the batch.")
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]
                            print(f'Size = {batch.batch["input_ids"].shape=}')
                            print(f'non_tensor_batch = {batch.non_tensor_batch.keys()=}')
                            print(f'meta_info = {batch.meta_info.keys()=}')
                            print(f'uids = {batch.non_tensor_batch["uid"]}')

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    old_log_prob_start_time = time.time()
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)
                    old_log_prob_end_time = time.time()
                    print(f"Time taken for old_log_prob: old_log_prob_{old_log_prob_end_time - old_log_prob_start_time} seconds")

                    ref_start_time = time.time()
                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)
                    ref_end_time = time.time()
                    print(f"Time taken for ref: ref_{ref_end_time - ref_start_time} seconds")

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    adv_start_time = time.time()
                    with _timer("adv", timing_raw):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )
                    adv_end_time = time.time()
                    print(f"Time taken for adv: adv_{adv_end_time - adv_start_time} seconds")

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    update_actor_start_time = time.time()
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)
                    update_actor_end_time = time.time()
                    print(f"Time taken for update_actor: update_actor_{update_actor_end_time - update_actor_start_time} seconds")

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                non_training_labels = ['testing', 'save_checkpoint']
                time_pure_training = timing_raw['step']
                for label in non_training_labels:
                    if label in timing_raw.keys():
                        time_pure_training -= timing_raw[label]
                timing_raw['time_pure_training'] = time_pure_training
                
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # Add total training time to metrics
                current_time = time.time()
                metrics["train/total_training_time_seconds"] = current_time - self.training_start_time
                metrics["train/global_steps"] = self.global_steps
                metrics['train/epoch'] = epoch
                metrics['train/num_prompts_in_batch'] = num_prompt_in_batch

                logger.log(data=metrics, step=self.global_steps)
                jf.write(json.dumps({'step': self.global_steps, **metrics}) + "\n") # Save metrics to JSON file
                jf.flush() # get flushed to disk immediately
                os.fsync(jf.fileno())

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    jf.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1