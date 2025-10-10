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
from verl.trainer.ppo.data_controller import DataController

class RayFastDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    This trainer runs Fast-DAPO algorithm.
    """

    def __init__(self, 
                 config, 
                 tokenizer, 
                 role_worker_mapping, 
                 resource_pool_manager, 
                 ray_worker_group_cls, 
                 processor, 
                 reward_fn, 
                 val_reward_fn):
        super().__init__(config, tokenizer, role_worker_mapping, resource_pool_manager, ray_worker_group_cls, processor, reward_fn, val_reward_fn)

        # Define the data controller.
        self.data_controller = DataController(
            curriculum_enable=config.curriculum.enable,
            gen_batch_size=config.data.gen_batch_size,
            train_batch_size=config.data.train_batch_size,
            initial_n=config.actor_rollout_ref.rollout.n,
            n_continue=config.actor_rollout_ref.rollout.n_continue,
            max_num_gen_batches=config.algorithm.filter_groups.max_num_gen_batches,
            n_gpus=self.resource_pool_manager.get_n_gpus(),
            max_prompt_length=self.config.data.max_prompt_length,
            max_buffer_size=config.data.train_batch_size*(config.actor_rollout_ref.rollout.n + config.actor_rollout_ref.rollout.n_continue)*3
        )

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

        # The corner cases have not been implemented for fast DAPO.
        if self.config.algorithm.adv_estimator== AdvantageEstimator.REMAX:
            raise NotImplementedError
        if not self.config.algorithm.filter_groups.enable:
            raise NotImplementedError(f"The self.config.algorith.filter_groups.enable must be True in fast-DAPO.")
        if self.config.algorithm.filter_groups.metric == "seq_final_reward" or self.config.algorithm.filter_groups.metric == "seq_reward":
            raise NotImplementedError("Filtering metrics = seq_final_reward or seq_reward is not supported in Fast-DAPO.")

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

        self.global_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        batch = None
        num_gen_batches = 0
        metrics = {}
        self.training_start_time = time.time()
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                is_last_step = self.global_steps >= self.total_training_steps

                if not self.data_controller.is_ready_for_training(): # not ready for training. Do Inference.
                    print(f"Not enough prompts for training. We have {self.data_controller.get_num_prompts_for_training()=} prompts ready for training.")

                    # get a new batch for inference
                    new_batch: DataProto = DataProto.from_single_dict(batch_dict) 
                    new_batch.non_tensor_batch["uid"] = np.array(
                            [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    ) # label all prompts with a unique id. Only label new batches.
                    num_gen_batches += 1
                    assert "multi_modal_inputs" not in new_batch.non_tensor_batch.keys(), "Multi-modal inputs are not supported yet."

                    # RZ: We add new batch to the data controller and get a batch ready for generation from the data controller. The batch for generation may contain the old prompts in data controller that passes the filter.
                    self.data_controller.add_new_prompts_first_phase(new_batch)
                    new_batch = self.data_controller.get_generation_inputs()
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"]
                    ) # DataProto.

                    with _timer("step", timing_raw):
                        # generate a batch
                        gen_start_time = time.time()
                        with _timer("gen", timing_raw):
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch) # DataProto with only batch, and the non_tensor_batch an meta_info are enpty.
                        gen_end_time = time.time()
                        print(f"Time taken for generation: {gen_end_time - gen_start_time} seconds")

                        new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True) # repeat to align with repeated responses in rollout
                        new_batch = new_batch.union(gen_batch_output)

                        with _timer("reward", timing_raw):
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
                            # if self.config.algorithm.use_kl_in_reward: # RZ: By default this is False
                            #     new_batch, kl_metrics = apply_kl_penalty(
                            #         new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            #     )
                            #     metrics.update(
                            #         kl_metrics
                            #     )  # TODO: This will be cleared if we use multiple genenration batches
                            # else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]
                        
                        # Update prompt statistics and get filtered prompts/trajectories
                        self.data_controller.update_prompts(new_batch, self.config.algorithm.filter_groups.metric)
                        continue
                            
                elif self.data_controller.is_ready_for_training(): # start training when we have enough qualified prompts.
                    print(f"We have {self.data_controller.get_num_prompts_for_training()=} prompts ready for training.")
                    if not metrics:
                        metrics = {} # if we start the training without doing any additional inferences (there coould be enough qualified prompts in the buffer), we need to initialize the metrics.
                    batch = self.data_controller.get_training_data(prompt_batch_size = self.config.data.train_batch_size) # get a batch of training data.

                    with _timer("step", timing_raw):
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

                    # Add total training time to metrics
                    current_time = time.time()
                    metrics["train/total_training_time_seconds"] = current_time - self.training_start_time
                    metrics["train/num_gen_batches"] = num_gen_batches #self.data_controller.get_num_gen_batches()
                    num_gen_batches = 0
                    
                    metrics["train/num_prompts_in_batch"] = self.data_controller.get_num_prompts_for_training()
                    metrics["train/global_steps"] = self.global_steps
                    metrics['train/epoch'] = epoch
                    self.data_controller.reset_num_gen_batches()
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
                    self.global_steps += 1 # Here, 'step' means actual RL step (so the number of training steps within one step is fixed).
                    metrics = {} # clear metrics for next batch.
