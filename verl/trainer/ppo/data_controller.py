from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from verl import DataProto
from collections import defaultdict

class DataController:
    def __init__(
        self,
        curriculum_enable: bool,
        gen_batch_size: int,
        train_batch_size: int,
        initial_n: int,
        n_continue: int,
        max_num_gen_batches: int,
        n_gpus: int,
        max_prompt_length: int,
        max_buffer_size: int,
    ):
        """Initialize the DataController.
        
        Args:
            curriculum_enable: Whether to use curriculum learning
            gen_batch_size: Number of prompts to generate in each batch
            train_batch_size: Number of qualified prompts needed for training
            initial_n: Number of initial responses per prompt (n)
            n_continue: Number of additional responses for qualified prompts (N-n)
            max_num_gen_batches: Maximum number of generation batches to try
        """
        self.curriculum_enable = curriculum_enable
        self.gen_batch_size = gen_batch_size
        self.train_batch_size = train_batch_size
        self.initial_n = initial_n
        self.n_continue = n_continue
        self.total_n_generations = initial_n + n_continue
        self.max_num_gen_batches = max_num_gen_batches
        self.n_gpus = n_gpus
        self.max_prompt_length = max_prompt_length
        self.prompts_for_first_generation_phase: Optional[DataProto] = None
        self.prompts_for_second_generation_phase: Optional[DataProto] = None
        self.prompts_for_training: Optional[DataProto] = None
        self.num_prompts_for_training = 0
        self.num_gen_batches = 0
        self.max_buffer_size = max_buffer_size
        assert self.max_buffer_size >= self.train_batch_size * self.total_n_generations, f"The maximum buffer size should be greater than or equal to the training batch size times the total number of generations. Got {self.max_buffer_size=} and {self.train_batch_size=} and {self.total_n_generations=}"
        assert self.n_continue % self.initial_n == 0, f"The number of additional responses should be divisible by the number of initial responses. Got {self.n_continue=} and {self.initial_n=}"
        self.multiplier = self.n_continue // self.initial_n

    def is_ready_for_training(self) -> bool:
        if self.prompts_for_training is None:
            return False
        elif self.num_prompts_for_training >= self.train_batch_size:
            return True
        else:
            return False
    
    def get_num_prompts_for_training(self) -> int:
        return self.num_prompts_for_training

    def get_num_gen_batches(self) -> int:
        return self.num_gen_batches

    def get_bsz_for_first_generation_phase(self) -> int:
        if self.prompts_for_first_generation_phase is None:
            return 0
        else: 
            return self.prompts_for_first_generation_phase.batch.batch_size[0]
        
    def get_bsz_for_second_generation_phase(self) -> int:
        if self.prompts_for_second_generation_phase is None:
            return 0
        else:
            return self.prompts_for_second_generation_phase.batch.batch_size[0]

    def add_new_prompts_first_phase(self, batch: DataProto) -> None:
        """Add new prompts to be processed in the current batch."""
        assert self.prompts_for_first_generation_phase is None, "First generation phase is already done. Before we call add_new_prompts, the prompts_for_first_generation_phase should be None."
        self.prompts_for_first_generation_phase = batch
    
    def get_generation_inputs(self) -> DataProto:
        """Get the prompts for the next generation step."""
        if self.prompts_for_first_generation_phase is None and self.prompts_for_second_generation_phase is None:
            raise ValueError("No prompts for generation.")

        self.num_gen_batches += 1
        if self.num_gen_batches >= self.max_num_gen_batches:
            raise ValueError(f"Generated too many batches. We have {self.num_gen_batches=} >= {self.max_num_gen_batches=}.")
        
        if self.prompts_for_second_generation_phase is None:
            assert self.prompts_for_first_generation_phase.batch.batch_size[0] % self.n_gpus == 0, "The number of prompts for the first generation phase should be divisible by the number of GPUs."
            
            batch = self.prompts_for_first_generation_phase
            self.prompts_for_first_generation_phase = None # remove the prompts.
            return batch
        else:
            num_second_phase_prompts = self.prompts_for_second_generation_phase.batch.batch_size[0]
            num_second_phase_prompts_for_generation = num_second_phase_prompts - (num_second_phase_prompts % self.n_gpus) # RZ: The number of prompts must be a multiple of the number of GPUs.
            print(f"Originally we have {num_second_phase_prompts} second phase prompts. After batchings, we have {num_second_phase_prompts_for_generation} second phase prompts for generation.")

            if num_second_phase_prompts_for_generation == 0:
                batch = self.prompts_for_first_generation_phase
                self.prompts_for_first_generation_phase = None # remove the prompts.
                return batch
            else:
                repeated_indices = []
                for i in range(num_second_phase_prompts_for_generation):
                    repeated_indices.extend([i] * self.multiplier)
                repeated_second_phase = self.prompts_for_second_generation_phase[repeated_indices] # multipy the second phase prompts.
                
                if self.prompts_for_first_generation_phase is not None:
                    batch = DataProto.concat([self.prompts_for_first_generation_phase, repeated_second_phase])
                else:
                    batch = repeated_second_phase
            
                self.prompts_for_first_generation_phase = None
                
                if num_second_phase_prompts % self.n_gpus == 0:
                    self.prompts_for_second_generation_phase = None
                else:
                    self.prompts_for_second_generation_phase = self.prompts_for_second_generation_phase[-(num_second_phase_prompts % self.n_gpus):]
                return batch

    def update_prompts(self, batch: DataProto, metric_name: str = "acc") -> Tuple[List[str], List[int]]:
        """Update prompt statistics based on the generated responses and their rewards.
        
        There are two types of prompts in 'batch':
            one with self.initial_n responses, and the other with self.n_continue responses.
        For the first type, those are the prompt that finish the first generation phase.
        For the second type, those are the prompt that finish the second generation phase and we can use those for training.
        """

        # Collect the sequence reward for each trajectory
        prompt_uid2metric_vals = defaultdict(list)
        for uid, metric_val in zip(
            batch.non_tensor_batch["uid"], batch.non_tensor_batch[metric_name]
        ):
            prompt_uid2metric_vals[uid].append(metric_val)

        # categorize the prompts into two types.
        prompts_uids_after_first_phase = [
            uid
            for uid, _ in prompt_uid2metric_vals.items()
            if len(prompt_uid2metric_vals[uid]) == self.initial_n
        ]
        prompts_uids_after_second_phase = [
            uid
            for uid, _ in prompt_uid2metric_vals.items()
            if len(prompt_uid2metric_vals[uid]) == self.n_continue
        ]

        # Deal with the first-class prompts.
        prompt_uid2metric_std = {}
        for prompt_uid in prompts_uids_after_first_phase:
            prompt_uid2metric_std[prompt_uid] = np.std(prompt_uid2metric_vals[prompt_uid])
        kept_prompt_uids = [
            uid
            for uid, std in prompt_uid2metric_std.items()
            if std > 0
        ] # RZ: The qualified prompts that have pass rate between 0 and 1.
        kept_traj_idxs = []
        kept_traj_idxs_unique = []
        kept_prompt_uids_unique = []
        for idx, traj_from_prompt_uid in enumerate(batch.non_tensor_batch["uid"]):
            if traj_from_prompt_uid in kept_prompt_uids:
                kept_traj_idxs.append(idx)
                if traj_from_prompt_uid not in kept_prompt_uids_unique:
                    kept_prompt_uids_unique.append(traj_from_prompt_uid)
                    kept_traj_idxs_unique.append(idx)

        qualified_prompts_responses_after_first_phase = batch[kept_traj_idxs] # DataProto
        qualified_prompts_after_first_phase = batch[kept_traj_idxs_unique].select(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=[key for key in batch.non_tensor_batch.keys() if key not in ['score', 'acc', 'reward']],
            meta_info_keys=None
        ).truncate(start=0, end=self.max_prompt_length) # DataProto. But only have the prompt part. No responses.

        self.prompts_for_second_generation_phase = qualified_prompts_after_first_phase if self.prompts_for_second_generation_phase is None else DataProto.concat([self.prompts_for_second_generation_phase, qualified_prompts_after_first_phase])

        # Add the responses from the first-class prompts to the training set.
        self.prompts_for_training = qualified_prompts_responses_after_first_phase if self.prompts_for_training is None else DataProto.concat([self.prompts_for_training, qualified_prompts_responses_after_first_phase])
        
        # Deal with the second-class prompts.
        traj_idxs = []
        if prompts_uids_after_second_phase:
            for idx, traj_from_prompt_uid in enumerate(batch.non_tensor_batch["uid"]):
                if traj_from_prompt_uid in prompts_uids_after_second_phase:
                    traj_idxs.append(idx)
        responses_after_second_phase = batch[traj_idxs]
        self.prompts_for_training = responses_after_second_phase if self.prompts_for_training is None else DataProto.concat([self.prompts_for_training, responses_after_second_phase])
        self.num_prompts_for_training += len(set(prompts_uids_after_second_phase))

    def get_training_data(self,
                          prompt_batch_size: int
                          ) -> DataProto:
        """Get qualified prompts for training."""

        assert prompt_batch_size == self.train_batch_size, f"The prompt batch size should be equal to the training batch size. Got {prompt_batch_size=} and {self.train_batch_size=}"

        # Get the indices of all qualified prompts (with self.total_n_generations responses).
        unique_uids = []
        for uid in self.prompts_for_training.non_tensor_batch["uid"]:
            if uid not in unique_uids:
                unique_uids.append(uid)
        uid_to_indices = defaultdict(list)
        for idx, uid in enumerate(self.prompts_for_training.non_tensor_batch["uid"]):
            uid_to_indices[uid].append(idx)
        training_indices = []
        for uid in unique_uids:
            if len(uid_to_indices[uid]) == self.total_n_generations:
                training_indices.extend(uid_to_indices[uid])

        # Check if the number of training indices is greater than or equal to the training response batch size.
        train_resp_bsz = prompt_batch_size * self.total_n_generations
        assert len(training_indices) >= train_resp_bsz, f"The number of training indices should be greater than or equal to the training response batch size. Got {len(training_indices)=} and {train_resp_bsz=}"

        training_indices = training_indices[:train_resp_bsz]
        training_data = self.prompts_for_training[training_indices]
        self.num_prompts_for_training -= self.train_batch_size
        remaining_indices = [i for i in range(self.prompts_for_training.batch.batch_size[0]) if i not in training_indices]
        self.prompts_for_training = self.prompts_for_training[remaining_indices]

        # Double Check if the training data has prompt_batch_size prompts and self.total_n_generations responses for each prompt.
        training_uids_to_indices = defaultdict(list)
        for idx, uid in enumerate(training_data.non_tensor_batch["uid"]):
            training_uids_to_indices[uid].append(idx)
        for uid in training_uids_to_indices.keys():
            assert len(training_uids_to_indices[uid]) == self.total_n_generations, f"The number of training indices for each prompt should be equal to the total number of generations. Got {len(training_uids_to_indices[uid])=} and {self.total_n_generations=}"
        assert len(training_uids_to_indices) == prompt_batch_size, f"The number of training uids should be equal to the prompt batch size. Got {len(training_uids_to_indices)=} and {prompt_batch_size=}"

        return training_data

    def reset_num_gen_batches(self):
        self.num_gen_batches = 0