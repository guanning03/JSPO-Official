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

import os
import random
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Union

import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.device import get_device_name, get_torch_device


class BaseCheckpointManager:
    """
    A checkpoint manager that saves and loads
    - model
    - optimizer
    - lr_scheduler
    - extra_states
    in a SPMD way.

    We save
    - sharded model states and optimizer states
    - full lr_scheduler states
    - huggingface tokenizer and config for ckpt merge
    """
    
    # Class-level thread pool for async uploads
    _upload_executor = None
    _upload_lock = threading.Lock()
    _active_uploads = set()  # Track active upload tasks

    def __init__(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        processing_class: Union[PreTrainedTokenizer, ProcessorMixin] = None,
        checkpoint_config: DictConfig = None,
        online_hf_name: str = None,
        online_hf_repo_name: str = None,
        experiment_name: str = None,
    ):
        self.checkpoint_config = checkpoint_config
        checkpoint_load_contents = checkpoint_config.get("load_contents", None) if checkpoint_config else None
        checkpoint_save_contents = checkpoint_config.get("save_contents", None) if checkpoint_config else None
        if checkpoint_load_contents is None:
            checkpoint_load_contents = ["model", "optimizer", "extra"]
        if checkpoint_save_contents is None:
            checkpoint_save_contents = ["model", "optimizer", "extra"]
        
        self.previous_global_step = None
        self.previous_saved_paths = []

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.processing_class = processing_class
        self.checkpoint_load_contents = checkpoint_load_contents
        self.checkpoint_save_contents = checkpoint_save_contents
        
        # HuggingFace online model upload settings
        self.online_hf_name = online_hf_name
        self.online_hf_repo_name = online_hf_repo_name
        self.experiment_name = experiment_name

        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        
        # Validate online HuggingFace model requirements
        self._validate_online_hf_model_config()
    
    @classmethod
    def _get_upload_executor(cls):
        """Get or create the class-level thread pool executor for async uploads."""
        with cls._upload_lock:
            if cls._upload_executor is None:
                # Use max 2 threads to avoid overwhelming network/API limits
                cls._upload_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="HF_Upload")
            return cls._upload_executor
    
    @classmethod
    def cleanup_upload_executor(cls):
        """Clean up the upload executor. Should be called at the end of training."""
        with cls._upload_lock:
            if cls._upload_executor is not None:
                print("Waiting for ongoing HuggingFace uploads to complete...")
                cls._upload_executor.shutdown(wait=True)
                cls._upload_executor = None
                cls._active_uploads.clear()
                print("All HuggingFace uploads completed.")

    @property
    def should_save_model(self) -> bool:
        """
        Returns True if 'model' is in checkpoint_save_contents, indicating the model state should be saved.
        """
        return "model" in self.checkpoint_save_contents

    @property
    def should_save_optimizer(self) -> bool:
        """
        Returns True if 'optimizer' is in checkpoint_save_contents, indicating the optimizer state should be saved.
        """
        return "optimizer" in self.checkpoint_save_contents

    @property
    def should_save_extra(self) -> bool:
        """
        Returns True if 'extra' is in checkpoint_save_contents, indicating the extra state should be saved.
        """
        return "extra" in self.checkpoint_save_contents

    @property
    def should_save_hf_model(self) -> bool:
        """
        Returns True if 'hf_model' is in checkpoint_save_contents, indicating the model should be converted to hf
        model and saved.
        """
        return "hf_model" in self.checkpoint_save_contents
    
    @property
    def should_save_online_hf_model(self) -> bool:
        """
        Returns True if 'online_hf_model' is in checkpoint_save_contents, indicating the model should be uploaded
        to HuggingFace Hub.
        """
        return "online_hf_model" in self.checkpoint_save_contents

    @property
    def should_load_model(self) -> bool:
        """
        Returns True if 'model' is in checkpoint_load_contents, indicating the model state should be loaded.
        """
        return "model" in self.checkpoint_load_contents

    @property
    def should_load_optimizer(self) -> bool:
        """
        Returns True if 'optimizer' is in checkpoint_load_contents, indicating the optimizer state should be loaded.
        """
        return "optimizer" in self.checkpoint_load_contents

    @property
    def should_load_extra(self) -> bool:
        """
        Returns True if 'extra' is in checkpoint_load_contents, indicating the extra state should be loaded.
        """
        return "extra" in self.checkpoint_load_contents

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load: bool = False):
        raise NotImplementedError

    def save_checkpoint(
        self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep: int = None
    ):
        raise NotImplementedError

    @staticmethod
    def checkpath(local_path: str, hdfs_path: str):
        assert local_path is not None or hdfs_path is not None, "local_path and hdfs_path cannot be both None"
        return local_path is not None, local_path if local_path is not None else hdfs_path

    def remove_previous_save_local_path(self, path):
        if isinstance(path, str):
            path = [path]
        for p in path:
            abs_path = os.path.abspath(p)
            print(f"Checkpoint manager remove previous save local path: {abs_path}")
            if not os.path.exists(abs_path):
                continue
            shutil.rmtree(abs_path, ignore_errors=True)

    @staticmethod
    def get_rng_state():
        rng_state = {
            "cpu": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }

        if get_device_name() != "cpu":
            rng_state[get_device_name()] = get_torch_device().get_rng_state()

        return rng_state

    @staticmethod
    def load_rng_state(rng_state):
        torch.set_rng_state(rng_state["cpu"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["random"])

        if get_device_name() != "cpu":
            get_torch_device().set_rng_state(rng_state[get_device_name()])

    def _validate_online_hf_model_config(self):
        """
        Validate configuration requirements for online HuggingFace model upload.
        
        Requirements:
        1. If 'online_hf_model' is in save_contents, then 'hf_model' must also be present
        2. If 'online_hf_model' is in save_contents, online_hf_name must be provided
        3. If 'online_hf_model' is in save_contents, online_hf_repo_name must be provided
        """
        if self.should_save_online_hf_model:
            if not self.should_save_hf_model:
                raise ValueError(
                    "When using 'online_hf_model', 'hf_model' must also be included in save_contents. "
                    "Current save_contents: {}".format(self.checkpoint_save_contents)
                )
            
            if not self.online_hf_name:
                raise ValueError(
                    "When using 'online_hf_model', online_hf_name (HuggingFace username/organization) must be provided."
                )
            
            if not self.online_hf_repo_name:
                raise ValueError(
                    "When using 'online_hf_model', online_hf_repo_name (HuggingFace repository name) must be provided."
                )
    
    def _sync_upload_to_huggingface(self, hf_model_path: str, global_step: int, repo_id: str, path_in_repo: str):
        """
        Internal method for synchronous upload to HuggingFace Hub.
        This method runs in a background thread.
        """
        upload_id = f"{repo_id}/{path_in_repo}"
        try:
            start_time = time.time()
            print(f"[ASYNC] Starting upload: {upload_id}")
            
            from huggingface_hub import HfApi
            
            api = HfApi()
            
            # Create repository if it doesn't exist
            api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
            
            # Upload folder
            api.upload_folder(
                folder_path=hf_model_path,
                repo_id=repo_id,
                path_in_repo=path_in_repo,
                repo_type="model"
            )
            
            elapsed = time.time() - start_time
            print(f"[ASYNC] ✅ Upload completed in {elapsed:.1f}s: https://huggingface.co/{repo_id}/tree/main/{path_in_repo}")
            
        except ImportError:
            print(f"[ASYNC] ❌ Upload failed: huggingface_hub is required for online HuggingFace model upload")
        except Exception as e:
            print(f"[ASYNC] ❌ Upload failed for {upload_id}: {e}")
        finally:
            # Remove from active uploads
            with self.__class__._upload_lock:
                self.__class__._active_uploads.discard(upload_id)
    
    def upload_to_huggingface(self, hf_model_path: str, global_step: int):
        """
        Asynchronously upload the HuggingFace model to HuggingFace Hub.
        This method returns immediately and runs the upload in a background thread.
        
        Args:
            hf_model_path: Local path to the HuggingFace model
            global_step: Current training step
        """
        if not self.should_save_online_hf_model:
            return
        
        repo_id = f"{self.online_hf_name}/{self.online_hf_repo_name}"
        path_in_repo = f"{self.experiment_name}/global_step_{global_step}/"
        upload_id = f"{repo_id}/{path_in_repo}"
        
        # Check if this upload is already in progress
        with self.__class__._upload_lock:
            if upload_id in self.__class__._active_uploads:
                print(f"[ASYNC] Upload already in progress: {upload_id}")
                return
            self.__class__._active_uploads.add(upload_id)
        
        # Submit async upload task
        executor = self._get_upload_executor()
        future = executor.submit(
            self._sync_upload_to_huggingface, 
            hf_model_path, 
            global_step, 
            repo_id, 
            path_in_repo
        )
        
        print(f"[ASYNC] Queued upload: {upload_id} (training continues...)")


def find_latest_ckpt_path(path, directory_format="global_step_{}"):
    """
    Return the most recent checkpoint directory based on a tracker file.

    Args:
        path (str): Base directory containing the checkpoint tracker.
        directory_format (str): Template for checkpoint subfolders with one
            placeholder for the iteration number (default "global_step_{}").

    Returns:
        str or None: Full path to the latest checkpoint directory, or
        None if the tracker or checkpoint folder is missing.
    """
    if path is None:
        return None

    tracker_file = get_checkpoint_tracker_filename(path)
    if not os.path.exists(tracker_file):
        print(f"Checkpoint tracker file does not exist: {tracker_file}")
        return None

    with open(tracker_file, "rb") as f:
        iteration = int(f.read().decode())
    ckpt_path = os.path.join(path, directory_format.format(iteration))
    if not os.path.exists(ckpt_path):
        print("Checkpoint does not exist: %s", ckpt_path)
        return None

    print("Found checkpoint: %s", ckpt_path)
    return ckpt_path


def get_checkpoint_tracker_filename(root_path: str):
    """
    Tracker file rescords the latest chckpoint during training to restart from.
    """
    return os.path.join(root_path, "latest_checkpointed_iteration.txt")


def should_save_ckpt_esi(max_steps_duration: float, save_ckpt_duration: float = 60, redundant_time: float = 0) -> bool:
    """
    Determine if checkpoint should be saved based on capacity esi expiration.

    Args:
        max_steps_duration: Max estimated time (seconds) required to complete one training step
        save_ckpt_duration: Estimated time (seconds) required to save checkpoint (default: 60)
        redundant_time: Additional buffer time (seconds) for unexpected delays (default: 0)
    """
    exp_ts_mlp = os.getenv("MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP")  # vemlp
    exp_ts_aws = os.getenv("SAGEMAKER_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP")  # aws
    if exp_ts_mlp:
        try:
            import time

            remaining = float(exp_ts_mlp) - time.time()
        except ValueError:
            return False
        return (
            remaining > 0
            and max_steps_duration > 0
            and remaining <= save_ckpt_duration + max_steps_duration + redundant_time
        )
    elif exp_ts_aws:
        from datetime import datetime, timedelta

        expiration_time = datetime.fromtimestamp(int(exp_ts_aws))
        time_difference = expiration_time - datetime.now()
        threshold_minutes = (save_ckpt_duration + max_steps_duration + redundant_time) / 60
        return time_difference < timedelta(minutes=threshold_minutes)
    else:
        return False
