# Copyright 2024-present the HuggingFace Inc. team.
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

from __future__ import annotations

import warnings
from typing import Optional

from transformers import PreTrainedModel

from .auto import MODEL_TYPE_TO_PEFT_MODEL_MAPPING
from .config import PeftConfig
from .mapping import PEFT_TYPE_TO_CONFIG_MAPPING, PEFT_TYPE_TO_PREFIX_MAPPING
from .mixed_model import PeftMixedModel
from .peft_model import PeftModel
from .tuners.tuners_utils import BaseTuner, BaseTunerLayer
from .utils import _prepare_prompt_learning_config
import copy

def get_peft_model(
    model: PreTrainedModel,
    peft_config: PeftConfig,
    num_lora_adapters: int = 0,
    autocast_adapter_dtype: bool = True,
    revision: Optional[str] = None,
    low_cpu_mem_usage: bool = False,
) -> PeftModel | PeftMixedModel:
    """
    Returns a Peft model object from a model and a config, where the model will be modified in-place.

    Args:
        model ([`transformers.PreTrainedModel`]):
            Model to be wrapped.
        peft_config ([`PeftConfig`]):
            Configuration object containing the parameters of the Peft model.
        adapter_name (`str`, `optional`, defaults to `"default"`):
            The name of the adapter to be injected, if not provided, the default adapter name is used ("default").
        mixed (`bool`, `optional`, defaults to `False`):
            Whether to allow mixing different (compatible) adapter types.
        autocast_adapter_dtype (`bool`, *optional*):
            Whether to autocast the adapter dtype. Defaults to `True`. Right now, this will only cast adapter weights
            using float16 or bfloat16 to float32, as this is typically required for stable training, and only affect
            select PEFT tuners.
        revision (`str`, `optional`, defaults to `main`):
            The revision of the base model. If this isn't set, the saved peft model will load the `main` revision for
            the base model
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading process. Leave this setting as
            False if you intend on training the model, unless the adapter weights will be replaced by different weights
            before training starts.
    """
    model_config = BaseTuner.get_model_config(model)
    old_name = peft_config.base_model_name_or_path
    new_name = model.__dict__.get("name_or_path", None)
    peft_config.base_model_name_or_path = new_name

    # Especially in notebook environments there could be a case that a user wants to experiment with different
    # configuration values. However, it is likely that there won't be any changes for new configs on an already
    # initialized PEFT model. The best we can do is warn the user about it.
    if any(isinstance(module, BaseTunerLayer) for module in model.modules()):
        warnings.warn(
            "You are trying to modify a model with PEFT for a second time. If you want to reload the model with a "
            "different config, make sure to call `.unload()` before."
        )

    if (old_name is not None) and (old_name != new_name):
        warnings.warn(
            f"The PEFT config's `base_model_name_or_path` was renamed from '{old_name}' to '{new_name}'. "
            "Please ensure that the correct base model is loaded when loading this checkpoint."
        )

    if revision is not None:
        if peft_config.revision is not None and peft_config.revision != revision:
            warnings.warn(
                f"peft config has already set base model revision to {peft_config.revision}, overwriting with revision {revision}"
            )
        peft_config.revision = revision

    if (
        (isinstance(peft_config, PEFT_TYPE_TO_CONFIG_MAPPING["LORA"]))
        and (peft_config.init_lora_weights == "eva")
        and not low_cpu_mem_usage
    ):
        warnings.warn(
            "lora with eva initialization used with low_cpu_mem_usage=False. "
            "Setting low_cpu_mem_usage=True can improve the maximum batch size possible for eva initialization."
        )

    # 多LoRA自动注册逻辑
    adapter_names = [f'LORA_{i}' for i in range(1, num_lora_adapters + 1)]
    peft_configs = [copy.deepcopy(peft_config) for _ in range(num_lora_adapters)]

    # 初始化第一个adapter
    peft_model = PeftModel(
        model,
        peft_configs[0],
        adapter_name=adapter_names[0],
        autocast_adapter_dtype=autocast_adapter_dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    # 批量添加剩余adapter
    for name, config in zip(adapter_names[1:], peft_configs[1:]):
        peft_model.add_adapter(name, 
                               config, 
                               low_cpu_mem_usage=low_cpu_mem_usage, 
                               autocast_adapter_dtype=autocast_adapter_dtype)

    return peft_model
