# Copyright 2025 The HuggingFace Team. All rights reserved.
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

# Copyright The Lightning AI team.
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

# Modified by OPPO in 2026 for caption-specific GRPO training.
# Adapted _ForwardRedirection from PyTorch Lightning.

import os
import textwrap
from collections import defaultdict
import time
from typing import Any, Callable, Optional, Union, Sized
import gc

import torch
import torch.utils.data
import transformers
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler
import datasets
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled, is_deepspeed_available
from transformers.utils import is_peft_available, is_rich_available,  is_datasets_available
from transformers.trainer_utils import seed_worker
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, print_prompt_completions_sample
from trl.import_utils import is_deepspeed_available
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
import PIL.Image

import copy
from torch.utils.data import Sampler
import warnings
import torch.distributed as dist


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

from open_r1.vlm_modules.vlm_module import VLMBaseModule
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss
from liger_kernel.transformers import LigerRMSNorm, LigerSwiGLUMLP
from liger_kernel.ops import LigerSiLUMulFunction

def apply_liger_to_omni(model):
    """
    针对 Qwen2_5-Omni 的精准手动替换
    """
    print("Starting Liger-Kernel injection...")

    def patch_norm(old_norm):
        # 保持参数一致：hidden_size 和 eps
        new_norm = LigerRMSNorm(
            old_norm.weight.shape[0], 
            eps=old_norm.variance_epsilon
        ).to(old_norm.weight.device).to(old_norm.weight.dtype)
        
        # 使用 load_state_dict 确保权重完整拷贝且属性正确
        new_norm.load_state_dict(old_norm.state_dict())
        return new_norm
    
    # 1. 替换文本解码器部分 (model.model.layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for i, layer in enumerate(model.model.layers):
            # # 替换 RMSNorm: Qwen2.5 默认使用 RMSNorm
            # # 注意：LigerRMSNorm 接受 (hidden_size, eps)
            # layer.input_layernorm = patch_norm(layer.input_layernorm)
            # layer.post_attention_layernorm = patch_norm(layer.post_attention_layernorm)

            # # --- 替换 MLP (关键修改) ---
            # # 我们不直接实例化 LigerSwiGLUMLP，因为它的 __init__ 会重置权重
            # # 我们通过修改原始 MLP 的 forward 逻辑来注入 Liger 算子
            
            def make_liger_forward(gate_proj, up_proj, down_proj):
                def forward(x):
                    # 调用 Liger 的融合算子：SiLU(gate(x)) * up(x)
                    return down_proj(LigerSiLUMulFunction.apply(gate_proj(x), up_proj(x)))
                return forward

            # 直接劫持原有 mlp 对象的 forward 方法
            # 这样既保留了 layer.mlp.gate_proj 等原本的权重，又换了算子
            layer.mlp.forward = make_liger_forward(
                layer.mlp.gate_proj, 
                layer.mlp.up_proj, 
                layer.mlp.down_proj
            )
        
        # # 替换最后的全局 Norm
        # model.model.norm = patch_norm(model.model.norm)
        print(f"Successfully patched {len(model.model.layers)} text layers.")

    print("Liger-Kernel manual injection completed!")
    return model


class _ForwardRedirection:
    """Implements the `forward-redirection`.

    Taken from Pytorch-lightning:
    https://github.com/Lightning-AI/pytorch-lightning/blob/02311d03fb982560246eead7c08104481fac9579/src/lightning/pytorch/strategies/strategy.py#L602

    A method call to a wrapped module gets rerouted through the wrapper's `forward` method instead.

    """

    def __call__(
        self, wrapper_module: nn.Module, original_module: nn.Module, method: Callable, *args: Any, **kwargs: Any
    ):
        """Reroutes a method call through the `wrapper_module`'s `forward` method.

        Args:
            wrapper_module: The module that has `original_module` wrapped.
            original_module: The module that was wrapped inside `wrapper_module`.
            method: The method that should be called on the `original_module` after inputs get
                redirected through the `wrapper_module`'s `forward` method.
            *args: The positional arguments to the `method`. They will get passed to a patched
                `forward` method instead.
            **kwargs: The keyword arguments to the `method`. They will get passed to a patched
                `forward` method instead.

        """
        original_forward = original_module.forward

        def wrapped_forward(*_args: Any, **_kwargs: Any) -> Any:
            # Unpatch ourselves immediately before calling the method `method_name`
            # because itself may want to call the real `forward`
            original_module.forward = original_forward  # type: ignore[method-assign]
            # Call the actual method e.g. `.training_step(...)`
            out = method(*_args, **_kwargs)
            self.on_after_inner_forward(wrapper_module, original_module)
            return out

        # Patch the original_module's forward so we can redirect the arguments back to the real method
        original_module.forward = wrapped_forward  # type: ignore[method-assign]

        wrapper_output = wrapper_module(*args, **kwargs)
        self.on_after_outer_forward(wrapper_module, original_module)
        return wrapper_output

    def on_after_inner_forward(self, wrapper_module: nn.Module, original_module: nn.Module) -> None:
        pass

    def on_after_outer_forward(self, wrapper_module: nn.Module, original_module: nn.Module) -> None:
        pass


def find_mask_between_patterns_1d(input_tensor: torch.Tensor, 
                                  start_pattern_list: list, 
                                  end_pattern_list: list) -> torch.Tensor:
    """
    (Helper function - same as before)
    Finds the mask for a single 1D tensor.
    """
    assert input_tensor.ndim == 1, "Input tensor must be 1-dimensional"
    
    device = input_tensor.device
    dtype = input_tensor.dtype # Use input tensor's dtype

    # Ensure patterns are tensors on the correct device and dtype
    start_pattern = torch.tensor(start_pattern_list, dtype=dtype, device=device)
    end_pattern = torch.tensor(end_pattern_list, dtype=dtype, device=device)

    n = input_tensor.shape[0]
    len_start = len(start_pattern)
    len_end = len(end_pattern)

    start_idx = -1
    end_idx = -1

    # --- Find start_pattern index ---
    if n >= len_start:
        start_windows = input_tensor.unfold(0, len_start, 1)
        start_matches = (start_windows == start_pattern).all(dim=1)
        start_indices = start_matches.nonzero(as_tuple=True)[0]
        if start_indices.numel() > 0:
            start_idx = start_indices[0].item() # Assume first match
        else:
            # Indicate pattern not found for this row
            return torch.zeros_like(input_tensor, dtype=torch.long, device=device) 
            # raise ValueError("Start pattern not found in the tensor.") # Original behavior
    else:
        return torch.zeros_like(input_tensor, dtype=torch.long, device=device) # Too short

    # --- Find end_pattern index ---
    if n >= len_end:
        # Search *after* the start pattern to ensure correct order if multiple end patterns exist
        # Although problem states "only one region", this adds robustness
        search_area_end = input_tensor[start_idx + len_start:] 
        if search_area_end.numel() >= len_end:
            end_windows = search_area_end.unfold(0, len_end, 1)
            end_matches = (end_windows == end_pattern).all(dim=1)
            end_indices = end_matches.nonzero(as_tuple=True)[0]
            if end_indices.numel() > 0:
                 # Index relative to the start of search_area_end, need to add offset
                relative_end_idx = end_indices[0].item()
                end_idx = start_idx + len_start + relative_end_idx 
            else:
                 # End pattern not found *after* start pattern
                return torch.zeros_like(input_tensor, dtype=torch.long, device=device)
        else:
            # Not enough elements after start pattern to contain end pattern
             return torch.zeros_like(input_tensor, dtype=torch.long, device=device)
    else:
       return torch.zeros_like(input_tensor, dtype=torch.long, device=device) # Too short

    # --- Calculate mask region ---
    mask_start = start_idx + len_start
    mask_end = end_idx # end_idx is the *start* of the end pattern

    # mask_start = start_idx 
    # mask_end = end_idx + len_end 

    # --- Create and fill mask ---
    mask = torch.zeros_like(input_tensor, dtype=torch.long, device=device)

    # if mask_start < mask_end:
    #     mask[mask_start:-1] = 1
    if mask_start < mask_end:
        mask[:mask_end] = 1

    # if mask_start < mask_end:
    #     mask[mask_start:mask_end] = 1
    # else: patterns adjacent or end before start, mask remains zero, no warning needed here

    return mask



def generate_2d_mask(input_tensor_2d: torch.Tensor, 
                       start_pattern_list: list, 
                       end_pattern_list: list) -> torch.Tensor:
    """
    Generates a 2D mask by applying the 1D pattern finding logic to each row.

    Args:
        input_tensor_2d: The input 2D PyTorch Tensor (Batch x SequenceLength).
        start_pattern_list: The start pattern list.
        end_pattern_list: The end pattern list.

    Returns:
        A 2D mask tensor of the same shape as input_tensor_2d (dtype=torch.long),
        where each row's mask is generated based on the patterns found in that row.
        Rows where patterns are not found (or order is wrong) will have a mask of all zeros.
    """
    assert input_tensor_2d.ndim == 2, "Input tensor must be 2-dimensional"
    
    num_rows = input_tensor_2d.shape[0]
    if num_rows == 0:
        return torch.empty_like(input_tensor_2d, dtype=torch.long) # Handle empty input

    row_masks = []
    for i in range(num_rows):
        current_row = input_tensor_2d[i]
        # Call the 1D function for the current row
        # Modify 1D function to return zeros instead of raising error if pattern not found
        mask_1d = find_mask_between_patterns_1d(current_row, start_pattern_list, end_pattern_list)
        row_masks.append(mask_1d)

    # Stack the generated 1D masks along the batch dimension (dim=0)
    mask_2d = torch.stack(row_masks, dim=0)
    
    return mask_2d

class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility.
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = 0 #len(data_source)
        for data_key, data_list in self.data_source.items():
            self.num_samples += len(data_list)
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    '''
    def __iter__(self):
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index
    '''

    def __iter__(self):
        all_chunks = []

        # 2. 对每个 type 组内打乱并切分 chunk
        for sample_type, indices in self.data_source.items():
            if len(indices) == 0:
                continue
            # 打乱当前 type 的 indices
            perm = torch.randperm(len(indices), generator=self.generator)
            shuffled_indices = [indices[i] for i in perm.tolist()]

            # 切分成 chunks（每个 chunk 大小为 batch_size）
            chunks = [
                shuffled_indices[i : i + self.batch_size]
                for i in range(0, len(shuffled_indices), self.batch_size)
            ]

            # 只保留完整 chunk
            full_chunks = [chunk for chunk in chunks if len(chunk) == self.batch_size]
            all_chunks.extend(full_chunks)

        # 3. 可选：打乱所有 chunk 的顺序，避免总是先训某 type
        if len(all_chunks) > 1:
            chunk_perm = torch.randperm(len(all_chunks), generator=self.generator).tolist()
            all_chunks = [all_chunks[i] for i in chunk_perm]

        # 4. 按 RepeatRandomSampler 原逻辑 yield
        for chunk in all_chunks:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


class GRPOTrainer_Caption(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        vlm_module: VLMBaseModule = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        train_dataloader = None,
        data_type_to_indices = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        freeze_vision_modules: Optional[bool] = True,
        attn_implementation: str = "flash_attention_2",
        torch_dtype: str = "bfloat16",
        **kwargs,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
        
        self.vlm_module = vlm_module

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        # FIXME
        # Remember to modify it in the invernvl
        model_init_kwargs["attn_implementation"] = attn_implementation
        if model_init_kwargs.get("torch_dtype") is None:
            model_init_kwargs["torch_dtype"] = torch_dtype
        
        assert isinstance(model, str), "model must be a string in the current implementation"
        model_id = model
        torch_dtype = model_init_kwargs.get("torch_dtype")
        if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
            pass  # torch_dtype is already a torch.dtype or "auto" or None
        elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
            torch_dtype = getattr(torch, torch_dtype)
        else:
            raise ValueError(
                "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
            )
        # model_init_kwargs["enable_audio_output"] = False
        # model_init_kwargs["use_cache"] = (
        #     False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        # )
        #     # Disable caching if gradient checkpointing is enabled (not supported)
        # model_init_kwargs["use_cache"] = (
        #     False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        # )
        model_cls = self.vlm_module.get_model_class(model_id, model_init_kwargs)
        model = model_cls.from_pretrained(model_id, **model_init_kwargs)
        print('model_origin', type(model.model.layers[0].self_attn))
        print('model_origin', model.model.layers[0].mlp.forward)
        model = apply_liger_to_omni(model)
        print('model_liger', model.model.layers[0].mlp.forward)
        # print(f"当前使用的 Attention 实现: {model.config._attn_implementation}")
        # model = model.thinker # for qwen-omni

        # LoRA
        self.vision_modules_keywords = self.vlm_module.get_vision_modules_keywords()
        if peft_config is not None:
            def find_all_linear_names(model, multimodal_keywords):
                cls = torch.nn.Linear
                lora_module_names = set()
                for name, module in model.named_modules():
                    # LoRA is not applied to the vision modules
                    if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                        continue
                    if isinstance(module, cls):
                        lora_module_names.add(name)
                for m in lora_module_names:  # needed for 16-bit
                    if "embed_tokens" in m:
                        lora_module_names.remove(m)
                return list(lora_module_names)
            target_modules = find_all_linear_names(model, self.vision_modules_keywords)
            peft_config.target_modules = target_modules
            model = get_peft_model(model, peft_config)

        # Freeze vision modules
        if freeze_vision_modules:
            print("Freezing vision modules...")
            for n, p in model.named_parameters():
                if any(keyword in n for keyword in self.vision_modules_keywords):
                    p.requires_grad = False

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Reference model
        if is_deepspeed_zero3_enabled() or is_deepspeed_available():

            self.ref_model = model_cls.from_pretrained(model_id, **model_init_kwargs)
            # self.ref_model = self.ref_model.thinker # for qwen-omni
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
            # self.ref_model = self.ref_model.thinker # for qwen-omni
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            processing_cls = self.vlm_module.get_processing_class()
            processing_class = processing_cls.from_pretrained(model_id, trust_remote_code=model_init_kwargs.get("trust_remote_code", None))
            for processing_keyword in self.vlm_module.get_custom_processing_keywords():
                if processing_keyword in kwargs:
                    setattr(processing_class, processing_keyword, kwargs[processing_keyword])
            if getattr(processing_class, "tokenizer",  None) is not None:
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
            else:
                assert isinstance(processing_class, PreTrainedTokenizerBase), "processing_class must be an instance of PreTrainedTokenizerBase if it has no tokenizer attribute"
                pad_token_id = processing_class.pad_token_id
        # print(processing_class.tokenizer)
        self.vlm_module.post_model_init(model, processing_class)
        self.vlm_module.post_model_init(self.ref_model, processing_class)


        total_trainable_params = 0
        for name, p in model.named_parameters():
            if p.requires_grad:
                # print(f'train param: {name}')
                total_trainable_params += p.numel()
        print(f"total_trainable_params: {total_trainable_params}")


        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)


        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_prompt_length = None  # TODO
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.num_generations_per_device = args.num_generations_per_device
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.markov_reward = args.markov_reward

        self.data_type_to_indices = data_type_to_indices
        self.custom_train_dataloader = train_dataloader
     
        if args.max_prompt_length is not None:
            warnings.warn("Setting max_prompt_length is currently not supported, it has been set to None")

        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            # num_return_sequences=self.num_generations,
            # num_return_sequences=self.num_generations_per_device,
            num_return_sequences=1,
            max_new_tokens=self.max_completion_length,
            do_sample=True,  
            pad_token_id=processing_class.tokenizer.pad_token_id,
            bos_token_id=processing_class.tokenizer.bos_token_id,
            eos_token_id=processing_class.tokenizer.eos_token_id,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            repetition_penalty=self.repetition_penalty,
            cache_implementation=args.cache_implementation,
        )
        if hasattr(self.vlm_module, "get_eos_token_id"): # For InternVL
            self.generation_config.eos_token_id = self.vlm_module.get_eos_token_id(processing_class)
        self.beta = args.beta
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon


        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        # Tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates
        self._buffered_inputs = [None] * args.gradient_accumulation_steps

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        # if self.num_generations not in possible_values:
        #     raise ValueError(
        #         f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
        #         f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
        #         f"batch size, the valid values for the number of generations are: {possible_values}."
        #     )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)
        # set_seed(args.seed+100*torch.distributed.get_rank())


        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            # 检查是否已经量化，量化模型直接跳过 prepare
            if hasattr(self.ref_model, "is_quantized") and self.ref_model.is_quantized:
                print("Ref model is quantized. Keeping it as is.")
            else:
                if self.is_deepspeed_enabled:
                    self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
                else:
                    self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

        self._forward_redirection = _ForwardRedirection()

        self.liger_grpo_loss = LigerFusedLinearGRPOLoss(
            beta=self.beta,
            epsilon_low=self.epsilon_low,
            epsilon_high=self.epsilon_high,
            temperature=self.temperature,
            use_ref_model=self.beta != 0.0,
            loss_type='grpo', #'dapo'
            max_completion_length=self.max_completion_length,
        )

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models
        else:
            try:
                model.gradient_checkpointing_enable()
            except:
                # For InternVL; these operations are copied from the original training script of InternVL
                model.language_model.config.use_cache = False
                model.vision_model.gradient_checkpointing = True
                model.vision_model.encoder.gradient_checkpointing = True
                model.language_model._set_gradient_checkpointing()
                # This line is necessary, otherwise the `model.gradient_checkpointing_enable()` will be executed during the training process, leading to an error since InternVL does not support this operation.
                args.gradient_checkpointing = False

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model
    
    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, **custom_multimodal_inputs):
        # print('input_ids', input_ids.shape)
        # print('attention_mask', attention_mask.shape)
        logits = model(input_ids=input_ids, attention_mask=attention_mask, **custom_multimodal_inputs).logits  # (B, L, V)
        # import time, sys
        # rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        # print(f"[RANK {rank}] before model forward: time={time.time()}", flush=True)
        # sys.stdout.flush()
        # t0 = time.time()
        # logits = model(input_ids=input_ids, attention_mask=attention_mask, **custom_multimodal_inputs).logits
        # t1 = time.time()
        # print(f"[RANK {rank}] after model forward: dt={t1-t0:.3f}s, time={t1}", flush=True)
        # sys.stdout.flush()
        
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)
    
    def _prepare_inputs(self, inputs):
        return self._generate_and_score_completions(inputs)

    def _get_key_from_inputs(self, x, key):
        ele = x.get(key, None)
        assert ele is not None, f"The key {key} is not found in the input"
        if isinstance(ele, list):
            return [e for e in ele]
        else:
            return [ele]

    def _generate_and_score_completions(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        # print('inputs', inputs)
        prompts = [x["prompt"] for x in inputs]
        # prompts_text = self.vlm_module.prepare_prompt(self.processing_class, inputs)[0]
        prompts_text = self.vlm_module.prepare_prompt(self.processing_class, inputs)
        use_audio_in_video = True #inputs[0].get("use_audio_in_video", False)
        # print('prompts_text', prompts_text)
        images, videos, audios = [], [], []


        for each in inputs:
            if each["images"] is not None:
                images.extend(each["images"])
            if each["audios"] is not None:
                audios.extend(each["audios"])
            if each["videos"] is not None:
                videos.extend(each["videos"])
        if len(images) == 0: images = None
        if len(audios) == 0: audios = None
        if len(videos) == 0: videos = None
    

        prompt_inputs = self.vlm_module.prepare_model_inputs(
            self.processing_class,
            copy.deepcopy(prompts_text),
            images,
            audios,
            videos,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            use_audio_in_video=use_audio_in_video,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_inputs["use_audio_in_video"] = use_audio_in_video

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]


        # max_prompt_length is not supported yet
        # if self.max_prompt_length is not None:
        #     prompt_ids = prompt_ids[:, -self.max_prompt_length :]
        #     prompt_inputs["input_ids"] = prompt_ids
        #     prompt_mask = prompt_mask[:, -self.max_prompt_length :]
        #     prompt_inputs["attention_mask"] = prompt_mask

        target_num_sequences = self.num_generations_per_device
        expanded_inputs = {}
        for k, v in prompt_inputs.items():
            if isinstance(v, torch.Tensor):
                if k in ['pixel_values', 'pixel_values_videos']:
                    # Flattened 视觉特征需要整体重复 (Block-wise repeat)
                    # 假设原始 batch_size=1，使用 repeat(N, 1) 实现 [V1, V1, ...]
                    expanded_inputs[k] = v.repeat(target_num_sequences, 1)
                else:
                    # 其他带 Batch 维度的 Tensor (input_ids, grids, audio 等) 使用 repeat_interleave
                    expanded_inputs[k] = v.repeat_interleave(target_num_sequences, dim=0)
            else:
                expanded_inputs[k] = v
        prompt_inputs = expanded_inputs

        start = time.perf_counter()
        
        # Generate completions
        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            generate_returned_result = unwrapped_model.generate(
                **{k: v for k, v in prompt_inputs.items() if k not in self.vlm_module.get_non_generate_params()}, 
                generation_config=self.generation_config
            )
            print('generate_returned_result', generate_returned_result.shape)
            prompt_length = prompt_ids.size(1)
            if not self.vlm_module.is_embeds_input():
                prompt_completion_ids = generate_returned_result
                prompt_ids = prompt_completion_ids[:, :prompt_length]
                completion_ids = prompt_completion_ids[:, prompt_length:]
            else:
                # In this case, the input of the LLM backbone is the embedding of the combination of the image and text prompt
                # So the returned result of the `generate` method only contains the completion ids
                completion_ids = generate_returned_result
                prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        end = time.perf_counter()
        elapsed = end - start
        print(f"生成耗时: {elapsed:.4f} 秒")

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_mask = prompt_mask.repeat_interleave(self.num_generations_per_device, dim=0)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)
        # print('prompt_completion_ids', prompt_completion_ids.shape)
        # print('attention_mask', attention_mask.shape)

        # Get the multimodal inputs
        multimodal_keywords = self.vlm_module.get_custom_multimodal_keywords()
        multimodal_inputs = {k: prompt_inputs[k] if k in prompt_inputs else None for k in multimodal_keywords}
        # print(f"Video shape: {prompt_inputs['pixel_values_videos'].shape}")
        # multimodal_inputs = {k: prompt_inputs[k] if k in prompt_inputs else None for k in multimodal_keywords}
        # multimodal_inputs = {k: (prompt_inputs[k].repeat_interleave(self.num_generations_per_device, dim=0) if hasattr(prompt_inputs[k], 'repeat_interleave') else prompt_inputs[k]) if k in prompt_inputs else None for k in multimodal_keywords}
        #for k in multimodal_keywords:
        #    if k in prompt_inputs:
        #        #print('multimodal_inputs', k, prompt_inputs[k].shape)
        #        print('multimodal_inputs', k, prompt_inputs[k], prompt_inputs[k].shape)
        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, **multimodal_inputs
                )
                old_per_token_logps = old_per_token_logps[:, prompt_length - 1:]
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, **multimodal_inputs
                )
                ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, **multimodal_inputs
                    )
                ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        print('completions_text', completions_text)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions_text]
            # print('completions', completions)

        # Compute the rewards
        # No need to duplicate prompts as we're not generating multiple completions per prompt
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations_per_device)]
        prompts_text = [prompt for prompt in prompts_text for _ in range(self.num_generations_per_device)]
        dict_in_processing = [{} for _ in range(len(prompts))]
        # print('prompts', prompts)
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        # No need to duplicate prompts as we're not generating multiple completions per prompt
                        reward_kwargs[key].extend([example[key]] * self.num_generations_per_device)
                        # reward_kwargs[key].extend([example[key]])
                output_reward_func = reward_func(prompts=prompts, completions=completions, dict_in_processing=dict_in_processing, **reward_kwargs)
                output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # markov
        if rewards_per_func.size(1) ==4 and self.markov_reward: # format, acc, reason, evi
            print("using markov")
            not_valid_evidence_index = rewards_per_func[:, -1]<=0.4
            not_valid_reason_index = rewards_per_func[:, -2]<=0.4
            rewards_per_func[not_valid_evidence_index, 1] = 0
            rewards_per_func[not_valid_evidence_index, 2] = 0
            rewards_per_func[not_valid_reason_index, 1] = 0

            # not_valid_format_index = rewards_per_func[:, 1]<=0.2
            # rewards_per_func[not_valid_format_index, 0] = 0

        # if rewards_per_func.size(1) ==2: # format, acc, reason, evi
        #     print("using markov")
        # not_valid_format_index = rewards_per_func[:, 1]<=0.2
        
        # rewards_per_func[not_valid_format_index, :] = 0

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )
        # Gather rewards across processes
        rewards_per_func = self.accelerator.gather(rewards_per_func)
        
        # Sum the rewards from all reward functions
        # rewards = rewards_per_func.sum(dim=1)

        def compute_advantage(rewards):
            #print('rewards', rewards)
            # Compute grouped-wise rewards
            # Each group consists of num_generations completions for the same prompt
            mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
            std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            
            # Normalize the rewards to compute the advantages
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            advantages = rewards - mean_grouped_rewards
            if self.args.scale_rewards:
                advantages = advantages / (std_grouped_rewards + 1e-4)
            return advantages, std_grouped_rewards

        full_rewards = []
        patial_rewards = []
        for i, reward_func in enumerate(self.reward_funcs):
            func_name = reward_func.__name__
            if "patial" not in func_name:
                full_rewards.append(rewards_per_func[:, i] * self.reward_weights[i].to(device))
            else:
                patial_rewards.append((func_name, rewards_per_func[:, i] * self.reward_weights[i].to(device)))
        
        
        
        # rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

         # Get only the local slice of advantages
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        # rewards_per_func = rewards_per_func[process_slice, :]

        #print('full_rewards', full_rewards, 'patial_rewards', patial_rewards)
        rewards = torch.stack(full_rewards, dim=-1).nansum(dim=1)
        # print('rewards', rewards.shape, rewards)
        advantages, std_grouped_rewards = compute_advantage(rewards)
        # print('advantages', advantages.shape, advantages)
        # print('std_grouped_rewards', std_grouped_rewards.shape, std_grouped_rewards)
        advantages = advantages[process_slice]

        patial_advantages = []
        
        if len(patial_rewards) > 0:
            for func_name, partial_reward in patial_rewards:
                partial_reward = compute_advantage(rewards)[0][process_slice]
                # evidence  : 27, 68, 27480 ... 522, 68 27480
                # think  : 13708, 766,  ... 522, 26865
                # if "evidence" in func_name:
                #     mask = generate_2d_mask(completion_ids, [27, 68, 27480], [522, 68, 27480])
                if "context" in func_name: # context
                    mask = generate_2d_mask(completion_ids, [34528], [522, 2147])
                else: # logical
                    mask = generate_2d_mask(completion_ids, [13708, 766], [522, 26865])
                patial_advantages.append({"name": func_name, "reward":partial_reward, "mask": mask})
                #print('partial_mask', completion_ids[mask])



        mode = "eval" if self.control.should_evaluate else "train"

        if mode == "train":
            self._total_train_tokens += self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]


        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)

        rewards_to_log = {}
        # reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            # Only calculate mean for samples where this reward function was applied (non-NaN values)
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
            rewards_to_log[reward_func_name] = rewards_per_func[:, i].tolist()
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            # prompts_to_log = gather_object([''.join(prompts_text)])
            prompts_to_log = gather_object(prompts_text)
            # print('prompts_to_log', len(prompts_to_log))
            completions_to_log = gather_object(completions_text)
            # print('completions_to_log', len(completions_to_log))
            #rewards_to_log = rewards.tolist()
            #print('advantages', advantages, 'patial_advantages', patial_advantages)
            advantages_to_log = gather_object(advantages)
            #advantages_to_log = advantages_to_log.tolist()
            #print('prompts_to_log', len(prompts_to_log))
            #print('completions_to_log', len(completions_to_log))
            #print('rewards_to_log', rewards_to_log)
            
            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        rewards_to_log,
                        advantages_to_log,
                        self.state.global_step,
                    )

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "multimodal_inputs": multimodal_inputs,
            "patial_advantages": patial_advantages
        }

    def compute_liger_loss_wrapper(self, unwrapped_model, inputs, advantages):
        # 1. 准备完整的序列输入（直接从 inputs 取，避免手动 cat 可能产生的错位）
        # 在 GRPO 中，inputs 通常已经包含了 concatenate 好的全量数据
        full_input_ids = inputs.get("input_ids")
        full_attention_mask = inputs.get("attention_mask")
        
        if full_input_ids is None:
            full_input_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
            full_attention_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)

        # 2. 安全调用：使用 unwrapped_model 而非 .model，确保所有 Hook 正常触发
        outputs = unwrapped_model(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            **inputs["multimodal_inputs"],
            return_dict=True,
            output_hidden_states=True, 
        )
        
        # 3. 精准切片：只保留 Completion 部分
        logits_to_keep = inputs["completion_ids"].size(1)
        
        # 注意：GRPO 通常需要 Shift 后的 logits 来对齐 labels
        # Liger 内部处理的是：input -> linear -> loss
        # 我们取最后 logits_to_keep 长度的 hidden_states
        last_hidden_state = outputs.hidden_states[-1][:, -(logits_to_keep + 1):-1, :].contiguous()

        del outputs
        gc.collect()
        torch.cuda.empty_cache()
        
        # 4. 数据检查（防止数值爆炸的最后一道防线）
        if torch.isnan(last_hidden_state).any() or torch.isinf(last_hidden_state).any():
            print("WARNING: last_hidden_state contains NaN or Inf!")

        # 5. 调用融合算子
        # 确保 advantages 的长度与 logits_to_keep 匹配
        loss, metrics = self.liger_grpo_loss(
            _input=last_hidden_state, # 确保内存连续
            lin_weight=unwrapped_model.lm_head.weight,
            selected_token_ids=inputs["completion_ids"],
            attention_mask=inputs["completion_mask"],
            advantages=advantages,
            bias=getattr(unwrapped_model.lm_head, "bias", None),
            old_per_token_logps=inputs["old_per_token_logps"],
            ref_per_token_logps=inputs.get("ref_per_token_logps"),
        )
        
        return loss, metrics

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        # Get the advantages from inputs
        advantages = inputs["advantages"]
        patial_advantages = inputs["patial_advantages"]

        for patial_advantage in patial_advantages:
            advantages = advantages + patial_advantage["reward"]*patial_advantage["mask"] 

        unwrapped_model = self.accelerator.unwrap_model(model)
    
        # 这里的关键是：Liger 内部会自己去调用 lm_head 映射并计算你代码里的 clip 和 kl 逻辑
        loss, metrics = self._forward_redirection(
            model, 
            unwrapped_model, 
            self.compute_liger_loss_wrapper, # 我们定义一个包装函数
            unwrapped_model, 
            inputs,
            advantages
        )

        # 记录指标 (Liger 会返回计算好的 KL 和 Clip Ratio)
        mode = "eval" if self.control.should_evaluate else "train"
        mean_kl, clip_ratio = metrics[0], metrics[-1]
        
        self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())

        mean_kl_gathered = self.accelerator.gather_for_metrics(mean_kl).mean().detach()
        total_loss_gathered = self.accelerator.gather_for_metrics(loss).mean().detach()
        kl_penalty_loss_val = self.beta * mean_kl_gathered
        adv_surrogate_loss_val = total_loss_gathered - kl_penalty_loss_val
        self._metrics[mode]["adv_loss"].append(adv_surrogate_loss_val.item())
        self._metrics[mode]["kl_penalty_loss"].append(kl_penalty_loss_val.item())

        # 5. 可选：在控制台打印
        if self.state.global_step % self.args.logging_steps == 0 and self.accelerator.is_main_process:
            print(f"\nStep {self.state.global_step} | Total Loss: {total_loss_gathered.item():.4f} | "
                f"Adv Loss: {adv_surrogate_loss_val.item():.4f} | KL Loss: {kl_penalty_loss_val.item():.4f}")

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "eval" if self.control.should_evaluate else "train"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics[mode].clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))

    def _get_train_sampler(self, dataset) -> Sampler:
        """Returns a sampler that ensures proper data sampling for GRPO training."""
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        
        print('effective_batch_size', effective_batch_size)
        return RepeatRandomSampler(
            data_source=self.data_type_to_indices,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations,
            seed=self.args.seed,
        )


    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        """Returns a sampler for evaluation."""
        return RepeatRandomSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )

    def get_train_dataloader(self):
        if self.custom_train_dataloader is not None:
            return self.custom_train_dataloader

        return super().get_train_dataloader()
