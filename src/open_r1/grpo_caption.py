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

# Modified by OPPO in 2026 to implement EVA-Cap caption reward training.

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass
import logging
import os
import sys
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import pathlib
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_SRC_ROOT, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from torch.utils.data.dataloader import DataLoader
from transformers import Qwen2VLForConditionalGeneration

from open_r1.trainer import GRPOTrainer_Caption, GRPOConfig, DistributedGroupRepeatSampler
from open_r1.vlm_modules import *
from open_r1.sft_datasets import CaptionDataset, VQADataset, create_concat_dataset, ASRDataset
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers import TrainingArguments
import yaml
import json
import random
import math

#import whisper
import librosa
from decord import VideoReader, cpu, AudioReader
import numpy as np

# ----------------------- Fix the flash attention bug in the current version of transformers -----------------------
#from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
import torch
from typing import Tuple
import copy
from qwen_omni_utils import process_mm_info
import av
from collections import defaultdict

# from liger_kernel.transformers import apply_liger_kernel_to_llama

# # 在 Trainer 初始化前调用，它会把模型的 Linear 层、LayerNorm 换成高效 Triton 算子
# apply_liger_kernel_to_llama()

def check_if_video_has_audio(video_path):
    try:
        container = av.open(video_path)
        audio_streams = [stream for stream in container.streams if stream.type == "audio"]
        if not audio_streams:
            return False
        return True
    except:
        return False

logger = logging.getLogger(__name__)

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["event_recall", "a2v_query", "event_attribution"],
        metadata={"help": "List of reward functions: event_recall, a2v_query, event_attribution"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image (for QwenVL)"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image (for QwenVL)"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )
    use_audio_in_video: Optional[bool] = field(
        default=True,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False

processor = None

DATASET_INFO = {
    'Caption': CaptionDataset,
    'VQA': VQADataset,
    'ASR': ASRDataset,
}

USE_AUDIO_IN_VIDEO = True

def load_json_dataset(train_data_path, script_args, seed):
    dataset_list = []
    repeat_list = []
    ds_collections = json.loads(open(train_data_path).read())
    for ds_idx, ds_name in enumerate(ds_collections.keys()):
        data_path = ds_collections[ds_name]["annotation"]
        repeat_time = ds_collections[ds_name]['repeat_time']
        data_type = ds_collections[ds_name]["type"]
        sample_rate = ds_collections[ds_name]["sample_rate"]
        video_root = ds_collections[ds_name].get("video_root", "")
        kwargs = {"video_root": video_root} if data_type == "Caption" else {}
        dataset_list.append(DATASET_INFO[data_type](data_path, sample_rate, script_args, seed, rl_mode=True, **kwargs))
        repeat_list.append(repeat_time)

    train_dataset, train_list_data_dict = create_concat_dataset(dataset_list, repeat_list)
    return train_dataset, train_list_data_dict

def build_type_to_indices(dataset):
    data_type_to_indices = defaultdict(list)
    for idx in range(len(dataset)):
        sample = dataset[idx]  # 这里只做一次，可以接受
        sample_type = sample.get("data_type", "default")  # 若无 type，默认归为 "default"
        data_type_to_indices[sample_type].append(idx)
    return data_type_to_indices

def get_vlm_module(model_name_or_path):
    if "qwen" in model_name_or_path.lower() and "omni" in model_name_or_path.lower():
        return QwenOmniModule
    elif "internvl" in model_name_or_path.lower():
        return InvernVLModule
    elif "ola" in model_name_or_path.lower():
        return QwenOlaModule
    elif "qwen" in model_name_or_path.lower() and "vl" in model_name_or_path.lower():
        return Qwen2VLModule
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")

def main(script_args, training_args, model_args):
    # Load the VLM module
    vlm_module_cls = get_vlm_module('qwen2.5omni')
    print("using vlm module:", vlm_module_cls.__name__)

    # Load the reward functions
    reward_funcs_registry = {
        "event_recall": vlm_module_cls.event_recall_reward,
        "a2v_query": vlm_module_cls.a2v_query_reward,
        "event_attribution": vlm_module_cls.event_attribution_reward,
    }
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print(script_args.reward_funcs)
    
    print("reward_funcs:", reward_funcs)
    # import ipdb;ipdb.set_trace()

    # Load the dataset
    dataset, list_data_dict = load_json_dataset(script_args.dataset_name, script_args, seed=training_args.seed)
    data_type_to_indices = build_type_to_indices(list_data_dict)
    print('data_type_to_indices', data_type_to_indices.keys())
    for data_type in data_type_to_indices:
        print(f"data_type: {data_type}, num_samples: {len(data_type_to_indices[data_type])}")

    sampler = DistributedGroupRepeatSampler(
        data_source_dict=data_type_to_indices,
        batch_size=training_args.per_device_train_batch_size,
        mini_repeat=training_args.num_generations//training_args.num_generations_per_device,
        repeat_count=training_args.num_train_epochs,
        shuffle=True,
        seed=training_args.seed,
        distributed=True,
        num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank(),
    )
    data_loader = DataLoader(
        dataset,
        batch_size=training_args.per_device_train_batch_size,
        sampler=sampler,
        collate_fn=lambda x: x,
        num_workers=training_args.dataloader_num_workers,
        pin_memory=training_args.dataloader_pin_memory,
        persistent_workers=training_args.dataloader_persistent_workers,
    )
    
    # Initialize the GRPO trainer
    print('training_args', training_args)
    trainer = GRPOTrainer_Caption(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=dataset,
        train_dataloader=data_loader,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        max_anyres_num=script_args.max_anyres_num,
        torch_dtype=model_args.torch_dtype,
    )

    # Train and push the model to the Hub
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    # if training_args.push_to_hub:
    #     trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
