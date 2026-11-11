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

# Modified by OPPO in 2026 to train EVA-Cap on audiovisual caption data.

"""
Supervised fine-tuning script for decoder language models.

Usage:

# One 1 node of 8 x H100s
accelerate launch --config_file=configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name HuggingFaceH4/Bespoke-Stratos-17k \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir data/Qwen2.5-1.5B-Open-R1-Distill
"""

import logging
import os
import sys
import re
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_SRC_ROOT, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import datasets
import torch
from torch.utils.data import Dataset
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed, AutoProcessor
from transformers import Qwen2_5OmniThinkerForConditionalGeneration
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from transformers.trainer_utils import get_last_checkpoint
from open_r1.configs import SFTConfig
from open_r1.utils.callbacks import get_callbacks
from open_r1.sft_datasets import CaptionDataset, VQADataset, create_concat_dataset, ASRDataset
import yaml
import json
import math
import random
from PIL import Image

from open_r1.trainer import SFTTrainer
from trl import (
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    # SFTTrainer
)
from dataclasses import field
from qwen_vl_utils import process_vision_info
from qwen_omni_utils import process_mm_info
import av

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
from dataclasses import dataclass

@dataclass
class SFTScriptArguments(ScriptArguments):
    image_root: str = field(default=None, metadata={"help": "The root directory of the image."})
    use_audio_in_video: bool = field(default=True)
    mask_non_assistant_labels: bool = field(default=False)

@dataclass
class SFTModelConfig(ModelConfig):
    freeze_vision_modules: bool = False

processor = None

DATASET_INFO = {
    'Caption': CaptionDataset,
    'VQA': VQADataset,
    'ASR': ASRDataset,
}

USE_AUDIO_IN_VIDEO = True

def load_json_dataset(train_data_path, script_args):
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
        dataset_list.append(DATASET_INFO[data_type](data_path, sample_rate, script_args, **kwargs))
        repeat_list.append(repeat_time)

    train_dataset, train_list_data_dict = create_concat_dataset(dataset_list, repeat_list)
    return train_dataset, train_list_data_dict

IGNORE_INDEX = -100
def mask_non_assistant_labels(input_ids, tokenizer):
    """
    将 <|im_start|>system ... <|im_end|> 和 <|im_start|>user ... <|im_end|> 段的 labels 置为 -100，
    仅保留 <|im_start|>assistant ... <|im_end|> 段用于计算 loss。

    参数:
        input_ids: torch.Tensor 或 list[int]，单个样本的 token ids
        tokenizer: HuggingFace tokenizer，需包含 <|im_start|> / <|im_end|>

    返回:
        labels: torch.Tensor，形状同 input_ids
    """

    im_end = tokenizer("<|im_end|>").input_ids[0]
    nl_tokens = tokenizer("\n").input_ids
    len_end = len([im_end] + nl_tokens)
    
    labels = []
    for ids in input_ids:
        text = tokenizer.decode(ids, skip_special_tokens=False)

        # 正则匹配所有 role 段落
        # 例如匹配：<|im_start|>system\n ... <|im_end|>\n
        pattern = r"<\|im_start\|\>(system|user|assistant)\n(.*?)<\|im_end\|\>\n"
        matches = list(re.finditer(pattern, text, flags=re.DOTALL))

        # 默认全部 -100
        label = torch.full((len(ids),), IGNORE_INDEX, dtype=torch.long)

        # 遍历匹配段，保留 assistant 部分的 token id
        for m in matches:
            role = m.group(1)
            content = m.group(2)
            if role != "assistant":
                continue  # mask 掉 system / user
            start_char, end_char = m.span(2)
            # 使用 offset 映射将字符范围转成 token 范围
            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                add_special_tokens=False
            )
            offsets = enc["offset_mapping"]

            tok_start, tok_end = None, None
            for i, (s, e) in enumerate(offsets):
                if tok_start is None and e > start_char:
                    tok_start = i
                if s < end_char:
                    tok_end = i + 1 + len_end
            if tok_start is not None and tok_end is not None:
                label[tok_start:tok_end] = ids[tok_start:tok_end].clone()

        labels.append(label)

    return torch.stack(labels)

def collate_fn(examples):
    # texts = [
    #     processor.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True)[0]
    #     for example in examples
    # ]
    # print(texts)

    # images = [item for sublist in examples  for item in sublist["images"]] if examples[0]['images'] is not None else None
    # audios = [item for sublist in examples  for item in sublist["audios"]] if examples[0]['audios'] is not None else None
    # videos = [item for sublist in examples  for item in sublist["videos"]] if examples[0]['videos'] is not None else None

    images, videos, audios, prompts = [], [], [], []
    for each in examples:
        prompts.append(each["prompt"])
        if each["images"] is not None:
            images.extend(each["images"])
        if each["audios"] is not None:
            audios.extend(each["audios"])
        if each["videos"] is not None:
            videos.extend(each["videos"])
    if len(images) == 0: images = None
    if len(audios) == 0: audios = None
    if len(videos) == 0: videos = None
    texts = processor.apply_chat_template(
                prompts,
                tokenize=False,
                add_generation_prompt=False,
            )
    # print('texts', texts)

    batch = processor(
        text=texts,
        images=images,
        audio=audios,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
        min_pixels=16*28*28,
    )
    # print(batch.keys())
    
    if labels_all:

        labels = mask_non_assistant_labels(batch["input_ids"], processor.tokenizer)
        labels[labels == processor.tokenizer.pad_token_id] = -100
    
    else:

        labels = batch["input_ids"].clone()

        # print(labels)
        labels[labels == processor.tokenizer.pad_token_id] = -100
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        video_token_id = processor.tokenizer.convert_tokens_to_ids(processor.video_token)
        audio_token_id = processor.tokenizer.convert_tokens_to_ids(processor.audio_token)
        labels[labels == image_token_id] = -100
        labels[labels == video_token_id] = -100
        labels[labels == audio_token_id] = -100

    # labels = mask_non_assistant_labels(batch["input_ids"], processor.tokenizer)
    # print('texts', [processor.tokenizer.decode(batch["input_ids"][0], skip_special_tokens=False)], 'texts_label', [processor.tokenizer.decode(batch["input_ids"][labels>0], skip_special_tokens=False)])

    batch["labels"] = labels
    batch["use_audio_in_video"] = USE_AUDIO_IN_VIDEO

    return batch


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    print(log_level, training_args)
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ################
    # Load datasets
    ################

    dataset, _ = load_json_dataset(script_args.dataset_name, script_args)

    ################
    # Load tokenizer
    ################
    global processor
    if "vl" in model_args.model_name_or_path.lower() or "omni" in model_args.model_name_or_path.lower():

        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
        )
        logger.info("Using AutoProcessor for vision-language model.")
    else:
        processor = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
        )
        logger.info("Using AutoTokenizer for text-only model.")
    if hasattr(processor, "pad_token") and processor.pad_token is None:
        processor.pad_token = processor.eos_token
    elif hasattr(processor.tokenizer, "pad_token") and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        # use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    # training_args.model_init_kwargs = model_kwargs

    if "Qwen2-VL" in model_args.model_name_or_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path, **model_kwargs
        )
    elif "Qwen2.5-VL" in model_args.model_name_or_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path, **model_kwargs
        )
    elif "qwen" in model_args.model_name_or_path.lower() and "omni" in model_args.model_name_or_path.lower():
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model_args.model_name_or_path, **model_kwargs)
        model.config.vocab_size = 152064

        vision_modules_keywords = ['visual','audio_tower']
    else:
        raise ValueError(f"Unsupported model: {model_args.model_name_or_path}")
    
    model.accepts_loss_kwargs = False # key to avoid passing loss kwargs, otherwise the gradient will be scaled

    if model_args.freeze_vision_modules:
        logger.info("Freezing vision modules...")
        for n, p in model.named_parameters():
            if any(keyword in n for keyword in vision_modules_keywords):
                p.requires_grad = False
    total_trainable_params = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            logger.info(f'train param: {name}')
            total_trainable_params += p.numel()
    logger.info(total_trainable_params)

    ############################
    # Initialize the SFT Trainer
    ############################
    training_args.dataset_kwargs = {
        "skip_prepare_dataset": True,
    }
    training_args.remove_unused_columns = False
    global labels_all
    labels_all=script_args.mask_non_assistant_labels
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        processing_class=processor.tokenizer,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    # metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    metrics["train_samples"] = len(dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(script_args.dataset_name),
        "dataset_tags": list(script_args.dataset_name),
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        # trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)
    #############
    # push to hub
    #############

    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)




if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, SFTModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    logger.info(script_args, training_args, model_args)
    main(script_args, training_args, model_args)
