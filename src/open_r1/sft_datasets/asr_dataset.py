
import logging
import os
import sys
import re

import torch
from torch.utils.data import Dataset
from trl import ScriptArguments
import yaml
import json
import math
import random
from PIL import Image
from qwen_omni_utils import process_mm_info


class ASRDataset(Dataset):

    PROMPTS = {
        'translate':[
            "请帮我把这句话翻译成中文",
        ],
        'transcribe':[
            "请输出这句话的内容",
        ]
    }

    def __init__(self, data_path: str, sample_rate: float, script_args: ScriptArguments):
        super(ASRDataset, self).__init__()
        self.script_args = script_args

        self.list_data_dict = []
        # 判断文件扩展名
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as json_file:
                self.list_data_dict = json.load(json_file)  # 读取为列表或字典
        elif data_path.endswith('.jsonl'):
            with open(data_path, 'r', encoding='utf-8') as jsonl_file:
                for line in jsonl_file:
                    if line.strip():  # 忽略空行
                        self.list_data_dict.append(json.loads(line))
        
        random.shuffle(self.list_data_dict)
        self.list_data_dict = self.list_data_dict[:int(len(self.list_data_dict)*sample_rate)]

        self.mel_size = 128
        self.frames_upbound = 16

    def __len__(self):
        return len(self.list_data_dict)

    def _make_conversation_image_and_video(self, example, use_audio_in_video=False):
        audio_path = example['audio_path']
        asr = example['asr_zh']
        task = example['type']
        text_prompt = random.choice(self.PROMPTS[task])

        msg =[{
                "role": "system", "content": [
                    {
                        "type": "text", 
                        "text": "You are a helpful assistant."
                    }
                ]
            },
            {
                "role": "user", "content": [
                    {
                        "type": "audio",
                        "audio": audio_path
                    },
                    {
                        "type": "text",
                        "text": text_prompt
                    }
                ]
            }]
        msg.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": asr
                }
            ]
        })

        print(msg)
        
        return msg

    def __getitem__(self, i):
        print('get_item', i)
        # Format into conversation
        num_base_retries = 10
        import traceback

        try:
            return self._get_item(i)
        except Exception as e:
            print(i)
            traceback.print_exc()

        for attempt_idx in range(num_base_retries):
            try:
                sample_idx = random.choice(range(len(self)))
                sample = self._get_item(sample_idx)
                return sample
            except Exception as e:
                # no need to sleep
                traceback.print_exc()
                print(f'[try other #{attempt_idx}] Failed to fetch sample {sample_idx}. Exception:', e)
                pass

    def _get_item(self, i):
        source = self.list_data_dict[i]
       
        messages  = self._make_conversation_image_and_video(source, use_audio_in_video=self.script_args.use_audio_in_video)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=self.script_args.use_audio_in_video)
        #print('messages', messages)
           
        if videos:
            print('videos', videos[0].shape)
        if audios:
            print('audios', audios[0].shape)
        return {
            'images': images,
            'audios': audios,
            'videos': videos,

            'messages': messages,
        #  
        }