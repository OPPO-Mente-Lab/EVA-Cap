
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


class VQADataset(Dataset):

    def __init__(self, data_path: str, sample_rate: float, script_args: ScriptArguments):
        super(VQADataset, self).__init__()
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
        video_path = example['video_path']
        conversations = example['conversations']
        #video_start = example.get('video_start', None)
        #video_end = example.get('video_end', None)

        if use_audio_in_video:
            msg =[{
                    "role": "system", "content": [
                        {
                            "type": "text", 
                            "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
                        }
                    ]
                }]
        else:
            msg =[{
                    "role": "system", "content": [
                        {
                            "type": "text", 
                            "text": "You are a helpful assistant."
                        }
                    ]
                }]

        for conv in conversations:
            if conv['from']=='human':
                if '<video>' in conv['value']:
                    if use_audio_in_video:
                        msg.append({
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "video",
                                            "video": video_path,
                                            #"video_start": video_start,
                                            #"video_end": video_end
                                            #"max_pixels": 61250
                                        },
                                        {
                                            "type": "text",
                                            "text": conv['value'].replace('<video>', '').strip()
                                        }
                                    ]
                                })
                    else:
                        msg.append({
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "video",
                                            "video": video_path,
                                            #"video_start": video_start,
                                            #"video_end": video_end
                                            #"max_pixels": 61250
                                        },
                                        {
                                            "type": "audio",
                                            "audio": video_path,
                                            #"video_start": video_start,
                                            #"video_end": video_end
                                        },
                                        {
                                            "type": "text",
                                            "text": conv['value'].replace('<video>', '').strip()
                                        }
                                    ]
                                })
                else:
                    msg.append({
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": conv['value'].strip()
                                    }
                                ]
                            })
            elif conv['from']=='gpt':
                msg.append({
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": conv['value'].strip()
                        }
                    ]
                })

        #print(msg)
        
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