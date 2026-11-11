
import json
import os
import random
import traceback

import torch
from qwen_omni_utils import process_mm_info
from torch.utils.data import Dataset
from trl import ScriptArguments

VIDEO_MIN_PIXELS = 64 * 28 * 28
VIDEO_MAX_PIXELS = 512 * 28 * 28
TOTAL_PIXELS = 30 * 512 * 28 * 28

class CaptionDataset(Dataset):

    PROMPTS = [
        "Provide a comprehensive description of all the content in the video, leaving out no details. Be sure to include as much of the audio information as possible, and ensure that your descriptions of the audio and video are closely aligned.",
        "Thoroughly describe everything in the video, capturing every detail. Include as much information from the audio as possible, and ensure that the descriptions of both audio and video are well-coordinated.",
        "Please describe all the information in the video without sparing every detail in it. As you describe, you should also describe as much of the information in the audio as possible, and pay attention to the synchronization between the audio and video descriptions.",
        "Offer a detailed description of the video, making sure to include every detail. Also, incorporate as much information from the audio as you can, and ensure that your descriptions of the audio and video are in sync.",
        "Describe every aspect of the video in full detail, covering all the information it contains. Additionally, include as much of the audio content as you can, and make sure your descriptions of the audio and video are synchronized.",
        "Please provide a thorough description of all the content in the video, including every detail. As you describe, ensure that you also cover as much information from the audio as possible, and be mindful of the synchronization between the audio and video as you do so.",
        "Give a detailed account of everything in the video, capturing all the specifics. While doing so, also include as much information from the audio as possible, ensuring that the descriptions of audio and video are well-synchronized."
    ]

    def __init__(self, data_path: str, sample_rate: float, script_args: ScriptArguments, seed=42, rl_mode: bool = False, video_root: str = ""):
        super(CaptionDataset, self).__init__()
        self.script_args = script_args
        self.rl_mode = rl_mode
        self.video_root = video_root

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
        
        random.seed(seed)
        random.shuffle(self.list_data_dict)
        self.list_data_dict = self.list_data_dict[:int(len(self.list_data_dict)*sample_rate)]

        self.mel_size = 128
        self.frames_upbound = 16

    def __len__(self):
        return len(self.list_data_dict)

    def get_data_dict(self):
        return self.list_data_dict

    def _make_conversation_image_and_video(self, prompt_id, example, use_audio_in_video=True):
        video_path = example['video_path']
        if self.video_root and not os.path.isabs(video_path):
            video_path = os.path.join(self.video_root, video_path)
        caption = example['caption']
        # text_prompt = random.choice(self.PROMPTS)
        text_prompt = self.PROMPTS[prompt_id%len(self.PROMPTS)]

        if use_audio_in_video:
            msg =[{
                    "role": "system", "content": [
                        {
                            "type": "text", 
                            "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
                        }
                    ]
                },
                {
                    "role": "user", "content": [
                        {
                            "type": "video",
                            "video": video_path,
                            "min_pixels": VIDEO_MIN_PIXELS,
                            "max_pixels": VIDEO_MAX_PIXELS,
                            "total_pixels": TOTAL_PIXELS
                        },
                        {
                            "type": "text",
                            "text": text_prompt
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
                },
                {
                    "role": "user", "content": [
                        {
                            "type": "video",
                            "video": video_path,
                            "min_pixels": VIDEO_MIN_PIXELS,
                            "max_pixels": VIDEO_MAX_PIXELS,
                            "total_pixels": TOTAL_PIXELS
                        },
                        {
                            "type": "audio",
                            "audio": video_path
                        },
                        {
                            "type": "text",
                            "text": text_prompt
                        }
                    ]
                }]
        if not self.rl_mode:
            msg.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": caption
                    }
                ]
            })

        #print(msg)
        
        return msg, caption

    def __getitem__(self, i):
        # print('get_item', i)
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

        event_list = source.get('event_list', None)
        sync_event_list = source.get('sync_event_list', None)
        a2v_query_list = source.get('a2v_query_list', None)
        messages, caption  = self._make_conversation_image_and_video(i, source, use_audio_in_video=self.script_args.use_audio_in_video)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=self.script_args.use_audio_in_video)
        # print('messages', messages, 'solution', caption, 'visual_list', event_list['visual'])
           
        # if videos:
        #     print('videos', videos[0].shape)
        # if audios:
        #     print('audios', audios[0].shape)
        return {
            'images': images,
            'audios': audios,
            'videos': videos,

            # 'messages': messages,
            'prompt': messages,
            'solution': caption,
            'event_list': event_list,
            'sync_event_list': sync_event_list,
            'a2v_query_list': a2v_query_list,
            # 'gt_event_list': gt_event_list,
            'raw_data': source,
            'use_audio_in_video': self.script_args.use_audio_in_video,
        }