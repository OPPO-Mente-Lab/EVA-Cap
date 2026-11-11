
import os
import json
import random
import re
from tqdm import tqdm
import torch
import itertools

import argparse
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration
from qwen_omni_utils import process_mm_info
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F

import av

from pathlib import Path

VIDEO_MAX_PIXELS = 401408  # 512*28*28
VIDEO_TOTAL_PIXELS = 20070400  # 512*28*28*50
USE_AUDIO_IN_VIDEO = True
os.environ['VIDEO_MAX_PIXELS'] = str(VIDEO_TOTAL_PIXELS)
video_dir = "bench/UGC-VideoCap/video" # TODO

class MyDataset(Dataset):
    def __init__(self, video_dir):
        super(MyDataset, self).__init__()
        self.data = os.listdir(video_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        video_path = os.path.join(video_dir, data)
        prompt = ("You are given a short video with both audio and visual content. Write a detailed and coherent paragraph that naturally integrates all modalities. "
        "Your description should include: (1) the primary scene and background setting; (2) key characters or objects and their actions or interactions; "
        "(3) significant audio cues such as voices, background music, sound effects, and their emotional tone; "
        "(4) any on-screen text (OCR) and its role in the video context; and (5) the overall theme or purpose of the video. "
        "Ensure the output is a fluent and objective paragraph, not a bullet-point list, and captures the video's content in a human-like, narrative style.")

        message = self._construct_message(video_path, prompt)
        audios, images, videos = process_mm_info(message, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        # inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True)
        if videos:
            print(video_path, 'videos', videos[0].shape)
        data_dict = {
            'images': images,
            'audios': audios,
            'videos': videos,
            'message': message,
            "video_id": data.replace(".mp4", "")
        }
        # import ipdb; ipdb.set_trace()
        return data_dict

    def _construct_message(self, file_path, prompt):

        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": file_path,
                        "max_pixels": VIDEO_MAX_PIXELS,
                    },
                    {
                        "type": "text",
                        "text": prompt
                    },
                ],
            },
        ]

        return conversation

        
def collate_fn(examples):
    
    return examples

class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        # shard_size = total_size // world_size
        # left = total_size % world_size
        # shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        # begin = sum(shard_sizes[:rank])
        # end = min(sum(shard_sizes[:rank + 1]), total_size)
        # return range(begin, end)
        indices = []
        for i in range(total_size):
            if i % world_size == rank:
                indices.append(i)
        return indices

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def main(args):
 
    model_path = args.model_path
    fout_path = args.save_path

    fout = open(fout_path, 'w', encoding='utf-8')

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="cuda",attn_implementation="flash_attention_2",)
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)

    dataset = MyDataset(video_dir)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=1,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    sources = []
    rets = []

    for inputs in tqdm(dataloader, desc="Processing batches"):

        images, videos, audios, prompts = [], [], [], []
        for each in inputs:
            prompts.append(each["message"])
            if each["images"] is not None:
                images.extend(each["images"])
            if each["audios"] is not None:
                audios.extend(each["audios"])
            if each["videos"] is not None:
                videos.extend(each["videos"])
        if len(images) == 0: images = None
        if len(audios) == 0: audios = None
        if len(videos) == 0: videos = None
        
        text = processor.apply_chat_template(
            prompts,
            tokenize=False,
            add_generation_prompt=True,
        )
        print('input', text)
        model_inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        model_inputs = model_inputs.to(model.device).to(model.dtype)
        with torch.inference_mode():
            text_ids = model.generate(
                **model_inputs, 
                use_audio_in_video=USE_AUDIO_IN_VIDEO, 
                do_sample=False,
                max_new_tokens=2048,  # Limit generation length
                eos_token_id=processor.tokenizer.eos_token_id, 
                pad_token_id=processor.tokenizer.pad_token_id,
                repetition_penalty=1.1
            )

        text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        model_generation = text.split("\nassistant\n")[-1]
        print('output', model_generation)

        rets.append(model_generation)
        sources.append({
            "video_id": inputs[0]['video_id']
        })

    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    merged_sources = [None for _ in range(world_size)]
    merged_responses = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_sources, sources)
    torch.distributed.all_gather_object(merged_responses, rets)

    merged_sources = [_ for _ in itertools.chain.from_iterable(merged_sources)]
    merged_responses = [
        _ for _ in itertools.chain.from_iterable(merged_responses)
    ]

    if torch.distributed.get_rank() == 0:
        for response, sample in zip(merged_responses, merged_sources):
            out_data = {
                "video_id": sample['video_id'],
                "caption": response, 
            }
            fout.write(json.dumps(out_data, ensure_ascii=False)+'\n')
            fout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model and save results.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the evaluation results.")
    args = parser.parse_args()

    main(args)
