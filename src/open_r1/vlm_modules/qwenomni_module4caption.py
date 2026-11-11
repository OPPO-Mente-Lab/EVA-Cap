# Modified by OPPO in 2026 from HumanOmniV2 for EVA-Cap caption rewards.

import asyncio
import json
import os
import random
import time
from typing import Any, Union

import numpy as np
import torch
from openai import AsyncOpenAI
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
from trl.data_utils import maybe_apply_chat_template

from open_r1.vlm_modules.vlm_module import VLMBaseModule

DEFAULT_REWARD_MODEL = "Qwen3.5-27B"
DEFAULT_REWARD_PORT = 8000
# 可通过环境变量 REWARD_VLLM_CONFIG 覆盖；默认读项目根目录 reward_config/reward_vllm.json
REWARD_VLLM_CONFIG = os.environ.get(
    "REWARD_VLLM_CONFIG",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "reward_config", "reward_vllm.json")),
)

_reward_clients = None
_reward_clients_key = None  # (tuple(addrs), port)
_reward_model = DEFAULT_REWARD_MODEL


class RewardVLLMConfigError(RuntimeError):
    """reward vLLM 配置无效时抛出，训练应立即终止。"""


def load_reward_vllm_config():
    """从 JSON / 环境变量读取 addrs/model/port；addrs 为空时直接抛异常。"""
    config_path = REWARD_VLLM_CONFIG
    # 优先用环境变量作为回退
    addrs = [x.strip() for x in os.environ.get("REWARD_VLLM_ADDRS", "").split(",") if x.strip()]
    model = DEFAULT_REWARD_MODEL
    port = DEFAULT_REWARD_PORT

    if not os.path.isfile(config_path):
        print(f"[reward_vllm] config not found: {config_path}, using REWARD_VLLM_ADDRS env")
    else:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            file_addrs = cfg.get("addrs") or cfg.get("ADDRS") or []
            if isinstance(file_addrs, str):
                file_addrs = [x.strip() for x in file_addrs.split(",") if x.strip()]
            file_addrs = [str(x).strip() for x in file_addrs if str(x).strip()]
            if file_addrs:
                addrs = file_addrs
            if cfg.get("model"):
                model = cfg["model"]
            if cfg.get("port") is not None:
                port = int(cfg["port"])
        except Exception as e:
            raise RewardVLLMConfigError(
                f"无法读取 reward vLLM 配置 {config_path}: {e}"
            ) from e

    if not addrs:
        raise RewardVLLMConfigError(
            f"reward vLLM addrs 为空，无法计算 reward。\n"
            f"请在 {config_path} 中配置非空 addrs，\n"
            f"或设置环境变量 REWARD_VLLM_ADDRS=host1,host2。"
        )
    return addrs, model, port


def get_reward_clients_and_model():
    """按当前 JSON 配置返回 (clients, model)；地址变化时重建 AsyncOpenAI clients。"""
    global _reward_clients, _reward_clients_key, _reward_model
    addrs, model, port = load_reward_vllm_config()
    key = (tuple(addrs), port)
    if _reward_clients is None or _reward_clients_key != key:
        _reward_clients = [
            AsyncOpenAI(base_url=f"http://{addr}:{port}/v1", api_key="EMPTY") for addr in addrs
        ]
        _reward_clients_key = key
        print(f"[reward_vllm] refreshed clients: addrs={addrs}, port={port}, model={model}")
    _reward_model = model
    return _reward_clients, _reward_model


# def gpt_api(prompt):

#     addrs, model, port = load_reward_vllm_config()
#     qwen_addr = random.choice(addrs)
    
#     # vllm的API地址
#     url = f'http://{qwen_addr}:{port}/v1/chat/completions'
#     headers = {'Content-Type': 'application/json'}
    
#     request_data = {
#         "model": model,
#         "messages": [{"role": "user", "content": prompt}],
#         "temperature": 0,
#         "max_tokens": 8192,
#         # "top_k": 20, 
#         "chat_template_kwargs": {"enable_thinking": False}
#     }

#     response = requests.post(url, json=request_data, headers=headers, timeout=300)
#     # print(response.json())
#     response = response.json()
#     response_message = response['choices'][0]['message']['content']

#     return response_message


# # 兼容旧代码：初始仍按 ADDRS 建一份 clients；实际调用走 get_reward_clients_and_model()
# clients = [AsyncOpenAI(base_url=f"http://{addr}:8000/v1", api_key="EMPTY") for addr in ADDRS]

async def gpt_api_async(prompt, semaphore, max_tokens=512):
    max_retries = 5
    # 使用信号量控制总并发数，防止瞬间压垮 vLLM
    async with semaphore:
        for attempt in range(max_retries):
            clients_now, model_now = get_reward_clients_and_model()
            client = random.choice(clients_now)
            try:
                response = await client.chat.completions.create(
                    model=model_now,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=max_tokens,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}}
                )
                
                content = response.choices[0].message.content
                # print({'qwen_api': prompt+content}) # 生产环境建议根据需要开启
                # print('prompt:', prompt, 'response', content) # 生产环境建议根据需要开启
                return content
            
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    # 指数退避：每次重试多等一点时间，例如 1s, 2s, 4s...
                    wait_time = (2 ** attempt) + random.random()
                    await asyncio.sleep(wait_time)
                else:
                    # 5次都失败了，返回空
                    print(f"All {max_retries} attempts failed for prompt: {prompt[:50]}...")
                    return ''

async def process_batch(prompts, max_tokens=512):
    # 每次 batch 先刷新一次 JSON 配置
    get_reward_clients_and_model()
    # 假设你有 32 张卡部署 vLLM，并发可以设大一点，比如 64 或 128
    semaphore = asyncio.Semaphore(800) 
    tasks = [gpt_api_async(p, semaphore, max_tokens=max_tokens) for p in prompts]
    return await asyncio.gather(*tasks)

# from pycocoevalcap.meteor.meteor import Meteor

class QwenOmniModule(VLMBaseModule):

    # tokenizer = Qwen2_5OmniProcessor.from_pretrained('ckpt/Qwen2.5-Omni-7B').tokenizer
    # meteor = Meteor()

    def __init__(self):
        super().__init__()

    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        
        return Qwen2_5OmniThinkerForConditionalGeneration
    
    def post_model_init(self, model, processing_class):
        pass
    
    def get_processing_class(self):
        return Qwen2_5OmniProcessor
    
    def get_vision_modules_keywords(self):  
        return ['visual','audio_tower']
    
    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw', 'video_second_per_grid', 'feature_attention_mask', 'input_features', 'audio_feature_lengths', 'use_audio_in_video', 'rope_deltas']

    def get_non_generate_params(self):
        return []
    
    def get_custom_processing_keywords(self):
        return ['max_pixels', 'min_pixels']
    
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
        return prompts_text
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, audios, videos, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False, use_audio_in_video=False):

        # print(audios)
        prompt_inputs = processing_class(
            text=prompts_text,
            images=images,
            audio=audios,
            videos=videos,
            return_tensors=return_tensors,
            padding=padding,
            padding_side=padding_side,
            add_special_tokens=add_special_tokens,
            use_audio_in_video=use_audio_in_video,
            min_pixels=16*28*28,
        )
       
        return prompt_inputs
    
    @staticmethod
    def get_question_template(task_type: str):
        match task_type:
            case "rec":
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
            case _:
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."

    @staticmethod
    def format_penalty(content, solution):

        # def has_repeated_tail(text, repeat=3, min_len=20):
        #     text = text.strip()
        #     for i in range(min_len, len(text)//repeat):
        #         unit = text[-i:]
        #         if text.endswith(unit * repeat):
        #             return True
        #     return False

        # if has_repeated_tail(content):
        #     pred_ratio = 0.0
        # else:
        #     content_len = len(QwenOmniModule.tokenizer(content, add_special_tokens=False).input_ids)
        #     solution_len = len(QwenOmniModule.tokenizer(solution, add_special_tokens=False).input_ids)

        #     if content_len/solution_len < 0.5:
        #         pred_ratio = 0.0
        #     elif 0.5 <= content_len/solution_len < 0.7:
        #         pred_ratio = 0.5
        #     elif content_len/solution_len >= 0.7:
        #         pred_ratio = 1.0

        content_len = len(QwenOmniModule.tokenizer(content, add_special_tokens=False).input_ids)
        if content_len > 2000:
            pred_ratio = 0.0
        else:
            pred_ratio = 1.0

        return pred_ratio

    @staticmethod
    def event_recall_reward(completions, event_list, solution, dict_in_processing, **kwargs):

        SINGLE_EVENT_HIT_PROMPT = '''
You are a professional factual verification expert. Your task is to verify whether the [Target Event] is logically entailed and supported by the [Caption].

### Rules:
1. **Modality-Specific Verification**:
   - **If modal is "audio"**: Focus on what is SAID or HEARD. The exact spoken words (ASR) or specific sound descriptions must be present in the [Caption].
     - **Note**: If type is "asr", use fuzzy matching. Allow homophones (same sound, different spelling), minor ASR errors, or slight variations in phrasing (e.g., "I'm going" vs "I am going"). The core meaning and key nouns must match.
   - **If modal is "visual"**: Focus EXCLUSIVELY on physical ACTIONS, movements, and visual appearances. 
     - **Note**: If the [Target Event]'s content for a visual modal includes descriptions of speaking, tone of voice, or dialogue (e.g., "says while jumping", "jumps angrily"), **IGNORE the speaking/tone part** and only verify the physical action (e.g., "jumping").
2. **Fact-Checking Logic**: 
   - **"Yes"**: The [Target Event]'s content is explicitly stated by the [Caption]. Synonym matches are allowed (e.g., "running" vs "jogging").
   - **"No"**: The [Caption] lacks this specific information, or provides contradictory information.
3. **Strict Constraints**: 
   - Base your judgment *only* on the [Caption]. 
   - If the [Target Event] contains multiple details, all details must be supported for a "Yes".
4. **Output Format**: 
   - Return ONLY a JSON object: {"modal": "...", "type": "...", "content": "...", "verified": "Yes/No"}.
   - After the JSON, provide a brief reasoning paragraph explaining the decision.
   - Do not include any conversational filler.

### Caption:
[[input omni caption]]

### Target Event:
[[input event dict]]

### Output (JSON):'''

        def extract_json_dict(text):
            try:
                # 定位第一个</think>标签后的JSON数据
                think_end = text.find('</think>')
                if think_end >=0:
                    json_text = text[think_end + len('</think>'):].strip()
                else:
                    json_text = text.strip()
                
                # 提取JSON对象（从第一个[到最后一个]）
                start_idx = json_text.find('{')
                end_idx = json_text.rfind('}') + 1
                if start_idx == -1 or end_idx <= start_idx:
                    return ''
                json_str = json_text[start_idx:end_idx]
                data = json.loads(json_str)
                return data
                
            except (json.JSONDecodeError, ValueError, TypeError, KeyError):
                # 捕获JSON解析错误、类型错误或键错误等
                return ''

        contents = [completion[0]["content"] for completion in completions]

        event_hit_prompts = []
        event_hit_nums = []
        event_list_all = []
        for content, event_checklist in zip(contents, event_list):
            event_checklist_all = []
            for key in ['audio', 'visual']:
                event_checklist_all.extend(event_checklist[key])
            event_hit_nums.append(len(event_checklist_all))
            event_list_all.append(event_checklist_all)
            for event in event_checklist_all:
                target_event = {"modal": event["modal"], "type": event["type"], "content": event["content"]}
                event_hit_prompts.append(SINGLE_EVENT_HIT_PROMPT.replace('[[input omni caption]]', content).replace("[[input event dict]]", json.dumps(target_event, ensure_ascii=False)))

        start = time.perf_counter()
        hit_check_results = asyncio.run(process_batch(event_hit_prompts))
        end = time.perf_counter()
        elapsed = end - start
        print(f"event_recall_reward 打分耗时: {elapsed:.4f} 秒")

        reward_list = []
        point4hit_result = 0
        for event_checklist_all, event_hit_cnt, dict_item in zip(event_list_all, event_hit_nums, dict_in_processing):
            hit_check_results_now = hit_check_results[point4hit_result: point4hit_result+event_hit_cnt]
            point4hit_result += event_hit_cnt

            right_answer_cnt = 0
            right_answer_cnt_audio = 0
            right_answer_cnt_visual = 0
            cnt_audio = 0
            cnt_visual = 0
            hit_events_for_character_check = []
            for event, query_item in zip(event_checklist_all, hit_check_results_now):
                if event['modal']=='audio':
                    cnt_audio += 1
                elif event['modal']=='visual':
                    cnt_visual += 1
                output = extract_json_dict(query_item)
                if output=='':
                    print('empty event list, skip')
                    # event['verified'] = 'No'
                elif type(output) != dict:
                    print('not dict format, skip')
                    # event['verified'] = 'No'
                elif 'verified' not in output:
                    print('no verified, skip')
                    # event['verified'] = 'No'
                else:
                    # print('success.')
                    # event['verified'] = output['verified']
                    if output['verified']=='Yes':
                        right_answer_cnt += 1
                        if event['character']!='N/A':
                            hit_events_for_character_check.append(event)
                        if output['modal']=='audio':
                            right_answer_cnt_audio += 1
                        elif output['modal']=='visual':
                            right_answer_cnt_visual += 1
                    # elif output['verified']=='Partial':
                    #     right_answer_cnt += 0.5  # 给予中间分，激活梯度

                dict_item['hit_events'] = hit_events_for_character_check

            # print(len(event_checklist_all), cnt_audio, cnt_visual)
            print(right_answer_cnt/len(event_checklist_all), cnt_audio, right_answer_cnt_audio/cnt_audio, cnt_visual, right_answer_cnt_visual/cnt_visual)
            reward_list.append(right_answer_cnt/len(event_checklist_all))
            
            # dict_item['event_hit_result'] = event_checklist_all

        # reward_list_final = []
        # for content, sol, reward_item in zip(contents, solution, reward_list):
        #     pred_ratio = QwenOmniModule.format_penalty(content, sol)
        #     if pred_ratio == 0.0:
        #         print('format penalty triggered, content:', content, 'solution:', sol)
        #     reward_list_final.append(pred_ratio*reward_item)

        # return reward_list_final
        return reward_list


    @staticmethod
    def event_precision_reward(completions, event_list, solution, dict_in_processing, **kwargs):

        EVENT_EXTRACTION_PROMPT = '''
You are a professional video/audio analysis expert. Your task is to extract significant, time-ordered dynamic events from the provided [Caption] into a structured JSON format.

### Rules:
1. **Chronological Order**: Events must be listed in the exact order they occur in the caption.
2. **Audio Logic**:
    - Types: "asr" (spoken words), "sound" (physical sounds, ambient effects, etc.).
    - Character: The speaker's description (e.g., "Off-screen narrator"). Use "N/A" for general sounds.
    - Content (ASR): Verbatim transcript. STRICTLY EXCLUDE filler words (e.g., "uh", "um", "oh").
    - Content (Sound): A complete sentence describing the sound (e.g., "A dog barks loudly").
3. **Visual Logic (Max 20 entries)**:
    - Types: "action" (physical movement), "camera" (pan/zoom/cut/angle), "environment" (text appearance, background).
    - Character: The specific discriminative agent (a person or an animal) performing the action. Use "N/A" for camera or environment types.
    - Content: A single, complete standalone sentence describing the change (e.g., "The cat jumps onto the table").
4. Constraint: Do not treat "speaking" as a visual action; it belongs in the audio section.
5. **Output Format (JSON)**: Return ONLY a valid JSON object with the following structure. Do not include any conversational text or Markdown code blocks.
   {
     "audio": [{"modal": "audio", "type": "asr/sound", "character": "...", "content": "..."}, ...],
     "visual": [{"modal": "visual", "type": "action/camera/environment", "character": "...", "content": "..."}, ...]
   }

### Caption:
[[input omni caption]]

### Output (JSON):'''

        SINGLE_EVENT_HIT_PROMPT = '''
You are a professional factual verification expert. Your task is to verify whether the [Target Event] is logically entailed and supported by the [Caption].

### Rules:
1. **Modality-Specific Verification**:
   - **If modal is "audio"**: Focus on what is SAID or HEARD. The exact spoken words (ASR) or specific sound descriptions must be present in the [Caption].
     - **Note**: If type is "asr", use fuzzy matching. Allow homophones (same sound, different spelling), minor ASR errors, or slight variations in phrasing (e.g., "I'm going" vs "I am going"). The core meaning and key nouns must match.
   - **If modal is "visual"**: Focus EXCLUSIVELY on physical ACTIONS, movements, and visual appearances. 
     - **Note**: If the [Target Event]'s content for a visual modal includes descriptions of speaking, tone of voice, or dialogue (e.g., "says while jumping", "jumps angrily"), **IGNORE the speaking/tone part** and only verify the physical action (e.g., "jumping").
2. **Fact-Checking Logic**: 
   - **"Yes"**: The [Target Event]'s content is explicitly stated by the [Caption]. Synonym matches are allowed (e.g., "running" vs "jogging").
   - **"No"**: The [Caption] lacks this specific information, or provides contradictory information.
3. **Strict Constraints**: 
   - Base your judgment *only* on the [Caption]. 
   - If the [Target Event] contains multiple details, all details must be supported for a "Yes".
4. **Output Format**: 
   - Return ONLY a JSON object: {"modal": "...", "type": "...", "content": "...", "verified": "Yes/No"}.
   - After the JSON, provide a brief reasoning paragraph explaining the decision.
   - Do not include any conversational filler.

### Caption:
[[input omni caption]]

### Target Event:
[[input event dict]]

### Output (JSON):'''

        def extract_json_dict(text):
            try:
                think_end = text.find('</think>')
                if think_end >= 0:
                    json_text = text[think_end + len('</think>'):].strip()
                else:
                    json_text = text.strip()

                start_idx = json_text.find('{')
                end_idx = json_text.rfind('}') + 1
                if start_idx == -1 or end_idx <= start_idx:
                    return ''
                json_str = json_text[start_idx:end_idx]
                data = json.loads(json_str)
                return data

            except (json.JSONDecodeError, ValueError, TypeError, KeyError):
                return ''

        contents = [completion[0]["content"] for completion in completions]

        # Step 1 (parallel): 从每个 completion 中抽取预测 event list
        extract_prompts = [
            EVENT_EXTRACTION_PROMPT.replace('[[input omni caption]]', content)
            for content in contents
        ]
        start = time.perf_counter()
        extract_results = asyncio.run(process_batch(extract_prompts, max_tokens=8192))
        print(f"event_precision_reward 抽取耗时: {time.perf_counter() - start:.4f} 秒")

        pred_event_lists = []
        for extract_item in extract_results:
            pred_dict = extract_json_dict(extract_item)
            pred_events = []
            if isinstance(pred_dict, dict):
                for key in ['audio', 'visual']:
                    events = pred_dict.get(key, [])
                    if isinstance(events, list):
                        for event in events:
                            if isinstance(event, dict) and 'content' in event:
                                pred_events.append({
                                    "modal": event.get("modal", key),
                                    "type": event.get("type", ""),
                                    "content": event["content"],
                                })
            pred_event_lists.append(pred_events)

        # Step 2 (parallel): 每个预测 event 相对 GT caption (solution) 做 hit 判断
        event_hit_prompts = []
        event_hit_nums = []
        for pred_events, sol in zip(pred_event_lists, solution):
            event_hit_nums.append(len(pred_events))
            for event in pred_events:
                target_event = {
                    "modal": event["modal"],
                    "type": event["type"],
                    "content": event["content"],
                }
                event_hit_prompts.append(
                    SINGLE_EVENT_HIT_PROMPT
                    .replace('[[input omni caption]]', sol)
                    .replace("[[input event dict]]", json.dumps(target_event, ensure_ascii=False))
                )

        start = time.perf_counter()
        hit_check_results = asyncio.run(process_batch(event_hit_prompts)) if event_hit_prompts else []
        print(f"event_precision_reward hit打分耗时: {time.perf_counter() - start:.4f} 秒")

        reward_list = []
        point4hit_result = 0
        for pred_events, event_hit_cnt in zip(pred_event_lists, event_hit_nums):
            hit_check_results_now = hit_check_results[point4hit_result: point4hit_result + event_hit_cnt]
            point4hit_result += event_hit_cnt

            if event_hit_cnt == 0:
                reward_list.append(0.0)
                print('event_precision: empty pred events, score=0.0')
                continue

            right_answer_cnt = 0
            right_answer_cnt_audio = 0
            right_answer_cnt_visual = 0
            cnt_audio = 0
            cnt_visual = 0
            for event, query_item in zip(pred_events, hit_check_results_now):
                if event['modal'] == 'audio':
                    cnt_audio += 1
                elif event['modal'] == 'visual':
                    cnt_visual += 1
                output = extract_json_dict(query_item)
                if output == '' or type(output) != dict or 'verified' not in output:
                    continue
                if output['verified'] == 'Yes':
                    right_answer_cnt += 1
                    if event['modal'] == 'audio':
                        right_answer_cnt_audio += 1
                    elif event['modal'] == 'visual':
                        right_answer_cnt_visual += 1

            score = right_answer_cnt / event_hit_cnt
            print(
                score,
                cnt_audio,
                right_answer_cnt_audio / cnt_audio if cnt_audio > 0 else 0,
                cnt_visual,
                right_answer_cnt_visual / cnt_visual if cnt_visual > 0 else 0,
            )
            reward_list.append(score)

        return reward_list
        

    @staticmethod
    def a2v_query_reward(completions, event_list, a2v_query_list, solution, dict_in_processing, **kwargs):

        AUDIO_VISUAL_SYNC_JUDGE_PROMPT = '''
You are a precise Audio-Visual Synchronization Auditor. Your task is to determine which [Visual Candidates] occur at the same time as the [Audio Event], using the [Caption] as the sole source of truth.

### Rules:
1. **Step 1: Audio Identification (The Gatekeeper)**:
   - Scan the [Caption] for the [Audio Event].
   - **If Type is "asr"**: Use fuzzy matching. Allow homophones (same sound, different spelling), minor ASR errors, or slight variations in phrasing (e.g., "I'm going" vs "I am going"). The core meaning and key nouns must match.
   - **If Type is "sound"**: Use semantic matching. Focus on the nature of the sound (e.g., "A dog barks" matches "Canine barking" or "The puppy yelps").
2. **Step 2: Individual Candidate Audit**
   For each candidate in the [Visual Candidates] list, it is ONLY considered synchronous if it meets BOTH conditions:
   - **Condition A (Temporal Connection)**: The visual event and the audio event are explicitly linked by words like "as", "while", "simultaneously", or described as a concurrent gesture (e.g., "says ..., gesturing with hands").
   - **Condition B (No Cut Barrier)**: There must be NO scene-change or camera-cut markers (e.g., "cuts to", "the scene shifts to", "returns to") between the audio description and the visual event in the [Caption]. A cut indicates a new timestamp.
3. **Step 3: Verification**
   - Return all indices that pass both conditions.
   - If the [Audio Event] cannot be found, or no visual candidate passes both conditions, return "N/A".
4. **Output Format**:
   - Return ONLY a JSON object: {"selected_indices": a list of integers (e.g., [1, 3]) or "N/A"}.
   - After the JSON, provide a brief reasoning paragraph explaining the decision:
    - **Audio Identification**: Compare [Audio Event] quote vs. [Caption] quote.
    - **Sync Logic**: Explain why the indices were selected or why "N/A" was chosen.

### Caption:
[[input omni caption]]

### Audio Event:
[[input audio event content]]

### Visual Candidates (Numbered 1 to N):
[[input visual event list]]

### Output (JSON):'''

        def extract_json_dict(text):
            try:
                # 定位第一个</think>标签后的JSON数据
                think_end = text.find('</think>')
                if think_end >=0:
                    json_text = text[think_end + len('</think>'):].strip()
                else:
                    json_text = text.strip()
                
                # 提取JSON对象（从第一个[到最后一个]）
                start_idx = json_text.find('{')
                end_idx = json_text.rfind('}') + 1
                if start_idx == -1 or end_idx <= start_idx:
                    return ''
                json_str = json_text[start_idx:end_idx]
                data = json.loads(json_str)
                return data
                
            except (json.JSONDecodeError, ValueError, TypeError, KeyError):
                # 捕获JSON解析错误、类型错误或键错误等
                return ''
        
        def calculate_f1_batch(ref_list, cand_list):
            """
            批量计算所有题目（Audio Query）的 F1 分数。
            
            参数:
                ref_list: 参考答案列表，例如 [[10], [1, 2], [5]]
                cand_list: 候选答案列表，例如 [[10], [1], [5, 6]]
                
            返回:
                individual_f1s: numpy 数组，包含每条记录的 F1 分数
                avg_f1: 所有记录的平均 F1 分数 (float)
            """
            if len(ref_list) != len(cand_list):
                raise ValueError("参考列表(ref_list)和候选列表(cand_list)的长度必须一致")

            individual_f1s = []

            for gts_raw, preds_raw in zip(ref_list, cand_list):
                # 1. 预处理：统一转为 set 并过滤掉非整数（如 "N/A"）
                gts = {i for i in gts_raw if isinstance(i, int)}
                preds = {i for i in preds_raw if isinstance(i, int)}

                # 2. 特殊情况处理
                if not gts:
                    # 如果标准答案为空，模型也给空则 1.0，否则 0.0
                    f1 = 1.0 if not preds else 0.0
                elif not preds:
                    # 标准答案不为空，但模型给空
                    f1 = 0.0
                else:
                    # 3. 计算 TP, Precision, Recall
                    tp = len(gts.intersection(preds))
                    precision = tp / len(preds)
                    recall = tp / len(gts)

                    # 4. 计算 F1
                    if (precision + recall) > 0:
                        f1 = 2 * (precision * recall) / (precision + recall)
                    else:
                        f1 = 0.0
                
                individual_f1s.append(f1)

            # 转换为 numpy 数组方便后续统计
            scores_array = np.array(individual_f1s)
            avg_score = np.mean(scores_array)

            return scores_array, avg_score.item()

        contents = [completion[0]["content"] for completion in completions]

        event_sync_prompts = []
        event_sync_nums = []
        event_list_all = []
        for content, a2v_query_list_now, event_list_now in zip(contents, a2v_query_list, event_list):
            
            event_sync_nums.append(len(a2v_query_list_now))
            event_list_all.append(a2v_query_list_now)
            for event in a2v_query_list_now:
                event_list_visual = event['visual_candidates_for_judge']
                input_visual_event_list = "\n".join([
                    f"{i+1}. type: {visual_event['type']}, content: {visual_event['content']}" 
                        for i, visual_event in enumerate(event_list_visual)
                    ])
                target_event = f"type: {event['type']}, content: {event['content']}"
                event_sync_prompts.append(AUDIO_VISUAL_SYNC_JUDGE_PROMPT.replace('[[input omni caption]]', content).replace("[[input audio event content]]", target_event).replace("[[input visual event list]]", input_visual_event_list))

        start = time.perf_counter()
        sync_check_results = asyncio.run(process_batch(event_sync_prompts))
        end = time.perf_counter()
        elapsed = end - start
        print(f"a2v_align_reward 打分耗时: {elapsed:.4f} 秒")

        reward_list = []
        point4sync_result = 0
        for content, a2v_query_list_now, event_sync_cnt, dict_item in zip(contents, event_list_all, event_sync_nums, dict_in_processing):
            sync_check_results_now = sync_check_results[point4sync_result: point4sync_result+event_sync_cnt]
            point4sync_result += event_sync_cnt

            ref_list = [item['selected_visual_indices_new'] for item in a2v_query_list_now]
            cand_list = []

            for query_item in sync_check_results_now:
                output = extract_json_dict(query_item)
                # print('output', output, query_item)
                if output=='' or output['selected_indices']=='N/A' or output['selected_indices']==['N/A']:
                    cand_list.append([])
                else:
                    cand_list.append(output['selected_indices'])
            
            scores_array, avg_score = calculate_f1_batch(ref_list, cand_list)

            # print('ref_list', ref_list, 'cand_list', cand_list, 'scores_list', scores_array, 'avg_score', avg_score)
            if avg_score < 0.2 or avg_score > 0.8:
                for a2v_item, ref_item, cand_item, score_item in zip(a2v_query_list_now, ref_list, cand_list, scores_array):
                    print('audio_query', a2v_item['content'], 'ref_item', ref_item, 'cand_item', cand_item, 'score_item', score_item, avg_score, 'content', content)
            reward_list.append(avg_score)
            
            # dict_item['event_hit_result'] = event_checklist_all

        # reward_list_final = []
        # for content, sol, reward_item in zip(contents, solution, reward_list):
        #     pred_ratio = QwenOmniModule.format_penalty(content, sol)
        #     if pred_ratio == 0.0:
        #         print('format penalty triggered, content:', content, 'solution:', sol)
        #     reward_list_final.append(pred_ratio*reward_item)

        # return reward_list_final
        return reward_list

    @staticmethod
    def event_attribution_reward(completions, event_list, solution, dict_in_processing, **kwargs):

        SINGLE_EVENT_ATTRIBUTION_PROMPT = '''
You are a factual consistency auditor. Your goal is to verify if the [Target Event] is accurately attributed to the correct [Character] in the [Caption] through a multi-step logical deduction.

### Rules:
1. **Step 1: Event Localization (Content Match)**
   - Scan the [Caption] to locate the specific event or speech that matches the [Content].
   - **Guidance**: This event is confirmed to exist. Identify the specific segment where it occurs.
   - **Modality-Specific**: 
     - For "audio": Focus on what is SAID or HEARD. Use fuzzy matching for "asr" type.
     - For "visual": Focus EXCLUSIVELY on physical ACTIONS, movements, and visual appearances. Ignore any mentioned tone or descriptions of speaking.
2. **Step 2: Global Character Profiling (Character Extraction)**
   - **Trace**: Identify the immediate subject (performer) of the event found in Step 1.
   - **Aggregate**: Scan the entire [Caption] to collect all descriptors (role, attire, gender) linked to this character via anaphora (e.g., "he", "she", "the man").
   - **Note**: For "asr" type, if the character is an "Off-screen narrator," focus on vocal descriptors to build the profile.
   - **Strict Rule**: If the [Caption] provides no identifying features for this character, report it as "Generic/Missing Profile". Do NOT hallucinate.
3. **Step 3: Entailment Judgment**
   - Compare the "Character Profile" (from Step 2) with the [Character] phrase from the Target Event.
     - **Yes**: The profile is logically consistent with and supports the [Character] description.
     - **No**: There is a conflict (wrong gender, wrong clothing) or the description is so vague (e.g., "someone", "a person") that it fails to uniquely identify the [Character].
4. **Output Format**: 
   - Return ONLY a JSON object: {"modal": "...", "type": "...", "character": "...", "content": "...", "verified": "Yes/No"}.
   - After the JSON, provide a brief reasoning paragraph explaining the decision:
     - **Character Profile**: List descriptions of this character found in caption.
     - **Entailment Logic**: Explain why the profile matches or fails to match the target character.

### Caption:
[[input omni caption]]

### Target Event:
[[input event dict]]

### Output (JSON):'''

        def extract_json_dict(text):
            try:
                # 定位第一个</think>标签后的JSON数据
                think_end = text.find('</think>')
                if think_end >=0:
                    json_text = text[think_end + len('</think>'):].strip()
                else:
                    json_text = text.strip()
                
                # 提取JSON对象（从第一个[到最后一个]）
                start_idx = json_text.find('{')
                end_idx = json_text.rfind('}') + 1
                if start_idx == -1 or end_idx <= start_idx:
                    return ''
                json_str = json_text[start_idx:end_idx]
                data = json.loads(json_str)
                return data
                
            except (json.JSONDecodeError, ValueError, TypeError, KeyError):
                # 捕获JSON解析错误、类型错误或键错误等
                return ''

        contents = [completion[0]["content"] for completion in completions]

        event_cnt_with_character = []
        for event_checklist in event_list:
            event_checklist_all = []
            for key in ['audio', 'visual']:
                event_checklist_all.extend(event_checklist[key])
            event_cnt_with_character.append(len([event for event in event_checklist_all if event['character']!='N/A']))

        event_hit_prompts = []
        event_hit_nums = []
        event_list_all = []
        
        for content, dict_item in zip(contents, dict_in_processing):
            event_checklist_all = dict_item['hit_events']
            event_hit_nums.append(len(event_checklist_all))
            event_list_all.append(event_checklist_all)
            for event in event_checklist_all:
                target_event = {"modal": event["modal"], "type": event["type"], "character": event["character"], "content": event["content"]}
                event_hit_prompts.append(SINGLE_EVENT_ATTRIBUTION_PROMPT.replace('[[input omni caption]]', content).replace("[[input event dict]]", json.dumps(target_event, ensure_ascii=False)))

        start = time.perf_counter()
        hit_check_results = asyncio.run(process_batch(event_hit_prompts))
        end = time.perf_counter()
        elapsed = end - start
        print(f"event_attribution_reward 打分耗时: {elapsed:.4f} 秒")

        reward_list = []
        point4hit_result = 0
        for event_checklist_all, event_hit_cnt, event_cnt_check in zip(event_list_all, event_hit_nums, event_cnt_with_character):
            hit_check_results_now = hit_check_results[point4hit_result: point4hit_result+event_hit_cnt]
            point4hit_result += event_hit_cnt

            right_answer_cnt = 0
            right_answer_cnt_audio = 0
            right_answer_cnt_visual = 0
            cnt_audio = 0
            cnt_visual = 0
            for event, query_item in zip(event_checklist_all, hit_check_results_now):
                if event['modal']=='audio':
                    cnt_audio += 1
                elif event['modal']=='visual':
                    cnt_visual += 1
                output = extract_json_dict(query_item)
                if output=='':
                    print('empty event list, skip')
                    # event['verified'] = 'No'
                elif type(output) != dict:
                    print('not dict format, skip')
                    # event['verified'] = 'No'
                elif 'verified' not in output:
                    print('no verified, skip')
                    # event['verified'] = 'No'
                else:
                    # print('success.')
                    # event['verified'] = output['verified']
                    if output['verified']=='Yes':
                        right_answer_cnt += 1
                        if output['modal']=='audio':
                            right_answer_cnt_audio += 1
                        elif output['modal']=='visual':
                            right_answer_cnt_visual += 1
                    # elif output['verified']=='Partial':
                    #     right_answer_cnt += 0.5  # 给予中间分，激活梯度

            # print(len(event_checklist_all), cnt_audio, cnt_visual)
            if event_cnt_check>0:
                print(right_answer_cnt/event_cnt_check, cnt_audio, right_answer_cnt_audio/cnt_audio if cnt_audio > 0 else 0, cnt_visual, right_answer_cnt_visual/cnt_visual if cnt_visual > 0 else 0)
                reward_list.append(right_answer_cnt/event_cnt_check)
            else:
                reward_list.append(0.0)
            
            # dict_item['event_hit_result'] = event_checklist_all

        # reward_list_final = []
        # for content, sol, reward_item in zip(contents, solution, reward_list):
        #     pred_ratio = QwenOmniModule.format_penalty(content, sol)
        #     if pred_ratio == 0.0:
        #         print('format penalty triggered, content:', content, 'solution:', sol)
        #     reward_list_final.append(pred_ratio*reward_item)

        # return reward_list_final
        return reward_list

