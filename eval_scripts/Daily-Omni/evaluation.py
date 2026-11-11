
import os
import sys
import time
import json
import traceback
import multiprocessing
import argparse
import hashlib
import hmac
from collections import OrderedDict

import requests
import tqdm

GEMINI_AK = os.environ.get("GEMINI_AK", "")
GEMINI_SK = os.environ.get("GEMINI_SK", "")
GEMINI_URL = os.environ.get("GEMINI_URL", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")

SYSTEM_PROMPT = (
    "You are a precise QA assistant. Your task is to answer multiple-choice questions based ONLY on the video caption provided. "
    "Do not use any outside knowledge or assumptions—your answer must strictly reflect information from the caption. "
    "Always output only the capital letter corresponding to your choice (e.g., A, B, C, D). "
    "If the caption does not provide enough information to answer the question, output \"N/A\" instead."
)


def sign(params, body, app_id, secret_key):
    auth_string_prefix = f"bot-auth-v1/{app_id}/{int(time.time() * 1000)}/"
    sb = [auth_string_prefix]
    if params:
        ordered_params = OrderedDict(sorted(params.items()))
        sb.extend([f"{k}={v}&" for k, v in ordered_params.items()])
    sign_str = "".join(sb) + body
    signature = hmac.new(
        secret_key.encode("utf-8"), sign_str.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return auth_string_prefix + signature

def generate(prompt):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    tries = 0
    while tries < 10:
        try:
            body = {
                "model": GEMINI_MODEL,
                "messages": messages
            }
            data = json.dumps(body)
            headers = {
                "recordId": hashlib.md5(data.encode("utf-8")).hexdigest(),
                "Authorization": sign(None, data, GEMINI_AK, GEMINI_SK),
                "Content-Type": "application/json",
            }
            resp = requests.post(GEMINI_URL, headers=headers, data=data, timeout=300)
            resp.raise_for_status()

            answer = json.loads(resp.text)["data"]["choices"][0]["message"]["content"]
            break  # 成功后退出循环
        except (KeyError, json.JSONDecodeError, requests.exceptions.RequestException) as e:
            resp_text = getattr(resp, "text", "")
            if resp_text:
                print(resp_text)
            print(f"Error: {e}. Retrying in 10 seconds...")
            tries += 1
            time.sleep(10)  # 等待10秒后继续
        except requests.exceptions.ConnectionError as e:
            print(f"连接错误: {e}, 尝试重新连接...")
            tries += 1
            time.sleep(30)  # 延迟 30 秒后重新尝试
    return answer

def worker(task):
    vid, video_duration, question, choices, answer, caption_key, answer_key, caption = task
    choices_text = "\n".join([f"{c}" for c in choices])
    prompt_filled = f'''
Here is the video caption:
"{caption}"

Question: {question}
Choices:
    {choices_text}'''
    try:
        # resp, _ = generate(prompt_filled)
        resp = generate(prompt_filled)
        return {
            "video_id": vid,
            "video_duration": video_duration,
            "question": question,
            "choices": choices,
            "answer": answer,
            caption_key: caption,
            answer_key: resp
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "video_id": vid,
            "video_duration": video_duration,
            "question": question,
            "choices": choices,
            "answer": answer,
            caption_key: caption,
            answer_key: None
        }

def run_multiprocess_tasks(tasks, num_processes=None, fout_path=None):
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=num_processes) as pool, \
         open(fout_path, "a", encoding="utf-8") as fout:
        # results = pool.map(worker, tasks)
        for item in tqdm.tqdm(
            pool.imap_unordered(worker, tasks),
            total=len(tasks)
        ):
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            fout.flush()

    # if fout_path:
    #     with open(fout_path, "w", encoding='utf-8') as f:
    #         for item in results:
    #             f.write(json.dumps(item, ensure_ascii=False) + '\n')
    #             f.flush()
    # return results

def eval_dailyomni_caption_qas(file_path, caption_keys=["omni_caption"]):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # all_results = []
    for caption_key in caption_keys:
        answer_key = caption_key.replace("_caption", "_resp")
        fout_path = f"{os.path.dirname(file_path)}/{caption_key}_result.jsonl"
        completed_keys = set()
        if os.path.exists(fout_path):
            with open(fout_path, "r", encoding="utf-8") as f:
                for line in f: 
                    line = line.strip() 
                    if line: 
                        completed_keys.add((json.loads(line)['video_id'], json.loads(line)['question']))
        
        tasks = [] 
        for video_info in data: 
            vid = video_info["video_id"] 
            video_duration = video_info["video_duration"]
            caption = video_info[caption_key]
            for q in video_info["questions"]:
                if (vid, q["Question"]) not in completed_keys:
                    task_item = (
                        vid,
                        video_duration,
                        q["Question"],
                        q["Choice"],
                        q["Answer"],
                        caption_key,
                        answer_key,
                        caption
                    )
                    tasks.append(task_item)

        run_multiprocess_tasks(tasks, num_processes=5, fout_path=fout_path)
        # results = run_multiprocess_tasks(tasks, num_processes=5, fout_path=fout_path)
        # all_results.extend(results)

    # return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate captions using Gemini.")
    parser.add_argument("--merged_file", type=str, required=True, help="Path to the merged caption file.")
    parser.add_argument(
        "--caption_keys", 
        type=str, 
        nargs='+',
        required=True, 
        help="A list of caption keys to evaluate"
    )
    args = parser.parse_args()

    eval_dailyomni_caption_qas(args.merged_file, caption_keys=args.caption_keys)