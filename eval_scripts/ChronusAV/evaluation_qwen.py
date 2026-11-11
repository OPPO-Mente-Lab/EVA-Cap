
import os
import sys
import time
import json
import traceback
import multiprocessing
import random
import argparse

import requests
import tqdm

QWEN_ADDRS = os.environ.get("QWEN_ADDRS", "").split(",") if os.environ.get("QWEN_ADDRS") else [] # IP
QWEN_MODEL = os.environ.get("QWEN_MODEL", "Qwen3.5-27B") # model path

SYSTEM_PROMPT = (
    "You are a precise QA assistant. Your task is to answer open end questions based ONLY on the video caption provided. "
    "Do not use any outside knowledge or assumptions—your answer must strictly reflect information from the caption. "
    "Always output ONLY the direct answer to the question as a single, short sentence. "
    "If the answer involves speech, you must provide the specific ASR content from the caption. "
    "Do not repeat the question or provide any introductory phrases. "
    "If the caption does not provide enough information to answer the question, output \"N/A\" instead."
)


def generate(prompt):
    url = f'http://{random.choice(QWEN_ADDRS)}:8000/v1/chat/completions'
    headers = {'Content-Type': 'application/json'}
    request_data = {
        "model": QWEN_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 512,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    tries = 0
    answer = None
    while tries < 10:
        try:
            response = requests.post(url, json=request_data, headers=headers, timeout=300)
            answer = response.json()['choices'][0]['message']['content']
            print(answer)
            break
        except (KeyError, requests.exceptions.RequestException) as e:
            print(f"Error: {e}. Retrying in 10 seconds...")
            tries += 1
            time.sleep(10)
    return answer


def worker(task):
    vid, video_duration, v2a_question, v2a_answer, a2v_question, a2v_answer, caption = task
    v2a_prompt = f'''
Here is the video caption:
"{caption}"

Question: {v2a_question}'''
    a2v_prompt = f'''
Here is the video caption:
"{caption}"

Question: {a2v_question}'''
    try:
        v2a_resp = generate(v2a_prompt)
        if v2a_resp == 'N/A':
            v2a_resp = ''
        a2v_resp = generate(a2v_prompt)
        if a2v_resp == 'N/A':
            a2v_resp = ''
        return {
            "video_id": vid,
            "video_duration": video_duration,
            "v2a_question": v2a_question,
            "v2a_answer": v2a_answer,
            "v2a_resp": v2a_resp.strip(),
            "a2v_question": a2v_question,
            "a2v_answer": a2v_answer,
            "a2v_resp": a2v_resp.strip(),
            "caption": caption,
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "video_id": vid,
            "video_duration": video_duration,
            "v2a_question": v2a_question,
            "v2a_answer": v2a_answer,
            "v2a_resp": '',
            "a2v_question": a2v_question,
            "a2v_answer": a2v_answer,
            "a2v_resp": '',
            "caption": caption,
        }


def run_multiprocess_tasks(tasks, num_processes=None, fout_path=None):
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=num_processes) as pool, \
         open(fout_path, "a", encoding="utf-8") as fout:
        for item in tqdm.tqdm(
            pool.imap_unordered(worker, tasks),
            total=len(tasks),
        ):
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            fout.flush()


def eval_chronus_caption_qas(file_path, caption_keys=["omni_caption"]):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for caption_key in caption_keys:
        fout_path = f"{os.path.dirname(file_path)}/{caption_key}_result_qwen35.jsonl"
        completed_keys = set()
        if os.path.exists(fout_path):
            with open(fout_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        completed_keys.add(json.loads(line)['video_id'])

        tasks = []
        for video_info in data:
            vid = video_info["video_id"]
            if caption_key not in video_info or not video_info[caption_key]:
                print(f"[SKIP] {vid}: missing {caption_key}")
                continue
            if vid in completed_keys:
                continue
            tasks.append((
                vid,
                video_info["video_duration"],
                video_info["v2a_question"],
                video_info["v2a_answer"],
                video_info["a2v_question"],
                video_info["a2v_answer"],
                video_info[caption_key],
            ))

        run_multiprocess_tasks(tasks, num_processes=40, fout_path=fout_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate captions using Qwen.")
    parser.add_argument("--merged_file", type=str, required=True, help="Path to the merged caption file.")
    parser.add_argument(
        "--caption_keys",
        type=str,
        nargs='+',
        required=True,
        help="A list of caption keys to evaluate",
    )
    args = parser.parse_args()

    eval_chronus_caption_qas(args.merged_file, caption_keys=args.caption_keys)
