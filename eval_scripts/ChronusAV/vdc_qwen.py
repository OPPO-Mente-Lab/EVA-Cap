# Copyright (c) 2024 LMMs-Lab
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
#
# Adapted from https://github.com/wenhaochai/aurora
# Original lmms-eval source: https://github.com/EvolvingLMMs-Lab/lmms-eval
# Modifications copyright 2026 OPPO and Fudan University

import ast
import copy
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

QA_SYSTEM_PROMPT = (
    "You are a precise QA assistant. Your task is to answer open end questions based ONLY on the video caption provided. "
    "Do not use any outside knowledge or assumptions—your answer must strictly reflect information from the caption. "
    "Always output ONLY the direct answer to the question as a single, short sentence. "
    "If the answer involves speech, you must provide the specific ASR content from the caption. "
    "Do not repeat the question or provide any introductory phrases. "
    "If the caption does not provide enough information to answer the question, output \"N/A\" instead."
)


def _call_qwen(messages):
    url = f'http://{random.choice(QWEN_ADDRS)}:8000/v1/chat/completions'
    headers = {'Content-Type': 'application/json'}
    request_data = {
        "model": QWEN_MODEL,
        "messages": messages,
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


def generate(prompt):
    messages = [
        {"role": "system", "content": QA_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    return _call_qwen(messages)


def run(message):
    answer = _call_qwen(message)
    return answer.strip('.') if answer else answer


def generate_answer_score(question, answer, pred_answer):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                "------"
                "##INSTRUCTIONS: "
                "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                "- Consider synonyms or paraphrases as valid matches.\n"
                "- Evaluate the correctness of the prediction compared to the answer."
            ),
        },
        {
            "role": "user",
            "content": (
                "Please evaluate the following video-based question-answer pair:\n\n"
                f"Question: {question}\n"
                f"Correct Answer: {answer}\n"
                f"Predicted Answer: {pred_answer}\n\n"
                "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
            ),
        },
    ]
    return run(messages)


def worker(task):
    vid, video_duration, v2a_question, v2a_answer, v2a_resp, a2v_question, a2v_answer, a2v_resp = task
    try:
        response = ast.literal_eval(generate_answer_score(v2a_question, v2a_answer, v2a_resp))
        if "pred" in response and "score" in response:
            v2a_score = copy.deepcopy(response["score"])
            v2a_acc = 1 if response["pred"] == 'yes' else 0
        else:
            v2a_score, v2a_acc = 0, 0

        response = ast.literal_eval(generate_answer_score(a2v_question, a2v_answer, a2v_resp))
        if "pred" in response and "score" in response:
            a2v_score = copy.deepcopy(response["score"])
            a2v_acc = 1 if response["pred"] == 'yes' else 0
        else:
            a2v_score, a2v_acc = 0, 0

        return {
            "video_id": vid,
            "video_duration": video_duration,
            "v2a_question": v2a_question,
            "v2a_answer": v2a_answer,
            "v2a_resp": v2a_resp,
            "v2a_score": v2a_score,
            "v2a_acc": v2a_acc,
            "a2v_question": a2v_question,
            "a2v_answer": a2v_answer,
            "a2v_resp": a2v_resp,
            "a2v_score": a2v_score,
            "a2v_acc": a2v_acc,
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "video_id": vid,
            "video_duration": video_duration,
            "v2a_question": v2a_question,
            "v2a_answer": v2a_answer,
            "v2a_resp": v2a_resp,
            "v2a_score": 0,
            "v2a_acc": 0,
            "a2v_question": a2v_question,
            "a2v_answer": a2v_answer,
            "a2v_resp": a2v_resp,
            "a2v_score": 0,
            "a2v_acc": 0,
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


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def vdc_aggregate_score_v2a(results):
    scores = [float(r["v2a_score"]) if r["v2a_score"] else 0.0 for r in results]
    return sum(scores) / len(scores)


def vdc_aggregate_score_a2v(results):
    scores = [float(r["a2v_score"]) if r["a2v_score"] else 0.0 for r in results]
    return sum(scores) / len(scores)


def vdc_aggregate_acc_v2a(results):
    accs = [float(r["v2a_acc"]) if r["v2a_acc"] else 0.0 for r in results]
    return sum(accs) / len(accs)


def vdc_aggregate_acc_a2v(results):
    accs = [float(r["a2v_acc"]) if r["a2v_acc"] else 0.0 for r in results]
    return sum(accs) / len(accs)


def eval_chronus_caption_qas(file_path):
    base_name = os.path.basename(file_path).split(".")[0]
    data = read_jsonl(file_path)

    fout_path = f"{os.path.dirname(file_path)}/{base_name}_vdc_qwen35.jsonl"
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
        if vid in completed_keys:
            continue
        tasks.append((
            vid,
            video_info["video_duration"],
            video_info["v2a_question"],
            video_info["v2a_answer"],
            video_info["v2a_resp"],
            video_info["a2v_question"],
            video_info["a2v_answer"],
            video_info["a2v_resp"],
        ))

    run_multiprocess_tasks(tasks, num_processes=40, fout_path=fout_path)
    return fout_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score QA responses using Qwen vllm.")
    parser.add_argument("--merged_file", type=str, required=True, help="Path to the QA response jsonl file.")
    args = parser.parse_args()

    fout_path = eval_chronus_caption_qas(args.merged_file)
    out_data = read_jsonl(fout_path)

    print(f"Final V2A Accuracy: {vdc_aggregate_acc_v2a(out_data):.4f}")
    print(f"Final V2A Score:    {vdc_aggregate_score_v2a(out_data):.4f}")
    print(f"Final A2V Accuracy: {vdc_aggregate_acc_a2v(out_data):.4f}")
    print(f"Final A2V Score:    {vdc_aggregate_score_a2v(out_data):.4f}")
