
import argparse
import json
from collections import defaultdict
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

def read_jsonl(file_path):
    """逐行读取 JSONL 文件并返回数据列表"""
    data = []
    with open(file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            if line.strip():  # 忽略空行
                data.append(json.loads(line))
    return data

def caption_metrics(data):
    eval_dict_all = {}
    for key in ['v2a', 'a2v']:
        pds = defaultdict(list)
        gts = defaultdict(list)
        for i, item in enumerate(data):
            video_id = item['video_id']
            if item['video_duration']<=300: #120
                pds[i].append({"video_id": video_id, "caption":item[f'{key}_resp']})
                gts[i].append({"video_id": video_id, "caption":item[f'{key}_answer']})

        tokenizer = PTBTokenizer() 
        pds = tokenizer.tokenize(pds)
        gts = tokenizer.tokenize(gts)
        scorers = [(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                (Meteor(),"METEOR"),
                (Rouge(), "ROUGE_L"),
                (Cider(), "CIDEr")]
        eval_dict = {}
        for scorer, method in scorers:
            score, scores = scorer.compute_score(gts, pds)
            if scorer.method() == "Bleu":
                eval_dict["BLEU4"] = score[3]
            else:
                eval_dict[scorer.method()] = score

        eval_dict_all[key] = eval_dict

    return eval_dict_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate metrics.")
    parser.add_argument("--result_file_path", type=str, required=True)
    args = parser.parse_args()
    data = read_jsonl(args.result_file_path)
    eval_dict_all = caption_metrics(data)
    for key, eval_dict in eval_dict_all.items():
        print(f"Metrics for {key}:")
        for k, v in eval_dict.items():
            print("%s: %.4f"%(k, v)) 