# EVA-Cap: Optimizing Audiovisual Video Captioning via Event-Centric Alignment

Jiapeng Shi*, Qiuxia Hou*, [Jiaqi Leng](https://jiaqileng2002.github.io/), [Junke Wang](https://wdrink.github.io/), [Yanhao Zhang](https://scholar.google.com/citations?user=J1OguXAAAAAJ), [Haonan Lu](https://scholar.google.com/citations?user=EPBgKu0AAAAJ), [Ziyi Ye✉](https://ziyiye.cn/), [Zuxuan Wu✉](https://zxwu.azurewebsites.net/)

[🎉 NeurIPS 2026](https://neurips.cc/virtual/2026/loc/sydney/poster/155915) | [Paper](https://arxiv.org/pdf/2610.22186) | [Code](https://github.com/OPPO-Mente-Lab/EVA-Cap) | [Model](https://huggingface.co/OPPOer/EVA-Cap-7B) | [Dataset](https://huggingface.co/datasets/OPPOer/EVA-data)


## 🔎 Overview

Recent advances in Omni Language Models (OLMs) provide a unified solution for audiovisual video captioning. Nevertheless, existing approaches predominantly treat captioning as an unstructured text generation process, neglecting the inherent semantic structure of videos and leading to suboptimal fine-grained audiovisual alignment. To bridge this gap, we propose **EVA-Cap**, a novel framework that decomposes captions into atomic audiovisual events and transforms unstructured text supervision into event-centric alignment. We train EVA-Cap via Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO) on a self-curated dataset, **EVA-Data**, which is built with event-centric quality control. During the GRPO phase, the model is explicitly optimized against **three event-centric alignment targets**: global event completeness, intra-event attribution, and inter-event synchronization. Extensive experiments demonstrate that EVA-Cap significantly outperforms existing methods in event-centric evaluation (e.g., +9.0% average gain on ChronusAV), while achieving competitive performance on general audiovisual captioning benchmarks (e.g., +3.3% on WorldSense).

## 🛠️ Setup

1. Install packages
```bash
conda create -n eva python=3.10 -y
conda activate eva
conda install ffmpeg
pip install --upgrade pip
pip install -r requirements.txt
pip install flash-attn==2.8.0.post2 --no-build-isolation
```

2. Download the base model [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)

3. Deploy the reward judge model [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) or directly use the api (Only for RL training)

## 🚀 Quick Start

1. Download the [EVA-Cap checkpoint](https://huggingface.co/OPPOer/EVA-Cap-7B).
2. In [inference.py](inference.py), set `MODEL_PATH` to your downloaded checkpoint directory and `VIDEO_PATH` to your local video file.
3. Run from the repository root:

```bash
python inference.py
```



## 🏋️‍♂️ Training

### Data Preparation

Download the five training annotation JSONL files from [EVA-data](https://huggingface.co/datasets/OPPOer/EVA-data) into `data/`. Download the corresponding source videos separately and arrange them as described in the [dataset README](https://huggingface.co/datasets/OPPOer/EVA-data/blob/main/README.md). The dataset README provides data sources, statistics, formats, and preparation instructions; video files are not included in EVA-data.

### Processor Patch

Apply the Qwen2.5-Omni processor patch to prevent audio truncation during training.
```bash
cp patches/processing_qwen2_5_omni.py "$(python -c 'import transformers, os; print(os.path.dirname(transformers.__file__))')/models/qwen2_5_omni/"
```

### SFT

Train with all the 63K EVA-data.

```bash
BASE=<qwenomni_path> RUN_NAME=<save_path> bash run_scripts/run_sft_caption.sh
```

Checkpoint: [SFT (`sft`)](https://huggingface.co/OPPOer/EVA-Cap-SFT-GRPO/tree/main/sft).

### GRPO

Train with the rollout-selected data. Configure [reward_config/reward_vllm.json](reward_config/reward_vllm.json) before training: set `addrs` to the judge server IP addresses or hostnames (without `http://` or a port). The provided server script exposes the model as `Qwen3.5-27B` on port `8000`, matching the configuration defaults. For another deployment, also set `model` to its served model name (as returned by `/v1/models`) and `port` to its listening port. The empty `addrs` list must be filled before training.
```bash
# Deploy reward judge LLM on another device and then manually set the configurations
# Note: We recommend creating a new env with vllm 0.18.0 and transformers 4.57.6
MODEL=<model_path> bash run_scripts/grpo_vllm_server.sh <run_name>
# Logs are written to output/<run_name>/vllm_server.log and output/<run_name>/ifconfig.log.
```

#### Stage 1

Optimize Global Event Completeness and Inter-Event Synchronization rewards.

```bash
STAGE0=<base_path> RUN_NAME=<save_path> bash run_scripts/run_grpo_caption_stage1.sh
```

Checkpoint: [GRPO stage 1 (`grpo_stage1`)](https://huggingface.co/OPPOer/EVA-Cap-SFT-GRPO/tree/main/grpo_stage1).

#### Stage 2

Optimize Global Event Completeness, Intra-Event Attribution and Inter-Event Synchronization rewards.

```bash
STAGE1=<base_path> RUN_NAME=<save_path> bash run_scripts/run_grpo_caption_stage2.sh
```

Checkpoint: [EVA-Cap](https://huggingface.co/OPPOer/EVA-Cap-7B).


## 📈 Evaluation

We provide evaluation scripts for all benchmarks evaluated in our paper.

- [Evaluation setup and data preparation](eval_scripts/README.md#evaluation-dependencies)
- [Benchmark evaluation commands](eval_scripts/README.md#benchmark-evaluation)


## 📝 Citation

If you find our work helpful, please consider giving a star ⭐ and citation 📝

```bibtex
@inproceedings{shi2026evacap,
      title={EVA-Cap: Optimizing Audiovisual Video Captioning via Event-Centric Alignment},
      author={Shi, Jiapeng and Hou, Qiuxia and Leng, Jiaqi and Wang, Junke and Zhang, Yanhao and Lu, Haonan and Ye, Ziyi and Wu, Zuxuan},
      booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
      year={2026}
}
```

## 🤝 Acknowledgements

Our project is based on [HumanOmniV2](https://github.com/HumanMLLM/HumanOmniV2) and [AVoCaDO](https://github.com/AVoCaDO-Captioner/AVoCaDO). We thank these projects for their contributions.

## License

Original EVA-Cap code is licensed under the Apache License 2.0. See [LICENSE](LICENSE). Third-party source attributions and license information are listed in [NOTICE](NOTICE).

Copyright 2026 OPPO and Fudan University.
