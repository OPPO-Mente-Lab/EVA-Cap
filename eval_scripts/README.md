# Evaluation

We provide evaluation scripts for all evaluated benchmarks in our paper. Please make sure the data paths and judge LLM configurations are correct before running the code.

Run all commands below from the repository root.

## Evaluation dependencies

After completing the setup in the [main README](../README.md), install the additional evaluation dependencies:

```bash
pip install -r requirements-eval.txt
```

ChronusAV caption metrics also require Java 8 with `java` on `PATH`.

## Judge API configuration

Set the judge API variables before evaluation (replace the placeholders with your gateway credentials):

```bash
# Daily-Omni, WorldSense, AV-SpeakerBench, and ChronusAV QA
export GEMINI_AK="your-gateway-access-key"
export GEMINI_SK="your-gateway-signing-secret"
export GEMINI_URL="https://your-gateway.example/gemini-endpoint"
export GEMINI_MODEL="gemini-2.5-pro"

# video-SALMONN2-testset and UGC-VideoCap
export GPT4_AK="your-gateway-access-key"
export GPT4_SK="your-gateway-signing-secret"
export GPT4_URL="https://your-gateway.example/gpt-endpoint"
# Choose gpt-4.1 for video-SALMONN2-testset or gpt-4o for UGC-VideoCap.
export GPT4_MODEL="gpt-4.1"

# ChronusAV VDC (the complete ChronusAV pipeline also needs GEMINI_* above)
export SEED_AK="your-gateway-access-key"
export SEED_SK="your-gateway-signing-secret"
export SEED_URL="https://your-gateway.example/seed-endpoint"
export SEED_MODEL="doubao-seed-1.6"
```

The scripts require a gateway supporting `bot-auth-v1` signing and `data.choices` responses; direct vendor APIs require adapting the authentication and response parsing.

## Evaluation data layout

Download and extract each benchmark from its link below. Run all evaluation commands from the repository root, and place the downloaded videos under `bench/` as shown here. The scripts use these relative paths; they do not provide a data-root CLI option.

```text
bench/
|-- video-SALMONN_2_testset/
|   `-- video/                         # paths from video_salmonn2_test.json
|-- UGC-VideoCap/
|   `-- video/                         # video files directly in this directory
|-- Daily-Omni/
|   `-- Videos/                        # .mp4 files, subdirectories allowed
|-- WorldSense/
|   `-- videos/                        # .mp4 files, subdirectories allowed
|-- ChronusAV/
|   `-- raw_videos/
|       `-- test/                      # <video_id>.mp4
`-- AV-SpeakerBench/
    `-- audiovisual/                  # clips referenced by audio_visual_path
```

Keep the original video filenames and annotation IDs. Directory names are case-sensitive on Linux. The evaluation annotations are already included under `eval_scripts/`; the video files must be downloaded separately.

| Benchmark | Video lookup | Included annotation |
|---|---|---|
| video-SALMONN2-testset | `bench/video-SALMONN_2_testset/<video>`; the annotation's `video` field already includes `video/`, e.g. `video/0.mp4` | `eval_scripts/video-SALMONN2-testset/video_salmonn2_test.json` |
| UGC-VideoCap | Every file directly under `bench/UGC-VideoCap/video/`; keep only the benchmark video files there | `eval_scripts/UGC-VideoCap/final_caption_qa.json` |
| Daily-Omni | Recursively reads `.mp4` files under `bench/Daily-Omni/Videos/` | `eval_scripts/Daily-Omni/grouped_data.json` |
| WorldSense | Recursively reads `.mp4` files under `bench/WorldSense/videos/` | `eval_scripts/WorldSense/worldsense_qa.json` |
| ChronusAV | `bench/ChronusAV/raw_videos/test/<video_id>.mp4`, using IDs from the included annotation | `eval_scripts/ChronusAV/grouped_data_sub200.json` |
| AV-SpeakerBench | `bench/AV-SpeakerBench/<audio_visual_path>`; preserve the referenced audiovisual clips and their relative paths, e.g. `audiovisual/I0CPeFpSIbE_280_296.mp4` | `eval_scripts/AVSpeaker/qa_speaker_recognition.json` |

If your downloaded files are stored elsewhere, arrange them in this layout or update `video_dir` / `video_base_dir` in the corresponding `eval_scripts/<benchmark>/generate_caption.py` before running its pipeline.

## Benchmark evaluation

### Direct Audiovisual Caption Evaluation

1. video-SALMONN2-testset [[Download link](https://huggingface.co/datasets/videoSALMONN2/video-SALMONN_2_testset)]:
```bash
bash eval_scripts/video-SALMONN2-testset/eval_video-SALMONN2-testset.sh <model_path> <save_directory>
```

2. UGC-VideoCap [[Download link](https://huggingface.co/datasets/openinterx/UGC-VideoCap)]:
```bash
bash eval_scripts/UGC-VideoCap/eval_UGC-VideoCap.sh <model_path> <save_directory>
```

### QA-based Audiovisual Caption Evaluation

1. Daily-Omni [[Download link](https://huggingface.co/datasets/liarliar/Daily-Omni)]:
```bash
bash eval_scripts/Daily-Omni/Daily-Omni_pipeline.sh <model_path> <save_directory>
```

2. WorldSense [[Download link](https://huggingface.co/datasets/lmms-lab/WorldSense)]:
```bash
bash eval_scripts/WorldSense/WorldSense_pipeline.sh <model_path> <save_directory>
```

### Event-Centric Audiovisual Caption Evaluation

1. ChronusAV (Audiovisual Synchronization) [[Download link](https://huggingface.co/datasets/mxxxxxxxxxxxxxxxxx/ChronusAV)]:
```bash
bash eval_scripts/ChronusAV/ChronusAV_pipeline.sh <model_path> <save_directory>
```

2. AV-SpeakerBench (Event Attribution) [[Download link](https://huggingface.co/datasets/plnguyen2908/AV-SpeakerBench)]:
```bash
bash eval_scripts/AVSpeaker/AVSpeaker_pipeline.sh <model_path> <save_directory>
```

## Evaluation annotation licenses

The included benchmark annotations retain their upstream authorship and data
license terms. These data licenses are separate from the Apache-2.0 license for
the original EVA-Cap code.

For Daily-Omni and ChronusAV, we selected evaluation subsets and converted the
annotation format for these scripts without changing the annotation content.

| Included annotation | Upstream authors | Data license / status |
|---|---|---|
| `eval_scripts/video-SALMONN2-testset/video_salmonn2_test.json` | Changli Tang, Yixuan Li, Yudong Yang, Jimin Zhuang, Guangzhi Sun, Wei Li, Zejun Ma, Chao Zhang | Not stated by the source. The official project's Apache-2.0 code license is not a data license; a separate dataset metadata declaration of Apache-2.0 has unverified applicability ([alternative source](https://huggingface.co/datasets/tsinghua-ee/video-SALMONN_2_testset)). |
| `eval_scripts/UGC-VideoCap/final_caption_qa.json` | Peiran Wu, Yunze Liu, Zhengdong Zhu, Enmin Zhou, Junxiao Shen | [MIT declaration](https://huggingface.co/datasets/openinterx/UGC-VideoCap/blob/main/README.md); [MIT terms](https://opensource.org/license/mit). Full data notices and applicable scope require confirmation. |
| `eval_scripts/Daily-Omni/grouped_data.json` | Ziwei Zhou, Rui Wang, Zuxuan Wu, Yu-Gang Jiang | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| `eval_scripts/WorldSense/worldsense_qa.json` | Jack Hong, Shilin Yan, Jiayin Cai, Xiaolong Jiang, Yao Hu, Weidi Xie | [Official dataset](https://huggingface.co/datasets/honglyhly/WorldSense): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); the download mirror linked above declares Apache-2.0 in its metadata. The version and applicable license scope require confirmation. |
| `eval_scripts/ChronusAV/grouped_data_sub200.json` | Yijing Chen, Yihan Wu, Kaisi Guan, Yuchen Ren, Yuyue Wang, Ruihua Song, Liyun Ru | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |
| `eval_scripts/AVSpeaker/qa_speaker_recognition.json` | Le Thien Phuc Nguyen, Zhuoran Yu, Samuel Low Yu Hang, Subin An, Jeongik Lee, Yohan Ban, SeungEun Chung, Thanh-Huy Nguyen, JuWan Maeng, Soochahn Lee, Yong Jae Lee | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), stated in the dataset README body and official project; MIT metadata is not used here. |

When sharing these annotations, preserve the authors, source and license links,
the processing statement above, and upstream notices. Daily-Omni, ChronusAV and
AV-SpeakerBench are non-commercial; adaptations of Daily-Omni must also comply
with ShareAlike. Uses outside an applicable license require separate permission.
An unresolved data license does not grant redistribution permission. Source
videos are not included here and their rights must be checked separately; other
benchmark assets retain their respective upstream licenses.
