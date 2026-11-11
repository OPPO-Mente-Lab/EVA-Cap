#!/bin/bash
MODEL_PATH=$1 # TODO
OUTPUT_DIR=$2

mkdir -p "$OUTPUT_DIR"

torchrun --nproc_per_node 8 --master_addr 127.0.0.1 --master-port 29502 --nnodes 1 \
    eval_scripts/UGC-VideoCap/generate_caption.py \
    --model_path "$MODEL_PATH" \
    --save_path "$OUTPUT_DIR/model_caption.jsonl"

python eval_scripts/UGC-VideoCap/evaluation.py "${OUTPUT_DIR}/model_caption.jsonl" "${OUTPUT_DIR}/eval_results.json"
