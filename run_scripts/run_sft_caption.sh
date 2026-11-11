#!/bin/bash

ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${2:-8}
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16666
ARG_RANK=0

# Multiple conditions
if [ -z "$WORLD_SIZE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
fi
if [ -z "$NPROC_PER_NODE" ]; then
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

export LOG_PATH="./debug_log_$RUN_NAME.txt"

mkdir -p output/$RUN_NAME/
cp $0  output/$RUN_NAME

torchrun  --nproc_per_node $NPROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    src/open_r1/sft_caption.py \
    --deepspeed run_scripts/zero2_offload.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path $BASE \
    --dataset_name data_config/sft_caption_datasets.json \
    --freeze_vision_modules true \
    --use_audio_in_video true \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --learning_rate 1.0e-5 \
    --bf16 True \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to tensorboard \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 2 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --log_level info \
    --dataloader_num_workers 8 \
    --save_only_model true 2>&1 | tee output/$RUN_NAME/train.log

