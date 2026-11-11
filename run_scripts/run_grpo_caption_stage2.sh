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

export NCCL_SOCKET_TIMEOUT=600000
export NCCL_MAX_RETRY=100
export NCCL_MAX_RETRY_COUNT=100
export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=3600000  # 单位是毫秒，这里设为 2 小时
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1200
export NCCL_ASYNC_ERROR_HANDLING=1

mkdir -p output/$RUN_NAME
cp $0  output/$RUN_NAME

torchrun  --nproc_per_node $NPROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    src/open_r1/grpo_caption.py \
    --deepspeed run_scripts/zero3_offload.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path $STAGE1 \
    --dataset_name data_config/rl_caption_datasets_stage2.json \
    --max_prompt_length 2048 \
    --max_completion_length 2048 \
    --num_generations 8 \
    --num_generations_per_device 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --freeze_vision_modules true \
    --logging_steps 1 \
    --learning_rate 2.0e-6 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to tensorboard \
    --scale_rewards true \
    --reward_funcs  event_recall a2v_query event_attribution\
    --reward_weights  1.0 1.0 1.0 \
    --use_audio_in_video true \
    --gradient_checkpointing true \
    --log_completions false \
    --attn_implementation sdpa \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 25 \
    --dataloader_num_workers 2 \
    --save_only_model false  2>&1 | tee output/$RUN_NAME/train.log


exit ${PIPESTATUS[0]}