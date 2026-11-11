#!/bin/bash

# source /opt/conda/bin/activate vllm
# vllm==0.18.0 
# transformers==4.57.6

# RUN_NAME="qwen_vllm"
RUN_NAME=$1
mkdir -p ./output/$RUN_NAME

#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=200000
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

# 检测系统GPU总数
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "检测到系统共有 $GPU_COUNT 块GPU"

# 分配vLLM使用最后两块GPU
VLLM_GPUS=$(seq $((GPU_COUNT-2)) $((GPU_COUNT-1)) | tr '\n' ',')
VLLM_GPUS=${VLLM_GPUS%,}  # 移除最后一个逗号

# 分配训练脚本使用剩余GPU
TRAIN_GPUS=$(seq 0 $((GPU_COUNT-3)) | tr '\n' ',')
TRAIN_GPUS=${TRAIN_GPUS%,}

echo "vLLM服务将使用GPU: $VLLM_GPUS"
echo "训练脚本将使用GPU: $TRAIN_GPUS"

# 启动vllm服务
echo "开始启动vllm服务..."

CUDA_VISIBLE_DEVICES=$VLLM_GPUS vllm serve $MODEL \
    --served-model-name Qwen3.5-27B \
    --port 8000 \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \
    --max-model-len 65536 \
    --reasoning-parser qwen3 \
    2> ./output/$RUN_NAME/vllm_server.log &

VLLM_PID=$!
echo "vllm服务进程ID: $VLLM_PID"

ifconfig > ./output/$RUN_NAME/ifconfig.log

echo "可以执行训练脚本..."
while true; do
    sleep 1
done

# 关闭vllm服务
echo "训练脚本执行完成，准备关闭vllm服务..."
if ps -p $VLLM_PID > /dev/null; then
    echo "正在关闭vllm服务（PID: $VLLM_PID）..."
    kill $VLLM_PID
    wait $VLLM_PID 2>/dev/null
    echo "vllm服务已成功关闭"
else
    echo "vllm服务未运行，无需关闭"
fi

echo "所有任务执行完毕"