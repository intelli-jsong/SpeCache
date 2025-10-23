#!/bin/bash

export MODEL_PATH="/mnt/cephfs/models/meta-llama/Meta-Llama-3-8B-Instruct"
# export MODEL_PATH="/mnt/cephfs/models/meta-llama/Llama-3.1-8B-Instruct"
# export MODEL_PATH="/mnt/cephfs/models/gradientai/Llama-3-8B-Instruct-Gradient-1048k"

MODEL_NAME=$(basename "$MODEL_PATH")

# For Checking Hit Rate
# ===============================================================
SEED_LIST="365"
for RAND_SEED in $SEED_LIST
do
    CUDA_VISIBLE_DEVICES=0 python test/pred_long_bench_v3.py \
    --method specache \
    --specache_bit 2 \
    --rand_seed ${RAND_SEED} \
    --check_hit_rate \
    --model_name $MODEL_PATH
done