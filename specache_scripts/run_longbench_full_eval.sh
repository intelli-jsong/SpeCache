#!/bin/bash

# export MODEL_PATH="/mnt/cephfs/models/meta-llama/Meta-Llama-3-8B-Instruct"
export MODEL_PATH="/mnt/cephfs/models/meta-llama/Llama-3.1-8B-Instruct"
# export MODEL_PATH="/mnt/cephfs/models/gradientai/Llama-3-8B-Instruct-Gradient-1048k"

MODEL_NAME=$(basename "$MODEL_PATH")

SEED_LIST="88 77 25 1 96"
for RAND_SEED in $SEED_LIST
do
    LOG_DIR="/mnt/cephfs/ShadowKV/longbench_pred/${MODEL_NAME}_131072_full_seed${RAND_SEED}"
    mkdir -p "$LOG_DIR"
    export LOG_FILE="${LOG_DIR}/pred_longbench.log"

    CUDA_VISIBLE_DEVICES=0 python test/pred_long_bench_v3.py \
    --method full \
    --rand_seed ${RAND_SEED} \
    --model_name "$MODEL_PATH" 2>&1 | tee "$LOG_FILE"

    CUDA_VISIBLE_DEVICES=0 python test/eval_long_bench.py \
    --model "${MODEL_NAME}_131072_full_seed${RAND_SEED}"
done