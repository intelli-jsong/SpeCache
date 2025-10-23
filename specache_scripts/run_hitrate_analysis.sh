#!/bin/bash

export MODEL_PATH="/mnt/cephfs/models/meta-llama/Meta-Llama-3-8B-Instruct"
# export MODEL_PATH="/mnt/cephfs/models/meta-llama/Llama-3.1-8B-Instruct"
# export MODEL_PATH="/mnt/cephfs/models/gradientai/Llama-3-8B-Instruct-Gradient-1048k"

MODEL_NAME=$(basename "$MODEL_PATH")

python test/utils_hitrate.py \
--model_name $MODEL_NAME \
--specache_bit 2 \
--dataset "gov_report" # {"multi_news", "gov_report"}
