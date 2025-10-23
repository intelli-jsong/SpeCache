
#!/bin/bash

export DATASETS="ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multikey_3,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2"

export MODEL_PATH="/mnt/cephfs/models/meta-llama/Llama-3.1-8B-Instruct"
export RAND_SEED=88
export DATALEN=32768

MODEL_NAME=$(basename "$MODEL_PATH")

LOG_DIR="/mnt/cephfs/ShadowKV/ruler_pred/${MODEL_NAME}_spec_2bit"
mkdir -p "$LOG_DIR"
export LOG_FILE="${LOG_DIR}/ruler_result.log"

OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 test/pred_ruler.py \
--datalen ${DATALEN} \
--method specache \
--dataset_name "$DATASETS" \
--specache_bit 2 \
--rand_seed ${RAND_SEED} \
--model_name "$MODEL_PATH" 2>&1 | tee "$LOG_FILE"
