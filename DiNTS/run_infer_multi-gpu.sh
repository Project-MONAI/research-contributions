#!/bin/bash
clear

ARCH_CKPT="search_code_20000.pth"
CONFIG="config.yaml"
DATA_ROOT="/home/dongy/Data/MSD/Task09_Spleen"
JSON_PATH="${DATA_ROOT}/dataset.json"
NUM_FOLDS=5

NUM_GPUS_PER_NODE=4
NUM_NODES=1

if [ ${NUM_GPUS_PER_NODE} -eq 2 ]
then
    export CUDA_VISIBLE_DEVICES=0,1
elif [ ${NUM_GPUS_PER_NODE} -eq 4 ]
then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
elif [ ${NUM_GPUS_PER_NODE} -eq 8 ]
then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi

for FOLD in 0 1 2 3 4
do
    CHECKPOINT="Fold${FOLD}/best_metric_model.pth"
    JSON_KEY="test"
    OUTPUT_ROOT="Seg_Fold${FOLD}_${JSON_KEY}"

    python -m torch.distributed.launch \
        --nproc_per_node=${NUM_GPUS_PER_NODE} \
        --nnodes=${NUM_NODES} \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=1234 \
        infer_multi-gpu.py  --arch_ckpt=${ARCH_CKPT} \
                            --checkpoint=${CHECKPOINT} \
                            --config=${CONFIG} \
                            --json=${JSON_PATH} \
                            --json_key=${JSON_KEY} \
                            --output_root=${OUTPUT_ROOT} \
                            --prob \
                            --root=${DATA_ROOT}
done
