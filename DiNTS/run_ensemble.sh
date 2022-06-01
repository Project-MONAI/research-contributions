#!/bin/sh
clear

ALGORITHM="AM"
CONFIG="config.yaml"
FOLDER_0="."
INPUT_ROOT="${PWD}"
OUTPUT_ROOT="${PWD}/${FOLDER_0}/Submission_${ALGORITHM}"

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

python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS_PER_NODE} \
    ensemble_multi-gpu.py   --algorithm ${ALGORITHM} \
                            --config=${CONFIG} \
                            --input_root ${INPUT_ROOT} \
                            --output_root ${OUTPUT_ROOT} \
                            --dir_list  "${FOLDER_0}/Seg_Fold0_test" \
                                        "${FOLDER_0}/Seg_Fold1_test" \
                                        "${FOLDER_0}/Seg_Fold2_test" \
                                        "${FOLDER_0}/Seg_Fold3_test" \
                                        "${FOLDER_0}/Seg_Fold4_test" \
