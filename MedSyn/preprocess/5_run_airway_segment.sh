#!/bin/bash

set -x

for IDX in {0..3}; do
    CUDA_VISIBLE_DEVICES=$IDX python batch_segment.py \
    --data_path /jet/home/lisun/work/r3/evaluation/seg_eval/results/img \
    --output_path /jet/home/lisun/work/r3/evaluation/seg_eval/results/img_seg \
    --seg_real '' \
    --num_job 4 \
    --job_id $IDX &
done
