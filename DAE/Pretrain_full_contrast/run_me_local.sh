#!/usr/bin/env bash

NPROC_PER_NODE=1
EXP_NAME='upsample_vae'
OUTPUT_PATH='./output/'${EXP_NAME}
LOG_PATH='./logdir/'${EXP_NAME}
BATCH_SIZE=3
IMG_SIZE=96
SW_BATCH_SIZE=1
MASK_RATIO=0.6
MASK_PATCH_SIZE=16
EPOCHS=100
WARMUP_EPOCHS=10
BASE_LR=2e-4
WARMUP_LR=1e-6
MIN_LR=1e-5
WEIGHT_DECAY=0.05
SAVE_FREQ=5
PRINT_FREQ=5
CACHE_RATE=0.5
DECODER='pixel_shuffle'
LOSS_TYPE='l2'
LOSS_TYPE='mask_only'
DECODER='deconv'
DECODER='swin'
DECODER='vae2'
MODEL_TYPE='swin_skip'

python -m torch.distributed.launch --nproc_per_node ${NPROC_PER_NODE} main.py \
--batch_size=${BATCH_SIZE} --sw_batch_size=${SW_BATCH_SIZE} --mask_ratio=${MASK_RATIO} \
--epoch=${EPOCHS} --mask_patch_size=${MASK_PATCH_SIZE} --img_size=${IMG_SIZE} \
 --min_lr=${MIN_LR} --warmpup_epoch=${WARMUP_EPOCHS} --decoder=${DECODER} --model_type=${MODEL_TYPE} --loss_type=${LOSS_TYPE} --base_lr=${BASE_LR} --warmup_lr=${WARMUP_LR} \
 --weight_decay=${WEIGHT_DECAY} --save_freq=${SAVE_FREQ} --print_freq=${PRINT_FREQ} --log_dir=${LOG_PATH} --cache_dataset --output=${OUTPUT_PATH}\
 --local --decoder_off\

# --use_grad_checkpoint
# --thread_loader
