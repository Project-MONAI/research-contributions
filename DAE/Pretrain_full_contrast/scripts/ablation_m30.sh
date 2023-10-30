#!/bin/bash

#####################  Setting up hardware
GPU_MEM=32g
N_GPU=8
NPROC_PER_NODE=8
INSTANCE="dgxa100.40g.${N_GPU}.norm"
INSTANCE="dgx1v.${GPU_MEM}.${N_GPU}.norm.beta"
INSTANCE="dgx1v.${GPU_MEM}.${N_GPU}.norm"
DOCKER_IMAGE="nvidian/dlmed/clara-train-sdk:v4.0"
#####################

#####################  Datasets
NGC_DATASET_ID1=81073  # luna
NGC_DATASET_ID2=80863  # covid
NGC_DATASET_ID3=81515  # HNSCC
NGC_DATASET_ID4=81798  # colon
NGC_DATASET_ID5=80329  # FLARE 2021 (id 80329)
NGC_DATASET_ID6=75547  # LiTS 2017 (id 75547)
NGC_DATASET_ID7=80490  # HECKTOR 2021 (id 80490)
NGC_DATASET_ID8=60659  # LiDC
NGC_DATASET_ID8=95352  # LiDC
#####################

FOLDER_PATH="/workspace/Swin_UNETR/Pretrain_code/pretrain_framework"


# Maximum batch sizes best on memory is specified below

BATCH_SIZE=12  # for A100 w checkpoint
BATCH_SIZE=10 # for V100 w checkpoint
BATCH_SIZE=6  # for A100
BATCH_SIZE=4 # for V100


#####################  Important hyper-parameters
   # masking ratio in the image
LOSS_TYPE='all_img'    # this loss works on all pixels
LOSS_TYPE='mask_only'  # this loss only works on masked patches, which is default
DECODER='deconv'       # this decoder uses a series of deconv layers without anything else
DECODER='pixel_shuffle' # this decoder only uses a deconv layer + pixel shuffling with stride 32 (maybe sub-optimal)
DECODER='upsample' # this decoder uses unet decoder blocks (2 conv blocks) progressively but no skip connections (maybe computationally heavy)
DECODER='vae2'  # this decoder uses conv + upsamling layers progressively but no skip connections

MODEL_TYPE='swin' # this is a swin transformer encoder but has no skip connections. Can be used with the different decoder types above
MODEL_TYPE='swin_skip' # this is a new architecture which uses : (1) conv + upsampling layers for decoder (2) skip connections between encoder and decoder
# when using this model_type, there is no need to choose another decoder type, since decoder is fixed

LOSS_TYPE='mask_only'

VERSION="new_b6_ep200_m50_p16_1node_lr2e4_vae_skip_v1_l2"
VERSION="new_b6_ep200_m50_p16_1node_lr2e4_vae_skip_v1_l2_cache1"
VERSION="new_b6_ep200_m50_p16_1node_lr2e4_vae_skip_v1_mask_only"

VERSION="new_b6_ep800_m60_p16_1node_lr2e4_vae_skip_v1_mask_only_new_embed"
VERSION="ablation_m30_v3"

EXP_NAME="pretrain.swin_unetr.${VERSION}"
OUTPUT_PATH='./output/'${EXP_NAME}
LOG_PATH='./logdir/'${EXP_NAME}
NAME="ml-model.${EXP_NAME}.${N_GPU}GPU.${GPU_MEM}"


IMG_SIZE=96
SW_BATCH_SIZE=1
EPOCHS=100
BASE_LR=2e-4
#####################

WARMUP_EPOCHS=10
WARMUP_LR=1e-6
MIN_LR=1e-5
WEIGHT_DECAY=0.05
SAVE_FREQ=5
PRINT_FREQ=5
CACHE_RATE=0.7

MASK_PATCH_SIZE=16
MASK_RATIO=0.30


ngc batch run --name "${NAME}" \
   --preempt RUNONCE --ace nv-us-west-2 --instance ${INSTANCE} \
   --result /results \
   --image ${DOCKER_IMAGE} \
   --org nvidian --team "dlmed" \
   --datasetid ${NGC_DATASET_ID1}:/dataset/dataset1\
   --datasetid ${NGC_DATASET_ID2}:/dataset/dataset2\
   --datasetid ${NGC_DATASET_ID3}:/dataset/dataset3\
   --datasetid ${NGC_DATASET_ID4}:/dataset/dataset4\
   --datasetid ${NGC_DATASET_ID5}:/dataset/dataset5\
   --datasetid ${NGC_DATASET_ID6}:/dataset/dataset6\
   --datasetid ${NGC_DATASET_ID7}:/dataset/dataset7\
   --datasetid ${NGC_DATASET_ID8}:/dataset/dataset8\
   --workspace pretrain-dlmed:/workspace:RW \
   --commandline "cd ${FOLDER_PATH} ; pip install -r requirements.txt ; pip install monai==0.8.0; python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE \
    main.py --batch_size=${BATCH_SIZE} --sw_batch_size=${SW_BATCH_SIZE} --mask_ratio=${MASK_RATIO} \
    --epoch=${EPOCHS} --mask_patch_size=${MASK_PATCH_SIZE} --img_size=${IMG_SIZE} \
    --min_lr=${MIN_LR} --warmpup_epoch=${WARMUP_EPOCHS} --decoder=${DECODER} --loss_type=${LOSS_TYPE} --base_lr=${BASE_LR} --warmup_lr=${WARMUP_LR} \
    --weight_decay=${WEIGHT_DECAY} --cache_dataset --cache_rate=${CACHE_RATE} --model_type=${MODEL_TYPE} --save_freq=${SAVE_FREQ} \
     --print_freq=${PRINT_FREQ} --log_dir=${LOG_PATH} --output=${OUTPUT_PATH} --thread_loader"

# --use_grad_checkpoint for gradient checkpointing ( allows for increasing batch size )
#--iso_spacing for resampling isotropic spacing
