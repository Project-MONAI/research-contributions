#!/bin/bash
EXP_NAME="test1"
POS_EMBED="perceptron"
BATCH_SIZE=2
NUM_STEPS=20000
EVAL_NUM=1000
LOSS_TYPE="dice_ce"
OPT="adamw"

cd ..; python __main__.py --pos_embedd=${POS_EMBED} --batch_size=${BATCH_SIZE}  --num_steps=${NUM_STEPS} --loss_type=${LOSS_TYPE} --opt=${OPT} --lrdecay --eval_num=${EVAL_NUM} --name=${EXP_NAME}
