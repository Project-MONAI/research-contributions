#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python fake_training_script_segresnet.py ${1} ${2} ${3}
