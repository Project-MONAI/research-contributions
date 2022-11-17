#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

python fake_training_script_segresnet2d.py ${1} ${2} ${3} ${4}
