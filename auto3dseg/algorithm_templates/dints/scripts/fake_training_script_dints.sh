#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python fake_training_script_dints.py ${1} ${2} ${3} ${4}
