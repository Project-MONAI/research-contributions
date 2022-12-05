#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

python fake_training_script.py ${1} ${2} ${3}
