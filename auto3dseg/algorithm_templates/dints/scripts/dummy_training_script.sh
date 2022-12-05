#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python dummy_training_script.py ${1} ${2} ${3} ${4}

