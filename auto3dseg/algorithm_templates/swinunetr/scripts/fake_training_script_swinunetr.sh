#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

python fake_training_script_swinunetr.py ${1} ${2} ${3} ${4}
