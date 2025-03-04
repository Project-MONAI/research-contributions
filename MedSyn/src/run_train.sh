#!/bin/bash

source /ocean/projects/asc170022p/yanwuxu/miniconda/etc/profile.d/conda.sh
conda activate medsyn

# Training with two nodes, low-res training
accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 --main_process_port=29816 train_low_res.py --data_dir <your training data dir> \
--prompt_dir <your prompt feature dir> --save_dir <your logs saving dir>

# Training with two nodes, high-res training
accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 --main_process_port=29816 train_super_res.py
