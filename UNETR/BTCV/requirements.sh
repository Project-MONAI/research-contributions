#!/bin/bash
pip install -q "monai-weekly[nibabel, tqdm, einops]"; pip install --upgrade --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex/; pip install tensorboardX
