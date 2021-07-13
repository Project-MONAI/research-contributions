#!/bin/bash
pip install monai-weekly==0.6.dev2127;
pip install nibabel==3.1.1;
pip install tqdm==4.59.0;
pip install einops==0.3.0;
pip install tensorboardX==2.1;
pip install --upgrade --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex/