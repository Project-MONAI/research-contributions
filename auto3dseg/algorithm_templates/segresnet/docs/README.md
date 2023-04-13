# Description

A 3D neural network based algorithm for volumetric segmentation of 3D medical images.

# Model Overview

This is a template for training the state-of-the-art algorithm [1] of the "Brain Tumor Segmentation (BraTS) Challenge 2018".

## Training configuration

The training was performed with at least 16GB-memory GPUs.

## commands example

Execute model training:

```
CUDA_VISIBLE_DEVICES=0 python scripts/train.py run --config_file=configs/hyper_parameters.yaml
```

Execute multi-GPU model training (recommended):

```
torchrun --nproc_per_node=gpu scripts/train.py run --config_file=configs/hyper_parameters.yaml
```

Execute validation:

```
python scripts/validate.py run --config_file=configs/hyper_parameters.yaml
```

Execute inference:

```
python scripts/infer.py run --config_file=configs/hyper_parameters.yaml
```

# References

[1] Myronenko, A., 2018, September. 3D MRI brain tumor segmentation using autoencoder regularization. In International MICCAI Brainlesion Workshop (pp. 311-320). Springer, Cham.
