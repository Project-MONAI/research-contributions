# Description

A 2D neural network based algorithm for volumetric segmentation of 3D medical images.

# Model Overview

This is a template for training the 2D version of the state-of-the-art algorithm [1] of the "Brain Tumor Segmentation (BraTS) Challenge 2018".

## Training configuration

The training was performed with at least 16GB-memory GPUs.

## commands example

Execute model training:

```
python -m scripts.train run --config_file "['configs/hyper_parameters.yaml','configs/network.yaml','configs/transforms_train.yaml','configs/transforms_validate.yaml']"
```

Execute multi-GPU model training (recommended):

```
torchrun --nnodes=1 --nproc_per_node=8 -m scripts.train run --config_file "['configs/hyper_parameters.yaml','configs/network.yaml','configs/transforms_train.yaml','configs/transforms_validate.yaml']"
```

Execute validation:

```
python -m scripts.validate run --config_file "['configs/hyper_parameters.yaml','configs/network.yaml','configs/transforms_infer.yaml']"
```

Execute inference:

```
python -m scripts.infer run --config_file "['configs/hyper_parameters.yaml','configs/network.yaml','configs/transforms_infer.yaml']"
```


# References

[1] Myronenko, A., 2018, September. 3D MRI brain tumor segmentation using autoencoder regularization. In International MICCAI Brainlesion Workshop (pp. 311-320). Springer, Cham.
