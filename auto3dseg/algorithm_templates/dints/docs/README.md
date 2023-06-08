# Description

A neural architecture search algorithm for volumetric segmentation of 3D medical images.

# Model Overview

This serves as a template for the state-of-the-art algorithm [1] of the "Medical Segmentation Decathlon Challenge 2018".

## Training Requirements

The training process was performed with at least 16GB-memory GPUs.

## Command Examples

Execute model searching:

```
python -m scripts.search run --config_file "['configs/hyper_parameters.yaml','configs/hyper_parameters_search.yaml','configs/network_search.yaml','configs/transforms_train.yaml','configs/transforms_validate.yaml']"
```

Execute multi-GPU model searching (recommended):

```
torchrun --nnodes=1 --nproc_per_node=8 -m scripts.search run --config_file "['configs/hyper_parameters.yaml','configs/hyper_parameters_search.yaml','configs/network_search.yaml','configs/transforms_train.yaml','configs/transforms_validate.yaml']"
```

Execute model training:

```
python -m scripts.train run --config_file "['configs/hyper_parameters.yaml','configs/network.yaml','configs/transforms_train.yaml','configs/transforms_validate.yaml','configs/transforms_infer.yaml']"
```

Execute multi-GPU model training (recommended):

```
torchrun --nnodes=1 --nproc_per_node=8 -m scripts.train run --config_file "['configs/hyper_parameters.yaml','configs/network.yaml','configs/transforms_train.yaml','configs/transforms_validate.yaml','configs/transforms_infer.yaml']"
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

[1] He, Y., Yang, D., Roth, H., Zhao, C. and Xu, D., 2021. DiNTS: Differentiable neural network topology search for 3d medical image segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 5841-5850).
