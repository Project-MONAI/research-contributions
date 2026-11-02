# Description

A 2D neural network based algorithm for volumetric segmentation of 3D medical images.

# Model Overview

This is a template for training the 2D version of the state-of-the-art algorithm [1] of the "Brain Tumor Segmentation (BraTS) Challenge 2018".

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

## Normalization for multiple inputs

Generated configs keep `normalize_mode` as a string when the primary image and all
`extra_modalities` use the same normalization. Otherwise, it is a list ordered by
the primary image key followed by the keys in `extra_modalities` (in insertion
order). Each entry applies to all channels of that input image. Modality aliases
`ct` and `mri` are saved as `range` and `meanstd`, respectively.

For example, an MRI image with an additional label image that must not be normalized produces:

```yaml
normalize_mode: [meanstd, none]
extra_modalities: {image2: none}
```

An explicit list overrides the normalization modes in `extra_modalities` and must
have one entry per input key. Legacy string configs continue to use the string
for the primary image and the values in `extra_modalities` for additional images.

Set `input_channels` to the actual total when extra images have multiple channels.
When export/refill overrides change the keys in `extra_modalities`, also provide
`input_channels`; counting image keys cannot determine the number of channels.
An indexed override such as `normalize_mode#1` requires a list-valued mode. For a
shared string mode, supply a complete `normalize_mode` list instead.

When an explicit list is in use, changing extra input keys or their order also
requires a complete `normalize_mode` list in the final input order. An indexed
override cannot replace this full-list update. Alternatively, provide a string
mode to regenerate the modes from `extra_modalities`. Derived modes without
indexed overrides are regenerated automatically when the input order changes.

# References

[1] Myronenko, A., 2018, September. 3D MRI brain tumor segmentation using autoencoder regularization. In International MICCAI Brainlesion Workshop (pp. 311-320). Springer, Cham.
