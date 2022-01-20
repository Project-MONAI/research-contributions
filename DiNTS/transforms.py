# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch

from monai.config import IndexSelection, KeysCollection
from monai.transforms import (
    AsDiscreted,
    AddChannel,
    AddChanneld,
    AsChannelFirstd,
    CastToTyped,
    Compose,
    ConcatItemsd,
    CopyItemsd,
    CropForegroundd,
    DivisiblePadd,
    EnsureChannelFirstd,
    EnsureTyped,
    KeepLargestConnectedComponent,
    Lambdad,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    ScaleIntensityRanged,
    ThresholdIntensityd,
    RandCropByLabelClassesd,
    RandCropByPosNegLabeld,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandShiftIntensityd,
    RandScaleIntensityd,
    RandSpatialCropd,
    RandSpatialCropSamplesd,
    RandFlipd,
    RandRotate90d,
    RandZoomd,
    Spacingd,
    SpatialPadd,
    SqueezeDimd,
    ToDeviced,
    ToNumpyd,
    ToTensord,
)
# from monai.transforms.compose import MapTransform
from monai.transforms.transform import MapTransform
from scipy import ndimage
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union


class CorrectLabelAffined(MapTransform):
    def __init__(self, keys) -> None:
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        d["label_meta_dict"]["affine"] = d["image_meta_dict"]["affine"]
        return d


# def str2aug(str_aug):

#     if str_aug == "RandShiftIntensity":
#         aug = RandShiftIntensityd(
#                   keys=["image"],
#                   offsets=0.1,
#                   prob=0.1
#               )
#     elif str_aug == "RandFlip":
#         aug = RandFlipd(
#                   keys=["image", "label"],
#                   prob=0.5
#               )
#     elif str_aug == "RandRotate90":
#         aug = RandRotate90d(
#                   keys=["image", "label"],
#                   prob=0.1
#               )
#     else:
#         aug = None

#     return aug


# def creating_label_interpolation_transform(method, spacing, output_classes):
#     if method.lower() == "nearest":
#         transform = [
#             Spacingd(
#                 keys=["image", "label"],
#                 pixdim=spacing,
#                 mode=("bilinear", "nearest"),
#                 align_corners=(True, True)
#                 )
#         ]
#     elif method.lower() == "linear":
#         transform = [
#             # multi-class to one-hot
#             AddChanneld(
#                 keys=["label"]
#             ),
#             ToTensord(
#                 keys=["label"]
#             ),
#             AsDiscreted(
#                 keys=["label"],
#                 to_onehot=True,
#                 n_classes=output_classes
#             ),
#             ToNumpyd(
#                 keys=["label"]
#             ),
#             SqueezeDimd(
#                 keys=["label"],
#                 dim=0
#             ),
#             CastToTyped(
#                 keys=["image", "label"],
#                 dtype=(np.float16, np.float16)
#             ),
#             # re-sampling
#             Spacingd(
#                 keys=["image", "label"],
#                 pixdim=spacing,
#                 mode=("bilinear", "bilinear"),
#                 align_corners=(True, True)
#             ),
#             # one-hot to multi-class
#             AddChanneld(
#                 keys=["label"]
#             ),
#             ToTensord(
#                 keys=["label"]
#             ),
#             AsDiscreted(
#                 keys=["label"],
#                 argmax=True
#             ),
#             ToNumpyd(
#                 keys=["label"]
#             ),
#             SqueezeDimd(
#                 keys=["label"],
#                 dim=0
#             )
#         ]
#     elif method.lower() == "raw":
#         transform = []

#     return transform


# def creating_transforms_training(foreground_crop_margin, label_interpolation_transform, num_patches_per_image, patch_size, scale_intensity_range, augmenations):
#     train_transforms = Compose(
#         [
#             LoadImaged(keys=["image", "label"]),
#             CorrectLabelAffined(keys=["image", "label"]),
#             EnsureChannelFirstd(keys=["image", "label"]),
#             Orientationd(keys=["image", "label"], axcodes="RAS"),
#             # CopyItemsd(keys=["label"], times=1, names=["label_before"]),
#             # Lambdad(keys="label_before", func=lambda x: np.array(np.sum(x))[None][None]),
#             CropForegroundd(
#                 keys=["image", "label"],
#                 source_key="image",
#                 select_fn=lambda x: x >= scale_intensity_range[0],
#                 margin=foreground_crop_margin
#             ),
#             # CopyItemsd(keys=["label"], times=1, names=["label_after"]),
#             # Lambdad(keys="label_after", func=lambda x: np.array(np.sum(x))[None][None]),
#             # ConcatItemsd(keys=["label_before", "label_after"], name="label_warning"),
#         ] +
#         label_interpolation_transform +
#         [
#             CastToTyped(
#                 keys=["image", "label"],
#                 dtype=(np.float16, np.uint8)
#             ),
#             # RandShiftIntensityd(
#             #     keys=["image"],
#             #     offsets=0.0,
#             #     prob=0.001
#             # ),
#             # image
#             CastToTyped(keys=["image"], dtype=(np.float32)),
#             # ThresholdIntensityd(keys=["image"], threshold=scale_intensity_range[0], above=True, cval=scale_intensity_range[0]),
#             # ThresholdIntensityd(keys=["image"], threshold=scale_intensity_range[1], above=False, cval=scale_intensity_range[1]),
#             # Lambdad(keys=["image"], func=lambda x: x - scale_intensity_range[0]),
#             # NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
#             ScaleIntensityRanged(
#                 keys=["image"],
#                 a_min=scale_intensity_range[0],
#                 a_max=scale_intensity_range[1],
#                 b_min=scale_intensity_range[2],
#                 b_max=scale_intensity_range[3],
#                 clip=True
#             ),
#             # label
#             # Lambdad(keys=["label"], func=lambda x: np.ceil(x.astype(np.float16) / 3.0).astype(np.uint8)),
#             SpatialPadd(
#                 keys=["image", "label"],
#                 spatial_size=patch_size,
#                 mode=["minimum", "constant"]
#             ),
#             RandCropByPosNegLabeld(
#                 keys=["image", "label"],
#                 label_key="label",
#                 spatial_size=patch_size,
#                 pos=1,
#                 neg=1,
#                 num_samples=num_patches_per_image
#             ),
#         ] + 
#             # RandSpatialCropd(
#             #     keys=["image", "label"],
#             #     roi_size=patch_size
#             # ),
#         augmenations + 
#         [
#             # RandZoomd(
#             #     keys=["image", "label"],
#             #     min_zoom=0.8,
#             #     max_zoom=1.2,
#             #     mode=("trilinear", "nearest"),
#             #     align_corners=(True, None),
#             #     prob=0.16,
#             # ),
#             # RandGaussianNoised(
#             #     keys=["image"],
#             #     std=0.01,
#             #     prob=0.15
#             # ),
#             # RandGaussianSmoothd(
#             #     keys=["image"],
#             #     sigma_x=(0.5, 1.15),
#             #     sigma_y=(0.5, 1.15),
#             #     sigma_z=(0.5, 1.15),
#             #     prob=0.15
#             # ),
#             # RandScaleIntensityd(
#             #     keys=["image"],
#             #     factors=0.3,
#             #     prob=0.15
#             # ),
#             # RandShiftIntensityd(
#             #     keys=["image"],
#             #     offsets=0.1,
#             #     prob=0.15
#             # ),
#             RandFlipd(
#                 keys=["image", "label"],
#                 spatial_axis=[0],
#                 prob=0.5
#             ),
#             RandFlipd(
#                 keys=["image", "label"],
#                 spatial_axis=[1],
#                 prob=0.5
#             ),
#             RandFlipd(
#                 keys=["image", "label"],
#                 spatial_axis=[2],
#             prob=0.5
#             ),
#             CastToTyped(
#                 keys=["image", "label"],
#                 dtype=(np.float32, np.uint8)
#             ),
#             ToTensord(
#                 keys=["image", "label"]
#             )
#         ]
#     )
#     return train_transforms


def creating_transforms_training(foreground_crop_margin, label_interpolation_transform, num_patches_per_image, patch_size, intensity_norm_transforms, augmenations, device, output_classes):
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            CorrectLabelAffined(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
        ] +
        label_interpolation_transform +
        [
            CastToTyped(
                keys=["image"],
                dtype=(torch.float32)
            ),
        ] +
        intensity_norm_transforms +
        [
            CastToTyped(
                keys=["image", "label"],
                dtype=(np.float16, np.uint8)
            ),
            CopyItemsd(
                keys=["label"],
                times=1,
                names=["label4crop"],
            ),
            Lambdad(
                keys=["label4crop"],
                func=lambda x: np.concatenate(tuple([ndimage.binary_dilation((x==_k).astype(x.dtype), iterations=48).astype(x.dtype) for _k in range(output_classes)]), axis=0),
                overwrite=True,
            ),
            EnsureTyped(
                keys=["image", "label"]
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.0,
                prob=0.001
            ),
            CastToTyped(keys=["image"], dtype=(torch.float32)),
            SpatialPadd(
                keys=["image", "label", "label4crop"],
                spatial_size=patch_size,
                mode=["reflect", "constant", "constant"]
            ),
            RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label4crop",
                num_classes=output_classes,
                ratios=[1,] * output_classes,
                spatial_size=patch_size,
                num_samples=num_patches_per_image
            ),
            Lambdad(keys=["label4crop"], func=lambda x: 0),
        ] + 
        augmenations + 
        [
            CastToTyped(
                keys=["image", "label"],
                dtype=(torch.float32, torch.uint8)
            ),
            ToTensord(
                keys=["image", "label"]
            )
        ]
    )
    return train_transforms


def creating_transforms_validation(foreground_crop_margin, label_interpolation_transform, patch_size, intensity_norm_transforms, device):
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            CorrectLabelAffined(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
        ] +
        label_interpolation_transform +
        [
            CastToTyped(
                keys=["image"],
                dtype=(torch.float32)
            ),
        ] +
        intensity_norm_transforms +
        [
            CastToTyped(
                keys=["image", "label"],
                dtype=(np.float16, np.uint8)
            ),
            EnsureTyped(
                keys=["image", "label"]
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.0,
                prob=0.001
            ),
            CastToTyped(
                keys=["image", "label"],
                dtype=(torch.float32, torch.uint8)
            ),
            ToTensord(
                keys=["image", "label"]
            )
        ]
    )
    return val_transforms


def creating_transforms_testing(foreground_crop_margin, intensity_norm_transforms, spacing):
    test_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            # CropForegroundd(keys=["image"], source_key="image", select_fn=lambda x: x >= intensity_range[0], margin=foreground_crop_margin),
            CastToTyped(keys=["image"], dtype=(np.float32)),
            Spacingd(keys=["image"], pixdim=spacing, mode=["bilinear"], align_corners=[True]),
        ] + 
        intensity_norm_transforms +
        [
            ToTensord(keys=["image"]),
        ]
    )
    return test_transforms
