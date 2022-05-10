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
from scipy import ndimage

from monai.transforms import (
    CastToTyped,
    Compose,
    ConcatItemsd,
    CopyItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    KeepLargestConnectedComponent,
    Lambdad,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByLabelClassesd,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    RandSpatialCropSamplesd,
    RandZoomd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    SqueezeDimd,
    ToDeviced,
    ToNumpyd,
    ToTensord,
)
from monai.transforms.transform import MapTransform


class CorrectLabelAffined(MapTransform):
    def __init__(self, keys) -> None:
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        d["label_meta_dict"]["affine"] = d["image_meta_dict"]["affine"]
        return d


def creating_transforms_training(
    foreground_crop_margin,
    label_interpolation_transform,
    num_patches_per_image,
    patch_size,
    intensity_norm_transforms,
    augmenations,
    device,
    output_classes,
):
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            CorrectLabelAffined(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
        ]
        + label_interpolation_transform
        + [CastToTyped(keys=["image"], dtype=(torch.float32))]
        + intensity_norm_transforms
        + [
            CastToTyped(keys=["image", "label"], dtype=(np.float16, np.uint8)),
            CopyItemsd(keys=["label"], times=1, names=["label4crop"]),
            Lambdad(
                keys=["label4crop"],
                func=lambda x: np.concatenate(
                    tuple(
                        [
                            ndimage.binary_dilation((x == _k).astype(x.dtype), iterations=48).astype(x.dtype)
                            for _k in range(output_classes)
                        ]
                    ),
                    axis=0,
                ),
                overwrite=True,
            ),
            EnsureTyped(keys=["image", "label"]),
            RandShiftIntensityd(keys=["image"], offsets=0.0, prob=0.001),
            CastToTyped(keys=["image"], dtype=(torch.float32)),
            SpatialPadd(
                keys=["image", "label", "label4crop"], spatial_size=patch_size, mode=["reflect", "constant", "constant"]
            ),
            RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label4crop",
                num_classes=output_classes,
                ratios=[1] * output_classes,
                spatial_size=patch_size,
                num_samples=num_patches_per_image,
            ),
            Lambdad(keys=["label4crop"], func=lambda x: 0),
        ]
        + augmenations
        + [CastToTyped(keys=["image", "label"], dtype=(torch.float32, torch.uint8)), ToTensord(keys=["image", "label"])]
    )
    return train_transforms


def creating_transforms_validation(
    foreground_crop_margin, label_interpolation_transform, patch_size, intensity_norm_transforms, device
):
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            CorrectLabelAffined(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
        ]
        + label_interpolation_transform
        + [CastToTyped(keys=["image"], dtype=(torch.float32))]
        + intensity_norm_transforms
        + [
            CastToTyped(keys=["image", "label"], dtype=(np.float16, np.uint8)),
            EnsureTyped(keys=["image", "label"]),
            RandShiftIntensityd(keys=["image"], offsets=0.0, prob=0.001),
            CastToTyped(keys=["image", "label"], dtype=(torch.float32, torch.uint8)),
            ToTensord(keys=["image", "label"]),
        ]
    )
    return val_transforms


def creating_transforms_testing(foreground_crop_margin, intensity_norm_transforms, spacing):
    test_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            CastToTyped(keys=["image"], dtype=(np.float32)),
            Spacingd(keys=["image"], pixdim=spacing, mode=["bilinear"], align_corners=[True]),
        ]
        + intensity_norm_transforms
        + [ToTensord(keys=["image"])]
    )
    return test_transforms
