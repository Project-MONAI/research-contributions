#!/usr/bin/env python

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


def str2aug(str_aug):

    if str_aug == "RandShiftIntensity":
        aug = RandShiftIntensityd(
                  keys=["image"],
                  offsets=0.1,
                  prob=0.1
              )
    elif str_aug == "RandFlip":
        aug = RandFlipd(
                  keys=["image", "label"],
                  prob=0.5
              )
    elif str_aug == "RandRotate90":
        aug = RandRotate90d(
                  keys=["image", "label"],
                  prob=0.1
              )
    else:
        aug = None

    return aug


def creating_label_interpolation_transform(method, spacing, output_classes):
    if method.lower() == "nearest":
        transform = [
            Spacingd(
                keys=["image", "label"],
                pixdim=spacing,
                mode=("bilinear", "nearest"),
                align_corners=(True, True)
                )
        ]
    elif method.lower() == "linear":
        transform = [
            # multi-class to one-hot
            AddChanneld(
                keys=["label"]
            ),
            ToTensord(
                keys=["label"]
            ),
            AsDiscreted(
                keys=["label"],
                to_onehot=True,
                n_classes=output_classes
            ),
            ToNumpyd(
                keys=["label"]
            ),
            SqueezeDimd(
                keys=["label"],
                dim=0
            ),
            CastToTyped(
                keys=["image", "label"],
                dtype=(np.float16, np.float16)
            ),
            # re-sampling
            Spacingd(
                keys=["image", "label"],
                pixdim=spacing,
                mode=("bilinear", "bilinear"),
                align_corners=(True, True)
            ),
            # one-hot to multi-class
            AddChanneld(
                keys=["label"]
            ),
            ToTensord(
                keys=["label"]
            ),
            AsDiscreted(
                keys=["label"],
                argmax=True
            ),
            ToNumpyd(
                keys=["label"]
            ),
            SqueezeDimd(
                keys=["label"],
                dim=0
            )
        ]
    elif method.lower() == "raw":
        transform = []

    return transform


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


def creating_transforms_training(foreground_crop_margin, label_interpolation_transform, num_patches_per_image, patch_size, intensity_range, intensity_norm_transforms, augmenations, device, output_classes):
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            CorrectLabelAffined(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # CropForegroundd(
            #     keys=["image", "label"],
            #     source_key="image",
            #     select_fn=lambda x: x >= intensity_range[0],
            #     margin=foreground_crop_margin
            # ),
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
                func=lambda x: np.concatenate(tuple([np.ones(shape=x.shape, dtype=x.dtype),] + [ndimage.binary_dilation((x==_k).astype(x.dtype), iterations=48).astype(x.dtype) for _k in range(1, output_classes)]), axis=0),
                overwrite=True,
            ),
            # Lambdad(
            #     keys=["label4crop"],
            #     func=lambda x: print(x.shape, x.dtype),
            #     overwrite=False,
            # ),
            EnsureTyped(
                keys=["image", "label"]
            ),
            # ToDeviced(
            #     keys=["image", "label"],
            #     device=device,
            # ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.0,
                prob=0.001
            ),
            # image
            CastToTyped(keys=["image"], dtype=(torch.float32)),
            # ThresholdIntensityd(keys=["image"], threshold=scale_intensity_range[0], above=True, cval=scale_intensity_range[0]),
            # ThresholdIntensityd(keys=["image"], threshold=scale_intensity_range[1], above=False, cval=scale_intensity_range[1]),
            # Lambdad(keys=["image"], func=lambda x: x - scale_intensity_range[0]),
            # NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            # label
            # Lambdad(keys=["label"], func=lambda x: np.ceil(x.astype(np.float16) / 3.0).astype(np.uint8)),
            SpatialPadd(
                keys=["image", "label"],
                spatial_size=patch_size,
                mode=["reflect", "constant"]
            ),
            RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label4crop",
                num_classes=output_classes,
                ratios=[1,] * output_classes,
                spatial_size=patch_size,
                num_samples=num_patches_per_image
            ),
            # RandCropByPosNegLabeld(
            #     keys=["image", "label"],
            #     label_key="label",
            #     spatial_size=patch_size,
            #     pos=1,
            #     neg=1,
            #     num_samples=num_patches_per_image
            # ),
            Lambdad(keys=["label4crop"], func=lambda x: 0),
        ] + 
        augmenations + 
        [
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=0.15
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.15
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=0.15
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
    return train_transforms


# def creating_transforms_validation(foreground_crop_margin, label_interpolation_transform, patch_size, scale_intensity_range):
#     val_transforms = Compose(
#         [
#             LoadImaged(keys=["image", "label"]),
#             CorrectLabelAffined(keys=["image", "label"]),
#             EnsureChannelFirstd(keys=["image", "label"]),
#             Orientationd(keys=["image", "label"], axcodes="RAS"),
#             CropForegroundd(
#                 keys=["image", "label"],
#                 source_key="image",
#                 select_fn=lambda x: x >= scale_intensity_range[0],
#                 margin=foreground_crop_margin
#             ),
#         ] +
#         label_interpolation_transform +
#         [CastToTyped(
#             keys=["image", "label"],
#             dtype=(np.float16, np.uint8)
#         ),
#         # RandShiftIntensityd(
#         #     keys=["image"],
#         #     offsets=0.0,
#         #     prob=0.001
#         # ),
#         # image
#         CastToTyped(keys=["image"], dtype=(np.float32)),
#         # ThresholdIntensityd(keys=["image"], threshold=scale_intensity_range[0], above=True, cval=scale_intensity_range[0]),
#         # ThresholdIntensityd(keys=["image"], threshold=scale_intensity_range[1], above=False, cval=scale_intensity_range[1]),
#         # Lambdad(keys=["image"], func=lambda x: x - scale_intensity_range[0]),
#         # NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
#         ScaleIntensityRanged(
#             keys=["image"],
#             a_min=scale_intensity_range[0],
#             a_max=scale_intensity_range[1],
#             b_min=scale_intensity_range[2],
#             b_max=scale_intensity_range[3],
#             clip=True
#         ),
#         # label
#         # Lambdad(keys=["label"], func=lambda x: np.ceil(x.astype(np.float16) / 3.0).astype(np.uint8)),
#         SpatialPadd(
#             keys=["image", "label"],
#             spatial_size=patch_size,
#             mode=["minimum", "constant"]
#         ),
#         CastToTyped(
#             keys=["image", "label"],
#             dtype=(np.float32, np.uint8)
#         ),
#         ToTensord(
#             keys=["image", "label"]
#         )
#     ])
#     return val_transforms


def creating_transforms_validation(foreground_crop_margin, label_interpolation_transform, patch_size, intensity_range, intensity_norm_transforms, device):
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            CorrectLabelAffined(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                select_fn=lambda x: x >= intensity_range[0],
                margin=foreground_crop_margin
            ),
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
            # ToDeviced(
            #     keys=["image", "label"],
            #     device=device,
            # ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.0,
                prob=0.001
            ),
            # image
            CastToTyped(keys=["image"], dtype=(torch.float32)),
            # ThresholdIntensityd(keys=["image"], threshold=scale_intensity_range[0], above=True, cval=scale_intensity_range[0]),
            # ThresholdIntensityd(keys=["image"], threshold=scale_intensity_range[1], above=False, cval=scale_intensity_range[1]),
            # Lambdad(keys=["image"], func=lambda x: x - scale_intensity_range[0]),
            # NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            # SpatialPadd(
            #     keys=["image", "label"],
            #     spatial_size=patch_size,
            #     mode=["minimum", "constant"]
            # ),
            # DivisiblePadd(
            #     keys=["image", "label"],
            #     k=32,
            #     mode=["minimum", "constant"]
            # ),
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


def creating_transforms_testing_legacy(foreground_crop_margin, label_interpolation_transform, scale_intensity_range):
    test_transforms = Compose(
        [
            LoadImaged(
                keys=["image", "label"],
                as_closest_canonical=True
            ),
            CustomAddChanneld(
                keys=["image", "label"]
            )
        ] +
        label_interpolation_transform +
        [
            CustomCropForegroundd(
                keys=["image", "label"],
                source_key="label",
                margin=foreground_crop_margin
            ),
            CastToTyped(
                keys=["image", "label"],
                dtype=(np.float32, np.uint8)
            ),
            # RandShiftIntensityd(
            #     keys=["image"],
            #     offsets=0.0,
            #     prob=0.001
            # ),
            CastToTyped(keys=["image"], dtype=(np.float32)),
            # ThresholdIntensityd(keys=["image"], threshold=scale_intensity_range[0], above=True, cval=scale_intensity_range[0]),
            # ThresholdIntensityd(keys=["image"], threshold=scale_intensity_range[1], above=False, cval=scale_intensity_range[1]),
            # Lambdad(keys=["image"], func=lambda x: x - scale_intensity_range[0]),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            CastToTyped(
                keys=["image", "label"],
                dtype=(np.float32, np.uint8)
            ),
            ToTensord(
                keys=["image", "label"]
            )
        ]
    )

    return test_transforms


# def creating_transforms_testing(foreground_crop_margin, scale_intensity_range, spacing):
#     s_range = scale_intensity_range
#     test_transforms = Compose(
#         [
#             LoadImaged(keys=["image"]),
#             EnsureChannelFirstd(keys=["image"]),
#             Orientationd(keys=["image"], axcodes="RAS"),
#             CropForegroundd(keys=["image"], source_key="image", select_fn=lambda x: x >= s_range[0], margin=foreground_crop_margin),
#             CastToTyped(keys=["image"], dtype=(np.float32)),
#             Spacingd(keys=["image"], pixdim=spacing, mode=["bilinear"], align_corners=[True]),
#             ScaleIntensityRanged(keys=["image"], a_min=s_range[0], a_max=s_range[1], b_min=s_range[2], b_max=s_range[3], clip=True),
#             ToTensord(keys=["image"])
#         ]
#     )
#     return test_transforms


def creating_transforms_testing(foreground_crop_margin, intensity_range, intensity_norm_transforms, spacing):
    test_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            CropForegroundd(keys=["image"], source_key="image", select_fn=lambda x: x >= intensity_range[0], margin=foreground_crop_margin),
            CastToTyped(keys=["image"], dtype=(np.float32)),
            Spacingd(keys=["image"], pixdim=spacing, mode=["bilinear"], align_corners=[True]),
        ] + 
        intensity_norm_transforms +
        [
            ToTensord(keys=["image"]),
        ]
    )
    return test_transforms


def creating_transforms_offline_validation(keys):
    transforms_offline_validation = Compose(
        [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
            ToTensord(keys=keys)
        ]
    )
    return transforms_offline_validation


def creating_transforms_ensemble(keys):
    transforms_ensemble = Compose(
        [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            ToTensord(keys=keys)
        ]
    )
    return transforms_ensemble


class CustomCropForegroundd(MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.CropForeground`.
    Crop only the foreground object of the expected images.
    The typical usage is to help training and evaluation if the valid part is small in the whole medical image.
    The valid part can be determined by any field in the data with `source_key`, for example:
    - Select values > 0 in image field as the foreground and crop on all fields specified by `keys`.
    - Select label = 3 in label field as the foreground to crop on all fields specified by `keys`.
    - Select label > 0 in the third channel of a One-Hot label field as the foreground to crop all `keys` fields.
    Users can define arbitrary function to select expected foreground from the whole source image or specified
    channels. And it can also add margin to every dim of the bounding box of foreground object.
    """

    def __init__(
        self,
        keys: KeysCollection,
        source_key: str,
        select_fn: Callable = lambda x: x > 0,
        channel_indices: Optional[IndexSelection] = None,
        margin: int = 0,
        start_coord_key: str = "foreground_start_coord",
        end_coord_key: str = "foreground_end_coord",
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            source_key: data source to generate the bounding box of foreground, can be image or label, etc.
            select_fn: function to select expected foreground, default is to select values > 0.
            channel_indices: if defined, select foreground only on the specified channels
                of image. if None, select foreground on the whole image.
            margin: add margin value to spatial dims of the bounding box, if only 1 value provided, use it for all dims.
            start_coord_key: key to record the start coordinate of spatial bounding box for foreground.
            end_coord_key: key to record the end coordinate of spatial bounding box for foreground.
        """
        super().__init__(keys)
        self.source_key = source_key
        self.select_fn = select_fn
        self.channel_indices = ensure_tuple(channel_indices) if channel_indices is not None else None
        self.margin = margin
        self.start_coord_key = start_coord_key
        self.end_coord_key = end_coord_key

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        box_start, box_end = generate_spatial_bounding_box(
            d[self.source_key], self.select_fn, self.channel_indices, self.margin
        )
        d[self.start_coord_key] = box_start
        d[self.end_coord_key] = box_end
        cropper = SpatialCrop(roi_start=box_start, roi_end=box_end)
        for key in self.keys:
            d["shape_before_cropping"] = d[key].shape
            d[key] = cropper(d[key])
        return d


class CustomAddChanneld(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AddChannel`.
    """

    def __init__(self, keys: KeysCollection) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
        """
        super().__init__(keys)
        self.adder = AddChannel()

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.adder(d[key])
            d[key + "_orig_shape"] = d[key].shape
        return d
