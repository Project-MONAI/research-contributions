"""
Author: Duy-Phuong Dao
Email: phuongdd.1997@gmail.com (or duyphuongcri@gmail.com)
"""

import torch
import monai
from torch.utils.data import DataLoader

from monai.transforms import (
    Compose,
    LoadImaged,
    AddChanneld,
    SpatialPadd,
    ToTensord,
    RandRotated,
    RandZoomd,
    RandSpatialCropd,
    ConcatItemsd,
)

def get_transforms_normal(shape):
    train_target_transforms = Compose(
        [
            LoadImaged(keys=["image", "text"]),
            ToTensord(keys=["image", "text"]),
        ]
    )

    return train_target_transforms


def get_transforms_normal_seg(shape):
    train_target_transforms = Compose(
        [
            LoadImaged(keys=["image", "lobe", "airway", "vessel", "text"]),
            ToTensord(keys=["image", "lobe", "airway", "vessel", "text"]),
        ]
    )

    return train_target_transforms


def get_transforms_text():
    train_target_transforms = Compose(
        [
            LoadImaged(keys=["text"], reader='NumpyReader'),
            ToTensord(keys=["text"]),
        ]
    )

    return train_target_transforms


def get_transforms_aug_seg(shape, crop_shape):
    train_target_transforms = Compose(
        [
            LoadImaged(keys=["image", "lobe", "airway", "vessel", "text"]),
            AddChanneld(keys=["image", "lobe", "airway", "vessel"]),
            ConcatItemsd(keys=["image", "lobe", "airway", "vessel"], dim=0, name="image"),
            # RandGaussianNoised(keys=["image"], prob=0.5, std=0.1),
            RandZoomd(keys=["image"], prob=0.5, mode='area'),
            RandRotated(keys=["image"], prob=0.5, mode="bilinear", range_x=0.2, range_y=0.2,
                        range_z=0.2),
            SpatialPadd(keys=["image"], spatial_size=shape),
            RandSpatialCropd(
                keys=["image"], roi_size=crop_shape,
                max_roi_size=shape, random_center=True, random_size=False,
            ),
            ToTensord(keys=["image", "text"]),
        ]
    )

    return train_target_transforms


def get_transforms_seg_multiple():
    train_target_transforms = Compose(
        [
            LoadImaged(keys=["image", "image_sr", "lobe", "airway", "vessel", "text"]),
            AddChanneld(keys=["image_sr", "lobe", "airway", "vessel"]),
            ConcatItemsd(keys=["image_sr", "lobe", "airway", "vessel"], dim=0, name="image_sr"),
            ToTensord(keys=["image", "image_sr", "text"]),
        ]
    )

    return train_target_transforms


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    worker_info.dataset.transform.set_random_state(worker_info.seed % (2 ** 32))


def cache_transformed_text(train_files):
    train_transforms = get_transforms_text()
    train_ds = monai.data.CacheDataset(
        data=train_files, transform=train_transforms, cache_rate=0.0
    )
    return train_ds


def cache_transformed_train_data(train_files, shape):
    train_transforms = get_transforms_normal(shape)
    train_ds = monai.data.CacheDataset(
        data=train_files, transform=train_transforms, cache_rate=1.0
    )
    return train_ds


def cache_transformed_train_data_marginalize(train_files, shape):
    train_transforms = get_transforms_seg_multiple()
    train_ds = monai.data.CacheDataset(
        data=train_files, transform=train_transforms, cache_rate=0.0
    )
    return train_ds


def cache_transformed_train_data_seg(train_files, shape):
    train_transforms = get_transforms_normal_seg(shape)
    train_ds = monai.data.CacheDataset(
        data=train_files, transform=train_transforms, cache_rate=0.0
    )
    return train_ds


def cache_transformed_train_data_aug_seg(train_files, shape, crop_shape):
    train_transforms = get_transforms_aug_seg(shape, crop_shape)
    train_ds = monai.data.CacheDataset(
        data=train_files, transform=train_transforms, cache_rate=0.0
    )

    return train_ds
