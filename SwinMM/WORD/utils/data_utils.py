# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Any, Callable, MutableMapping, Sequence

import timm.data
import torch
import torch.utils.data
from utils import dataset_in_memory

from monai import data, transforms
from monai.data import load_decathlon_datalist
from monai.data.utils import list_data_collate


def get_dataset_kwargs(dataset_name: str, stage: str, use_normal_dataset: bool, args) -> MutableMapping[str, Any]:
    dataset_kwargs = {}
    if not use_normal_dataset:
        dataset_kwargs = dict(
            dataset_name=f"{stage}_{dataset_name}",
            hosts=[{"host": "localhost", "port": str(port)} for port in args.redis_ports],
            cluster_mode=True,
            capacity_per_node=200 * 1024 * 1024 * 1024,
            writer_buffer_size=0,  # Disable write buffer
        )
    return dataset_kwargs


def create_dataloader(
    data_files: Sequence[Any],
    is_testing: bool,
    transform: Callable,
    num_workers: int,
    is_distributed: bool,
    with_cache: bool,
    use_multi_epochs_loader: bool,
    batch_size: int,
    dataset_kwargs: MutableMapping[str, Any],
) -> torch.utils.data.DataLoader:
    if not with_cache:
        dataset = data.Dataset(data=data_files, transform=transform, **dataset_kwargs)
    else:
        dataset = dataset_in_memory.CachedDataset(data=data_files, transform=transform, **dataset_kwargs)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=not is_testing) if is_distributed else None

    loader_class = data.DataLoader
    if use_multi_epochs_loader:
        loader_class = timm.data.loader.MultiEpochsDataLoader
    loader = loader_class(
        dataset,
        batch_size=batch_size,
        shuffle=False if is_distributed or is_testing else True,
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True,
        persistent_workers=True,
        # NOTE(meijieru): otherwise `too many open`
        collate_fn=list_data_collate,
    )
    return loader


def get_loader(args):
    data_dir = args.data_dir
    datalist_json = os.path.join(data_dir, args.json_list)
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            # transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            # transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(keys="image", pixdim=(args.space_x, args.space_y, args.space_z), mode="bilinear"),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    def _get_subset_loader(subset_name: str, transform: Callable, is_testing: bool, use_normal_dataset: bool):
        datalist = load_decathlon_datalist(datalist_json, True, subset_name, base_dir=data_dir)
        dataset_kwargs = get_dataset_kwargs(subset_name, "finetune", use_normal_dataset, args)
        loader = create_dataloader(
            datalist,
            is_testing,
            transform,
            args.workers,
            args.distributed,
            not use_normal_dataset,
            not args.nouse_multi_epochs_loader,
            args.batch_size,
            dataset_kwargs,
        )
        return loader

    if args.test_mode:
        # Never cache as only go through once.
        test_files = load_decathlon_datalist(datalist_json, True, "testing", base_dir=data_dir)
        test_ds = data.Dataset(data=test_files, transform=test_transform)
        test_sampler = torch.utils.data.DistributedSampler(test_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=test_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        loader = test_loader
    else:
        train_loader = _get_subset_loader("training", train_transform, False, args.use_normal_dataset)
        val_loader = _get_subset_loader("validation", val_transform, True, args.use_normal_dataset_val)

        if args.unsupervised:
            unsupervised_loader = _get_subset_loader("unsupervised", train_transform, False, args.use_normal_dataset)
            loader = [train_loader, val_loader, unsupervised_loader]
        else:
            loader = [train_loader, val_loader]

    return loader
