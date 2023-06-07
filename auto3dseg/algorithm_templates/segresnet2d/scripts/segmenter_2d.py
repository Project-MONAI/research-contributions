# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import copy
import csv
import logging
import os
import shutil
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import psutil
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from monai.apps.auto3dseg.transforms import EnsureSameShaped
from monai.bundle.config_parser import ConfigParser
from monai.config import KeysCollection
from monai.inferers import SlidingWindowInfererAdapt

# from monai.optimizers.lr_scheduler import WarmupCosineSchedule
from monai.transforms import (
    AsDiscreted,
    CastToTyped,
    ClassesToIndicesd,
    Compose,
    ConcatItemsd,
    CopyItemsd,
    CropForegroundd,
    DataStatsd,
    DeleteItemsd,
    EnsureTyped,
    Identityd,
    Invertd,
    Lambdad,
    LoadImaged,
    NormalizeIntensityd,
    RandAffined,
    RandCropByLabelClassesd,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    ResampleToMatchd,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
)
from monai.transforms.transform import MapTransform
from monai.utils import MetricReduction, convert_to_dst_type, optional_import, set_determinism

if __package__ in (None, ""):
    from segmenter import DataTransformBuilder, Segmenter
else:
    from .segmenter import Segmenter, DataTransformBuilder

from monai.apps.auto3dseg.auto_runner import logger

print = logger.debug


class WrappedModel2D(torch.nn.Module):
    def __init__(self, net, memory_format=torch.preserve_format):
        super().__init__()
        self.net = net
        self.memory_format = memory_format
        print("WrappedModel2D is initialized")

        self.net = self.net.to(memory_format=memory_format)

    def reshape2D(self, x):
        return x.permute(0, 4, 1, 2, 3).reshape(-1, x.shape[1], x.shape[2], x.shape[3])

    def reshape3D(self, x, sh):
        return x.reshape(sh[0], sh[4], x.shape[1], x.shape[2], x.shape[3]).permute(0, 2, 3, 4, 1)

    def forward(self, x: torch.Tensor):
        # print('WrappedModel2D input', x.shape,  'local', self.memory_format)
        sh = x.shape
        x = self.reshape2D(x)

        x = x.to(memory_format=self.memory_format)
        x = self.net(x)

        if isinstance(x, (list, tuple)):
            x = [self.reshape3D(x0, sh) for x0 in x]
            # print('end reshaped loop', x[0].shape)
        else:
            x = self.reshape3D(x, sh)
            # print('end reshaped', x.shape)

        return x


class DataTransformBuilder2D(DataTransformBuilder):
    def get_resample_transforms(self, resample_label=True):
        ts = self.get_custom("resample_transforms")
        if len(ts) > 0:
            return ts

        keys = [self.image_key, self.label_key] if resample_label else [self.image_key]
        mode = ["bilinear", "nearest"] if resample_label else ["bilinear"]
        extra_keys = list(self.extra_modalities)

        if self.extra_options.get("crop_foreground", False) and len(extra_keys) == 0:
            ts.append(
                CropForegroundd(
                    keys=keys, source_key=self.image_key, allow_missing_keys=True, margin=10, allow_smaller=True
                )
            )

        if self.resample:
            if self.resample_resolution is None:
                raise ValueError("resample_resolution is not provided")

            pixdim = self.resample_resolution
            ts.append(
                Spacingd(
                    keys=keys,
                    pixdim=pixdim[:2] + [-1],
                    mode=mode,
                    dtype=torch.float,
                    min_pixdim=np.array(pixdim) * 0.75,
                    max_pixdim=np.array(pixdim) * 1.25,
                    allow_missing_keys=True,
                )
            )

            if resample_label:
                ts.append(
                    EnsureSameShaped(
                        keys=self.label_key, source_key=self.image_key, allow_missing_keys=True, warn=self.debug
                    )
                )

        for extra_key in extra_keys:
            ts.append(ResampleToMatchd(keys=extra_key, key_dst=self.image_key, dtype=np.float32))

        ts.extend(self.get_custom("after_resample_transforms", resample_label=resample_label))

        return ts

    def get_augment_transforms(self):
        ts = self.get_custom("augment_transforms")
        if len(ts) > 0:
            return ts

        if self.roi_size is None:
            raise ValueError("roi_size is not specified")

        ts = []
        ts.append(
            RandAffined(
                keys=[self.image_key, self.label_key],
                prob=0.2,
                rotate_range=[0, 0, 0.26],
                scale_range=[0.2, 0.2, 0],
                mode=["bilinear", "nearest"],
                spatial_size=self.roi_size,
                cache_grid=True,
                padding_mode="border",
            )
        )
        ts.append(
            RandGaussianSmoothd(keys=self.image_key, prob=0.2, sigma_x=[0.5, 1.0], sigma_y=[0.5, 1.0], sigma_z=[0, 0])
        )
        ts.append(RandFlipd(keys=[self.image_key, self.label_key], prob=0.5, spatial_axis=0))
        ts.append(RandFlipd(keys=[self.image_key, self.label_key], prob=0.5, spatial_axis=1))
        ts.append(RandScaleIntensityd(keys=self.image_key, prob=0.5, factors=0.3))
        ts.append(RandShiftIntensityd(keys=self.image_key, prob=0.5, offsets=0.1))
        ts.append(RandGaussianNoised(keys=self.image_key, prob=0.2, mean=0.0, std=0.1))

        ts.extend(self.get_custom("after_augment_transforms"))

        return ts


class Segmenter2D(Segmenter):
    def __init__(
        self,
        config_file: Optional[Union[str, Sequence[str]]] = None,
        config_dict: Dict = {},
        rank: int = 0,
        global_rank: int = 0,
    ) -> None:
        super().__init__(config_file=config_file, config_dict=config_dict, rank=rank, global_rank=global_rank)

        config = self.config

        if config.get("sliding_inferrer") is not None:
            self.sliding_inferrer = ConfigParser(config["sliding_inferrer"]).get_parsed_content()
        else:
            self.sliding_inferrer = SlidingWindowInfererAdapt(
                roi_size=config["roi_size"],
                sw_batch_size=1,
                overlap=[0.625, 0.625, 0],
                mode="gaussian",
                cache_roi_weight_map=False,
                progress=False,
            )

    # change model to be wrapped 2D

    def setup_model(self, pretrained_ckpt_name=None):
        config = self.config

        memory_format = torch.channels_last if self.config["channels_last"] else torch.preserve_format
        self.config["channels_last"] = False

        model = ConfigParser(config["network"]).get_parsed_content()

        model = WrappedModel2D(model, memory_format=memory_format)  # wrap in 2D

        if self.global_rank == 0:
            print(str(model))

        if pretrained_ckpt_name is not None:
            self.checkpoint_load(ckpt=pretrained_ckpt_name, model=model)

        model = model.to(self.device)

        if self.distributed and not config["infer"]["enabled"]:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DistributedDataParallel(
                module=model, device_ids=[self.rank], output_device=self.rank, find_unused_parameters=False
            )

        if self.global_rank == 0:
            pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total parameters count: {pytorch_total_params} distributed: {self.distributed}")

        return model

    # change augmentations to be 2D
    def get_data_transform_builder(self):
        if self._data_transform_builder is None:
            config = self.config
            custom_transforms = self.get_custom_transforms()

            self._data_transform_builder = DataTransformBuilder2D(
                roi_size=config["roi_size"],
                resample=config["resample"],
                resample_resolution=config["resample_resolution"],
                normalize_mode=config["normalize_mode"],
                normalize_params={
                    "intensity_bounds": config["intensity_bounds"],
                    "label_dtype": torch.uint8 if config["input_channels"] < 255 else torch.int16,
                },
                crop_mode=config["crop_mode"],
                crop_params={
                    "output_classes": config["output_classes"],
                    "crop_ratios": config["crop_ratios"],
                    "cache_class_indices": config["cache_class_indices"],
                    "num_crops_per_image": config["num_crops_per_image"],
                    "max_samples_per_class": config["max_samples_per_class"],
                },
                extra_modalities=config["extra_modalities"],
                custom_transforms=custom_transforms,
                crop_foreground=config.get("crop_foreground", True),
            )

        return self._data_transform_builder


def run_segmenter_worker(rank=0, config_file: Optional[Union[str, Sequence[str]]] = None, override: Dict = {}):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    dist_available = dist.is_available()
    global_rank = rank

    if type(config_file) == str and "," in config_file:
        config_file = config_file.split(",")

    if dist_available:
        mgpu = override.get("mgpu", None)
        if mgpu is not None:
            logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.WARNING)
            dist.init_process_group(backend="nccl", rank=rank, **mgpu)  # we spawn this process
            mgpu.update({"rank": rank, "global_rank": rank})
            if rank == 0:
                print(f"Distributed: initializing multi-gpu tcp:// process group {mgpu}")

        elif dist_launched():
            rank = int(os.getenv("LOCAL_RANK"))
            global_rank = int(os.getenv("RANK"))
            world_size = int(os.getenv("LOCAL_WORLD_SIZE"))
            logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.WARNING)
            dist.init_process_group(backend="nccl", init_method="env://")  # torchrun spawned it
            override["mgpu"] = {"world_size": world_size, "rank": rank, "global_rank": global_rank}

            print(f"Distributed launched: initializing multi-gpu env:// process group {override['mgpu']}")

    segmenter = Segmenter2D(config_file=config_file, config_dict=override, rank=rank, global_rank=global_rank)
    best_metric = segmenter.run()
    segmenter = None

    if dist_available and dist.is_initialized():
        dist.destroy_process_group()

    return best_metric


def dist_launched() -> bool:
    return dist.is_torchelastic_launched() or (
        os.getenv("NGC_ARRAY_SIZE") is not None and int(os.getenv("NGC_ARRAY_SIZE")) > 1
    )


def run_segmenter(config_file: Optional[Union[str, Sequence[str]]] = None, **kwargs):
    """
    if multiple gpu available, start multiprocessing for all gpus
    """

    nprocs = torch.cuda.device_count()

    if nprocs > 1 and not dist_launched():
        print("Manually spawning processes {nprocs}")
        kwargs["mgpu"] = {"world_size": nprocs, "init_method": kwargs.get("init_method", "tcp://127.0.0.1:23456")}
        torch.multiprocessing.spawn(run_segmenter_worker, nprocs=nprocs, args=(config_file, kwargs))
    else:
        print("Not spawning processes, dist is already launched {nprocs}")
        run_segmenter_worker(0, config_file, kwargs)


if __name__ == "__main__":
    fire, fire_is_imported = optional_import("fire")
    if fire_is_imported:
        fire.Fire(run_segmenter)
    else:
        warnings.warn("Fire commandline parser cannot be imported, using options from config/hyper_parameters.yaml")
        run_segmenter(config_file="config/hyper_parameters.yaml")
