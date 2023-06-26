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

from __future__ import annotations

import copy
import csv
import gc
import logging
import multiprocessing as mp
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
import torch.multiprocessing as mp
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from monai.apps.auto3dseg.auto_runner import logger
from monai.apps.auto3dseg.transforms import EnsureSameShaped
from monai.auto3dseg.utils import datafold_read
from monai.bundle.config_parser import ConfigParser
from monai.config import KeysCollection
from monai.data import CacheDataset, DataLoader, Dataset, DistributedSampler, decollate_batch, list_data_collate
from monai.inferers import SlidingWindowInfererAdapt
from monai.losses import DeepSupervisionLoss
from monai.metrics import CumulativeAverage, DiceHelper
from monai.networks.layers.factories import split_args
from monai.optimizers.lr_scheduler import WarmupCosineSchedule
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
    ToDeviced,
)
from monai.transforms.transform import MapTransform
from monai.utils import ImageMetaKey, convert_to_dst_type, optional_import, set_determinism

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048"


print = logger.debug
tqdm, has_tqdm = optional_import("tqdm", name="tqdm")

if __package__ in (None, ""):
    from utils import auto_adjust_network_settings, logger_configure
else:
    from .utils import auto_adjust_network_settings, logger_configure


class LabelEmbedClassIndex(MapTransform):
    """
    Label embedding according to class_index
    """

    def __init__(
        self, keys: KeysCollection = "label", allow_missing_keys: bool = False, class_index: Optional[List] = None
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be compared to the source_key item shape.
            allow_missing_keys: do not raise exception if key is missing.
            class_index: a list of class indices
        """
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.class_index = class_index

    def label_mapping(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        return torch.cat([sum([x == i for i in c]) for c in self.class_index], dim=0).to(dtype=dtype)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        if self.class_index is not None:
            for key in self.key_iterator(d):
                d[key] = self.label_mapping(d[key])
        return d


def schedule_validation_epochs(num_epochs, num_epochs_per_validation=None, fraction=0.16) -> list:
    """
    Schedule of epochs to validate (progressively more frequently)
        num_epochs - total number of epochs
        num_epochs_per_validation - if provided use a linear schedule with this step
        init_step
    """

    if num_epochs_per_validation is None:
        x = (np.sin(np.linspace(0, np.pi / 2, max(10, int(fraction * num_epochs)))) * num_epochs).astype(int)
        x = np.cumsum(np.sort(np.diff(np.unique(x)))[::-1])
        x[-1] = num_epochs
        x = x.tolist()
    else:
        if num_epochs_per_validation >= num_epochs:
            x = [num_epochs_per_validation]
        else:
            x = list(range(num_epochs_per_validation, num_epochs, num_epochs_per_validation))

    if len(x) == 0:
        x = [0]

    return x


class DataTransformBuilder:
    def __init__(
        self,
        roi_size: list,
        image_key: str = "image",
        label_key: str = "label",
        resample: bool = False,
        resample_resolution: Optional[list] = None,
        normalize_mode: str = "meanstd",
        normalize_params: Optional[dict] = None,
        crop_mode: str = "ratio",
        crop_params: Optional[dict] = None,
        extra_modalities: Optional[dict] = None,
        custom_transforms=None,
        debug: bool = False,
        rank: int = 0,
        **kwargs,
    ) -> None:
        self.roi_size, self.image_key, self.label_key = roi_size, image_key, label_key

        self.resample, self.resample_resolution = resample, resample_resolution
        self.normalize_mode = normalize_mode
        self.normalize_params = normalize_params if normalize_params is not None else {}
        self.crop_mode = crop_mode
        self.crop_params = crop_params if crop_params is not None else {}

        self.extra_modalities = extra_modalities if extra_modalities is not None else {}
        self.custom_transforms = custom_transforms if custom_transforms is not None else {}

        self.extra_options = kwargs
        self.debug = debug
        self.rank = rank

    def get_custom(self, name, **kwargs):
        tr = []
        for t in self.custom_transforms.get(name, []):
            if isinstance(t, dict):
                t.update(kwargs)
                t = ConfigParser(t).get_parsed_content(instantiate=True)
            tr.append(t)

        return tr

    def get_load_transforms(self):
        ts = self.get_custom("load_transforms")
        if len(ts) > 0:
            return ts

        keys = [self.image_key, self.label_key] + list(self.extra_modalities)
        ts.append(
            LoadImaged(keys=keys, ensure_channel_first=True, dtype=None, allow_missing_keys=True, image_only=True)
        )
        ts.append(EnsureTyped(keys=keys, data_type="tensor", dtype=torch.float, allow_missing_keys=True))
        ts.append(
            EnsureSameShaped(keys=self.label_key, source_key=self.image_key, allow_missing_keys=True, warn=self.debug)
        )

        ts.extend(self.get_custom("after_load_transforms"))

        return ts

    def get_resample_transforms(self, resample_label=True):
        ts = self.get_custom("resample_transforms", resample_label=resample_label)
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
                    pixdim=pixdim,
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

    def get_normalize_transforms(self):
        ts = self.get_custom("normalize_transforms")
        if len(ts) > 0:
            return ts

        modalities = {self.image_key: self.normalize_mode}
        modalities.update(self.extra_modalities)

        for key, normalize_mode in modalities.items():
            if normalize_mode == "none":
                pass
            elif normalize_mode in ["range", "ct"]:
                intensity_bounds = self.normalize_params.get("intensity_bounds", None)
                if intensity_bounds is None:
                    intensity_bounds = [-250, 250]
                    warnings.warn(f"intensity_bounds is not specified, assuming {intensity_bounds}")

                ts.append(
                    ScaleIntensityRanged(
                        keys=key, a_min=intensity_bounds[0], a_max=intensity_bounds[1], b_min=-1, b_max=1, clip=False
                    )
                )
                ts.append(Lambdad(keys=key, func=lambda x: torch.sigmoid(x)))
            elif normalize_mode in ["meanstd", "mri"]:
                ts.append(NormalizeIntensityd(keys=key, nonzero=True, channel_wise=True))
            elif normalize_mode in ["pet"]:
                ts.append(Lambdad(keys=key, func=lambda x: torch.sigmoid((x - x.min()) / x.std())))
            else:
                raise ValueError("Unsupported normalize_mode" + str(self.normalize_mode))

        if len(self.extra_modalities) > 0:
            ts.append(ConcatItemsd(keys=list(modalities), name=self.image_key))  # concat
            ts.append(DeleteItemsd(keys=list(self.extra_modalities)))  # release memory

        label_dtype = self.normalize_params.get("label_dtype", None)
        if label_dtype is not None:
            ts.append(CastToTyped(keys=self.label_key, dtype=label_dtype, allow_missing_keys=True))

        ts.extend(self.get_custom("after_normalize_transforms"))
        return ts

    def get_crop_transforms(self):
        ts = self.get_custom("crop_transforms")
        if len(ts) > 0:
            return ts

        if self.roi_size is None:
            raise ValueError("roi_size is not specified")

        keys = [self.image_key, self.label_key]
        ts = []
        ts.append(SpatialPadd(keys=keys, spatial_size=self.roi_size))

        if self.crop_mode == "ratio":
            output_classes = self.crop_params.get("output_classes", None)
            if output_classes is None:
                raise ValueError("crop_params option output_classes must be specified")

            crop_ratios = self.crop_params.get("crop_ratios", None)
            cache_class_indices = self.crop_params.get("cache_class_indices", False)
            max_samples_per_class = self.crop_params.get("max_samples_per_class", None)
            if max_samples_per_class <= 0:
                max_samples_per_class = None
            indices_key = None

            if cache_class_indices:
                ts.append(
                    ClassesToIndicesd(
                        keys=self.label_key,
                        num_classes=output_classes,
                        indices_postfix="_cls_indices",
                        max_samples_per_class=max_samples_per_class,
                    )
                )

                indices_key = self.label_key + "_cls_indices"

            num_crops_per_image = self.crop_params.get("num_crops_per_image", 1)
            # if num_crops_per_image > 1:
            #     print(f"Cropping with num_crops_per_image {num_crops_per_image}")

            ts.append(
                RandCropByLabelClassesd(
                    keys=keys,
                    label_key=self.label_key,
                    num_classes=output_classes,
                    spatial_size=self.roi_size,
                    num_samples=num_crops_per_image,
                    ratios=crop_ratios,
                    indices_key=indices_key,
                    warn=False,
                )
            )
        elif self.crop_mode == "rand":
            ts.append(RandSpatialCropd(keys=keys, roi_size=self.roi_size, random_size=False))
        else:
            raise ValueError("Unsupported crop mode" + str(self.crop_mode))

        ts.extend(self.get_custom("after_crop_transforms"))

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
                rotate_range=[0.26, 0.26, 0.26],
                scale_range=[0.2, 0.2, 0.2],
                mode=["bilinear", "nearest"],
                spatial_size=self.roi_size,
                cache_grid=True,
                padding_mode="border",
            )
        )
        ts.append(RandFlipd(keys=[self.image_key, self.label_key], prob=0.5, spatial_axis=0))
        ts.append(RandFlipd(keys=[self.image_key, self.label_key], prob=0.5, spatial_axis=1))
        ts.append(RandFlipd(keys=[self.image_key, self.label_key], prob=0.5, spatial_axis=2))
        ts.append(
            RandGaussianSmoothd(
                keys=self.image_key, prob=0.2, sigma_x=[0.5, 1.0], sigma_y=[0.5, 1.0], sigma_z=[0.5, 1.0]
            )
        )
        ts.append(RandScaleIntensityd(keys=self.image_key, prob=0.5, factors=0.3))
        ts.append(RandShiftIntensityd(keys=self.image_key, prob=0.5, offsets=0.1))
        ts.append(RandGaussianNoised(keys=self.image_key, prob=0.2, mean=0.0, std=0.1))

        ts.extend(self.get_custom("after_augment_transforms"))

        return ts

    def get_final_transforms(self):
        return self.get_custom("final_transforms")

    @classmethod
    def get_postprocess_transform(
        cls,
        save_mask=False,
        invert=False,
        transform=None,
        sigmoid=False,
        output_path=None,
        resample=False,
        data_root_dir="",
        output_dtype=np.uint8,
    ) -> Compose:
        ts = []
        if invert and transform is not None:
            # if resample:
            #     ts.append(ToDeviced(keys="pred", device=torch.device("cpu")))
            ts.append(Invertd(keys="pred", orig_keys="image", transform=transform, nearest_interp=False))

        if save_mask and output_path is not None:
            ts.append(CopyItemsd(keys="pred", times=1, names="seg"))
            ts.append(AsDiscreted(keys="seg", argmax=True) if not sigmoid else AsDiscreted(keys="seg", threshold=0.5))
            ts.append(
                SaveImaged(
                    keys=["seg"],
                    output_dir=output_path,
                    output_postfix="",
                    data_root_dir=data_root_dir,
                    output_dtype=output_dtype,
                    separate_folder=False,
                    squeeze_end_dims=True,
                    resample=False,
                    print_log=False,
                )
            )

        return Compose(ts)

    def __call__(self, augment=False, resample_label=False) -> Compose:
        ts = []
        ts.extend(self.get_load_transforms())
        ts.extend(self.get_resample_transforms(resample_label=resample_label))
        ts.extend(self.get_normalize_transforms())

        if augment:
            ts.extend(self.get_crop_transforms())
            ts.extend(self.get_augment_transforms())

        ts.extend(self.get_final_transforms())

        compose_ts = Compose(ts)

        return compose_ts

    def __repr__(self) -> str:
        out: str = f"DataTransformBuilder: with image_key: {self.image_key}, label_key: {self.label_key} \n"
        out += f"roi_size {self.roi_size} resample {self.resample} resample_resolution {self.resample_resolution} \n"
        out += f"normalize_mode {self.normalize_mode} normalize_params {self.normalize_params} \n"
        out += f"crop_mode {self.crop_mode} crop_params {self.crop_params} \n"
        out += f"extra_modalities {self.extra_modalities} \n"
        for k, trs in self.custom_transforms.items():
            out += f"Custom {k} : {str(trs)} \n"
        return out


class Segmenter:
    def __init__(
        self,
        config_file: Optional[Union[str, Sequence[str]]] = None,
        config_dict: Dict = {},
        rank: int = 0,
        global_rank: int = 0,
    ) -> None:
        self.rank = rank
        self.global_rank = global_rank
        self.distributed = dist.is_initialized()

        if self.global_rank == 0:
            print(f"Segmenter started  config_file: {config_file}, config_dict: {config_dict}")

        np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
        logging.getLogger("torch.nn.parallel.distributed").setLevel(logging.WARNING)

        config = self.parse_input_config(config_file=config_file, override=config_dict)
        self.config = config
        self.config_file = config_file if not isinstance(config_file, (list, tuple)) else config_file[0]
        self.override = config_dict

        if config["ckpt_path"] is not None and not os.path.exists(config["ckpt_path"]):
            os.makedirs(config["ckpt_path"], exist_ok=True)

        if config["log_output_file"] is None:
            config["log_output_file"] = os.path.join(self.config["ckpt_path"], "training.log")
        logger_configure(log_output_file=config["log_output_file"], debug=config["debug"], global_rank=self.global_rank)

        if config["fork"] and "fork" in mp.get_all_start_methods():
            mp.set_start_method("fork", force=True)  # lambda functions fail to pickle without it
        else:
            warnings.warn(
                "Multiprocessing method fork is not available, some non-picklable objects (e.g. lambda ) may fail"
            )

        if config["cuda"] and torch.cuda.is_available():
            self.device = torch.device(self.rank)
            if self.distributed and dist.get_backend() == dist.Backend.NCCL:
                torch.cuda.set_device(rank)
        else:
            self.device = torch.device("cpu")

        if self.global_rank == 0:
            print(yaml.dump(config))

        if config["determ"]:
            set_determinism(seed=0)
        elif torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        # auto adjust network settings
        if config["auto_scale_allowed"]:
            if config["auto_scale_batch"] or config["auto_scale_roi"] or config["auto_scale_filters"]:
                roi_size, _, init_filters, batch_size = auto_adjust_network_settings(
                    auto_scale_batch=config["auto_scale_batch"],
                    auto_scale_roi=config["auto_scale_roi"],
                    auto_scale_filters=config["auto_scale_filters"],
                    image_size_mm=config["image_size_mm_median"],
                    spacing=config["resample_resolution"],
                    anisotropic_scales=config["anisotropic_scales"],
                    levels=len(config["network"]["blocks_down"]),
                    output_classes=config["output_classes"],
                )

                config["roi_size"] = roi_size
                if config["auto_scale_batch"]:
                    config["batch_size"] = batch_size
                if config["auto_scale_filters"] and config["pretrained_ckpt_name"] is None:
                    config["network"]["init_filters"] = init_filters

        self.model = self.setup_model(pretrained_ckpt_name=config["pretrained_ckpt_name"])

        loss_function = ConfigParser(config["loss"]).get_parsed_content(instantiate=True)
        self.loss_function = DeepSupervisionLoss(loss_function)

        self.acc_function = DiceHelper(sigmoid=config["sigmoid"])
        self.grad_scaler = GradScaler(enabled=config["amp"])

        if config.get("sliding_inferrer") is not None:
            self.sliding_inferrer = ConfigParser(config["sliding_inferrer"]).get_parsed_content()
        else:
            self.sliding_inferrer = SlidingWindowInfererAdapt(
                roi_size=config["roi_size"],
                sw_batch_size=1,
                overlap=0.625,
                mode="gaussian",
                cache_roi_weight_map=True,
                progress=False,
            )

        self._data_transform_builder: DataTransformBuilder = None
        self.lr_scheduler = None
        self.optimizer = None

    def get_custom_transforms(self):
        config = self.config

        # check for custom transforms
        custom_transforms = {}
        for tr in config.get("custom_data_transforms", []):
            must_include_keys = ("key", "path", "transform")
            if not all(k in tr for k in must_include_keys):
                raise ValueError("custom transform must include " + str(must_include_keys))

            if os.path.abspath(tr["path"]) not in sys.path:
                sys.path.append(os.path.abspath(tr["path"]))

            custom_transforms.setdefault(tr["key"], [])
            custom_transforms[tr["key"]].append(tr["transform"])

        if len(custom_transforms) > 0 and self.global_rank == 0:
            print(f"Using custom transforms {custom_transforms}")

        if isinstance(config["class_index"], list) and len(config["class_index"]) > 0:
            # custom label embedding, if class_index provided
            custom_transforms.setdefault("final_transforms", [])
            custom_transforms["final_transforms"].append(
                LabelEmbedClassIndex(keys="label", class_index=config["class_index"], allow_missing_keys=True)
            )

        return custom_transforms

    def get_data_transform_builder(self):
        if self._data_transform_builder is None:
            config = self.config
            custom_transforms = self.get_custom_transforms()

            self._data_transform_builder = DataTransformBuilder(
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
                debug=config["debug"],
            )

        return self._data_transform_builder

    def setup_model(self, pretrained_ckpt_name=None):
        config = self.config
        spatial_dims = config["network"].get("spatial_dims", 3)
        norm_name, norm_args = split_args(config["network"].get("norm", ""))
        norm_name = norm_name.upper()

        if norm_name == "INSTANCE_NVFUSER":
            _, has_nvfuser = optional_import("apex.normalization", name="InstanceNorm3dNVFuser")
            if has_nvfuser and spatial_dims == 3:
                act = config["network"].get("act", "relu")
                if isinstance(act, str):
                    config["network"]["act"] = [act, {"inplace": False}]
            else:
                norm_name = "INSTANCE"

        if len(norm_name) > 0:
            config["network"]["norm"] = norm_name if len(norm_args) == 0 else [norm_name, norm_args]

        if spatial_dims == 3:
            if config.get("anisotropic_scales", False) and "SegResNetDS" in config["network"]["_target_"]:
                config["network"]["resolution"] = copy.deepcopy(config["resample_resolution"])
                if self.global_rank == 0:
                    print(f"Using anisotropic scales {config['network']}")

        model = ConfigParser(config["network"]).get_parsed_content()

        if self.global_rank == 0:
            print(str(model))

        if pretrained_ckpt_name is not None:
            self.checkpoint_load(ckpt=pretrained_ckpt_name, model=model)

        model = model.to(self.device)

        if spatial_dims == 3:
            memory_format = torch.channels_last_3d if config["channels_last"] else torch.preserve_format
            model = model.to(memory_format=memory_format)

        if self.distributed and not config["infer"]["enabled"]:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DistributedDataParallel(
                module=model, device_ids=[self.rank], output_device=self.rank, find_unused_parameters=False
            )

        if self.global_rank == 0:
            pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total parameters count: {pytorch_total_params} distributed: {self.distributed}")

        return model

    def parse_input_config(
        self, config_file: Optional[Union[str, Sequence[str]]] = None, override: Dict = {}
    ) -> Tuple[ConfigParser, Dict]:
        config = {}
        if config_file is None or override.get("use_ckpt_config", False):
            # attempt to load config from model ckpt file
            for ckpt_key in ["pretrained_ckpt_name", "validate#ckpt_name", "infer#ckpt_name", "finetune#ckpt_name"]:
                ckpt = override.get(ckpt_key, None)
                if ckpt and os.path.exists(ckpt):
                    checkpoint = torch.load(ckpt, map_location="cpu")
                    config = checkpoint.get("config", {})
                    if self.global_rank == 0:
                        print(f"Initializing config from the checkpoint {ckpt}: {yaml.dump(config)}")

            if len(config) == 0 and config_file is None:
                warnings.warn("No input config_file provided, and no valid checkpoints found")

        if config_file is not None and len(config) == 0:
            config = ConfigParser.load_config_files(config_file)
            config.setdefault("finetune", {"enabled": False, "ckpt_name": None})
            config.setdefault(
                "validate", {"enabled": False, "ckpt_name": None, "save_mask": False, "output_path": None}
            )
            config.setdefault("infer", {"enabled": False, "ckpt_name": None})

        parser = ConfigParser(config=config)
        parser.update(pairs=override)
        config = parser.config  # just in case

        if config.get("data_file_base_dir", None) is None or config.get("data_list_file_path", None) is None:
            raise ValueError("CONFIG: data_file_base_dir and  data_list_file_path must be provided")

        if config.get("bundle_root", None) is None:
            config["bundle_root"] = str(Path(__file__).parent.parent)

        if "modality" not in config:
            if self.global_rank == 0:
                warnings.warn("CONFIG: modality is not provided, assuming MRI")
            config["modality"] = "mri"

        if "normalize_mode" not in config:
            config["normalize_mode"] = "range" if config["modality"].lower() == "ct" else "meanstd"
            if self.global_rank == 0:
                print(f"CONFIG: normalize_mode is not provided, assuming: {config['normalize_mode']}")

        # assign defaults
        config.setdefault("debug", False)

        config.setdefault("loss", None)
        config.setdefault("acc", None)
        config.setdefault("amp", True)
        config.setdefault("cuda", True)
        config.setdefault("fold", 0)
        config.setdefault("batch_size", 1)
        config.setdefault("determ", False)
        config.setdefault("quick", False)
        config.setdefault("sigmoid", False)
        config.setdefault("cache_rate", None)
        config.setdefault("cache_class_indices", None)

        config.setdefault("channels_last", True)
        config.setdefault("fork", True)

        config.setdefault("num_epochs", 300)
        config.setdefault("num_warmup_epochs", 3)
        config.setdefault("num_epochs_per_validation", None)
        config.setdefault("num_epochs_per_saving", 10)
        config.setdefault("num_steps_per_image", None)
        config.setdefault("num_crops_per_image", 1)
        config.setdefault("max_samples_per_class", None)

        config.setdefault("calc_val_loss", False)
        config.setdefault("validate_final_original_res", False)
        config.setdefault("early_stopping_fraction", 0)
        config.setdefault("start_epoch", 0)

        config.setdefault("ckpt_path", None)
        config.setdefault("ckpt_save", True)
        config.setdefault("log_output_file", None)

        config.setdefault("crop_mode", "ratio")
        config.setdefault("crop_ratios", None)
        config.setdefault("resample_resolution", [1.0, 1.0, 1.0])
        config.setdefault("resample", False)
        config.setdefault("roi_size", [128, 128, 128])
        config.setdefault("num_workers", 4)
        config.setdefault("extra_modalities", {})
        config.setdefault("intensity_bounds", [-250, 250])
        config.setdefault("stop_on_lowacc", True)

        config.setdefault("class_index", None)
        config.setdefault("class_names", [])
        if not isinstance(config["class_names"], (list, tuple)):
            config["class_names"] = []

        if len(config["class_names"]) == 0:
            n_foreground_classes = int(config["output_classes"])
            if not config["sigmoid"]:
                n_foreground_classes -= 1
            config["class_names"] = ["acc_" + str(i) for i in range(n_foreground_classes)]

        pretrained_ckpt_name = config.get("pretrained_ckpt_name", None)
        if pretrained_ckpt_name is None:
            if config["validate"]["enabled"]:
                pretrained_ckpt_name = config["validate"]["ckpt_name"]
            elif config["infer"]["enabled"]:
                pretrained_ckpt_name = config["infer"]["ckpt_name"]
            elif config["finetune"]["enabled"]:
                pretrained_ckpt_name = config["finetune"]["ckpt_name"]
        config["pretrained_ckpt_name"] = pretrained_ckpt_name

        config.setdefault("auto_scale_allowed", False)
        config.setdefault("auto_scale_batch", False)
        config.setdefault("auto_scale_roi", False)
        config.setdefault("auto_scale_filters", False)

        if pretrained_ckpt_name is not None:
            config["auto_scale_roi"] = False
            config["auto_scale_filters"] = False

        if config["max_samples_per_class"] is None:
            config["max_samples_per_class"] = 10 * config["num_epochs"]

        if not torch.cuda.is_available() and config["cuda"]:
            print("No cuda is available.! Running on CPU!!!")
            config["cuda"] = False

        config["amp"] = config["amp"] and config["cuda"]
        config["rank"] = self.rank
        config["global_rank"] = self.global_rank

        # resolve content
        for k, v in config.items():
            if isinstance(v, dict) and "_target_" in v:
                config[k] = parser.get_parsed_content(k, instantiate=False).config
            elif "_target_" in str(v):
                config[k] = copy.deepcopy(v)
            else:
                config[k] = parser.get_parsed_content(k)

        return config

    def config_save_updated(self, save_path=None):
        if self.global_rank == 0 and self.config["auto_scale_allowed"]:
            # reload input config
            config = ConfigParser.load_config_files(self.config_file)
            parser = ConfigParser(config=config)
            parser.update(pairs=self.override)
            config = parser.config

            config["batch_size"] = self.config["batch_size"]
            config["roi_size"] = self.config["roi_size"]
            config["num_crops_per_image"] = self.config["num_crops_per_image"]

            if "init_filters" in self.config["network"]:
                config["network"]["init_filters"] = self.config["network"]["init_filters"]

            if save_path is None:
                save_path = self.config_file

            print(f"Re-saving main config to {save_path}.")
            ConfigParser.export_config_file(config, save_path, fmt="yaml", default_flow_style=None, sort_keys=False)

    def config_with_relpath(self, config=None):
        if config is None:
            config = self.config
        config = copy.deepcopy(config)
        bundle_root = config["bundle_root"]

        def convert_rel_path(conf):
            for k, v in conf.items():
                if isinstance(v, str) and v.startswith(bundle_root):
                    conf[k] = f"$@bundle_root + '/{os.path.relpath(v, bundle_root)}'"

        convert_rel_path(config)
        convert_rel_path(config["finetune"])
        convert_rel_path(config["validate"])
        convert_rel_path(config["infer"])
        config["bundle_root"] = bundle_root

        return config

    def checkpoint_save(self, ckpt: str, model: torch.nn.Module, **kwargs):
        save_time = time.time()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        config = self.config_with_relpath()

        torch.save({"state_dict": state_dict, "config": config, **kwargs}, ckpt)

        save_time = time.time() - save_time
        print(f"Saving checkpoint process: {ckpt}, {kwargs}, save_time {save_time:.2f}s")

        return save_time

    def checkpoint_load(self, ckpt: str, model: torch.nn.Module, **kwargs):
        if not os.path.isfile(ckpt):
            if self.global_rank == 0:
                warnings.warn("Invalid checkpoint file: " + str(ckpt))
        else:
            checkpoint = torch.load(ckpt, map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            epoch = checkpoint.get("epoch", 0)
            best_metric = checkpoint.get("best_metric", 0)

            if self.config.pop("continue", False):
                if "epoch" in checkpoint:
                    self.config["start_epoch"] = checkpoint["epoch"]
                if "best_metric" in checkpoint:
                    self.config["best_metric"] = checkpoint["best_metric"]

            print(
                f"=> loaded checkpoint {ckpt} (epoch {epoch}) (best_metric {best_metric}) setting start_epoch {self.config['start_epoch']}"
            )
            # print(f"config after {self.config}")

    def get_shared_memory_list(self, length=0):
        mp.current_process().authkey = np.arange(32, dtype=np.uint8).tobytes()
        shl0 = mp.Manager().list([None] * length)

        if self.distributed:
            # to support multi-node training, we need check for a local process group
            is_multinode = False

            if dist_launched():
                local_world_size = int(os.getenv("LOCAL_WORLD_SIZE"))
                world_size = int(os.getenv("WORLD_SIZE"))
                group_rank = int(os.getenv("GROUP_RANK"))
                if world_size > local_world_size:
                    is_multinode = True
                    # we're in multi-node, get local world sizes
                    lw = torch.tensor(local_world_size, dtype=torch.int, device=self.device)
                    lw_sizes = [torch.zeros_like(lw) for _ in range(world_size)]
                    dist.all_gather(tensor_list=lw_sizes, tensor=lw)

                    src = g_rank = 0
                    while src < world_size:
                        # create sub-groups local to a node, to share memory only within a node
                        # and broadcast shared list within a node
                        group = dist.new_group(ranks=list(range(src, src + local_world_size)))
                        if group_rank == g_rank:
                            shl_list = [shl0]
                            dist.broadcast_object_list(shl_list, src=src, group=group, device=self.device)
                            shl = shl_list[0]
                        dist.destroy_process_group(group)
                        src = src + lw_sizes[src].item()  # rank of first process in the next node
                        g_rank += 1

            if not is_multinode:
                shl_list = [shl0]
                dist.broadcast_object_list(shl_list, src=0, device=self.device)
                shl = shl_list[0]

        else:
            shl = shl0

        return shl

    def get_train_loader(self, data, cache_rate=0, persistent_workers=False):
        distributed = self.distributed
        num_workers = self.config["num_workers"]
        batch_size = self.config["batch_size"]

        train_transform = self.get_data_transform_builder()(augment=True, resample_label=True)

        if cache_rate > 0:
            runtime_cache = self.get_shared_memory_list(length=len(data))
            train_ds = CacheDataset(
                data=data,
                transform=train_transform,
                copy_cache=False,
                cache_rate=cache_rate,
                runtime_cache=runtime_cache,
            )
        else:
            train_ds = Dataset(data=data, transform=train_transform)

        train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            sampler=train_sampler,
            persistent_workers=persistent_workers and num_workers > 0,
            pin_memory=True,
        )

        return train_loader

    def get_val_loader(self, data, cache_rate=0, resample_label=False, persistent_workers=False):
        distributed = self.distributed
        num_workers = self.config["num_workers"]

        val_transform = self.get_data_transform_builder()(augment=False, resample_label=resample_label)

        if cache_rate > 0:
            runtime_cache = self.get_shared_memory_list(length=len(data))
            val_ds = CacheDataset(
                data=data, transform=val_transform, copy_cache=False, cache_rate=cache_rate, runtime_cache=runtime_cache
            )
        else:
            val_ds = Dataset(data=data, transform=val_transform)

        val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            sampler=val_sampler,
            persistent_workers=persistent_workers and num_workers > 0,
            pin_memory=True,
        )

        return val_loader

    def train(self):
        if self.global_rank == 0:
            print("Segmenter train called")

        if self.loss_function is None:
            raise ValueError("CONFIG loss function is not provided")
        if self.acc_function is None:
            raise ValueError("CONFIG accuracy function is not provided")

        config = self.config
        distributed = self.distributed
        sliding_inferrer = self.sliding_inferrer

        loss_function = self.loss_function
        acc_function = self.acc_function
        grad_scaler = self.grad_scaler

        use_amp = config["amp"]
        use_cuda = config["cuda"]
        ckpt_path = config["ckpt_path"]
        sigmoid = config["sigmoid"]
        channels_last = config["channels_last"]
        calc_val_loss = config["calc_val_loss"]

        data_list_file_path = config["data_list_file_path"]
        if not os.path.isabs(data_list_file_path):
            data_list_file_path = os.path.abspath(os.path.join(config["bundle_root"], data_list_file_path))

        if config.get("validation_key", None) is not None:
            train_files, _ = datafold_read(datalist=data_list_file_path, basedir=config["data_file_base_dir"], fold=-1)
            validation_files, _ = datafold_read(
                datalist=data_list_file_path,
                basedir=config["data_file_base_dir"],
                fold=-1,
                key=config["validation_key"],
            )
        else:
            train_files, validation_files = datafold_read(
                datalist=data_list_file_path, basedir=config["data_file_base_dir"], fold=config["fold"]
            )

        if config["quick"]:  # quick run on a smaller subset of files
            train_files, validation_files = train_files[:8], validation_files[:8]
        if self.global_rank == 0:
            print(f"train_files files {len(train_files)}, validation files {len(validation_files)}")

        if len(validation_files) == 0:
            warnings.warn("No validation files found!")

        cache_rate_train, cache_rate_val = self.get_cache_rate(
            train_cases=len(train_files), validation_cases=len(validation_files)
        )

        if config["cache_class_indices"] is None:
            config["cache_class_indices"] = cache_rate_train > 0

        if self.global_rank == 0:
            print(
                f"Auto setting max_samples_per_class: {config['max_samples_per_class']} cache_class_indices: {config['cache_class_indices']}"
            )

        num_steps_per_image = config["num_steps_per_image"]
        if config["auto_scale_allowed"] and num_steps_per_image is None:
            be = config["batch_size"]

            if config["crop_mode"] == "ratio":
                config["num_crops_per_image"] = config["batch_size"]
                config["batch_size"] = 1
            else:
                config["num_crops_per_image"] = 1

            if cache_rate_train < 0.75:
                num_steps_per_image = max(1, 4 // be)
            else:
                num_steps_per_image = 1

        elif num_steps_per_image is None:
            num_steps_per_image = 1

        num_crops_per_image = int(config["num_crops_per_image"])
        num_epochs_per_saving = max(1, config["num_epochs_per_saving"] // num_crops_per_image)
        num_warmup_epochs = max(3, config["num_warmup_epochs"] // num_crops_per_image)
        num_epochs_per_validation = config["num_epochs_per_validation"]
        num_epochs = max(1, config["num_epochs"] // min(3, num_crops_per_image))
        if self.global_rank == 0:
            print(
                f"Given num_crops_per_image {num_crops_per_image}, num_epochs was adjusted {config['num_epochs']} => {num_epochs}"
            )

        if num_epochs_per_validation is not None:
            num_epochs_per_validation = max(1, num_epochs_per_validation // num_crops_per_image)

        val_schedule_list = schedule_validation_epochs(
            num_epochs=num_epochs,
            num_epochs_per_validation=num_epochs_per_validation,
            fraction=min(0.3, 0.16 * num_crops_per_image),
        )
        if self.global_rank == 0:
            print(f"Scheduling validation loops at epochs: {val_schedule_list}")

        train_loader = self.get_train_loader(data=train_files, cache_rate=cache_rate_train, persistent_workers=True)

        val_loader = self.get_val_loader(
            data=validation_files, cache_rate=cache_rate_val, resample_label=True, persistent_workers=True
        )

        optim_name = config.get("optim_name", None)  # experimental
        if optim_name is not None:
            if self.global_rank == 0:
                print(f"Using optimizer: {optim_name}")
            if optim_name == "fusednovograd":
                import apex

                optimizer = apex.optimizers.FusedNovoGrad(
                    params=self.model.parameters(), lr=config["learning_rate"], weight_decay=1.0e-5
                )
            elif optim_name == "sgd":
                momentum = config.get("sgd_momentum", 0.9)
                optimizer = torch.optim.SGD(
                    params=self.model.parameters(), lr=config["learning_rate"], weight_decay=1.0e-5, momentum=momentum
                )
                if self.global_rank == 0:
                    print(f"Using momentum: {momentum}")
            else:
                raise ValueError("Unsupported optim_name" + str(optim_name))

        elif self.optimizer is None:
            optimizer_part = ConfigParser(config["optimizer"]).get_parsed_content(instantiate=False)
            optimizer = optimizer_part.instantiate(params=self.model.parameters())
        else:
            optimizer = self.optimizer

        tb_writer = None
        csv_path = progress_path = None

        if self.global_rank == 0 and ckpt_path is not None:
            # rank 0 is responsible for heavy lifting of logging/saving
            progress_path = os.path.join(ckpt_path, "progress.yaml")

            tb_writer = SummaryWriter(log_dir=ckpt_path)
            print(f"Writing Tensorboard logs to {tb_writer.log_dir}")

            csv_path = os.path.join(ckpt_path, "accuracy_history.csv")
            self.save_history_csv(
                csv_path=csv_path,
                header=["epoch", "metric", "loss", "iter", "time", "train_time", "validation_time", "epoch_time"],
            )

        do_torch_save = (self.global_rank == 0) and ckpt_path is not None and config["ckpt_save"]
        best_ckpt_path = os.path.join(ckpt_path, "model.pt")
        intermediate_ckpt_path = os.path.join(ckpt_path, "model_final.pt")

        best_metric = -1
        best_metric_epoch = -1
        pre_loop_time = time.time()
        report_num_epochs = num_epochs * num_crops_per_image
        train_time = validation_time = 0
        val_acc_history = []

        start_epoch = config["start_epoch"]
        if "best_metric" in config:
            best_metric = float(config["best_metric"])

        start_epoch = start_epoch // num_crops_per_image
        if start_epoch > 0:
            val_schedule_list = [v for v in val_schedule_list if v >= start_epoch]
            if len(val_schedule_list) == 0:
                val_schedule_list = [start_epoch]
            print(f"adjusted schedule_list {val_schedule_list}")

        if self.global_rank == 0:
            print(
                f"Using num_epochs => {num_epochs}\n "
                f"Using start_epoch => {start_epoch}\n "
                f"batch_size => {config['batch_size']} \n "
                f"num_crops_per_image => {config['num_crops_per_image']} \n "
                f"num_steps_per_image => {num_steps_per_image} \n "
                f"num_warmup_epochs => {num_warmup_epochs} \n "
            )

        if self.lr_scheduler is None:
            lr_scheduler = WarmupCosineSchedule(
                optimizer=optimizer, warmup_steps=num_warmup_epochs, warmup_multiplier=0.1, t_total=num_epochs
            )
        else:
            lr_scheduler = self.lr_scheduler
        if lr_scheduler is not None and start_epoch > 0:
            lr_scheduler.last_epoch = start_epoch

        range_num_epochs = range(start_epoch, num_epochs)
        if self.global_rank == 0 and has_tqdm and not config["debug"]:
            range_num_epochs = tqdm(
                range(start_epoch, num_epochs),
                desc=str(os.path.basename(config["bundle_root"])) + " - training",
                unit="epoch",
            )

        if distributed:
            dist.barrier()
        self.config_save_updated(save_path=self.config_file)  # overwriting main input config

        for epoch in range_num_epochs:
            report_epoch = epoch * num_crops_per_image

            if distributed:
                if isinstance(train_loader.sampler, DistributedSampler):
                    train_loader.sampler.set_epoch(epoch)
                dist.barrier()

            epoch_time = start_time = time.time()

            train_loss, train_acc = 0, 0
            if not config.get("skip_train", False):
                train_loss, train_acc = self.train_epoch(
                    model=self.model,
                    train_loader=train_loader,
                    optimizer=optimizer,
                    loss_function=loss_function,
                    acc_function=acc_function,
                    grad_scaler=grad_scaler,
                    epoch=report_epoch,
                    rank=self.rank,
                    global_rank=self.global_rank,
                    num_epochs=report_num_epochs,
                    sigmoid=sigmoid,
                    use_amp=use_amp,
                    use_cuda=use_cuda,
                    channels_last=channels_last,
                    num_steps_per_image=num_steps_per_image,
                )

            train_time = time.time() - start_time

            if self.global_rank == 0:
                print(
                    f"Final training  {report_epoch}/{report_num_epochs - 1} "
                    f"loss: {train_loss:.4f} acc_avg: {np.mean(train_acc):.4f} "
                    f"acc {train_acc} time {train_time:.2f}s  "
                    f"lr: {optimizer.param_groups[0]['lr']:.4e}"
                )

                if tb_writer is not None:
                    tb_writer.add_scalar("train/loss", train_loss, report_epoch)
                    tb_writer.add_scalar("train/acc", np.mean(train_acc), report_epoch)

            # validate every num_epochs_per_validation epochs (defaults to 1, every epoch)
            val_acc_mean = -1
            if (
                len(val_schedule_list) > 0
                and epoch + 1 >= val_schedule_list[0]
                and val_loader is not None
                and len(val_loader) > 0
            ):
                val_schedule_list.pop(0)

                start_time = time.time()
                torch.cuda.empty_cache()

                val_loss, val_acc = self.val_epoch(
                    model=self.model,
                    val_loader=val_loader,
                    sliding_inferrer=sliding_inferrer,
                    loss_function=loss_function,
                    acc_function=acc_function,
                    epoch=report_epoch,
                    rank=self.rank,
                    global_rank=self.global_rank,
                    num_epochs=report_num_epochs,
                    sigmoid=sigmoid,
                    use_amp=use_amp,
                    use_cuda=use_cuda,
                    channels_last=channels_last,
                    calc_val_loss=calc_val_loss,
                )

                torch.cuda.empty_cache()
                validation_time = time.time() - start_time

                val_acc_mean = float(np.mean(val_acc))
                val_acc_history.append((report_epoch, val_acc_mean))

                if self.global_rank == 0:
                    print(
                        f"Final validation {report_epoch}/{report_num_epochs - 1} "
                        f"loss: {val_loss:.4f} acc_avg: {val_acc_mean:.4f} acc: {val_acc} time: {validation_time:.2f}s"
                    )

                    if tb_writer is not None:
                        tb_writer.add_scalar("val/acc", val_acc_mean, report_epoch)
                        for i in range(min(len(config["class_names"]), len(val_acc))):  # accuracy per class
                            tb_writer.add_scalar("val_class/" + config["class_names"][i], val_acc[i], report_epoch)
                        if calc_val_loss:
                            tb_writer.add_scalar("val/loss", val_loss, report_epoch)

                    timing_dict = dict(
                        time="{:.2f} hr".format((time.time() - pre_loop_time) / 3600),
                        train_time="{:.2f}s".format(train_time),
                        validation_time="{:.2f}s".format(validation_time),
                        epoch_time="{:.2f}s".format(time.time() - epoch_time),
                    )

                    if val_acc_mean > best_metric:
                        print(f"New best metric ({best_metric:.6f} --> {val_acc_mean:.6f}). ")
                        best_metric, best_metric_epoch = val_acc_mean, report_epoch
                        save_time = 0
                        if do_torch_save:
                            save_time = self.checkpoint_save(
                                ckpt=best_ckpt_path, model=self.model, epoch=best_metric_epoch, best_metric=best_metric
                            )

                        if progress_path is not None:
                            self.save_progress_yaml(
                                progress_path=progress_path,
                                ckpt=best_ckpt_path if do_torch_save else None,
                                best_avg_dice_score_epoch=best_metric_epoch,
                                best_avg_dice_score=best_metric,
                                save_time=save_time,
                                **timing_dict,
                            )
                    if csv_path is not None:
                        self.save_history_csv(
                            csv_path=csv_path,
                            epoch=report_epoch,
                            metric="{:.4f}".format(val_acc_mean),
                            loss="{:.4f}".format(train_loss),
                            iter=report_epoch * len(train_loader.dataset),
                            **timing_dict,
                        )

                # sanity check
                if epoch > max(20, num_epochs / 4) and 0 <= val_acc_mean < 0.01 and config["stop_on_lowacc"]:
                    raise ValueError(
                        f"Accuracy seems very low at epoch {report_epoch}, acc {val_acc_mean}. "
                        f"Most likely optimization diverged, try setting  a smaller learning_rate than {config['learning_rate']}"
                    )

                # early stopping
                if config["early_stopping_fraction"] > 0 and epoch > num_epochs / 2 and len(val_acc_history) > 10:
                    check_interval = int(0.1 * num_epochs * num_crops_per_image)
                    check_stats = [
                        va[1] for va in val_acc_history if report_epoch - va[0] < check_interval
                    ]  # at least 10% epochs
                    if len(check_stats) < 10:
                        check_stats = [va[1] for va in val_acc_history[-10:]]  # at least 10 sample points
                    mac, mic = max(check_stats), min(check_stats)

                    early_stopping_fraction = (mac - mic) / (abs(mac) + 1e-8)
                    if mac > 0 and mic > 0 and early_stopping_fraction < config["early_stopping_fraction"]:
                        if self.global_rank == 0:
                            print(
                                f"Early stopping at epoch {report_epoch} fraction {early_stopping_fraction} !!! max {mac} min {mic} samples count {len(check_stats)} {check_stats[-50:]}"
                            )
                        break
                    else:
                        if self.global_rank == 0:
                            print(
                                f"No stopping at epoch {report_epoch} fraction {early_stopping_fraction} !!! max {mac} min {mic} samples count {len(check_stats)} {check_stats[-50:]}"
                            )

            # save intermediate checkpoint every num_epochs_per_saving epochs
            if do_torch_save and ((epoch + 1) % num_epochs_per_saving == 0 or (epoch + 1) >= num_epochs):
                if report_epoch != best_metric_epoch:
                    self.checkpoint_save(
                        ckpt=intermediate_ckpt_path, model=self.model, epoch=report_epoch, best_metric=val_acc_mean
                    )
                else:
                    shutil.copyfile(best_ckpt_path, intermediate_ckpt_path)  # if already saved once

            if lr_scheduler is not None:
                lr_scheduler.step()

            if self.global_rank == 0:
                # report time estimate
                time_remaining_estimate = train_time * (num_epochs - epoch)
                if val_loader is not None and len(val_loader) > 0:
                    if validation_time == 0:
                        validation_time = train_time
                    time_remaining_estimate += validation_time * len(val_schedule_list)

                print(
                    f"Estimated remaining training time for the current model fold {config['fold']} is "
                    f"{time_remaining_estimate/3600:.2f} hr, "
                    f"running time {(time.time() - pre_loop_time)/3600:.2f} hr, "
                    f"est total time {(time.time() - pre_loop_time + time_remaining_estimate)/3600:.2f} hr \n"
                )

        # end of main epoch loop

        train_loader = val_loader = optimizer = None

        # optionally validate best checkpoint at the original image resolution
        orig_res = config["resample"] == False
        if config["validate_final_original_res"] and config["resample"]:
            pretrained_ckpt_name = best_ckpt_path if os.path.exists(best_ckpt_path) else intermediate_ckpt_path
            if os.path.exists(pretrained_ckpt_name):
                self.model = None
                gc.collect()
                torch.cuda.empty_cache()

                best_metric = self.original_resolution_validate(
                    pretrained_ckpt_name=pretrained_ckpt_name,
                    progress_path=progress_path,
                    best_metric_epoch=best_metric_epoch,
                    pre_loop_time=pre_loop_time,
                )
                orig_res = True
            else:
                if self.global_rank == 0:
                    print(
                        f"Unable to validate at the original res since no model checkpoints found {best_ckpt_path}, {intermediate_ckpt_path}"
                    )

        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()

        if self.global_rank == 0:
            print(
                f"=== DONE: best_metric: {best_metric:.4f} at epoch: {best_metric_epoch} of {report_num_epochs} orig_res {orig_res}. Training time {(time.time() - pre_loop_time)/3600:.2f} hr."
            )

        return best_metric

    def original_resolution_validate(self, pretrained_ckpt_name, progress_path, best_metric_epoch, pre_loop_time):
        if self.global_rank == 0:
            print("Running final best model validation on the original image resolution!")

        self.model = self.setup_model(pretrained_ckpt_name=pretrained_ckpt_name)

        # validate
        start_time = time.time()
        val_acc_mean, val_loss, val_acc = self.validate()
        validation_time = "{:.2f}s".format(time.time() - start_time)
        val_acc_mean = float(np.mean(val_acc))
        if self.global_rank == 0:
            print(
                f"Original resolution validation: "
                f"loss: {val_loss:.4f} acc_avg: {val_acc_mean:.4f} "
                f"acc {val_acc} time {validation_time}"
            )

            if progress_path is not None:
                self.save_progress_yaml(
                    progress_path=progress_path,
                    ckpt=pretrained_ckpt_name,
                    best_avg_dice_score_epoch=best_metric_epoch,
                    best_avg_dice_score=val_acc_mean,
                    validation_time=validation_time,
                    inverted_best_validation=True,
                    time="{:.2f} hr".format((time.time() - pre_loop_time) / 3600),
                )

        return val_acc_mean

    def validate(self, validation_files=None):
        config = self.config
        resample = config["resample"]

        val_config = self.config["validate"]
        output_path = val_config.get("output_path", None)
        save_mask = val_config.get("save_mask", False) and output_path is not None
        invert = val_config.get("invert", True)

        data_list_file_path = config["data_list_file_path"]
        if not os.path.isabs(data_list_file_path):
            data_list_file_path = os.path.abspath(os.path.join(config["bundle_root"], data_list_file_path))

        if validation_files is None:
            if config.get("validation_key", None) is not None:
                validation_files, _ = datafold_read(
                    datalist=data_list_file_path,
                    basedir=config["data_file_base_dir"],
                    fold=-1,
                    key=config["validation_key"],
                )
            else:
                _, validation_files = datafold_read(
                    datalist=data_list_file_path, basedir=config["data_file_base_dir"], fold=config["fold"]
                )

        if self.global_rank == 0:
            print(f"validation files {len(validation_files)}")

        if len(validation_files) == 0:
            warnings.warn("No validation files found!")
            return

        val_loader = self.get_val_loader(data=validation_files, resample_label=not invert)
        val_transform = val_loader.dataset.transform

        post_transforms = None
        if save_mask or invert:
            post_transforms = DataTransformBuilder.get_postprocess_transform(
                save_mask=save_mask,
                invert=invert,
                transform=val_transform,
                sigmoid=self.config["sigmoid"],
                output_path=output_path,
                resample=resample,
                data_root_dir=self.config["data_file_base_dir"],
                output_dtype=np.uint8 if self.config["output_classes"] < 255 else np.uint16,
            )

        start_time = time.time()
        val_loss, val_acc = self.val_epoch(
            model=self.model,
            val_loader=val_loader,
            sliding_inferrer=self.sliding_inferrer,
            loss_function=self.loss_function,
            acc_function=self.acc_function,
            rank=self.rank,
            global_rank=self.global_rank,
            sigmoid=self.config["sigmoid"],
            use_amp=self.config["amp"],
            use_cuda=self.config["cuda"],
            post_transforms=post_transforms,
            channels_last=self.config["channels_last"],
            calc_val_loss=self.config["calc_val_loss"],
        )
        val_acc_mean = float(np.mean(val_acc))

        if self.global_rank == 0:
            print(
                f"Validation complete, loss_avg: {val_loss:.4f} "
                f"acc_avg: {val_acc_mean:.4f} acc {val_acc} time {time.time() - start_time:.2f}s"
            )

        return val_acc_mean, val_loss, val_acc

    def infer(self, testing_files=None):
        output_path = self.config["infer"].get("output_path", None)
        testing_key = self.config["infer"].get("data_list_key", "testing")

        if output_path is None:
            if self.global_rank == 0:
                print("Inference output_path is not specified")
            return

        if testing_files is None:
            data_list_file_path = self.config["data_list_file_path"]
            if not os.path.isabs(data_list_file_path):
                data_list_file_path = os.path.abspath(os.path.join(self.config["bundle_root"], data_list_file_path))

            testing_files, _ = datafold_read(
                datalist=data_list_file_path, basedir=self.config["data_file_base_dir"], fold=-1, key=testing_key
            )

        if self.global_rank == 0:
            print(f"testing_files files {len(testing_files)}")

        if len(testing_files) == 0:
            warnings.warn("No testing_files files found!")
            return

        inf_loader = self.get_val_loader(data=testing_files, resample_label=False)
        inf_transform = inf_loader.dataset.transform

        post_transforms = DataTransformBuilder.get_postprocess_transform(
            save_mask=True,
            invert=True,
            transform=inf_transform,
            sigmoid=self.config["sigmoid"],
            output_path=output_path,
            resample=self.config["resample"],
            data_root_dir=self.config["data_file_base_dir"],
            output_dtype=np.uint8 if self.config["output_classes"] < 255 else np.uint16,
        )

        start_time = time.time()
        self.val_epoch(
            model=self.model,
            val_loader=inf_loader,
            sliding_inferrer=self.sliding_inferrer,
            rank=self.rank,
            global_rank=self.global_rank,
            sigmoid=self.config["sigmoid"],
            use_amp=self.config["amp"],
            use_cuda=self.config["cuda"],
            post_transforms=post_transforms,
            channels_last=self.config["channels_last"],
            calc_val_loss=self.config["calc_val_loss"],
        )

        if self.global_rank == 0:
            print(f"Inference complete, time {time.time() - start_time:.2f}s")

    @torch.no_grad()
    def infer_image(self, image_file):
        self.model.eval()

        infer_config = self.config["infer"]
        output_path = infer_config.get("output_path", None)
        save_mask = infer_config.get("save_mask", False) and output_path is not None
        invert_on_gpu = infer_config.get("invert_on_gpu", False)

        start_time = time.time()
        sigmoid = self.config["sigmoid"]
        resample = self.config["resample"]
        channels_last = self.config["channels_last"]

        inf_transform = self.get_data_transform_builder()(augment=False, resample_label=False)

        batch_data = inf_transform([image_file])
        batch_data = list_data_collate([batch_data])

        memory_format = torch.channels_last_3d if channels_last else torch.preserve_format
        data = batch_data["image"].as_subclass(torch.Tensor).to(memory_format=memory_format, device=self.device)

        with autocast(enabled=self.config["amp"]):
            logits = self.sliding_inferrer(inputs=data, network=self.model)

        data = None

        logits = logits.float().contiguous()
        pred = self.logits2pred(logits=logits, sigmoid=sigmoid, inplace=True)
        logits = None

        if not invert_on_gpu:
            pred = pred.cpu()  # invert on cpu (default)

        post_transforms = DataTransformBuilder.get_postprocess_transform(
            save_mask=save_mask,
            invert=True,
            transform=inf_transform,
            sigmoid=sigmoid,
            output_path=output_path,
            resample=resample,
            data_root_dir=self.config["data_file_base_dir"],
            output_dtype=np.uint8 if self.config["output_classes"] < 255 else np.uint16,
        )

        batch_data["pred"] = convert_to_dst_type(pred, batch_data["image"], dtype=pred.dtype, device=pred.device)[
            0
        ]  # make Meta tensor
        pred = [post_transforms(x)["pred"] for x in decollate_batch(batch_data)]

        pred = pred[0]

        print(f"Inference complete, time {time.time() - start_time:.2f}s shape {pred.shape} {image_file}")

        return pred

    def train_epoch(
        self,
        model,
        train_loader,
        optimizer,
        loss_function,
        acc_function,
        grad_scaler,
        epoch,
        rank,
        global_rank=0,
        num_epochs=0,
        sigmoid=False,
        use_amp=True,
        use_cuda=True,
        channels_last=False,
        num_steps_per_image=1,
    ):
        model.train()
        device = torch.device(rank) if use_cuda else torch.device("cpu")
        memory_format = torch.channels_last_3d if channels_last else torch.preserve_format

        run_loss = CumulativeAverage()
        run_acc = CumulativeAverage()

        start_time = time.time()
        avg_loss = avg_acc = 0
        for idx, batch_data in enumerate(train_loader):
            data = batch_data["image"].as_subclass(torch.Tensor).to(memory_format=memory_format, device=device)
            target = batch_data["label"].as_subclass(torch.Tensor).to(memory_format=memory_format, device=device)

            data_list = data.chunk(num_steps_per_image) if num_steps_per_image > 1 else [data]
            target_list = target.chunk(num_steps_per_image) if num_steps_per_image > 1 else [target]

            for ich in range(min(num_steps_per_image, len(data_list))):
                data = data_list[ich]
                target = target_list[ich]

                # optimizer.zero_grad(set_to_none=True)
                for param in model.parameters():
                    param.grad = None

                with autocast(enabled=use_amp):
                    logits = model(data)

                loss = loss_function(logits, target)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                with torch.no_grad():
                    pred = self.logits2pred(logits, sigmoid=sigmoid, skip_softmax=True)
                    acc = acc_function(pred, target)

                batch_size_adjusted = batch_size = data.shape[0]
                if isinstance(acc, (list, tuple)):
                    acc, batch_size_adjusted = acc

                run_loss.append(loss, count=batch_size)
                run_acc.append(acc, count=batch_size_adjusted)

            avg_loss = run_loss.aggregate()
            avg_acc = run_acc.aggregate()

            if global_rank == 0:
                print(
                    f"Epoch {epoch}/{num_epochs} {idx}/{len(train_loader)} "
                    f"loss: {avg_loss:.4f} acc {avg_acc}  time {time.time() - start_time:.2f}s "
                )
                start_time = time.time()

        # optimizer.zero_grad(set_to_none=True)
        for param in model.parameters():
            param.grad = None

        data = None
        target = None
        data_list = None
        target_list = None
        batch_data = None

        return avg_loss, avg_acc

    @torch.no_grad()
    def val_epoch(
        self,
        model,
        val_loader,
        sliding_inferrer,
        loss_function=None,
        acc_function=None,
        epoch=0,
        rank=0,
        global_rank=0,
        num_epochs=0,
        sigmoid=False,
        use_amp=True,
        use_cuda=True,
        post_transforms=None,
        channels_last=False,
        calc_val_loss=False,
    ):
        model.eval()
        device = torch.device(rank) if use_cuda else torch.device("cpu")
        memory_format = torch.channels_last_3d if channels_last else torch.preserve_format
        distributed = dist.is_initialized()

        run_loss = CumulativeAverage()
        run_acc = CumulativeAverage()
        run_loss.append(torch.tensor(0, device=device), count=0)
        run_acc.append(torch.tensor(0, device=device), count=0)

        avg_loss = avg_acc = 0
        start_time = time.time()

        # In DDP, each replica has a subset of data, but if total data length is not evenly divisible by num_replicas, then some replicas has 1 extra repeated item.
        # For proper validation with batch of 1, we only want to collect metrics for non-repeated items, hence let's compute a proper subset length
        nonrepeated_data_length = len(val_loader.dataset)
        sampler = val_loader.sampler
        if dist.is_initialized and isinstance(sampler, DistributedSampler) and not sampler.drop_last:
            nonrepeated_data_length = len(range(sampler.rank, len(sampler.dataset), sampler.num_replicas))

        for idx, batch_data in enumerate(val_loader):
            data = batch_data["image"].as_subclass(torch.Tensor).to(memory_format=memory_format, device=device)
            filename = batch_data["image"].meta[ImageMetaKey.FILENAME_OR_OBJ]
            batch_size = data.shape[0]

            with autocast(enabled=use_amp):
                logits = sliding_inferrer(inputs=data, network=model)

            data = None

            if post_transforms:
                logits = logits.float().contiguous()
                pred = self.logits2pred(logits, sigmoid=sigmoid, inplace=not calc_val_loss)
                if not calc_val_loss:
                    logits = None

                batch_data["pred"] = convert_to_dst_type(
                    pred, batch_data["image"], dtype=pred.dtype, device=pred.device
                )[0]
                pred = None

                try:
                    # inverting on gpu can OOM due inverse resampling or un-cropping
                    pred = torch.stack([post_transforms(x)["pred"] for x in decollate_batch(batch_data)])
                except RuntimeError as e:
                    if not batch_data["pred"].is_cuda:
                        raise e
                    print(f"post_transforms failed on GPU pred retrying on CPU {batch_data['pred'].shape}")
                    batch_data["pred"] = batch_data["pred"].cpu()
                    pred = torch.stack([post_transforms(x)["pred"] for x in decollate_batch(batch_data)])

                batch_data["pred"] = None
                if logits is not None and pred.shape != logits.shape:
                    logits = None  # if shape has changed due to inverse resampling or un-cropping
            else:
                pred = self.logits2pred(logits, sigmoid=sigmoid, inplace=not calc_val_loss, skip_softmax=True)

            if "label" in batch_data and loss_function is not None and acc_function is not None:
                loss = acc = None
                if idx < nonrepeated_data_length:
                    target = batch_data["label"].as_subclass(torch.Tensor)

                    if calc_val_loss:
                        if logits is not None:
                            loss = loss_function(logits, target.to(device=logits.device))
                            run_loss.append(loss.to(device=device), count=batch_size)
                            logits = None

                    with torch.no_grad():
                        try:
                            acc = acc_function(pred.to(device=device), target.to(device=device))  # try GPU
                        except RuntimeError as e:
                            if "OutOfMemoryError" not in str(type(e).__name__):
                                raise e
                            print(
                                f"acc_function val failed on GPU pred: {pred.shape} on {pred.device}, target: {target.shape} on {target.device}. retrying on CPU"
                            )
                            acc = acc_function(pred.cpu(), target.cpu())

                        batch_size_adjusted = batch_size
                        if isinstance(acc, (list, tuple)):
                            acc, batch_size_adjusted = acc
                        acc = acc.detach().clone()
                        run_acc.append(acc.to(device=device), count=batch_size_adjusted)

                avg_loss = loss.cpu() if loss is not None else 0
                avg_acc = acc.cpu().numpy() if acc is not None else 0
                pred, target = None, None

                if global_rank == 0:
                    print(
                        f"Val {epoch}/{num_epochs} {idx}/{len(val_loader)}  loss: {avg_loss:.4f} "
                        f"acc {avg_acc}  time {time.time() - start_time:.2f}s"
                    )

            else:
                if global_rank == 0:
                    print(f"Val {epoch}/{num_epochs} {idx}/{len(val_loader)} time {time.time() - start_time:.2f}s")

            start_time = time.time()

        pred = target = data = batch_data = None

        if distributed:
            dist.barrier()

        avg_loss = run_loss.aggregate()
        avg_acc = run_acc.aggregate()

        if np.any(avg_acc < 0):
            dist.barrier()
            warnings.warn("Avg dice accuracy is negative, something went wrong!!!!!")

        return avg_loss, avg_acc

    def logits2pred(self, logits, sigmoid=False, dim=1, skip_softmax=False, inplace=False):
        if isinstance(logits, (list, tuple)):
            logits = logits[0]

        if sigmoid:
            pred = torch.sigmoid(logits, out=logits if inplace else None)
        else:
            pred = logits if skip_softmax else torch.softmax(logits, dim=dim, out=logits if inplace else None)

        return pred

    def get_avail_cpu_memory(self):
        avail_memory = psutil.virtual_memory().available

        # check if in docker
        memory_limit_filename = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
        if os.path.exists(memory_limit_filename):
            with open(memory_limit_filename, "r") as f:
                docker_limit = int(f.read())
                avail_memory = min(docker_limit, avail_memory)  # could be lower limit in docker

        return avail_memory

    def get_cache_rate(self, train_cases=0, validation_cases=0, prioritise_train=True):
        config = self.config
        cache_rate = config["cache_rate"]
        avail_memory = None

        total_cases = train_cases + validation_cases

        image_size_mm_90 = config.get("image_size_mm_90", None)
        if config["resample"] and image_size_mm_90 is not None:
            image_size = (
                (np.array(image_size_mm_90) / np.array(config["resample_resolution"])).astype(np.int32).tolist()
            )
        else:
            image_size = config["image_size"]

        approx_data_cache_required = (4 * config["input_channels"] + 1) * np.prod(image_size) * total_cases
        approx_os_cache_required = 50 * 1024**3  # reserve 50gb

        if cache_rate is None:
            cache_rate = 0

            if image_size is not None:
                avail_memory = self.get_avail_cpu_memory()
                cache_rate = min(avail_memory / float(approx_data_cache_required + approx_os_cache_required), 1.0)
                if cache_rate < 0.1:
                    cache_rate = 0.0  # don't cache small

                if self.global_rank == 0:
                    print(
                        f"Calculating cache required {approx_data_cache_required >> 30}GB, available RAM {avail_memory >> 30}GB given avg image size {image_size}."
                    )
                    if cache_rate < 1:
                        print(
                            f"Available RAM is not enought to cache full dataset, caching a fraction {cache_rate:.2f}"
                        )
                    else:
                        print("Caching full dataset in RAM")
            else:
                print("Cant calculate cache_rate since image_size is not provided!!!!")

        else:
            if self.global_rank == 0:
                print(f"Using user specified cache_rate={cache_rate} to cache data in RAM")

        # allocate cache_rate to training files first
        cache_rate_train = cache_rate_val = cache_rate

        if prioritise_train:
            if cache_rate > 0 and cache_rate < 1:
                cache_num = cache_rate * total_cases
                cache_rate_train = min(1.0, cache_num / train_cases) if train_cases > 0 else 0
                if (cache_rate_train < 1 and train_cases > 0) or validation_cases == 0:
                    cache_rate_val = 0
                else:
                    cache_rate_val = (cache_num - cache_rate_train * train_cases) / validation_cases

                if self.global_rank == 0:
                    print(f"Prioritizing cache_rate training {cache_rate_train} validation {cache_rate_val}")

        return cache_rate_train, cache_rate_val

    def save_history_csv(self, csv_path=None, header=None, **kwargs):
        if csv_path is not None:
            if header is not None:
                with open(csv_path, "a") as myfile:
                    wrtr = csv.writer(myfile, delimiter="\t")
                    wrtr.writerow(header)
            if len(kwargs):
                with open(csv_path, "a") as myfile:
                    wrtr = csv.writer(myfile, delimiter="\t")
                    wrtr.writerow(list(kwargs.values()))

    def save_progress_yaml(self, progress_path=None, ckpt=None, **report):
        if ckpt is not None:
            report["model"] = ckpt

        report["date"] = str(datetime.now())[:19]

        if progress_path is not None:
            yaml.add_representer(
                float, lambda dumper, value: dumper.represent_scalar("tag:yaml.org,2002:float", "{0:.4f}".format(value))
            )
            with open(progress_path, "a") as progress_file:
                yaml.dump([report], stream=progress_file, allow_unicode=True, default_flow_style=None, sort_keys=False)

        print("Progress:" + ",".join(f" {k}: {v}" for k, v in report.items()))

    def run(self):
        if self.config["validate"]["enabled"]:
            self.validate()
        elif self.config["infer"]["enabled"]:
            self.infer()
        else:
            self.train()


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

        elif dist_launched() and torch.cuda.device_count() > 1:
            rank = int(os.getenv("LOCAL_RANK"))
            global_rank = int(os.getenv("RANK"))
            world_size = int(os.getenv("LOCAL_WORLD_SIZE"))
            logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.WARNING)
            dist.init_process_group(backend="nccl", init_method="env://")  # torchrun spawned it
            override["mgpu"] = {"world_size": world_size, "rank": rank, "global_rank": global_rank}

            print(f"Distributed launched: initializing multi-gpu env:// process group {override['mgpu']}")

    segmenter = Segmenter(config_file=config_file, config_dict=override, rank=rank, global_rank=global_rank)
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
