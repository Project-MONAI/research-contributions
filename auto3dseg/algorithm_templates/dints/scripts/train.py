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

import contextlib
import ctypes
import gc
import io
import logging
import math
import os
import random
import sys
import time
import warnings
from datetime import datetime, timedelta
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import monai
from monai import transforms
from monai.apps.auto3dseg.auto_runner import logger
from monai.apps.utils import DEFAULT_FMT
from monai.auto3dseg.utils import datafold_read
from monai.bundle import ConfigParser
from monai.bundle.scripts import _pop_args, _update_args
from monai.data import DataLoader, partition_dataset
from monai.inferers import sliding_window_inference
from monai.metrics import compute_dice
from monai.networks.utils import pytorch_after
from monai.utils import set_determinism

try:
    from apex.contrib.clip_grad import clip_grad_norm_
except ModuleNotFoundError:
    from torch.nn.utils import clip_grad_norm_


_libcudart = ctypes.CDLL("libcudart.so")
# Set device limit on the current device
# cudaLimitMaxL2FetchGranularity = 0x05
p_value = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
_libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
_libcudart.cudaDeviceGetLimit(p_value, ctypes.c_int(0x05))
assert p_value.contents.value == 128

torch.backends.cudnn.benchmark = True


CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"monai_default": {"format": DEFAULT_FMT}},
    "loggers": {
        "monai.apps.auto3dseg.auto_runner": {"handlers": ["file", "console"], "level": "DEBUG", "propagate": False}
    },
    "filters": {"rank_filter": {"{}": "__main__.RankFilter"}},
    "handlers": {
        "file": {
            "class": "logging.FileHandler",
            "filename": "runner.log",
            "mode": "a",  # append or overwrite
            "level": "DEBUG",
            "formatter": "monai_default",
            "filters": ["rank_filter"],
        },
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "monai_default",
            "filters": ["rank_filter"],
        },
    },
}


class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = -1

    def __call__(self, val_acc):
        if self.best_score is None:
            self.best_score = val_acc
        elif val_acc + self.delta < self.best_score:
            self.counter += 1
            if self.verbose:
                logger.debug(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_acc
            self.counter = 0


def get_mem_from_visible_gpus():
    available_mem_visible_gpus = []
    for d in range(torch.cuda.device_count()):
        available_mem_visible_gpus.append(torch.cuda.mem_get_info(device=d)[0])
    return available_mem_visible_gpus


def pre_operation(config_file, **override):
    # update hyper-parameter configuration
    rank = int(os.getenv("RANK", "0"))
    if rank == 0:
        if isinstance(config_file, str) and "," in config_file:
            config_file = config_file.split(",")

        for _file in config_file:
            if "hyper_parameters.yaml" in _file:
                parser = ConfigParser(globals=False)
                parser.read_config(_file)

                auto_scale_allowed = parser["training"]["auto_scale_allowed"]
                if "training#auto_scale_allowed" in override:
                    auto_scale_allowed = override["training#auto_scale_allowed"]

                if auto_scale_allowed:
                    output_classes = parser["training"]["output_classes"]
                    mem = get_mem_from_visible_gpus()
                    mem = min(mem) if isinstance(mem, list) else mem
                    mem = float(mem) / (1024.0**3)
                    mem = max(1.0, mem - 1.0)
                    mem_bs2 = 6.0 + (20.0 - 6.0) * (output_classes - 2) / (105 - 2)
                    mem_bs9 = 24.0 + (74.0 - 24.0) * (output_classes - 2) / (105 - 2)
                    batch_size = 2 + (9 - 2) * (mem - mem_bs2) / (mem_bs9 - mem_bs2)
                    batch_size = int(batch_size)
                    batch_size = max(batch_size, 1)

                    parser["training"].update({"num_patches_per_iter": batch_size})
                    parser["training"].update({"num_patches_per_image": 2 * batch_size})

                    # estimate data size based on number of images and image size
                    _factor = 1.0

                    try:
                        _factor *= 1251.0 / float(parser["stats_summary"]["n_cases"])
                        _mean_shape = parser["stats_summary"]["image_stats"]["shape"]["mean"]
                        _factor *= float(_mean_shape[0]) / 240.0
                        _factor *= float(_mean_shape[1]) / 240.0
                        _factor *= float(_mean_shape[2]) / 155.0
                    except BaseException:
                        pass

                    _patch_size = parser["training"]["patch_size"]
                    _factor *= 96.0 / float(_patch_size[0])
                    _factor *= 96.0 / float(_patch_size[1])
                    _factor *= 96.0 / float(_patch_size[2])

                    _factor /= 6.0
                    _factor = max(1.0, _factor)

                    _estimated_epochs = 400.0
                    _estimated_epochs *= _factor

                    parser["training"].update({"num_epochs": int(_estimated_epochs / float(batch_size))})

                    ConfigParser.export_config_file(parser.get(), _file, fmt="yaml", default_flow_style=None)

    return


def run(config_file: Optional[Union[str, Sequence[str]]] = None, **override):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # pre-operations
    logger.debug(f"number of GPUs: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 1:
        logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.WARNING)
        dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(seconds=7200))
        world_size = dist.get_world_size()
    else:
        world_size = 1
    logger.debug(f"world_size: {world_size}")

    pre_operation(config_file, **override)

    if torch.cuda.device_count() > 1:
        dist.barrier()

    if isinstance(config_file, str) and "," in config_file:
        config_file = config_file.split(",")

    torch.set_float32_matmul_precision("high")

    _args = _update_args(config_file=config_file, **override)
    config_file_ = _pop_args(_args, "config_file")[0]

    parser = ConfigParser()
    parser.read_config(config_file_)
    parser.update(pairs=_args)

    amp = parser.get_parsed_content("training#amp")
    bundle_root = parser.get_parsed_content("bundle_root")
    ckpt_path = parser.get_parsed_content("ckpt_path")
    data_file_base_dir = parser.get_parsed_content("data_file_base_dir")
    data_list_file_path = parser.get_parsed_content("data_list_file_path")
    finetune = parser.get_parsed_content("finetune")
    fold = parser.get_parsed_content("fold")
    log_output_file = parser.get_parsed_content("training#log_output_file")
    num_images_per_batch = parser.get_parsed_content("training#num_images_per_batch")
    num_epochs = parser.get_parsed_content("training#num_epochs")
    num_epochs_per_validation = parser.get_parsed_content("training#num_epochs_per_validation")
    num_patches_per_iter = parser.get_parsed_content("training#num_patches_per_iter")
    num_sw_batch_size = parser.get_parsed_content("training#num_sw_batch_size")
    output_classes = parser.get_parsed_content("training#output_classes")
    overlap_ratio = parser.get_parsed_content("training#overlap_ratio")
    overlap_ratio_train = parser.get_parsed_content("training#overlap_ratio_train")
    patch_size_valid = parser.get_parsed_content("training#patch_size_valid")
    random_seed = parser.get_parsed_content("training#random_seed")
    softmax = parser.get_parsed_content("training#softmax")
    sw_input_on_cpu = parser.get_parsed_content("training#sw_input_on_cpu")
    valid_at_orig_resolution_at_last = parser.get_parsed_content("training#valid_at_orig_resolution_at_last")
    valid_at_orig_resolution_only = parser.get_parsed_content("training#valid_at_orig_resolution_only")

    if not valid_at_orig_resolution_only:
        train_transforms = parser.get_parsed_content("transforms_train")
        val_transforms = parser.get_parsed_content("transforms_validate")

    if valid_at_orig_resolution_at_last or valid_at_orig_resolution_only:
        infer_transforms = parser.get_parsed_content("transforms_infer")
        infer_transforms = transforms.Compose(
            [
                infer_transforms,
                transforms.LoadImaged(keys="label", image_only=False),
                transforms.EnsureChannelFirstd(keys="label"),
                transforms.EnsureTyped(keys="label"),
            ]
        )

        if "class_names" in parser and isinstance(parser["class_names"], list) and "index" in parser["class_names"][0]:
            class_index = [x["index"] for x in parser["class_names"]]

            infer_transforms = transforms.Compose(
                [
                    infer_transforms,
                    transforms.Lambdad(
                        keys="label",
                        func=lambda x: torch.cat([sum([x == i for i in c]) for c in class_index], dim=0).to(
                            dtype=x.dtype
                        ),
                    ),
                ]
            )

    class_names = None
    try:
        class_names = parser.get_parsed_content("class_names")
        if isinstance(class_names[0], dict):
            class_names = [class_names[_i]["name"] for _i in range(len(class_names))]
    except BaseException:
        pass

    ad = parser.get_parsed_content("training#adapt_valid_mode")
    if ad:
        ad_progress_percentages = parser.get_parsed_content("training#adapt_valid_progress_percentages")
        ad_num_epochs_per_validation = parser.get_parsed_content("training#adapt_valid_num_epochs_per_validation")

        sorted_indices = np.argsort(ad_progress_percentages)
        ad_progress_percentages = [ad_progress_percentages[_i] for _i in sorted_indices]
        ad_num_epochs_per_validation = [ad_num_epochs_per_validation[_i] for _i in sorted_indices]

    es = parser.get_parsed_content("training#early_stop_mode")
    if es:
        es_delta = parser.get_parsed_content("training#early_stop_delta")
        es_patience = parser.get_parsed_content("training#early_stop_patience")

    ad = parser.get_parsed_content("training#adapt_valid_mode")
    if ad:
        ad_progress_percentages = parser.get_parsed_content("training#adapt_valid_progress_percentages")
        ad_num_epochs_per_validation = parser.get_parsed_content("training#adapt_valid_num_epochs_per_validation")

        sorted_indices = np.argsort(ad_progress_percentages)
        ad_progress_percentages = [ad_progress_percentages[_i] for _i in sorted_indices]
        ad_num_epochs_per_validation = [ad_num_epochs_per_validation[_i] for _i in sorted_indices]

    es = parser.get_parsed_content("training#early_stop_mode")
    if es:
        es_delta = parser.get_parsed_content("training#early_stop_delta")
        es_patience = parser.get_parsed_content("training#early_stop_patience")

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path, exist_ok=True)

    if random_seed is not None and (isinstance(random_seed, int) or isinstance(random_seed, float)):
        set_determinism(seed=random_seed)

    CONFIG["handlers"]["file"]["filename"] = log_output_file
    logging.config.dictConfig(CONFIG)

    train_data_list_key = parser.get_parsed_content("training#data_list_key")
    valid_data_list_key = parser.get_parsed_content("validate#data_list_key")
    if valid_data_list_key is not None:
        train_files, _ = datafold_read(
            datalist=data_list_file_path, basedir=data_file_base_dir, fold=-1, key=train_data_list_key
        )
        val_files, _ = datafold_read(
            datalist=data_list_file_path, basedir=data_file_base_dir, fold=-1, key=valid_data_list_key
        )
    else:
        train_files, val_files = datafold_read(datalist=data_list_file_path, basedir=data_file_base_dir, fold=fold)

    random.shuffle(train_files)

    if torch.cuda.device_count() > 1:
        train_files = partition_dataset(data=train_files, shuffle=True, num_partitions=world_size, even_divisible=True)[
            dist.get_rank()
        ]
    logger.debug(f"train_files: {len(train_files)}")

    if torch.cuda.device_count() > 1:
        if len(val_files) < world_size:
            val_files = val_files * math.ceil(float(world_size) / float(len(val_files)))

        val_files = partition_dataset(data=val_files, shuffle=False, num_partitions=world_size, even_divisible=False)[
            dist.get_rank()
        ]
    logger.debug(f"val_files: {len(val_files)}")

    train_cache_rate = float(parser.get_parsed_content("training#train_cache_rate"))
    validate_cache_rate = float(parser.get_parsed_content("training#validate_cache_rate"))

    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        warnings.simplefilter(action="ignore", category=Warning)

        if not valid_at_orig_resolution_only:
            train_ds = monai.data.CacheDataset(
                data=train_files * num_epochs_per_validation,
                transform=train_transforms,
                cache_rate=train_cache_rate,
                hash_as_key=True,
                num_workers=parser.get_parsed_content("training#num_cache_workers"),
                progress=parser.get_parsed_content("show_cache_progress"),
            )
            val_ds = monai.data.CacheDataset(
                data=val_files,
                transform=val_transforms,
                cache_rate=validate_cache_rate,
                hash_as_key=True,
                num_workers=parser.get_parsed_content("training#num_cache_workers"),
                progress=parser.get_parsed_content("show_cache_progress"),
            )

        if valid_at_orig_resolution_at_last or valid_at_orig_resolution_only:
            orig_val_ds = monai.data.Dataset(data=val_files, transform=infer_transforms)

    if not valid_at_orig_resolution_only:
        train_loader = DataLoader(
            train_ds,
            num_workers=parser.get_parsed_content("training#num_workers"),
            batch_size=num_images_per_batch,
            shuffle=True,
            persistent_workers=True,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            num_workers=parser.get_parsed_content("training#num_workers_validation"),
            batch_size=1,
            shuffle=False,
        )

    if valid_at_orig_resolution_at_last or valid_at_orig_resolution_only:
        orig_val_loader = DataLoader(orig_val_ds, num_workers=2, batch_size=1, shuffle=False)

    device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}") if world_size > 1 else torch.device("cuda:0")

    if world_size > 1:
        parser["training_network"]["dints_space"]["device"] = device

    with io.StringIO() as buffer, contextlib.redirect_stdout(buffer):
        model = parser.get_parsed_content("training_network#network")
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if softmax:
        post_pred = transforms.Compose([transforms.EnsureType(), transforms.AsDiscrete(argmax=True, to_onehot=None)])
    else:
        post_pred = transforms.Compose(
            [
                transforms.EnsureType(),
                transforms.Activations(sigmoid=True),
                transforms.AsDiscrete(threshold=0.5 + np.finfo(np.float32).eps),
            ]
        )

    if valid_at_orig_resolution_at_last or valid_at_orig_resolution_only:
        post_transforms = [
            transforms.Invertd(
                keys="pred",
                transform=infer_transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            )
        ]

        if softmax:
            post_transforms += [transforms.AsDiscreted(keys="pred", argmax=True)]
        else:
            post_transforms += [
                transforms.Activationsd(keys="pred", sigmoid=True),
                transforms.AsDiscreted(keys="pred", threshold=0.5 + np.finfo(np.float32).eps),
            ]

        post_transforms = transforms.Compose(post_transforms)

    loss_function = parser.get_parsed_content("training#loss")

    optimizer_part = parser.get_parsed_content("training#optimizer", instantiate=False)
    optimizer = optimizer_part.instantiate(params=model.parameters())

    if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
        logger.debug(f"num_epochs: {num_epochs}")
        logger.debug(f"num_epochs_per_validation: {num_epochs_per_validation}")

    # patch fix to support PolynomialLR use in PyTorch <= 1.12
    if "PolynomialLR" in parser.get("training#lr_scheduler#_target_") and not pytorch_after(1, 13):
        dints_dir = os.path.dirname(os.path.dirname(__file__))
        sys.path.insert(0, dints_dir)
        parser["training#lr_scheduler#_target_"] = "scripts.utils.PolynomialLR"

    lr_scheduler_part = parser.get_parsed_content("training#lr_scheduler", instantiate=False)
    lr_scheduler = lr_scheduler_part.instantiate(optimizer=optimizer)

    if torch.cuda.device_count() > 1:
        model = DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)

    if finetune["activate"] and os.path.isfile(finetune["pretrained_ckpt_name"]):
        logger.debug("fine-tuning pre-trained checkpoint {:s}".format(finetune["pretrained_ckpt_name"]))
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(torch.load(finetune["pretrained_ckpt_name"], map_location=device))
        else:
            model.load_state_dict(torch.load(finetune["pretrained_ckpt_name"], map_location=device))
    else:
        logger.debug("training from scratch")

    if amp:
        from torch.cuda.amp import GradScaler, autocast

        scaler = GradScaler()
        if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
            logger.debug("amp enabled")

    best_metric = -1
    best_metric_epoch = -1
    idx_iter = 0
    metric_dim = output_classes - 1 if softmax else output_classes
    val_devices_input = {}
    val_devices_output = {}

    if es:
        stop_train = torch.tensor(False).to(device)

    if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
        writer = SummaryWriter(log_dir=os.path.join(ckpt_path, "Events"))

        with open(os.path.join(ckpt_path, "accuracy_history.csv"), "a") as f:
            f.write("epoch\tmetric\tloss\tlr\ttime\titer\n")

        if es:
            # instantiate the early stopping object
            early_stopping = EarlyStopping(patience=es_patience, delta=es_delta, verbose=True)

    start_time = time.time()

    num_rounds = int(np.ceil(float(num_epochs) // float(num_epochs_per_validation)))

    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        warnings.simplefilter(action="ignore", category=Warning)

        if not valid_at_orig_resolution_only:
            if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
                progress_bar = tqdm(
                    range(num_rounds), desc=f"{os.path.basename(bundle_root)} - training ...", unit="round"
                )

            for _round in range(num_rounds) if torch.cuda.device_count() > 1 and dist.get_rank() != 0 else progress_bar:
                epoch = (_round + 1) * num_epochs_per_validation
                lr = lr_scheduler.get_last_lr()[0]
                if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
                    logger.debug("----------")
                    logger.debug(f"epoch {_round * num_epochs_per_validation + 1}/{num_epochs}")
                    logger.debug(f"learning rate is set to {lr}")

                model.train()
                epoch_loss = 0
                loss_torch = torch.zeros(2, dtype=torch.float, device=device)
                step = 0

                for batch_data in train_loader:
                    step += 1

                    inputs_l = (
                        batch_data["image"].as_tensor()
                        if isinstance(batch_data["image"], monai.data.MetaTensor)
                        else batch_data["image"]
                    )
                    labels_l = (
                        batch_data["label"].as_tensor()
                        if isinstance(batch_data["label"], monai.data.MetaTensor)
                        else batch_data["label"]
                    )

                    _idx = torch.randperm(inputs_l.shape[0])
                    inputs_l = inputs_l[_idx]
                    labels_l = labels_l[_idx]

                    for _k in range(inputs_l.shape[0] // num_patches_per_iter):
                        inputs = inputs_l[_k * num_patches_per_iter : (_k + 1) * num_patches_per_iter, ...]
                        labels = labels_l[_k * num_patches_per_iter : (_k + 1) * num_patches_per_iter, ...]

                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        for param in model.parameters():
                            param.grad = None

                        if amp:
                            with autocast():
                                outputs = model(inputs)
                                loss = loss_function(outputs.float(), labels)

                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                            clip_grad_norm_(model.parameters(), 0.5)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            outputs = model(inputs)
                            loss = loss_function(outputs.float(), labels)

                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                            optimizer.step()

                        epoch_loss += loss.item()
                        loss_torch[0] += loss.item()
                        loss_torch[1] += 1.0
                        epoch_len = len(train_loader)
                        idx_iter += 1

                        if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
                            logger.debug(
                                f"[{str(datetime.now())[:19]}] " + f"{step}/{epoch_len}, train_loss: {loss.item():.4f}"
                            )
                            writer.add_scalar("train/loss", loss.item(), epoch_len * _round + step)

                lr_scheduler.step()

                if torch.cuda.device_count() > 1:
                    dist.all_reduce(loss_torch, op=torch.distributed.ReduceOp.SUM)

                loss_torch = loss_torch.tolist()
                if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
                    loss_torch_epoch = loss_torch[0] / loss_torch[1]
                    logger.debug(
                        f"epoch {epoch} average loss: {loss_torch_epoch:.4f}, "
                        f"best mean dice: {best_metric:.4f} at epoch {best_metric_epoch}"
                    )

                del inputs, labels, outputs
                torch.cuda.empty_cache()
                gc.collect()

                if ad:
                    _percentage = float(_round) / float(num_rounds) * 100.0

                    target_num_epochs_per_validation = -1
                    for _j in range(len(ad_progress_percentages)):
                        if _percentage <= ad_progress_percentages[-1 - _j]:
                            if (
                                _j == (len(ad_progress_percentages) - 1)
                                or _percentage > ad_progress_percentages[-2 - _j]
                            ):
                                target_num_epochs_per_validation = ad_num_epochs_per_validation[-1 - _j]
                                break

                    if target_num_epochs_per_validation > 0 and (_round + 1) < num_rounds:
                        if (_round + 1) % (target_num_epochs_per_validation // num_epochs_per_validation) != 0:
                            continue

                model.eval()
                with torch.no_grad():
                    metric = torch.zeros(metric_dim * 2, dtype=torch.float, device=device)

                    _index = 0
                    for val_data in val_loader:
                        finished = None
                        device_list_input = None
                        device_list_output = None

                        val_filename = val_data["image_meta_dict"]["filename_or_obj"][0]
                        if sw_input_on_cpu:
                            device_list_input = ["cpu"]
                            device_list_output = ["cpu"]
                        elif val_filename not in val_devices_input or val_filename not in val_devices_output:
                            device_list_input = [device, device, "cpu"]
                            device_list_output = [device, "cpu", "cpu"]
                        elif val_filename in val_devices_input and val_filename in val_devices_output:
                            device_list_input = [val_devices_input[val_filename]]
                            device_list_output = [val_devices_output[val_filename]]

                        for _device_in, _device_out in zip(device_list_input, device_list_output):
                            try:
                                val_devices_input[val_filename] = _device_in
                                val_devices_output[val_filename] = _device_out

                                val_images = val_data["image"].to(_device_in)
                                val_labels = val_data["label"].to(_device_out)

                                if num_sw_batch_size is None:
                                    sw_batch_size = num_patches_per_iter * 12 if _device_out == "cpu" else 1
                                else:
                                    sw_batch_size = num_sw_batch_size

                                with autocast(enabled=amp):
                                    val_outputs = sliding_window_inference(
                                        inputs=val_images,
                                        roi_size=patch_size_valid,
                                        sw_batch_size=sw_batch_size,
                                        predictor=model,
                                        mode="gaussian",
                                        overlap=overlap_ratio_train,
                                        sw_device=device,
                                        device=_device_out,
                                    )

                                finished = True

                            except RuntimeError as e:
                                if not any(x in str(e).lower() for x in ("memory", "cuda", "cudnn")):
                                    raise e

                                finished = False

                            if finished:
                                break

                        del val_images
                        val_labels = val_labels.cpu()
                        val_outputs = val_outputs.cpu()
                        torch.cuda.empty_cache()
                        gc.collect()

                        val_outputs = post_pred(val_outputs[0, ...])
                        val_outputs = val_outputs[None, ...]

                        val_labels = val_labels.to(_device_in)
                        val_outputs = val_outputs.to(_device_in)

                        if softmax:
                            val_labels = val_labels.int()
                            value = torch.zeros(1, metric_dim).to(device)
                            for _k in range(1, metric_dim + 1):
                                value[0, _k - 1] = compute_dice(
                                    y_pred=(val_outputs == _k).float(),
                                    y=(val_labels == _k).float(),
                                    include_background=not softmax,
                                )
                        else:
                            value = compute_dice(y_pred=val_outputs, y=val_labels, include_background=not softmax)
                            value = value.to(device)

                        logger.debug(f"{_index + 1} / {len(val_loader)}: {value}")

                        del val_labels, val_outputs
                        torch.cuda.empty_cache()
                        gc.collect()

                        for _c in range(metric_dim):
                            val0 = torch.nan_to_num(value[0, _c], nan=0.0)
                            val1 = 1.0 - torch.isnan(value[0, _c]).float()
                            metric[2 * _c] += val0
                            metric[2 * _c + 1] += val1

                        _index += 1

                    if torch.cuda.device_count() > 1:
                        dist.all_reduce(metric, op=torch.distributed.ReduceOp.SUM)

                    metric = metric.tolist()
                    if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
                        for _c in range(metric_dim):
                            logger.debug(f"evaluation metric - class {_c + 1}: {metric[2 * _c] / metric[2 * _c + 1]}")
                            try:
                                writer.add_scalar(
                                    f"val_class/acc_{class_names[_c]}", metric[2 * _c] / metric[2 * _c + 1], epoch
                                )
                            except BaseException:
                                writer.add_scalar(f"val_class/acc_{_c}", metric[2 * _c] / metric[2 * _c + 1], epoch)

                        avg_metric = 0
                        for _c in range(metric_dim):
                            avg_metric += metric[2 * _c] / metric[2 * _c + 1]
                        avg_metric = avg_metric / float(metric_dim)
                        logger.debug(f"avg_metric: {avg_metric}")

                        writer.add_scalar("val/acc", avg_metric, epoch)

                        if avg_metric > best_metric:
                            best_metric = avg_metric
                            best_metric_epoch = epoch
                            if torch.cuda.device_count() > 1:
                                torch.save(model.module.state_dict(), os.path.join(ckpt_path, "best_metric_model.pt"))
                            else:
                                torch.save(model.state_dict(), os.path.join(ckpt_path, "best_metric_model.pt"))
                            logger.debug("saved new best metric model")

                            dict_file = {}
                            dict_file["best_avg_dice_score"] = float(best_metric)
                            dict_file["best_avg_dice_score_epoch"] = int(best_metric_epoch)
                            dict_file["best_avg_dice_score_iteration"] = int(idx_iter)
                            dict_file["inverted_best_validation"] = False
                            with open(os.path.join(ckpt_path, "progress.yaml"), "a") as out_file:
                                yaml.dump([dict_file], stream=out_file)

                        logger.debug(
                            "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                                epoch, avg_metric, best_metric, best_metric_epoch
                            )
                        )

                        current_time = time.time()
                        elapsed_time = (current_time - start_time) / 60.0
                        with open(os.path.join(ckpt_path, "accuracy_history.csv"), "a") as f:
                            f.write(
                                "{:d}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.1f}\t{:d}\n".format(
                                    epoch, avg_metric, loss_torch_epoch, lr, elapsed_time, idx_iter
                                )
                            )

                        if es:
                            early_stopping(val_acc=avg_metric)
                            stop_train = torch.tensor(early_stopping.early_stop).to(device)

                    if torch.cuda.device_count() > 1:
                        dist.barrier()

                    if es:
                        if torch.cuda.device_count() > 1:
                            dist.broadcast(stop_train, src=0)
                        if stop_train:
                            break

                torch.cuda.empty_cache()
                gc.collect()

        if valid_at_orig_resolution_at_last or valid_at_orig_resolution_only:
            if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
                print(f"{os.path.basename(bundle_root)} - validation at original spacing/resolution")
                logger.debug("validation at original spacing/resolution")

            if torch.cuda.device_count() > 1:
                model.module.load_state_dict(
                    torch.load(os.path.join(ckpt_path, "best_metric_model.pt"), map_location=device)
                )
            else:
                model.load_state_dict(torch.load(os.path.join(ckpt_path, "best_metric_model.pt"), map_location=device))
            logger.debug("checkpoints loaded")

            model.eval()
            with torch.no_grad():
                metric = torch.zeros(metric_dim * 2, dtype=torch.float, device=device)

                _index = 0
                for val_data in orig_val_loader:
                    finished = None
                    device_list_input = None
                    device_list_output = None

                    if sw_input_on_cpu:
                        device_list_input = ["cpu"]
                        device_list_output = ["cpu"]
                    else:
                        device_list_input = [device, device, "cpu"]
                        device_list_output = [device, "cpu", "cpu"]

                    for _device_in, _device_out in zip(device_list_input, device_list_output):
                        try:
                            val_images = val_data["image"].to(_device_in)
                            val_labels = val_data["label"].to(_device_out)

                            if num_sw_batch_size is None:
                                sw_batch_size = num_patches_per_iter * 12 if _device_out == "cpu" else 1
                            else:
                                sw_batch_size = num_sw_batch_size

                            with autocast(enabled=amp):
                                val_data["pred"] = sliding_window_inference(
                                    inputs=val_images,
                                    roi_size=patch_size_valid,
                                    sw_batch_size=sw_batch_size,
                                    predictor=model,
                                    mode="gaussian",
                                    overlap=overlap_ratio,
                                    sw_device=device,
                                    device=_device_out,
                                )

                            finished = True

                        except RuntimeError as e:
                            if not any(x in str(e).lower() for x in ("memory", "cuda", "cudnn")):
                                raise e

                            finished = False

                        if finished:
                            break

                    del val_images
                    val_data["image"] = val_data["image"].cpu()
                    val_data["label"] = val_data["label"].cpu()
                    val_data["pred"] = val_data["pred"].cpu()
                    torch.cuda.empty_cache()
                    gc.collect()

                    val_data = [post_transforms(i) for i in monai.data.decollate_batch(val_data)]
                    val_outputs = val_data[0]["pred"][None, ...]

                    val_labels = val_labels.to(_device_in)
                    val_outputs = val_outputs.to(_device_in)

                    if softmax:
                        val_labels = val_labels.int()
                        value = torch.zeros(1, metric_dim)
                        for _k in range(1, metric_dim + 1):
                            value[0, _k - 1] = compute_dice(
                                y_pred=(val_outputs == _k).float(),
                                y=(val_labels == _k).float(),
                                include_background=not softmax,
                            )
                    else:
                        value = compute_dice(y_pred=val_outputs, y=val_labels, include_background=not softmax)

                    logger.debug(f"validation Dice score at original spacing/resolution: {value}")

                    for _c in range(metric_dim):
                        val0 = torch.nan_to_num(value[0, _c], nan=0.0)
                        val1 = 1.0 - torch.isnan(value[0, _c]).float()
                        metric[2 * _c] += val0
                        metric[2 * _c + 1] += val1

                    _index += 1

                if torch.cuda.device_count() > 1:
                    dist.all_reduce(metric, op=torch.distributed.ReduceOp.SUM)

                metric = metric.tolist()
                if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
                    for _c in range(metric_dim):
                        logger.debug(
                            f"evaluation metric at original spacing/resolution - class {_c + 1}: {metric[2 * _c] / metric[2 * _c + 1]}"
                        )

                    avg_metric = 0
                    for _c in range(metric_dim):
                        avg_metric += metric[2 * _c] / metric[2 * _c + 1]
                    avg_metric = avg_metric / float(metric_dim)
                    logger.debug(f"avg_metric at original spacing/resolution: {avg_metric}")

                    with open(os.path.join(ckpt_path, "progress.yaml"), "r") as out_file:
                        progress = yaml.safe_load(out_file)

                    dict_file = {}
                    dict_file["best_avg_dice_score"] = float(avg_metric)
                    dict_file["best_avg_dice_score_epoch"] = int(progress[-1]["best_avg_dice_score_epoch"])
                    dict_file["best_avg_dice_score_iteration"] = int(progress[-1]["best_avg_dice_score_iteration"])
                    dict_file["inverted_best_validation"] = True
                    with open(os.path.join(ckpt_path, "progress.yaml"), "a") as out_file:
                        yaml.dump([dict_file], stream=out_file)

                if torch.cuda.device_count() > 1:
                    dist.barrier()

    if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
        logger.debug(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

        writer.flush()
        writer.close()

    if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
        if (not valid_at_orig_resolution_only) and es and (_round + 1) < num_rounds:
            logger.warning(f"{os.path.basename(bundle_root)} - training: finished with early stop")
        else:
            logger.warning(f"{os.path.basename(bundle_root)} - training: finished")

    if torch.cuda.device_count() > 1:
        dist.destroy_process_group()

    return


if __name__ == "__main__":
    from monai.utils import optional_import

    fire, _ = optional_import("fire")
    fire.Fire()
