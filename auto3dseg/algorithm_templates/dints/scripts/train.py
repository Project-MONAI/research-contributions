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
import io
import logging
import math
import os
import random
import sys
import time
import warnings
from datetime import datetime
from tqdm import tqdm
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

import monai
from apex.contrib.clip_grad import clip_grad_norm_
from monai import transforms
from monai.apps.auto3dseg.auto_runner import logger
from monai.apps.utils import DEFAULT_FMT
from monai.bundle import ConfigParser
from monai.bundle.scripts import _pop_args, _update_args
from monai.data import DataLoader, partition_dataset
from monai.inferers import sliding_window_inference
from monai.metrics import compute_dice
from monai.utils import set_determinism


_libcudart = ctypes.CDLL("libcudart.so")
# Set device limit on the current device
# cudaLimitMaxL2FetchGranularity = 0x05
p_value = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
_libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
_libcudart.cudaDeviceGetLimit(p_value, ctypes.c_int(0x05))
assert p_value.contents.value == 128

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


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
                logger.debug(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_acc
            self.counter = 0


def run(config_file: Optional[Union[str, Sequence[str]]] = None, **override):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    if isinstance(config_file, str) and ',' in config_file:
        config_file = config_file.split(',')

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
    num_images_per_batch = parser.get_parsed_content(
        "training#num_images_per_batch")
    num_epochs = parser.get_parsed_content("training#num_epochs")
    num_epochs_per_validation = parser.get_parsed_content(
        "training#num_epochs_per_validation")
    num_sw_batch_size = parser.get_parsed_content("training#num_sw_batch_size")
    num_patches_per_iter = parser.get_parsed_content(
        "training#num_patches_per_iter")
    output_classes = parser.get_parsed_content("training#output_classes")
    overlap_ratio = parser.get_parsed_content("training#overlap_ratio")
    patch_size_valid = parser.get_parsed_content("training#patch_size_valid")
    random_seed = parser.get_parsed_content("training#random_seed")
    sw_input_on_cpu = parser.get_parsed_content("training#sw_input_on_cpu")
    softmax = parser.get_parsed_content("training#softmax")

    train_transforms = parser.get_parsed_content("transforms_train")
    val_transforms = parser.get_parsed_content("transforms_validate")

    ad = parser.get_parsed_content("training#adapt_valid_mode")
    if ad:
        ad_progress_percentages = parser.get_parsed_content(
            "training#adapt_valid_progress_percentages")
        ad_num_epochs_per_validation = parser.get_parsed_content(
            "training#adapt_valid_num_epochs_per_validation")

        sorted_indices = np.argsort(ad_progress_percentages)
        ad_progress_percentages = [
            ad_progress_percentages[_i] for _i in sorted_indices]
        ad_num_epochs_per_validation = [
            ad_num_epochs_per_validation[_i] for _i in sorted_indices]

    es = parser.get_parsed_content("training#early_stop_mode")
    if es:
        es_delta = parser.get_parsed_content("training#early_stop_delta")
        es_patience = parser.get_parsed_content(
            "training#early_stop_patience")

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path, exist_ok=True)

    if random_seed is not None and (
        isinstance(
            random_seed,
            int) or isinstance(
            random_seed,
            float)):
        set_determinism(seed=random_seed)

    CONFIG["handlers"]["file"]["filename"] = log_output_file
    logging.config.dictConfig(CONFIG)

    logger.debug(f"[info] number of GPUs: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 1:
        logging.getLogger("torch.distributed.distributed_c10d").setLevel(
            logging.WARNING)
        dist.init_process_group(backend="nccl", init_method="env://")
        world_size = dist.get_world_size()
    else:
        world_size = 1
    logger.debug(f"[info] world_size: {world_size}")

    datalist = ConfigParser.load_config_file(data_list_file_path)

    list_train = []
    list_valid = []
    for item in datalist["training"]:
        if item["fold"] == fold:
            item.pop("fold", None)
            list_valid.append(item)
        else:
            item.pop("fold", None)
            list_train.append(item)

    files = []
    for _i in range(len(list_train)):
        str_img = os.path.join(data_file_base_dir, list_train[_i]["image"])
        str_seg = os.path.join(data_file_base_dir, list_train[_i]["label"])

        if (not os.path.exists(str_img)) or (not os.path.exists(str_seg)):
            continue

        files.append({"image": str_img, "label": str_seg})

    train_files = files
    random.shuffle(train_files)

    if torch.cuda.device_count() > 1:
        train_files = partition_dataset(
            data=train_files,
            shuffle=True,
            num_partitions=world_size,
            even_divisible=True)[
            dist.get_rank()]
    logger.debug(f"train_files: {len(train_files)}")

    files = []
    for _i in range(len(list_valid)):
        str_img = os.path.join(data_file_base_dir, list_valid[_i]["image"])
        str_seg = os.path.join(data_file_base_dir, list_valid[_i]["label"])

        if (not os.path.exists(str_img)) or (not os.path.exists(str_seg)):
            continue

        files.append({"image": str_img, "label": str_seg})

    val_files = files

    if torch.cuda.device_count() > 1:
        if len(val_files) < world_size:
            val_files = val_files * \
                math.ceil(float(world_size) / float(len(val_files)))

        val_files = partition_dataset(
            data=val_files,
            shuffle=False,
            num_partitions=world_size,
            even_divisible=False)[
            dist.get_rank()]
    logger.debug(f"val_files: {len(val_files)}")

    train_cache_rate = float(parser.get_parsed_content("train_cache_rate"))
    validate_cache_rate = float(
        parser.get_parsed_content("validate_cache_rate"))

    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        warnings.simplefilter(action="ignore", category=Warning)

        train_ds = monai.data.CacheDataset(
            data=train_files * num_epochs_per_validation,
            transform=train_transforms,
            cache_rate=train_cache_rate,
            hash_as_key=True,
            num_workers=parser.get_parsed_content("training#num_cache_workers"),
            progress=parser.get_parsed_content("show_cache_progress"))
        val_ds = monai.data.CacheDataset(
            data=val_files,
            transform=val_transforms,
            cache_rate=validate_cache_rate,
            hash_as_key=True,
            num_workers=parser.get_parsed_content("training#num_cache_workers"),
            progress=parser.get_parsed_content("show_cache_progress"))

    train_loader = DataLoader(
        train_ds,
        num_workers=parser.get_parsed_content("training#num_workers"),
        batch_size=num_images_per_batch,
        shuffle=True,
        persistent_workers=True,
        pin_memory=True)
    val_loader = DataLoader(
        val_ds,
        num_workers=parser.get_parsed_content("training#num_workers_validation"),
        batch_size=1,
        shuffle=False,
        persistent_workers=True,
        pin_memory=True)

    device = torch.device(
        f"cuda:{os.environ['LOCAL_RANK']}") if world_size > 1 else torch.device("cuda:0")

    with io.StringIO() as buffer, contextlib.redirect_stdout(buffer):
        model = parser.get_parsed_content("training_network#network")
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if softmax:
        post_pred = transforms.Compose([transforms.EnsureType(
        ), transforms.AsDiscrete(argmax=True, to_onehot=output_classes)])
    else:
        post_pred = transforms.Compose([transforms.EnsureType(), transforms.Activations(
            sigmoid=True), transforms.AsDiscrete(threshold=0.5)])

    loss_function = parser.get_parsed_content("training#loss")

    optimizer_part = parser.get_parsed_content(
        "training#optimizer", instantiate=False)
    optimizer = optimizer_part.instantiate(params=model.parameters())

    if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
        logger.debug(f"num_epochs: {num_epochs}")
        logger.debug(f"num_epochs_per_validation: {num_epochs_per_validation}")

    lr_scheduler_part = parser.get_parsed_content(
        "training#lr_scheduler", instantiate=False)
    lr_scheduler = lr_scheduler_part.instantiate(optimizer=optimizer)

    if torch.cuda.device_count() > 1:
        model = DistributedDataParallel(
            model, device_ids=[device], find_unused_parameters=True)

    if finetune["activate"] and os.path.isfile(
            finetune["pretrained_ckpt_name"]):
        logger.debug(
            "[info] fine-tuning pre-trained checkpoint {:s}".format(finetune["pretrained_ckpt_name"]))
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(
                torch.load(
                    finetune["pretrained_ckpt_name"],
                    map_location=device))
        else:
            model.load_state_dict(
                torch.load(
                    finetune["pretrained_ckpt_name"],
                    map_location=device))
    else:
        logger.debug("[info] training from scratch")

    if amp:
        from torch.cuda.amp import GradScaler, autocast

        scaler = GradScaler()
        if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
            logger.debug("[info] amp enabled")

    best_metric = -1
    best_metric_epoch = -1
    idx_iter = 0
    metric_dim = output_classes - 1 if softmax else output_classes
    val_devices = {}

    if es:
        stop_train = torch.tensor(False).to(device)

    if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
        writer = SummaryWriter(log_dir=os.path.join(ckpt_path, "Events"))

        with open(os.path.join(ckpt_path, "accuracy_history.csv"), "a") as f:
            f.write("epoch\tmetric\tloss\tlr\ttime\titer\n")

        if es:
            # instantiate the early stopping object
            early_stopping = EarlyStopping(
                patience=es_patience,
                delta=es_delta,
                verbose=True)

    start_time = time.time()

    num_rounds = int(
        np.ceil(
            float(num_epochs) //
            float(num_epochs_per_validation)))

    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        warnings.simplefilter(action="ignore", category=Warning)

        if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
            progress_bar = tqdm(
                range(num_rounds),
                desc=f"{os.path.basename(bundle_root)} - training ...",
                unit="round")

        for _round in range(num_rounds) if torch.cuda.device_count(
        ) > 1 and dist.get_rank() != 0 else progress_bar:
            epoch = (_round + 1) * num_epochs_per_validation
            lr = lr_scheduler.get_last_lr()[0]
            if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
                logger.debug("----------")
                logger.debug(
                    f"epoch {_round * num_epochs_per_validation + 1}/{num_epochs}")
                logger.debug(f"learning rate is set to {lr}")

            model.train()
            epoch_loss = 0
            loss_torch = torch.zeros(2, dtype=torch.float, device=device)
            step = 0

            for batch_data in train_loader:
                step += 1

                inputs_l = batch_data["image"].as_tensor() if isinstance(
                    batch_data["image"], monai.data.MetaTensor) else batch_data["image"]
                labels_l = batch_data["label"].as_tensor() if isinstance(
                    batch_data["label"], monai.data.MetaTensor) else batch_data["label"]

                _idx = torch.randperm(inputs_l.shape[0])
                inputs_l = inputs_l[_idx]
                labels_l = labels_l[_idx]

                for _k in range(inputs_l.shape[0] // num_patches_per_iter):
                    inputs = inputs_l[_k *
                                      num_patches_per_iter:(_k +
                                                            1) *
                                      num_patches_per_iter, ...]
                    labels = labels_l[_k *
                                      num_patches_per_iter:(_k +
                                                            1) *
                                      num_patches_per_iter, ...]

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
                        clip_grad_norm_(model.parameters(), 0.5)
                        optimizer.step()

                    epoch_loss += loss.item()
                    loss_torch[0] += loss.item()
                    loss_torch[1] += 1.0
                    epoch_len = len(train_loader)
                    idx_iter += 1

                    if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
                        logger.debug(
                            f"[{str(datetime.now())[:19]}] " +
                            f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
                        writer.add_scalar(
                            "train/loss",
                            loss.item(),
                            epoch_len *
                            num_rounds +
                            step)

            lr_scheduler.step()

            if torch.cuda.device_count() > 1:
                dist.barrier()
                dist.all_reduce(loss_torch, op=torch.distributed.ReduceOp.SUM)

            loss_torch = loss_torch.tolist()
            if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
                loss_torch_epoch = loss_torch[0] / loss_torch[1]
                logger.debug(
                    f"epoch {epoch} average loss: {loss_torch_epoch:.4f}, "
                    f"best mean dice: {best_metric:.4f} at epoch {best_metric_epoch}")

            del inputs, labels, outputs
            torch.cuda.empty_cache()

            if ad:
                _percentage = float(_round) / float(num_rounds) * 100.0

                target_num_epochs_per_validation = -1
                for _j in range(len(ad_progress_percentages)):
                    if _percentage <= ad_progress_percentages[-1 - _j]:
                        target_num_epochs_per_validation = ad_num_epochs_per_validation[-1 - _j]
                        break

                if target_num_epochs_per_validation > 0 and (_round + 1) < num_rounds:
                    if (_round + 1) % (target_num_epochs_per_validation //
                                       num_epochs_per_validation) != 0:
                        continue

            model.eval()
            with torch.no_grad():
                metric = torch.zeros(
                    metric_dim * 2, dtype=torch.float, device=device)
                metric_sum = 0.0
                metric_mat = []
                val_images = None
                val_labels = None
                val_outputs = None

                _index = 0
                for val_data in val_loader:
                    val_images = val_data["image"]
                    val_labels = val_data["label"]

                    val_filename = val_data["image_meta_dict"]["filename_or_obj"][0]
                    if sw_input_on_cpu:
                        val_devices[val_filename] = "cpu"
                    elif val_filename not in val_devices:
                        val_devices[val_filename] = device

                    try:
                        val_images = val_images.to(val_devices[val_filename])
                        val_labels = val_labels.to(val_devices[val_filename])

                        with autocast(enabled=amp):
                            val_outputs = sliding_window_inference(
                                inputs=val_images,
                                roi_size=patch_size_valid,
                                sw_batch_size=num_sw_batch_size,
                                predictor=model,
                                mode="gaussian",
                                overlap=overlap_ratio,
                                sw_device=device)
                    except BaseException:
                        val_devices[val_filename] = "cpu"

                        with autocast(enabled=amp):
                            val_outputs = sliding_window_inference(
                                val_images,
                                patch_size_valid,
                                sw_batch_size=num_sw_batch_size,
                                predictor=model,
                                mode="gaussian",
                                overlap=overlap_ratio,
                                sw_device=device)

                    val_outputs = post_pred(val_outputs[0, ...])
                    val_outputs = val_outputs[None, ...]

                    if softmax:
                        val_labels = val_labels.int()
                        value = torch.zeros(1, metric_dim).to(device)
                        for _k in range(1, metric_dim + 1):
                            value[0, _k - 1] = compute_dice(
                                y_pred=val_outputs[:, _k: _k + 1],
                                y=(val_labels == _k).float(),
                                include_background=not softmax)
                    else:
                        value = compute_dice(
                            y_pred=val_outputs,
                            y=val_labels,
                            include_background=not softmax)
                        value = value.to(device)

                    logger.debug(f"{_index + 1} / {len(val_loader)}: {value}")

                    del val_images, val_labels, val_outputs
                    torch.cuda.empty_cache()

                    metric_sum += value.sum().item()
                    metric_vals = value.cpu().numpy()
                    if len(metric_mat) == 0:
                        metric_mat = metric_vals
                    else:
                        metric_mat = np.concatenate(
                            (metric_mat, metric_vals), axis=0)

                    for _c in range(metric_dim):
                        val0 = torch.nan_to_num(value[0, _c], nan=0.0)
                        val1 = 1.0 - torch.isnan(value[0, 0]).float()
                        metric[2 * _c] += val0 * val1
                        metric[2 * _c + 1] += val1

                    _index += 1

                if torch.cuda.device_count() > 1:
                    dist.barrier()
                    dist.all_reduce(metric, op=torch.distributed.ReduceOp.SUM)

                metric = metric.tolist()
                if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
                    for _c in range(metric_dim):
                        logger.debug(
                            f"evaluation metric - class {_c + 1}: {metric[2 * _c] / metric[2 * _c + 1]}")
                        writer.add_scalar(
                            f"val/acc/class{_c}", metric[2 * _c] / metric[2 * _c + 1], epoch)
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
                            torch.save(
                                model.module.state_dict(), os.path.join(
                                    ckpt_path, "best_metric_model.pt"))
                        else:
                            torch.save(
                                model.state_dict(), os.path.join(
                                    ckpt_path, "best_metric_model.pt"))
                        logger.debug("saved new best metric model")

                        dict_file = {}
                        dict_file["best_avg_dice_score"] = float(best_metric)
                        dict_file["best_avg_dice_score_epoch"] = int(
                            best_metric_epoch)
                        dict_file["best_avg_dice_score_iteration"] = int(
                            idx_iter)
                        with open(os.path.join(ckpt_path, "progress.yaml"), "a") as out_file:
                            yaml.dump([dict_file], stream=out_file)

                    logger.debug(
                        "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                            epoch, avg_metric, best_metric, best_metric_epoch))

                    current_time = time.time()
                    elapsed_time = (current_time - start_time) / 60.0
                    with open(os.path.join(ckpt_path, "accuracy_history.csv"), "a") as f:
                        f.write("{:d}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.1f}\t{:d}\n".format(
                            epoch, avg_metric, loss_torch_epoch, lr, elapsed_time, idx_iter))

                    if es:
                        early_stopping(val_acc=avg_metric)
                        stop_train = torch.tensor(
                            early_stopping.early_stop).to(device)

                if torch.cuda.device_count() > 1:
                    dist.barrier()

                if es:
                    if torch.cuda.device_count() > 1:
                        dist.broadcast(stop_train, src=0)
                    if stop_train:
                        break

            torch.cuda.empty_cache()

    if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
        logger.debug(
            f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

        writer.flush()
        writer.close()

    if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
        if es and (_round + 1) < num_rounds:
            logger.warning(
                f"{os.path.basename(bundle_root)} - training: finished with early stop")
        else:
            logger.warning(f"{os.path.basename(bundle_root)} - training: finished")

    if torch.cuda.device_count() > 1:
        dist.destroy_process_group()

    return


if __name__ == "__main__":
    from monai.utils import optional_import

    fire, _ = optional_import("fire")
    fire.Fire()
