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

import logging
import math
import os
import random
import sys
import time
from datetime import datetime
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

import monai
from monai import transforms
from monai.apps.auto3dseg.auto_runner import logger
from monai.apps.utils import DEFAULT_FMT
from monai.auto3dseg.utils import datafold_read
from monai.bundle import ConfigParser
from monai.bundle.scripts import _pop_args, _update_args
from monai.data import ThreadDataLoader, partition_dataset
from monai.inferers import sliding_window_inference
from monai.metrics import compute_dice
from monai.utils import set_determinism

try:
    from apex.contrib.clip_grad import clip_grad_norm_
except ModuleNotFoundError:
    from torch.nn.utils import clip_grad_norm_


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


def try_except(func, default=None, expected_exc=(Exception,)):
    try:
        return func()
    except expected_exc:
        return default


def run(config_file: Optional[Union[str, Sequence[str]]] = None, **override):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    if isinstance(config_file, str) and "," in config_file:
        config_file = config_file.split(",")

    _args = _update_args(config_file=config_file, **override)
    config_file_ = _pop_args(_args, "config_file")[0]

    parser = ConfigParser()
    parser.read_config(config_file_)
    parser.update(pairs=_args)

    amp = parser.get_parsed_content("searching#amp")
    arch_path = parser.get_parsed_content("searching#arch_path")
    data_file_base_dir = parser.get_parsed_content("data_file_base_dir")
    data_list_file_path = parser.get_parsed_content("data_list_file_path")
    determ = parser.get_parsed_content("searching#determ")
    fold = parser.get_parsed_content("fold")
    log_output_file = parser.get_parsed_content("searching#log_output_file")
    num_images_per_batch = parser.get_parsed_content("searching#num_images_per_batch")
    num_epochs = parser.get_parsed_content("searching#num_epochs")
    num_epochs_per_validation = parser.get_parsed_content("searching#num_epochs_per_validation")
    num_epochs_warmup = parser.get_parsed_content("searching#num_warmup_epochs")
    num_patches_per_image = parser.get_parsed_content("searching#num_patches_per_image")
    num_sw_batch_size = parser.get_parsed_content("searching#num_sw_batch_size")
    output_classes = parser.get_parsed_content("searching#output_classes")
    overlap_ratio = parser.get_parsed_content("searching#overlap_ratio")
    patch_size_valid = parser.get_parsed_content("searching#patch_size_valid")
    ram_cost_factor = parser.get_parsed_content("searching#ram_cost_factor")
    sw_input_on_cpu = parser.get_parsed_content("training#sw_input_on_cpu")
    softmax = parser.get_parsed_content("searching#softmax")

    # update transforms
    for _i in range(len(parser["transforms_train"]["transforms"])):
        if (
            "crop" in parser["transforms_train"]["transforms"][_i]["_target_"].lower()
            and "num_samples" in parser["transforms_train"]["transforms"][_i]
        ):
            parser["transforms_train"]["transforms"][_i]["num_samples"] = num_patches_per_image

    train_transforms = parser.get_parsed_content("transforms_train")
    val_transforms = parser.get_parsed_content("transforms_validate")

    if not os.path.exists(arch_path):
        os.makedirs(arch_path, exist_ok=True)

    if determ:
        set_determinism(seed=0)

    CONFIG["handlers"]["file"]["filename"] = log_output_file
    logging.config.dictConfig(CONFIG)

    logger.debug(f"number of GPUs: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        world_size = dist.get_world_size()
    else:
        world_size = 1
    logger.debug(f"world_size: {world_size}")

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

    train_files_w = train_files[: len(train_files) // 2]
    if torch.cuda.device_count() > 1:
        train_files_w = partition_dataset(
            data=train_files_w, shuffle=True, num_partitions=world_size, even_divisible=True
        )[dist.get_rank()]
    logger.debug(f"train_files_w: {len(train_files_w)}")

    train_files_a = train_files[len(train_files) // 2 :]
    if torch.cuda.device_count() > 1:
        train_files_a = partition_dataset(
            data=train_files_a, shuffle=True, num_partitions=world_size, even_divisible=True
        )[dist.get_rank()]
    logger.debug(f"train_files_a: {len(train_files_a)}")

    if torch.cuda.device_count() > 1:
        if len(val_files) < world_size:
            val_files = val_files * math.ceil(float(world_size) / float(len(val_files)))

        val_files = partition_dataset(data=val_files, shuffle=False, num_partitions=world_size, even_divisible=False)[
            dist.get_rank()
        ]
    logger.debug(f"val_files: {len(val_files)}")

    train_cache_rate = float(parser.get_parsed_content("searching#train_cache_rate"))
    validate_cache_rate = float(parser.get_parsed_content("searching#validate_cache_rate"))

    train_ds_a = monai.data.CacheDataset(
        data=train_files_a,
        transform=train_transforms,
        cache_rate=train_cache_rate,
        num_workers=parser.get_parsed_content("searching#num_cache_workers"),
        progress=False,
    )
    train_ds_w = monai.data.CacheDataset(
        data=train_files_w,
        transform=train_transforms,
        cache_rate=train_cache_rate,
        num_workers=parser.get_parsed_content("searching#num_cache_workers"),
        progress=False,
    )
    val_ds = monai.data.CacheDataset(
        data=val_files,
        transform=val_transforms,
        cache_rate=validate_cache_rate,
        num_workers=parser.get_parsed_content("searching#num_cache_workers"),
        progress=False,
    )

    train_loader_a = ThreadDataLoader(
        train_ds_a,
        num_workers=parser.get_parsed_content("searching#num_workers"),
        batch_size=num_images_per_batch,
        shuffle=True,
    )
    train_loader_w = ThreadDataLoader(
        train_ds_w,
        num_workers=parser.get_parsed_content("searching#num_workers"),
        batch_size=num_images_per_batch,
        shuffle=True,
    )
    val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1, shuffle=False)

    device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}") if world_size > 1 else torch.device("cuda:0")

    if world_size > 1:
        parser["searching_network"]["dints_space"]["device"] = device

    dints_space = parser.get_parsed_content("searching_network#dints_space")
    model = parser.get_parsed_content("searching_network#network")
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if softmax:
        post_pred = transforms.Compose(
            [transforms.EnsureType(), transforms.AsDiscrete(argmax=True, to_onehot=output_classes)]
        )
        post_label = transforms.Compose([transforms.EnsureType(), transforms.AsDiscrete(to_onehot=output_classes)])
    else:
        post_pred = transforms.Compose(
            [
                transforms.EnsureType(),
                transforms.Activations(sigmoid=True),
                transforms.AsDiscrete(threshold=0.5 + np.finfo(np.float32).eps),
            ]
        )

    loss_function = parser.get_parsed_content("searching#loss")

    optimizer_part = parser.get_parsed_content("searching#optimizer", instantiate=False)
    optimizer = optimizer_part.instantiate(params=model.parameters())

    arch_optimizer_a_part = parser.get_parsed_content("searching#arch_optimizer_a", instantiate=False)
    arch_optimizer_a = arch_optimizer_a_part.instantiate(params=[dints_space.log_alpha_a])

    arch_optimizer_c_part = parser.get_parsed_content("searching#arch_optimizer_c", instantiate=False)
    arch_optimizer_c = arch_optimizer_c_part.instantiate(params=[dints_space.log_alpha_c])

    if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
        logger.debug(f"num_epochs: {num_epochs}")
        logger.debug(f"num_epochs_warmup: {num_epochs_warmup}")
        logger.debug(f"num_epochs_per_validation: {num_epochs_per_validation}")

    lr_scheduler_part = parser.get_parsed_content("searching#lr_scheduler", instantiate=False)
    lr_scheduler = lr_scheduler_part.instantiate(optimizer=optimizer)

    if torch.cuda.device_count() > 1:
        model = DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)

    if amp:
        from torch.cuda.amp import GradScaler, autocast

        scaler = GradScaler()
        if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
            logger.debug("amp enabled")

    val_interval = num_epochs_per_validation
    best_metric = -1
    best_metric_epoch = -1
    idx_iter = 0
    metric_dim = output_classes - 1 if softmax else output_classes
    val_devices = {}

    if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
        writer = SummaryWriter(log_dir=os.path.join(arch_path, "Events"))

        with open(os.path.join(arch_path, "accuracy_history.csv"), "a") as f:
            f.write("epoch\tmetric\tloss\tlr\ttime\titer\n")

    dataloader_a_iterator = iter(train_loader_a)

    start_time = time.time()
    for epoch in range(num_epochs):
        lr = lr_scheduler.get_last_lr()[0]
        if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
            logger.debug("-" * 10)
            logger.debug(f"epoch {epoch + 1}/{num_epochs}")
            logger.debug(f"learning rate is set to {lr}")

        model.train()
        epoch_loss = 0
        loss_torch = torch.zeros(2, dtype=torch.float, device=device)
        epoch_loss_arch = 0
        loss_torch_arch = torch.zeros(2, dtype=torch.float, device=device)
        step = 0

        for batch_data in train_loader_w:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)

            if world_size == 1:
                for _ in model.weight_parameters():
                    _.requires_grad = True
            else:
                for _ in model.module.weight_parameters():
                    _.requires_grad = True

            dints_space.log_alpha_a.requires_grad = False
            dints_space.log_alpha_c.requires_grad = False

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
            epoch_len = len(train_loader_w)
            idx_iter += 1

            if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
                logger.debug(f"[{str(datetime.now())[:19]}] " + f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
                writer.add_scalar("Loss/train", loss.item(), epoch_len * epoch + step)

            if epoch < num_epochs_warmup:
                continue

            try:
                sample_a = next(dataloader_a_iterator)
            except StopIteration:
                dataloader_a_iterator = iter(train_loader_a)
                sample_a = next(dataloader_a_iterator)

            inputs_search, labels_search = (sample_a["image"].to(device), sample_a["label"].to(device))
            if world_size == 1:
                for _ in model.weight_parameters():
                    _.requires_grad = False
            else:
                for _ in model.module.weight_parameters():
                    _.requires_grad = False

            dints_space.log_alpha_a.requires_grad = True
            dints_space.log_alpha_c.requires_grad = True

            entropy_alpha_c = torch.tensor(0.0).to(device)
            entropy_alpha_a = torch.tensor(0.0).to(device)
            ram_cost_full = torch.tensor(0.0).to(device)
            ram_cost_usage = torch.tensor(0.0).to(device)
            ram_cost_loss = torch.tensor(0.0).to(device)
            topology_loss = torch.tensor(0.0).to(device)

            probs_a, arch_code_prob_a = dints_space.get_prob_a(child=True)
            entropy_alpha_a = -((probs_a) * torch.log(probs_a + 1e-5)).mean()
            entropy_alpha_c = -(
                F.softmax(dints_space.log_alpha_c, dim=-1) * F.log_softmax(dints_space.log_alpha_c, dim=-1)
            ).mean()
            topology_loss = dints_space.get_topology_entropy(probs_a)

            ram_cost_full = dints_space.get_ram_cost_usage(inputs.shape, full=True)
            ram_cost_usage = dints_space.get_ram_cost_usage(inputs.shape)
            ram_cost_loss = torch.abs(ram_cost_factor - ram_cost_usage / ram_cost_full)

            arch_optimizer_a.zero_grad()
            arch_optimizer_c.zero_grad()

            combination_weights = (epoch - num_epochs_warmup) / (num_epochs - num_epochs_warmup)

            if amp:
                with autocast():
                    outputs_search = model(inputs_search)
                    loss = loss_function(outputs_search.float(), labels_search)
                    loss += combination_weights * (
                        (entropy_alpha_a + entropy_alpha_c) + ram_cost_loss + 0.001 * topology_loss
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(arch_optimizer_a)
                scaler.unscale_(arch_optimizer_c)
                clip_grad_norm_([dints_space.log_alpha_a], 0.5)
                clip_grad_norm_([dints_space.log_alpha_c], 0.5)
                scaler.step(arch_optimizer_a)
                scaler.step(arch_optimizer_c)
                scaler.update()
            else:
                outputs_search = model(inputs_search)
                loss = loss_function(outputs_search.float(), labels_search)
                loss += combination_weights * (
                    (entropy_alpha_a + entropy_alpha_c) + ram_cost_loss + 0.001 * topology_loss
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_([dints_space.log_alpha_a], 0.5)
                torch.nn.utils.clip_grad_norm_([dints_space.log_alpha_c], 0.5)
                arch_optimizer_a.step()
                arch_optimizer_c.step()

            epoch_loss_arch += loss.item()
            loss_torch_arch[0] += loss.item()
            loss_torch_arch[1] += 1.0

            if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
                logger.debug(
                    f"[{str(datetime.now())[:19]}] " + f"{step}/{epoch_len}, train_loss_arch: {loss.item():.4f}"
                )
                writer.add_scalar("train_loss_arch", loss.item(), epoch_len * epoch + step)

        lr_scheduler.step()

        if torch.cuda.device_count() > 1:
            dist.barrier()
            dist.all_reduce(loss_torch, op=torch.distributed.ReduceOp.SUM)

        loss_torch = loss_torch.tolist()
        loss_torch_arch = loss_torch_arch.tolist()
        if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
            loss_torch_epoch = loss_torch[0] / loss_torch[1]
            logger.debug(
                f"epoch {epoch + 1} average loss: {loss_torch_epoch:.4f}, "
                f"best mean dice: {best_metric:.4f} at epoch {best_metric_epoch}"
            )

            if epoch >= num_epochs_warmup:
                loss_torch_arch_epoch = loss_torch_arch[0] / loss_torch_arch[1]
                logger.debug(
                    f"epoch {epoch + 1} average arch loss: {loss_torch_arch_epoch:.4f}, "
                    f"best mean dice: {best_metric:.4f} at epoch {best_metric_epoch}"
                )

        if (epoch + 1) % val_interval == 0 or (epoch + 1) == num_epochs:
            torch.cuda.empty_cache()
            model.eval()
            with torch.no_grad():
                metric = torch.zeros(metric_dim * 2, dtype=torch.float, device=device)
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

                        with torch.cuda.amp.autocast(enabled=amp):
                            val_outputs = sliding_window_inference(
                                inputs=val_images,
                                roi_size=patch_size_valid,
                                sw_batch_size=num_sw_batch_size,
                                predictor=model,
                                mode="gaussian",
                                overlap=overlap_ratio,
                                sw_device=device,
                            )
                    except RuntimeError as e:
                        if not any(x in str(e).lower() for x in ("memory", "cuda", "cudnn")):
                            raise e

                        val_devices[val_filename] = "cpu"

                        with torch.cuda.amp.autocast(enabled=amp):
                            val_outputs = sliding_window_inference(
                                val_images,
                                patch_size_valid,
                                sw_batch_size=num_sw_batch_size,
                                predictor=model,
                                mode="gaussian",
                                overlap=overlap_ratio,
                                sw_device=device,
                            )

                    val_outputs = post_pred(val_outputs[0, ...])
                    val_outputs = val_outputs[None, ...]

                    if softmax:
                        val_labels = post_label(val_labels[0, ...])
                        val_labels = val_labels[None, ...]

                    value = compute_dice(y_pred=val_outputs, y=val_labels, include_background=not softmax)

                    logger.debug(f"{_index + 1}, /, {len(val_loader)}, {value}")

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
                        logger.debug(f"evaluation metric - class {_c + 1:d}: {metric[2 * _c] / metric[2 * _c + 1]}")
                    avg_metric = 0
                    for _c in range(metric_dim):
                        avg_metric += metric[2 * _c] / metric[2 * _c + 1]
                    avg_metric = avg_metric / float(metric_dim)
                    logger.debug(f"avg_metric, {avg_metric}")

                    if avg_metric > best_metric:
                        best_metric = avg_metric
                        best_metric_epoch = epoch + 1
                        best_metric_iterations = idx_iter

                    (node_a_d, arch_code_a_d, arch_code_c_d, arch_code_a_max_d) = dints_space.decode()

                    torch.save(
                        {
                            "code_a": arch_code_a_d,
                            "code_a_max": arch_code_a_max_d,
                            "code_c": arch_code_c_d,
                            "best_dsc": best_metric,
                            "best_path": best_metric_iterations,
                            "dsc": avg_metric,
                            "epochs": epoch + 1,
                            "iter_num": idx_iter,
                            "node_a": node_a_d,
                        },
                        os.path.join(arch_path, "search_code_" + str(idx_iter) + ".pt"),
                    )

                    torch.save(
                        {
                            "code_a": arch_code_a_d,
                            "code_a_max": arch_code_a_max_d,
                            "code_c": arch_code_c_d,
                            "best_dsc": best_metric,
                            "best_path": best_metric_iterations,
                            "dsc": avg_metric,
                            "epochs": epoch + 1,
                            "iter_num": idx_iter,
                            "node_a": node_a_d,
                        },
                        os.path.join(arch_path, "search_code_latest.pt"),
                    )
                    logger.debug("saved new best metric model")

                    dict_file = {}
                    dict_file["best_avg_dice_score"] = float(best_metric)
                    dict_file["best_avg_dice_score_epoch"] = int(best_metric_epoch)
                    dict_file["best_avg_dice_score_iteration"] = int(idx_iter)
                    with open(os.path.join(arch_path, "progress.yaml"), "a") as out_file:
                        _ = yaml.dump([dict_file], stream=out_file)

                    logger.debug(
                        "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                            epoch + 1, avg_metric, best_metric, best_metric_epoch
                        )
                    )

                    current_time = time.time()
                    elapsed_time = (current_time - start_time) / 60.0
                    with open(os.path.join(arch_path, "accuracy_history.csv"), "a") as f:
                        f.write(
                            "{:d}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.1f}\t{:d}\n".format(
                                epoch + 1, avg_metric, loss_torch_epoch, lr, elapsed_time, idx_iter
                            )
                        )

                if torch.cuda.device_count() > 1:
                    dist.barrier()

            torch.cuda.empty_cache()

    logger.debug(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

    if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
        writer.flush()
        writer.close()

    if torch.cuda.device_count() > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    from monai.utils import optional_import

    fire, _ = optional_import("fire")
    fire.Fire()
