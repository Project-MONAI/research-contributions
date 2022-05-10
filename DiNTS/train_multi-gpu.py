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

import argparse
import copy
import json
import logging
import os
import pathlib
import shutil
import sys
import tempfile
import time
from datetime import datetime
from glob import glob

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from transforms import creating_transforms_training, creating_transforms_validation
from utils import parse_monai_specs

import monai
from monai.data import (
    DataLoader,
    Dataset,
    DistributedSampler,
    ThreadDataLoader,
    create_test_image_3d,
    decollate_batch,
    list_data_collate,
    partition_dataset,
)
from monai.inferers import sliding_window_inference
from monai.metrics import compute_meandice
from monai.transforms import AsDiscrete, Compose, EnsureType, Randomizable, Transform, apply_transform
from monai.utils import set_determinism


def main():
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument("--arch_ckpt", action="store", required=True, help="data root")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint full path")
    parser.add_argument("--config", action="store", required=True, help="configuration")
    parser.add_argument("--fold", action="store", required=True, help="fold index in N-fold cross-validation")
    parser.add_argument("--json", action="store", required=True, help="full path of .json file")
    parser.add_argument("--json_key", action="store", required=True, help="selected key in .json data list")
    parser.add_argument("--local_rank", required=int, help="local process rank")
    parser.add_argument("--num_folds", action="store", required=True, help="number of folds in cross-validation")
    parser.add_argument("--output_root", action="store", required=True, help="output root")
    parser.add_argument("--root", action="store", required=True, help="data root")
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root)

    # configuration
    with open(args.config) as in_file:
        config = yaml.full_load(in_file)
    print("\n", pd.DataFrame(config), "\n")

    # core
    config_core = config["core"]
    amp = config_core["amp"]
    determ = config_core["deterministic"]
    fold = int(args.fold)
    foreground_crop_margin = int(config_core["foreground_crop_margin"])
    input_channels = config_core["input_channels"]
    intensity_norm = config_core["intensity_norm"]
    interpolation = config_core["interpolation"]
    learning_rate = config_core["learning_rate"]
    learning_rate_milestones = np.array(list(map(float, config_core["learning_rate_milestones"].split(","))))
    num_images_per_batch = config_core["num_images_per_batch"]
    num_epochs = config_core["num_epochs"]
    num_epochs_per_validation = config_core["num_epochs_per_validation"]
    num_folds = int(args.num_folds)
    num_patches_per_image = config_core["num_patches_per_image"]
    num_sw_batch_size = config_core["num_sw_batch_size"]
    output_classes = config_core["output_classes"]
    overlap_ratio = config_core["overlap_ratio"]
    patch_size = tuple(map(int, config_core["patch_size"].split(",")))
    patch_size_valid = tuple(map(int, config_core["infer_patch_size"].split(",")))

    # deterministic training
    if determ:
        set_determinism(seed=config_core["random_seed"])

    # initialize the distributed training process, every GPU runs in a process
    dist.init_process_group(backend="nccl", init_method="env://")
    dist.barrier()
    world_size = dist.get_world_size()

    # augmentation
    config_aug = config["augmentation_monai"]
    num_augmentations = len(config_aug.keys())
    augmenations = []
    for _k in range(num_augmentations):
        transform_string = config_aug["aug_{0:d}".format(_k)]
        transform_name, transform_dict = parse_monai_specs(transform_string)
        if dist.get_rank() == 0:
            print("\naugmenation {0:d}:\t{1:s}".format(_k + 1, transform_name))
            for _key in transform_dict.keys():
                print("  {0}:\t{1}".format(_key, transform_dict[_key]))
        transform_class = getattr(monai.transforms, transform_name)
        transform_func = transform_class(**transform_dict)
        augmenations.append(transform_func)

    # intensity normalization
    intensity_norm = intensity_norm.split("||")
    intensity_norm_transforms = []
    for _k in range(len(intensity_norm)):
        transform_string = intensity_norm[_k]
        transform_name, transform_dict = parse_monai_specs(transform_string)
        if dist.get_rank() == 0:
            print("\nintensity normalization {0:d}:\t{1:s}".format(_k + 1, transform_name))
            for _key in transform_dict.keys():
                print("  {0}:\t{1}".format(_key, transform_dict[_key]))
        transform_class = getattr(monai.transforms, transform_name)
        transform_func = transform_class(**transform_dict)
        intensity_norm_transforms.append(transform_func)

    # interpolation (re-sampling)
    interpolation = interpolation.split("||")
    interpolation_transforms = []
    for _k in range(len(interpolation)):
        transform_string = interpolation[_k]
        transform_name, transform_dict = parse_monai_specs(transform_string)
        if dist.get_rank() == 0:
            print("\ninterpolation {0:d}:\t{1:s}".format(_k + 1, transform_name))
            for _key in transform_dict.keys():
                print("  {0}:\t{1}".format(_key, transform_dict[_key]))
        transform_class = getattr(monai.transforms, transform_name)
        transform_func = transform_class(**transform_dict)
        interpolation_transforms.append(transform_func)

    # data
    with open(args.json, "r") as f:
        json_data = json.load(f)

    split = len(json_data[args.json_key]) // num_folds
    list_train = json_data[args.json_key][: (split * fold)] + json_data[args.json_key][(split * (fold + 1)) :]
    list_valid = json_data[args.json_key][(split * fold) : (split * (fold + 1))]

    # training data
    files = []
    for _i in range(len(list_train)):
        str_img = os.path.join(args.root, list_train[_i]["image"])
        str_seg = os.path.join(args.root, list_train[_i]["label"])

        if (not os.path.exists(str_img)) or (not os.path.exists(str_seg)):
            continue

        files.append({"image": str_img, "label": str_seg})

    train_files = files
    train_files = partition_dataset(
        data=train_files, shuffle=True, num_partitions=dist.get_world_size(), even_divisible=True
    )[dist.get_rank()]
    print("train_files:", len(train_files))

    # validation data
    files = []
    for _i in range(len(list_valid)):
        str_img = os.path.join(args.root, list_valid[_i]["image"])
        str_seg = os.path.join(args.root, list_valid[_i]["label"])

        if (not os.path.exists(str_img)) or (not os.path.exists(str_seg)):
            continue

        files.append({"image": str_img, "label": str_seg})

    val_files = files
    val_files = partition_dataset(
        data=val_files, shuffle=False, num_partitions=dist.get_world_size(), even_divisible=False
    )[dist.get_rank()]
    print("val_files:", len(val_files))

    device = torch.device(f"cuda:{args.local_rank}")
    torch.cuda.set_device(device)

    train_transforms = creating_transforms_training(
        foreground_crop_margin,
        interpolation_transforms,
        num_patches_per_image,
        patch_size,
        intensity_norm_transforms,
        augmenations,
        device,
        output_classes,
    )
    val_transforms = creating_transforms_validation(
        foreground_crop_margin, interpolation_transforms, patch_size, intensity_norm_transforms, device
    )

    # alternative Dataset
    # train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    # val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)

    train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=8)
    val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=2)

    # alternative DataLoader
    # train_loader = DataLoader(train_ds, batch_size=num_images_per_batch, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available())
    # val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=num_images_per_batch, shuffle=True)
    val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1, shuffle=False)

    ckpt = torch.load(args.arch_ckpt)
    node_a = ckpt["node_a"]
    arch_code_a = ckpt["arch_code_a"]
    arch_code_c = ckpt["arch_code_c"]

    dints_space = monai.networks.nets.TopologyInstance(
        channel_mul=1.0,
        num_blocks=12,
        num_depths=4,
        use_downsample=True,
        arch_code=[arch_code_a, arch_code_c],
        device=device,
    )

    model = monai.networks.nets.DiNTS(
        dints_space=dints_space,
        in_channels=input_channels,
        num_classes=output_classes,
        use_downsample=True,
        node_a=node_a,
    )

    model = model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=output_classes)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=output_classes)])

    # loss function
    loss_func = monai.losses.DiceCELoss(
        include_background=False,
        to_onehot_y=True,
        softmax=True,
        squared_pred=True,
        batch=True,
        smooth_nr=0.00001,
        smooth_dr=0.00001,
    )

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate * world_size, momentum=0.9, weight_decay=0.00004)

    print()

    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            print("Let's use", torch.cuda.device_count(), "GPUs!")

        model = DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)

    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        print("[info] fine-tuning pre-trained checkpoint {0:s}".format(args.checkpoint))
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        torch.cuda.empty_cache()
    else:
        print("[info] training from scratch")

    # amp
    if amp:
        from torch.cuda.amp import GradScaler, autocast

        scaler = GradScaler()
        if dist.get_rank() == 0:
            print("[info] amp enabled")

    # start a typical PyTorch training
    val_interval = num_epochs_per_validation
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    idx_iter = 0
    metric_values = list()

    if dist.get_rank() == 0:
        writer = SummaryWriter(log_dir=os.path.join(args.output_root, "Events"))

        with open(os.path.join(args.output_root, "accuracy_history.csv"), "a") as f:
            f.write("epoch\tmetric\tloss\tlr\ttime\titer\n")

    start_time = time.time()
    for epoch in range(num_epochs):
        decay = 0.5 ** np.sum([epoch / num_epochs > learning_rate_milestones])
        lr = learning_rate * decay
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if dist.get_rank() == 0:
            print("-" * 10)
            print(f"epoch {epoch + 1}/{num_epochs}")
            print("learning rate is set to {}".format(lr))

        model.train()
        epoch_loss = 0
        loss_torch = torch.zeros(2, dtype=torch.float, device=device)
        step = 0

        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)

            for param in model.parameters():
                param.grad = None

            if amp:
                with autocast():
                    outputs = model(inputs)
                    if output_classes == 2:
                        loss = loss_func(torch.flip(outputs, dims=[1]), 1 - labels)
                    else:
                        loss = loss_func(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                if output_classes == 2:
                    loss = loss_func(torch.flip(outputs, dims=[1]), 1 - labels)
                else:
                    loss = loss_func(outputs, labels)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            loss_torch[0] += loss.item()
            loss_torch[1] += 1.0
            epoch_len = len(train_loader)
            idx_iter += 1

            if dist.get_rank() == 0:
                print("[{0}] ".format(str(datetime.now())[:19]) + f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
                writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

        # synchronizes all processes and reduce results
        dist.barrier()
        dist.all_reduce(loss_torch, op=torch.distributed.ReduceOp.SUM)
        loss_torch = loss_torch.tolist()
        if dist.get_rank() == 0:
            loss_torch_epoch = loss_torch[0] / loss_torch[1]
            print(
                f"epoch {epoch + 1} average loss: {loss_torch_epoch:.4f}, best mean dice: {best_metric:.4f} at epoch {best_metric_epoch}"
            )

        if (epoch + 1) % val_interval == 0 or (epoch + 1) == num_epochs:
            torch.cuda.empty_cache()
            model.eval()
            with torch.no_grad():
                metric = torch.zeros((output_classes - 1) * 2, dtype=torch.float, device=device)
                metric_sum = 0.0
                metric_count = 0
                metric_mat = []
                val_images = None
                val_labels = None
                val_outputs = None

                _index = 0
                for val_data in val_loader:
                    val_images = val_data["image"].to(device)
                    val_labels = val_data["label"].to(device)

                    roi_size = patch_size_valid
                    sw_batch_size = num_sw_batch_size

                    ct = 1.0
                    with torch.cuda.amp.autocast():
                        pred = sliding_window_inference(
                            val_images,
                            roi_size,
                            sw_batch_size,
                            lambda x: model(x),
                            mode="gaussian",
                            overlap=overlap_ratio,
                        )

                    val_outputs = pred / ct

                    val_outputs = post_pred(val_outputs[0, ...])
                    val_outputs = val_outputs[None, ...]
                    val_labels = post_label(val_labels[0, ...])
                    val_labels = val_labels[None, ...]

                    value = compute_meandice(y_pred=val_outputs, y=val_labels, include_background=False)

                    print(_index + 1, "/", len(val_loader), value)

                    metric_count += len(value)
                    metric_sum += value.sum().item()
                    metric_vals = value.cpu().numpy()
                    if len(metric_mat) == 0:
                        metric_mat = metric_vals
                    else:
                        metric_mat = np.concatenate((metric_mat, metric_vals), axis=0)

                    for _c in range(output_classes - 1):
                        val0 = torch.nan_to_num(value[0, _c], nan=0.0)
                        val1 = 1.0 - torch.isnan(value[0, 0]).float()
                        metric[2 * _c] += val0 * val1
                        metric[2 * _c + 1] += val1

                    _index += 1

                # synchronizes all processes and reduce results
                dist.barrier()
                dist.all_reduce(metric, op=torch.distributed.ReduceOp.SUM)
                metric = metric.tolist()
                if dist.get_rank() == 0:
                    for _c in range(output_classes - 1):
                        print("evaluation metric - class {0:d}:".format(_c + 1), metric[2 * _c] / metric[2 * _c + 1])
                    avg_metric = 0
                    for _c in range(output_classes - 1):
                        avg_metric += metric[2 * _c] / metric[2 * _c + 1]
                    avg_metric = avg_metric / float(output_classes - 1)
                    print("avg_metric", avg_metric)

                    if avg_metric > best_metric:
                        best_metric = avg_metric
                        best_metric_epoch = epoch + 1
                        torch.save(model.state_dict(), os.path.join(args.output_root, "best_metric_model.pth"))
                        print("saved new best metric model")

                        dict_file = {}
                        dict_file["best_avg_dice_score"] = float(best_metric)
                        dict_file["best_avg_dice_score_epoch"] = int(best_metric_epoch)
                        dict_file["best_avg_dice_score_iteration"] = int(idx_iter)
                        with open(os.path.join(args.output_root, "progress.yaml"), "w") as out_file:
                            documents = yaml.dump(dict_file, stream=out_file)

                    print(
                        "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                            epoch + 1, avg_metric, best_metric, best_metric_epoch
                        )
                    )

                    current_time = time.time()
                    elapsed_time = (current_time - start_time) / 60.0
                    with open(os.path.join(args.output_root, "accuracy_history.csv"), "a") as f:
                        f.write(
                            "{0:d}\t{1:.5f}\t{2:.5f}\t{3:.5f}\t{4:.1f}\t{5:d}\n".format(
                                epoch + 1, avg_metric, loss_torch_epoch, lr, elapsed_time, idx_iter
                            )
                        )

                dist.barrier()

            torch.cuda.empty_cache()

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

    if dist.get_rank() == 0:
        writer.close()

    dist.destroy_process_group()

    return


if __name__ == "__main__":
    main()
