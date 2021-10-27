#!/usr/bin/env python

import argparse
import copy
import json
import logging
import monai
import nibabel as nib
import numpy as np
import os
import pandas as pd
import pathlib
import shutil
import sys
import tempfile
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml

from auto_unet import AutoUnet
# from config import *
from datetime import datetime
from glob import glob
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from monai.data import (
    DataLoader,
    ThreadDataLoader,
    decollate_batch,
)
# from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from monai.transforms import (
    apply_transform,
    AsDiscrete,
    Compose,
    EnsureType,
    Randomizable,
    Transform,
)
from monai.data import Dataset, create_test_image_3d, DistributedSampler, list_data_collate, partition_dataset
from monai.inferers import sliding_window_inference
# from monai.losses import DiceLoss, FocalLoss, GeneralizedDiceLoss
from monai.metrics import compute_meandice
from monai.utils import set_determinism
from transforms import (
    creating_label_interpolation_transform,
    creating_transforms_training,
    creating_transforms_validation,
    str2aug
)
from utils import (
    # BoundaryLoss,
    parse_monai_specs,
    # parse_monai_transform_specs,
)


class DupCacheDataset(monai.data.CacheDataset):                                                                                                                                                                                                                          
    def __init__(self, times: int, **kwargs):                                                                                                                                                                                                                 
        super().__init__(**kwargs)                                                                                                                                                                                                                            
        self.times = times                                                                                                                                                                                                                                    
                                                                                                                                                                                                                                                              
    def __len__(self):                                                                                                                                                                                                                                        
        return self.times * super().__len__()                                                                                                                                                                                                                 
                                                                                                                                                                                                                                                              
    def _transform(self, index: int):                                                                                                                                                                                                                         
        # print("index", index)                                                                                                                                                                                                                               
        index = index // self.times                                                                                                                                                                                                                           
        if index % len(self) >= self.cache_num:  # support negative index                                                                                                                                                                                     
            # no cache for this index, execute all the transforms directly                                                                                                                                                                                    
            return super()._transform(index)                                                                                                                                                                                                                  
        # load data from cache and execute from the first random transform                                                                                                                                                                                    
        start_run = False                                                                                                                                                                                                                                     
        if self._cache is None:                                                                                                                                                                                                                               
            self._cache = self._fill_cache()                                                                                                                                                                                                                  
        data = self._cache[index]                                                                                                                                                                                                                             
        if not isinstance(self.transform, Compose):                                                                                                                                                                                                           
            raise ValueError("transform must be an instance of monai.transforms.Compose.")                                                                                                                                                                    
        for _transform in self.transform.transforms:                                                                                                                                                                                                          
            if start_run or isinstance(_transform, Randomizable) or not isinstance(_transform, Transform):                                                                                                                                                    
                # only need to deep copy data on first non-deterministic transform                                                                                                                                                                            
                if not start_run:                                                                                                                                                                                                                             
                    start_run = True                                                                                                                                                                                                                          
                    data = copy.deepcopy(data)                                                                                                                                                                                                                     
                data = apply_transform(_transform, data)                                                                                                                                                                                                      
        return data                                                                                                                                                                                                                                           
                                                                                                                                                                                                                                                              
    def __item__(self, index: int):                                                                                                                                                                                                                           
        return super().__item__(index // self.times)    


def main():
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="checkpoint full path",
    )
    parser.add_argument(
        "--config",
        action="store",
        required=True,
        help="configuration",
    )
    parser.add_argument(
        "--fold",
        action="store",
        required=True,
        help="fold index in N-fold cross-validation",
    )
    parser.add_argument(
        "--json",
        action="store",
        required=True,
        help="full path of .json file",
    )
    parser.add_argument(
        "--json_key",
        action="store",
        required=True,
        help="selected key in .json data list",
    )
    parser.add_argument(
        "--local_rank",
        required=int,
        help="local process rank",
    )
    parser.add_argument(
        "--num_folds",
        action="store",
        required=True,
        help="number of folds in cross-validation",
    )
    parser.add_argument(
        "--output_root",
        action="store",
        required=True,
        help="output root",
    )
    parser.add_argument(
        "--resume_arch_ckpt",
        action="store",
        required=True,
        help="data root",
    )
    parser.add_argument(
        "--root",
        action="store",
        required=True,
        help="data root",
    )
    args = parser.parse_args()

    # # disable logging for processes except 0 on every node
    # if args.local_rank != 0:
    #     f = open(os.devnull, "w")
    #     sys.stdout = sys.stderr = f

    monai.config.print_config()
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
    intensity_range = list(map(float, config_core["intensity_range"].split(',')))
    label_interpolation = config_core["label_interpolation"]
    learning_rate = config_core["learning_rate"]
    learning_rate_gamma = config_core["learning_rate_gamma"]
    learning_rate_step_size = config_core["learning_rate_step_size"]
    loss_string = config_core["loss"]
    num_images_per_batch = config_core["num_images_per_batch"]
    num_epochs = config_core["num_epochs"]
    num_epochs_per_validation = config_core["num_epochs_per_validation"]
    num_folds = int(args.num_folds)
    num_patches_per_image = config_core["num_patches_per_image"]
    num_sw_batch_size = config_core["num_sw_batch_size"]
    num_tta = config_core["num_tta"]
    optim_string = config_core["optimizer"]
    output_classes = config_core["output_classes"]
    overlap_ratio = config_core["overlap_ratio"]
    patch_size = tuple(map(int, config_core["patch_size"].split(',')))
    # patch_size_valid = tuple(map(int, config_core["infer_patch_size"].split(',')))
    patch_size_valid = patch_size
    # scale_intensity_range = list(map(float, config_core["scale_intensity_range"].split(',')))
    spacing = list(map(float, config_core["spacing"].split(',')))

    # augmentation
    config_aug = config["augmentation_monai"]

    # deterministic training
    if determ:
        set_determinism(seed=config_core["random_seed"])

    # initialize the distributed training process, every GPU runs in a process
    dist.init_process_group(backend="nccl", init_method="env://")

    # augmentation
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

    # data
    with open(args.json, "r") as f:
        json_data = json.load(f)

    split = len(json_data[args.json_key]) // num_folds
    list_train = json_data[args.json_key][:(split * fold)] + json_data[args.json_key][(split * (fold + 1)):]
    list_valid = json_data[args.json_key][(split * fold):(split * (fold + 1))]

    # training data
    files = []
    for _i in range(len(list_train)):
        str_img = os.path.join(args.root, list_train[_i]["image"])
        str_seg = os.path.join(args.root, list_train[_i]["label"])

        if (not os.path.exists(str_img)) or (not os.path.exists(str_seg)):
            continue

        files.append({"image": str_img, "label": str_seg})
    
    train_files = files
    train_files = partition_dataset(data=train_files, shuffle=True, num_partitions=dist.get_world_size(), even_divisible=True)[dist.get_rank()]
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
    val_files = partition_dataset(data=val_files, shuffle=False, num_partitions=dist.get_world_size(), even_divisible=False)[dist.get_rank()]
    print("val_files:", len(val_files))

    # network architecture
    device = torch.device(f"cuda:{args.local_rank}")
    torch.cuda.set_device(device)

    label_interpolation_transform = creating_label_interpolation_transform(label_interpolation, spacing, output_classes)
    train_transforms = creating_transforms_training(foreground_crop_margin, label_interpolation_transform, num_patches_per_image, patch_size, intensity_range, intensity_norm_transforms, augmenations, device, output_classes)
    val_transforms = creating_transforms_validation(foreground_crop_margin, label_interpolation_transform, patch_size, intensity_range, intensity_norm_transforms, device)

    if True:
        # train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=8)
        train_ds = DupCacheDataset(                                                                                                                                                                                                                                   
            data=train_files,
            transform=train_transforms,
            cache_rate=1.0,
            num_workers=8,
            times=num_epochs_per_validation,
        )
        val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=2)
    else:
        train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
        val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=num_images_per_batch, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
    
    # train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=num_images_per_batch, shuffle=True)
    # val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1, shuffle=False)

    # monai network
    # config_network = config["network_monai"]
    # network_string = config_network["network"]
    # network_name, network_dict = parse_monai_specs(network_string)
    # network_dict["in_channels"] = input_channels
    # network_dict["out_channels"] = output_classes
    # if dist.get_rank() == 0:
    #     print("\nnetwork: " + network_name)
    #     for _key in network_dict.keys():
    #         print("  {0}:\t{1}".format(_key, network_dict[_key]))
    # network_class = getattr(monai.networks.nets, network_name)
    # model = network_class(**network_dict)
    # model = model.to(device)

    ckpt = torch.load(args.resume_arch_ckpt)
    node_a = ckpt['node_a']
    code_a = ckpt['code_a']
    code_c = ckpt['code_c']

    model = AutoUnet(
        in_channels=input_channels,
        num_classes=output_classes,
        cell_ops=5,
        k=1, 
        num_blocks=12,
        num_depths=4,
        channel_mul=1.0,
        affine=False,
        use_unet=False,
        probs=0.9,
        ef=0.3,
        use_stem=True,
        code=[node_a, code_a, code_c]
    )
    
    code_a = torch.from_numpy(code_a).to(torch.float32).cuda()
    code_c = F.one_hot(torch.from_numpy(code_c), model.cell_ops).to(torch.float32).cuda()
    model = model.to(device)

    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=True, num_classes=output_classes)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=True, num_classes=output_classes)])

    # loss function
    loss_name, loss_dict = parse_monai_specs(loss_string)
    if dist.get_rank() == 0:
        print("\nloss: " + loss_name)
        for _key in loss_dict.keys():
            print("  {0}:\t{1}".format(_key, loss_dict[_key]))
    loss_class = getattr(monai.losses, loss_name)
    loss_func = loss_class(**loss_dict)

    # optimizer
    optim_name, optim_dict = parse_monai_specs(optim_string)
    optim_dict["lr"] = learning_rate
    optim_dict["params"] = model.parameters()
    if dist.get_rank() == 0:
        print("\noptimizer: " + optim_name)
        for _key in optim_dict.keys():
            print("  {0}:\t{1}".format(_key, optim_dict[_key]))

    if optim_name.lower() == "novograd":
        optim_class = getattr(monai.optimizers, optim_name)
    else:
        optim_class = getattr(torch.optim, optim_name)
    optimizer = optim_class(**optim_dict)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    print()

    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            print("Let's use", torch.cuda.device_count(), "GPUs!")

        model = DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)

    if args.checkpoint != None and os.path.isfile(args.checkpoint):
        print("[info] fine-tuning pre-trained checkpoint {0:s}".format(args.checkpoint))
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        torch.cuda.empty_cache()
    else:
        print("[info] training from scratch")

    # amp
    if amp:
        from torch.cuda.amp import autocast, GradScaler
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

    if num_tta == 0 or num_tta == 1:
        flip_tta = []
    elif num_tta == 4:
        flip_tta = [[2], [3], [4]]
    elif num_tta == 8:
        flip_tta = [[2], [3], [4], (2, 3), (2, 4), (3, 4), (2, 3, 4)]

    if dist.get_rank() == 0:
        print("num_tta", num_tta)
        print("flip_tta", flip_tta)

    if dist.get_rank() == 0:
        writer = SummaryWriter(log_dir=os.path.join(args.output_root, "Events"))

        with open(os.path.join(args.output_root, "accuracy_history.csv"), "a") as f:
            f.write("epoch\tmetric\tloss\tlr\ttime\titer\n")

    start_time = time.time()
    for epoch in range(num_epochs // num_epochs_per_validation):
        # if learning_rate_final > -0.000001 and learning_rate_final < learning_rate:
        #     # lr = learning_rate - epoch / (num_epochs - 1) * (learning_rate - learning_rate_final)
        #     lr = (learning_rate - learning_rate_final) * (1 - epoch / (num_epochs - 1)) ** 0.9 + learning_rate_final
        #     for param_group in optimizer.param_groups:
        #         param_group["lr"] = lr
        # else:
        #     lr = learning_rate
        
        # lr = learning_rate * (learning_rate_gamma ** (epoch // learning_rate_step_size))
        # for param_group in optimizer.param_groups:
        #     param_group["lr"] = lr

        lr = optimizer.param_groups[0]["lr"]

        if dist.get_rank() == 0:
            print("-" * 10)
            print(f"epoch {epoch * num_epochs_per_validation + 1}/{num_epochs}")
            print('learning rate is set to {}'.format(lr))

        model.train()
        epoch_loss = 0
        loss_torch = torch.zeros(2, dtype=torch.float, device=device)
        step = 0
        # train_sampler.set_epoch(epoch)
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)

            # optimizer.zero_grad()
            for param in model.parameters():
                param.grad = None

            if amp:
                with autocast():
                    outputs = model(inputs, [node_a, code_a, code_c], ds=False)
                    outputs = outputs[0]
                    loss = loss_func(outputs, labels)

                scaler.scale(loss).backward()
                # scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs, [node_a, code_a, code_c], ds=False)
                outputs = outputs[0]
                loss = loss_func(outputs, labels)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
            print(f"epoch {(epoch + 1) * num_epochs_per_validation} average loss: {loss_torch_epoch:.4f}, best mean dice: {best_metric:.4f} at epoch {best_metric_epoch}")

        # if (epoch + 1) % val_interval == 0:
        if True:
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

                    # test time augmentation
                    ct = 1.0
                    with torch.cuda.amp.autocast():
                        pred = sliding_window_inference(
                            val_images,
                            roi_size,
                            sw_batch_size,
                            lambda x: model(x, [node_a, code_a, code_c], ds=False)[0],
                            mode="gaussian",
                            overlap=overlap_ratio,
                        )

                    for dims in flip_tta:
                        with torch.cuda.amp.autocast():
                            flip_pred = torch.flip(
                                sliding_window_inference(
                                    torch.flip(
                                        val_images,
                                        dims=dims
                                    ),
                                    roi_size,
                                    sw_batch_size,
                                    lambda x: model(x, [node_a, code_a, code_c], ds=False)[0],
                                    mode="gaussian",
                                    overlap=overlap_ratio,
                                ),
                                dims=dims,
                            )

                        pred += flip_pred
                        ct += 1.0

                    val_outputs = pred / ct

                    val_outputs = post_pred(val_outputs[0, ...])
                    val_outputs = val_outputs[None, ...]
                    val_labels = post_label(val_labels[0, ...])
                    val_labels = val_labels[None, ...]
                    # val_outputs = post_processing(val_outputs)

                    value = compute_meandice(
                                y_pred=val_outputs,
                                y=val_labels,
                                include_background=False
                            )

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
                            (epoch + 1) * num_epochs_per_validation, avg_metric, best_metric, best_metric_epoch
                        )
                    )

                    current_time = time.time()
                    elapsed_time = (current_time - start_time) / 60.0
                    with open(os.path.join(args.output_root, "accuracy_history.csv"), "a") as f:
                        f.write("{0:d}\t{1:.5f}\t{2:.5f}\t{3:.5f}\t{4:.1f}\t{5:d}\n".format((epoch + 1) * num_epochs_per_validation, avg_metric, loss_torch_epoch, lr, elapsed_time, idx_iter))

                dist.barrier()

            torch.cuda.empty_cache()

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

    if dist.get_rank() == 0:
        writer.close()

    dist.destroy_process_group()

    return


if __name__ == "__main__":
    main()
