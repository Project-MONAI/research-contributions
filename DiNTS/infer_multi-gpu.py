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
from auto_unet import AutoUnet
from torch import nn
from torch.nn.parallel import DistributedDataParallel

# from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transforms import creating_transforms_testing, str2aug
from utils import parse_monai_specs  # parse_monai_transform_specs,

import monai
from monai.data import (
    DataLoader,
    Dataset,
    DistributedSampler,
    create_test_image_3d,
    list_data_collate,
    partition_dataset,
)
from monai.inferers import sliding_window_inference

# from monai.losses import DiceLoss, FocalLoss, GeneralizedDiceLoss
from monai.metrics import compute_meandice
from monai.transforms import AsDiscrete, BatchInverseTransform, Invertd, KeepLargestConnectedComponent
from monai.utils import set_determinism
from monai.utils.enums import InverseKeys


def main():
    parser = argparse.ArgumentParser(description="inference")
    parser.add_argument("--arch_ckpt", action="store", required=True, help="data root")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint")
    parser.add_argument("--config", action="store", required=True, help="configuration")
    parser.add_argument("--json", action="store", required=True, help="full path of .json file")
    parser.add_argument("--json_key", action="store", required=True, help=".json data list key")
    parser.add_argument("--local_rank", required=int, help="local process rank")
    parser.add_argument("--output_root", action="store", required=True, help="output root")
    parser.add_argument("--prob", default=False, action="store_true", help="probility map")
    parser.add_argument("--root", action="store", required=True, help="data root")
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
    # amp = config_core["amp"]
    # cache_replace_rate = config_core["cache_replace_rate"]
    # determ = config_core["deterministic"]
    foreground_crop_margin = int(config_core["foreground_crop_margin"])
    input_channels = config_core["input_channels"]
    intensity_norm = config_core["intensity_norm"]
    # intensity_range = list(map(float, config_core["intensity_range"].split(',')))
    # label_interpolation = config_core["label_interpolation"]
    # learning_rate = config_core["learning_rate"]
    # learning_rate_gamma = config_core["learning_rate_gamma"]
    # learning_rate_step_size = config_core["learning_rate_step_size"]
    # loss_string = config_core["loss"]
    # num_cache_images = config_core["num_cache_images"]
    # num_images_per_batch = config_core["num_images_per_batch"]
    # num_epochs = config_core["num_epochs"]
    # num_epochs_per_validation = config_core["num_epochs_per_validation"]
    # num_patches_per_image = config_core["num_patches_per_image"]
    num_sw_batch_size = config_core["infer_num_sw_batch_size"]
    num_tta = config_core["infer_num_tta"]
    # optim_string = config_core["optimizer"]
    output_classes = config_core["output_classes"]
    overlap_ratio = config_core["infer_overlap_ratio"]
    patch_size = tuple(map(int, config_core["infer_patch_size"].split(",")))
    patch_size_valid = patch_size
    # scale_intensity_range = list(map(float, config_core["scale_intensity_range"].split(',')))
    spacing = list(map(float, config_core["spacing"].split(",")))

    # if args.debug:
    #     num_epochs_per_validation = 1

    # initialize the distributed training process, every GPU runs in a process
    dist.init_process_group(backend="nccl", init_method="env://")

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

    # inference data
    dataset_key = args.json_key
    files = []
    for _i in range(len(json_data[dataset_key])):
        str_img = os.path.join(args.root, json_data[dataset_key][_i])
        # str_seg = os.path.join(args.root, json_data[dataset_key][_i]["label"])

        # if (not os.path.exists(str_img)) or (not os.path.exists(str_seg)):
        if not os.path.exists(str_img):
            continue

        # files.append({"image": str_img, "label": str_seg})
        files.append({"image": str_img})

    infer_files = files
    infer_files = partition_dataset(
        data=infer_files, shuffle=False, num_partitions=dist.get_world_size(), even_divisible=False
    )[dist.get_rank()]
    print("infer_files", len(infer_files))

    # label_interpolation_transform = creating_label_interpolation_transform(label_interpolation, spacing, output_classes)
    # train_transforms = creating_transforms_training(foreground_crop_margin, label_interpolation_transform, num_patches_per_image, patch_size, scale_intensity_range, augmenations)
    infer_transforms = creating_transforms_testing(foreground_crop_margin, intensity_norm_transforms, spacing)

    argmax = AsDiscrete(argmax=True, to_onehot=False, n_classes=output_classes)
    onehot = AsDiscrete(argmax=False, to_onehot=True, n_classes=output_classes)

    # train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
    # infer_ds = monai.data.CacheDataset(data=infer_files, transform=infer_transforms, cache_rate=1.0, num_workers=4)
    # train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    # infer_ds = monai.data.Dataset(data=infer_files, transform=infer_transforms)
    infer_ds = monai.data.Dataset(data=infer_files, transform=infer_transforms)

    # train_loader = DataLoader(train_ds, batch_size=num_images_per_batch, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
    # infer_loader = DataLoader(infer_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())
    infer_loader = DataLoader(
        infer_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available()
    )

    # inverter = Invertd(
    #     # `image` was not copied, invert the original value directly
    #     keys=["image"],
    #     transform=infer_transforms,
    #     loader=infer_loader,
    #     orig_keys="image",
    #     meta_keys=["image_meta_dict"],
    #     orig_meta_keys="image_meta_dict",
    #     nearest_interp=False,
    #     to_tensor=[True],
    #     device="cpu",
    #     num_workers=0 if sys.platform == "darwin" or torch.cuda.is_available() else 2,
    # )

    # def no_collation(x):
    #     return x

    # batch_inverter = BatchInverseTransform(infer_transforms, infer_loader, collate_fn=no_collation)

    # network architecture
    device = torch.device(f"cuda:{args.local_rank}")
    torch.cuda.set_device(device)

    # monai network
    # config_network = config["network_monai"]
    # network_string = config_network["network"]
    # network_name, network_dict = parse_monai_specs(network_string)
    # network_dict["in_channels"] = input_channels
    # network_dict["out_channels"] = output_classes
    # if dist.get_rank() == 0:
    #     print("\nnetwork:")
    #     for _key in network_dict.keys():
    #         print("  {0}:\t{1}".format(_key, network_dict[_key]))
    # network_class = getattr(monai.networks.nets, network_name)
    # model = network_class(**network_dict)
    # model = model.to(device)

    ckpt = torch.load(args.arch_ckpt)
    node_a = ckpt["node_a"]
    code_a = ckpt["code_a"]
    code_c = ckpt["code_c"]

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
        code=[node_a, code_a, code_c],
    )

    code_a = torch.from_numpy(code_a).to(torch.float32).cuda()
    code_c = F.one_hot(torch.from_numpy(code_c), model.cell_ops).to(torch.float32).cuda()
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            print("Let's use", torch.cuda.device_count(), "GPUs!")

        model = DistributedDataParallel(model, device_ids=[device])

    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        print("[info] loading pre-trained checkpoint {0:s}".format(args.checkpoint))
        model.load_state_dict(torch.load(args.checkpoint))
    else:
        print("[info] cannot find pre-trained checkpoint!")
        input()

    saver = monai.data.NiftiSaver(
        output_dir=args.output_root, output_postfix="seg", resample=False, output_dtype=np.uint8
    )

    # # amp
    # if amp:
    #     from torch.cuda.amp import autocast, GradScaler
    #     scaler = GradScaler()
    #     if dist.get_rank() == 0:
    #         print("[info] amp enabled")

    # # start a typical PyTorch training
    # infer_interval = num_epochs_per_validation
    # best_metric = -1
    # best_metric_epoch = -1
    # epoch_loss_values = list()
    # idx_iter = 0
    # metric_values = list()

    if num_tta == 0 or num_tta == 1:
        flip_tta = []
    elif num_tta == 4:
        flip_tta = [[2], [3], [4]]
    elif num_tta == 8:
        flip_tta = [[2], [3], [4], (2, 3), (2, 4), (3, 4), (2, 3, 4)]

    if dist.get_rank() == 0:
        print("num_tta", num_tta)
        print("flip_tta", flip_tta)

    # if dist.get_rank() == 0:
    #     writer = SummaryWriter(log_dir=os.path.join(args.output_root, "Events"))

    # with open(os.path.join(args.output_root, "accuracy_history.csv"), "a") as f:
    #     f.write("epoch\tmetric\tloss\tlr\ttime\titer\n")

    start_time = time.time()
    # # for epoch in range(num_epochs):
    #     # if learning_rate_final > -0.000001 and learning_rate_final < learning_rate:
    #     #     # lr = learning_rate - epoch / (num_epochs - 1) * (learning_rate - learning_rate_final)
    #     #     lr = (learning_rate - learning_rate_final) * (1 - epoch / (num_epochs - 1)) ** 0.9 + learning_rate_final
    #     #     for param_group in optimizer.param_groups:
    #     #         param_group["lr"] = lr
    #     # else:
    #     #     lr = learning_rate
    #     # lr = learning_rate * (learning_rate_gamma ** (epoch // learning_rate_step_size))
    #     # for param_group in optimizer.param_groups:
    #     #     param_group["lr"] = lr

    #     # if dist.get_rank() == 0:
    #     #     print("-" * 10)
    #     #     print(f"epoch {epoch + 1}/{num_epochs}")
    #     #     print('learning rate is set to {}'.format(lr))

    #     model.train()
    #     epoch_loss = 0
    #     loss_torch = torch.zeros(2, dtype=torch.float, device=device)
    #     step = 0
    #     # train_sampler.set_epoch(epoch)
    #     for batch_data in train_loader:
    #         step += 1
    #         inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
    #         optimizer.zero_grad()

    #         if amp:
    #             with autocast():
    #                 outputs = model(inputs)
    #                 loss = loss_func(outputs, labels)

    #             scaler.scale(loss).backward()
    #             scaler.step(optimizer)
    #             scaler.update()
    #         else:
    #             outputs = model(inputs)
    #             loss = loss_func(outputs, labels)
    #             loss.backward()
    #             optimizer.step()

    #         epoch_loss += loss.item()
    #         loss_torch[0] += loss.item()
    #         loss_torch[1] += 1.0
    #         epoch_len = len(train_loader)
    #         idx_iter += 1

    #         if dist.get_rank() == 0:
    #             print("[{0}] ".format(str(datetime.now())[:19]) + f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
    #             writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

    #     # synchronizes all processes and reduce results
    #     dist.barrier()
    #     dist.all_reduce(loss_torch, op=torch.distributed.ReduceOp.SUM)
    #     loss_torch = loss_torch.tolist()
    #     if dist.get_rank() == 0:
    #         loss_torch_epoch = loss_torch[0] / loss_torch[1]
    #         print(f"epoch {epoch + 1} average loss: {loss_torch_epoch:.4f}, best mean dice: {best_metric:.4f} at epoch {best_metric_epoch}")

    #     # epoch_loss /= step
    #     # epoch_loss_values.append(epoch_loss)
    #     # print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}, best mean dice: {best_metric:.4f} at epoch {best_metric_epoch}")

    model.eval()
    with torch.no_grad():
        # metric = torch.zeros((output_classes - 1) * 2, dtype=torch.float, device=device)
        # metric_sum = 0.0
        # metric_count = 0
        # metric_mat = []
        infer_images = None
        # infer_labels = None
        infer_outputs = None

        _index = 0
        for infer_data in infer_loader:
            infer_images = infer_data["image"].to(device)

            roi_size = patch_size_valid
            sw_batch_size = num_sw_batch_size

            # test time augmentation
            ct = 1.0
            # pred = sliding_window_inference(infer_images, roi_size, sw_batch_size, nn.Sequential(model, nn.Softmax(1)), mode="gaussian", overlap=overlap_ratio)
            pred = sliding_window_inference(
                infer_images,
                roi_size,
                sw_batch_size,
                lambda x: model(x, [node_a, code_a, code_c], ds=False)[-1],
                mode="gaussian",
                overlap=overlap_ratio,
            )

            for dims in flip_tta:
                # flip_pred = torch.flip(sliding_window_inference(torch.flip(infer_images, dims=dims), roi_size, sw_batch_size, nn.Sequential(model, nn.Softmax(1)), mode="gaussian", overlap=overlap_ratio), dims=dims)
                flip_pred = torch.flip(
                    sliding_window_inference(
                        torch.flip(infer_images, dims=dims),
                        roi_size,
                        sw_batch_size,
                        lambda x: model(x, [node_a, code_a, code_c], ds=False)[-1],
                        mode="gaussian",
                        overlap=overlap_ratio,
                    ),
                    dims=dims,
                )
                pred += flip_pred
                ct += 1.0

            infer_outputs = pred / ct

            # infer_outputs = sliding_window_inference(infer_images, roi_size, sw_batch_size, nn.Sequential(model, nn.Softmax(1)), mode="gaussian", overlap=overlap_ratio)
            # infer_outputs = sliding_window_inference(infer_images, roi_size, sw_batch_size, nn.Sequential(model, nn.Softmax(1)), mode="gaussian", overlap=overlap_ratio, device=torch.device("cpu"))

            infer_outputs = nn.Softmax(dim=1)(infer_outputs)
            # print(infer_outputs.size())

            # label_transform_key = "image" + InverseKeys.KEY_SUFFIX
            # segs_dict = {
            #     "image": infer_outputs,
            #     "image_meta_dict": infer_data["image_meta_dict"],
            #     label_transform_key: infer_data[label_transform_key]
            # }
            # inv_batch = batch_inverter(segs_dict)
            # # print("inv_batch", type(inv_batch), len(inv_batch))

            # # infer_output_data = copy.deepcopy(infer_data)
            # # infer_output_data["image"] = infer_outputs
            # # infer_output_data = inverter(infer_output_data)
            # # infer_outputs = infer_output_data["image"][0][None]
            # print("inv_batch[0]", type(inv_batch[0]["image"]))
            # infer_outputs = inv_batch[0]["image"]
            # # infer_outputs = infer_outputs[None]
            # infer_outputs = torch.from_numpy(infer_outputs)

            infer_outputs = infer_outputs.cpu().detach().numpy()
            infer_outputs = np.squeeze(infer_outputs)
            out_nda = np.argmax(infer_outputs, axis=0)
            out_nda = out_nda.astype(np.uint8)
            print(out_nda.shape, np.unique(out_nda))

            # if args.post:
            #     out_nda = torch.as_tensor(out_nda[None][None], device="cpu")
            #     out_nda = post_processing(out_nda)
            #     out_nda = out_nda.detach().numpy().squeeze()
            #     out_nda = out_nda.astype(np.uint8)
            #     print("post-processing")

            out_filename = os.path.join(
                args.output_root, infer_data["image_meta_dict"]["filename_or_obj"][0].split(os.sep)[-1]
            )
            # out_filename = out_filename.replace("case_", "prediction_") + ".nii.gz"
            out_affine = infer_data["image_meta_dict"]["affine"].numpy().squeeze()
            out_img = nib.Nifti1Image(out_nda.astype(np.uint8), out_affine)
            nib.save(out_img, out_filename)
            print(out_filename)

            if args.prob:
                for _k in range(1, output_classes):
                    out_filename = os.path.join(
                        args.output_root, infer_data["image_meta_dict"]["filename_or_obj"][0].split(os.sep)[-1]
                    )
                    # out_filename = out_filename.replace("case_", "prediction_") + ".nii.gz"
                    out_filename = out_filename.replace(".nii", "_prob{0:d}.nii".format(_k))
                    out_affine = infer_data["image_meta_dict"]["affine"].numpy().squeeze()

                    # out_img = nib.Nifti1Image(infer_outputs[_k:_k+1, ...].squeeze().astype(np.float32), out_affine)
                    infer_outputs_indiv = infer_outputs[_k : _k + 1, ...].squeeze()
                    # infer_outputs_indiv[infer_outputs_indiv < 0.0] = 0.0
                    # infer_outputs_indiv[infer_outputs_indiv > 1.0] = 1.0
                    # infer_outputs_indiv = infer_outputs_indiv * 255.0
                    # infer_outputs_indiv = np.round(infer_outputs_indiv).astype(np.float32)
                    out_img = nib.Nifti1Image(infer_outputs_indiv, out_affine)

                    nib.save(out_img, out_filename)
                    print(out_filename)

    dist.destroy_process_group()

    return


if __name__ == "__main__":
    main()
