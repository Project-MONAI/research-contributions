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
#!/usr/bin/env python

import argparse
import copy
import glob
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
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import yaml
from scipy import ndimage as ndi

# from monai.utils.enums import InverseKeys
from transforms import creating_transforms_ensemble, str2aug
from utils import keep_largest_cc, parse_monai_specs  # parse_monai_transform_specs,

import monai
from monai.data import (
    DataLoader,
    Dataset,
    DistributedSampler,
    create_test_image_3d,
    list_data_collate,
    partition_dataset,
)
from monai.transforms import AsDiscrete  # BatchInverseTransform,; Invertd,
from monai.transforms import KeepLargestConnectedComponent

# from monai.inferers import sliding_window_inference
# from monai.losses import DiceLoss, FocalLoss, GeneralizedDiceLoss
# from monai.metrics import compute_meandice
from monai.utils import set_determinism


def main():
    parser = argparse.ArgumentParser(description="inference")
    parser.add_argument("--algorithm", type=str, default=None, help="ensemble algorithm")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint")
    parser.add_argument("--config", action="store", required=True, help="configuration")
    parser.add_argument("--local_rank", required=int, help="local process rank")
    parser.add_argument("--input_root", action="store", required=True, help="input root")
    parser.add_argument("--original_root", action="store", required=True, help="orignal dataset root")
    parser.add_argument("--output_root", action="store", required=True, help="output root")
    parser.add_argument("--post", default=False, action="store_true", help="post-processing")
    parser.add_argument("--dir_list", nargs="*", type=str, default=[])
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
    # foreground_crop_margin = int(config_core["foreground_crop_margin"])
    input_channels = config_core["input_channels"]
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
    # num_sw_batch_size = config_core["num_sw_batch_size"]
    # num_tta = config_core["num_tta"]
    # optim_string = config_core["optimizer"]
    output_classes = config_core["output_classes"]
    # overlap_ratio = config_core["overlap_ratio"]
    # patch_size = tuple(map(int, config_core["patch_size"].split(',')))
    # patch_size_valid = patch_size
    # scale_intensity_range = list(map(float, config_core["scale_intensity_range"].split(',')))
    # spacing = list(map(float, config_core["spacing"].split(',')))

    # if args.debug:
    #     num_epochs_per_validation = 1

    # initialize the distributed training process, every GPU runs in a process
    dist.init_process_group(backend="nccl", init_method="env://")

    # # data
    # with open(args.json, "r") as f:
    #     json_data = json.load(f)

    # ensemble data
    num_folds = len(args.dir_list)
    all_filenames = []

    for _i in range(num_folds):
        list_filenames = []
        for root, dirs, files in os.walk(os.path.join(args.input_root, args.dir_list[_i])):
            for basename in files:
                if "_prob1.nii" in basename:
                    filename = os.path.join(root, basename)
                    filename = filename.replace(args.input_root, "")
                    filename = filename[1:]
                    list_filenames.append(filename)
        list_filenames.sort()
        all_filenames.append(list_filenames)
        num_cases = len(list_filenames)

    files = []
    for _i in range(num_cases):
        case_dict = {}
        for _j in range(num_folds):
            volume_list = []
            for _k in range(1, output_classes):
                volume_list.append(
                    os.path.join(args.input_root, all_filenames[_j][_i].replace("_prob1", "_prob" + str(_k)))
                )
            case_dict["fold" + str(_j)] = volume_list
        # print(case_dict)
        files.append(case_dict)

    ensemble_files = files
    ensemble_files = partition_dataset(
        data=ensemble_files, shuffle=False, num_partitions=dist.get_world_size(), even_divisible=False
    )[dist.get_rank()]
    print("ensemble_files", len(ensemble_files))

    key_list = ["fold" + str(_item) for _item in range(num_folds)]
    ensemble_transforms = creating_transforms_ensemble(keys=key_list)

    ensemble_ds = monai.data.Dataset(data=ensemble_files, transform=ensemble_transforms)
    ensemble_loader = DataLoader(ensemble_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=False)

    start_time = time.time()
    for ensemble_data in ensemble_loader:
        for _k in range(num_folds):
            print(ensemble_data["fold" + str(_k) + "_meta_dict"]["filename_or_obj"][0])

            ensemble_outputs = None
            ensemble_outputs = ensemble_data["fold" + str(_k)]

            nda_all = ensemble_outputs.numpy()
            nda_all = nda_all.squeeze().astype(np.float32) / 255.0
            # print(nda_all.shape, np.amax(nda_all), np.amin(nda_all))

            nda_cls0 = 1.0 - np.sum(nda_all, axis=0, keepdims=True)
            nda_all = np.concatenate((nda_cls0, nda_all), axis=0)
            # print(nda_all.shape)

            if args.algorithm.lower() == "am":
                nda_all = nda_all / float(num_folds)
                nda_out = nda_all if _k == 0 else nda_out + nda_all
            elif args.algorithm.lower() == "gm":
                nda_all = nda_all ** (1.0 / float(num_folds))
                nda_out = nda_all if _k == 0 else nda_out * nda_all
            elif args.algorithm.lower() == "wam":
                nda_all = nda_all**2
                nda_out = nda_all if _k == 0 else nda_out + nda_all
            else:
                "[error] wrong algorithm!"
                return

            print(np.amax(nda_out), np.amin(nda_out), np.mean(nda_out))
        # resize to orignal data size
        # find orignal data
        file_basename = ensemble_data["fold0_meta_dict"]["filename_or_obj"][0].split(os.sep)[-1]
        original_data_path = list(Path(args.original_root).glob(file_basename))[0]
        original_data = nib.load(original_data_path)
        # get affine matrix
        seg_affine = ensemble_data["fold0_meta_dict"]["original_affine"].numpy().squeeze()
        img_affine = original_data.affine
        img_shape = original_data.shape
        T = np.matmul(np.linalg.inv(seg_affine), img_affine)
        # output segmentation affine matches original image
        out_affine = img_affine
        nda_out = np.argmax(nda_out, axis=0)
        nda_out = ndi.affine_transform(nda_out, T, output_shape=img_shape)
        nda_out = nda_out.astype(np.uint8)

        # post-processing
        if args.post:
            print("[info] keep largest connected component")
            nda_mask = copy.deepcopy(nda_out)
            nda_mask = (nda_mask > 0).astype(np.uint8)
            print(nda_mask.sum())
            nda_mask = keep_largest_cc(nda_mask)
            nda_mask = nda_mask.astype(np.uint8)
            print(nda_mask.sum())
            nda_out[nda_mask == 0] = 0

        out_img = nib.Nifti1Image(nda_out, out_affine)
        out_filename = os.path.join(
            args.output_root, ensemble_data["fold0_meta_dict"]["filename_or_obj"][0].split(os.sep)[-1]
        )
        out_filename = out_filename.replace("_prob1", "")
        print("out_filename", out_filename)
        nib.save(out_img, os.path.join(args.output_root, out_filename))

    dist.destroy_process_group()

    return


if __name__ == "__main__":
    main()
