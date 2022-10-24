#!/usr/bin/env python

import nvidia_smi
import numpy as np
import os
import torch
import sys
import yaml

from monai.bundle import ConfigParser
from monai.inferers import sliding_window_inference
from torch.cuda.amp import GradScaler, autocast


config_file = []
config_file.append(
    os.path.join("work_dir", "swinunetr_0", "configs", "hyper_parameters.yaml")
)
config_file.append(os.path.join("work_dir", "swinunetr_0", "configs", "network.yaml"))
config_file.append(
    os.path.join("work_dir", "swinunetr_0", "configs", "transforms_train.yaml")
)
config_file.append(
    os.path.join("work_dir", "swinunetr_0", "configs", "transforms_validate.yaml")
)

parser = ConfigParser()
parser.read_config(config_file)

device = torch.device("cuda:0")
torch.cuda.set_device(device)

input_channels = parser.get_parsed_content("input_channels")
output_classes = parser.get_parsed_content("output_classes")
patch_size = parser.get_parsed_content("patch_size")
patch_size_valid = parser.get_parsed_content("patch_size_valid")
overlap_ratio = parser.get_parsed_content("overlap_ratio")

model = parser.get_parsed_content("network")
model = model.to(device)

loss_function = parser.get_parsed_content("loss")
optimizer_part = parser.get_parsed_content("optimizer", instantiate=False)
optimizer = optimizer_part.instantiate(params=model.parameters())

train_transforms = parser.get_parsed_content("transforms_train")

data_stats_file = os.path.join("work_dir", "datastats.yaml")
with open(data_stats_file) as f_data_stat:
    data_stat = yaml.full_load(f_data_stat)

pixdim = parser.get_parsed_content("transforms_train#transforms#3#pixdim")
max_shape = [-1, -1, -1]
for _k in range(len(data_stat["stats_by_cases"])):
    image_shape = data_stat["stats_by_cases"][_k]["image_stats"]["shape"][0]
    image_spacing = data_stat["stats_by_cases"][_k]["image_stats"]["spacing"][0]

    for _l in range(3):
        max_shape[_l] = max(
            max_shape[_l],
            int(np.ceil(float(image_shape[_l]) * image_spacing[_l] / pixdim[_l])),
        )
print("max_shape", max_shape)

scaler = GradScaler()

num_epochs = 2
num_iterations = 6
num_iterations_validation = 1

nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

num_images_per_batch = int(sys.argv[1])
num_patches_per_image = int(sys.argv[2])
num_sw_batch_size = int(sys.argv[3])

validation_data_device = str(sys.argv[4]).lower()
if validation_data_device != "cpu" and validation_data_device != "gpu":
    raise ValueError("only cpu or gpu allowed for validation_data_device!")

print("num_images_per_batch", num_images_per_batch)
print("num_patches_per_image", num_patches_per_image)
print("num_sw_batch_size", num_sw_batch_size)
print("validation_data_device", validation_data_device)

percentage = []
for _i in range(num_epochs):
    # training
    print("------  training  ------")

    model.train()

    for _j in range(num_iterations):
        print("iteration", _j + 1)

        inputs = torch.rand(
            (
                num_images_per_batch * num_patches_per_image,
                input_channels,
                patch_size[0],
                patch_size[1],
                patch_size[2],
            )
        )
        labels = torch.rand(
            (
                num_images_per_batch * num_patches_per_image,
                1,
                patch_size[0],
                patch_size[1],
                patch_size[2],
            )
        )
        inputs, labels = inputs.to(device), labels.to(device)

        for param in model.parameters():
            param.grad = None

        with autocast():
            outputs = model(inputs)
            loss = loss_function(outputs.float(), labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(optimizer)
        scaler.update()

        if _j == num_iterations - 1:
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            print(
                "Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(
                    0,
                    nvidia_smi.nvmlDeviceGetName(handle),
                    100 * info.free / info.total,
                    info.total,
                    info.free,
                    info.used,
                )
            )
            percentage.append(100 - 100 * info.free / info.total)

    # validation
    print("------  validation  ------")
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        for _k in range(num_iterations_validation):
            print("validation iteration", _k + 1)

            val_images = torch.rand(
                (1, input_channels, max_shape[0], max_shape[1], max_shape[2])
            )

            if validation_data_device == "gpu":
                val_images = val_images.to(device)

            with autocast():
                val_outputs = sliding_window_inference(
                    val_images,
                    patch_size_valid,
                    num_sw_batch_size,
                    model,
                    mode="gaussian",
                    overlap=overlap_ratio,
                    sw_device=device,
                )

    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    print(
        "Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(
            0,
            nvidia_smi.nvmlDeviceGetName(handle),
            100 * info.free / info.total,
            info.total,
            info.free,
            info.used,
        )
    )
    percentage.append(100 - 100 * info.free / info.total)

    torch.cuda.empty_cache()

nvidia_smi.nvmlShutdown()
