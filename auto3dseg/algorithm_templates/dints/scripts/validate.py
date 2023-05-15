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

import csv
import logging
import os
import sys
from typing import Optional, Sequence, Union

import numpy as np
import torch
import yaml

import monai
from monai import transforms
from monai.apps.auto3dseg.auto_runner import logger
from monai.apps.utils import DEFAULT_FMT
from monai.bundle import ConfigParser
from monai.bundle.scripts import _pop_args, _update_args
from monai.data import ThreadDataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import compute_dice


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


def get_mem_from_visible_gpus():
    available_mem_visible_gpus = []
    for d in range(torch.cuda.device_count()):
        available_mem_visible_gpus.append(torch.cuda.mem_get_info(device=d)[0])
    return available_mem_visible_gpus


def pre_operation(config_file, **override):
    # update hyper-parameter configuration
    rank = int(os.getenv("RANK", "0"))
    if rank == 0:
        if isinstance(config_file, str) and ',' in config_file:
            config_file = config_file.split(',')

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
                    mem_bs2 = 6.0 + (20.0 - 6.0) * \
                        (output_classes - 2) / (105 - 2)
                    mem_bs9 = 24.0 + (74.0 - 24.0) * \
                        (output_classes - 2) / (105 - 2)
                    batch_size = 2 + (9 - 2) * \
                        (mem - mem_bs2) / (mem_bs9 - mem_bs2)
                    batch_size = int(batch_size)
                    batch_size = max(batch_size, 1)

                    parser["training"].update(
                        {"num_patches_per_iter": batch_size})
                    parser["training"].update(
                        {"num_patches_per_image": 2 * batch_size})
                    parser["training"].update(
                        {"num_epochs": int(400.0 / float(batch_size))})

                    ConfigParser.export_config_file(
                        parser.get(), _file, fmt="yaml", default_flow_style=None)

    return


def run(config_file: Optional[Union[str, Sequence[str]]] = None, **override):
    pre_operation(config_file, **override)

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    _args = _update_args(config_file=config_file, **override)
    config_file_ = _pop_args(_args, "config_file")[0]

    parser = ConfigParser()
    parser.read_config(config_file_)
    parser.update(pairs=_args)

    amp = parser.get_parsed_content("training#amp")
    data_file_base_dir = parser.get_parsed_content("data_file_base_dir")
    data_list_file_path = parser.get_parsed_content("data_list_file_path")
    fold = parser.get_parsed_content("fold")
    log_output_file = parser.get_parsed_content("validate#log_output_file")
    num_sw_batch_size = parser.get_parsed_content("training#num_sw_batch_size")
    output_classes = parser.get_parsed_content("training#output_classes")
    overlap_ratio = parser.get_parsed_content("training#overlap_ratio")
    patch_size_valid = parser.get_parsed_content("training#patch_size_valid")
    softmax = parser.get_parsed_content("training#softmax")

    ckpt_name = parser.get_parsed_content("validate")["ckpt_name"]
    output_path = parser.get_parsed_content("validate")["output_path"]
    save_mask = parser.get_parsed_content("validate")["save_mask"]

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    CONFIG["handlers"]["file"]["filename"] = log_output_file
    logging.config.dictConfig(CONFIG)

    infer_transforms = parser.get_parsed_content("transforms_infer")
    validate_transforms = transforms.Compose(
        [
            infer_transforms,
            transforms.LoadImaged(keys="label"),
            transforms.EnsureChannelFirstd(keys="label"),
            transforms.EnsureTyped(keys="label"),
        ]
    )

    datalist = ConfigParser.load_config_file(data_list_file_path)

    list_valid = []
    for item in datalist["training"]:
        if item["fold"] == fold:
            item.pop("fold", None)
            list_valid.append(item)

    files = []
    for _i in range(len(list_valid)):
        str_img = os.path.join(data_file_base_dir, list_valid[_i]["image"])
        str_seg = os.path.join(data_file_base_dir, list_valid[_i]["label"])

        if (not os.path.exists(str_img)) or (not os.path.exists(str_seg)):
            continue

        files.append({"image": str_img, "label": str_seg})

    val_files = files

    val_ds = monai.data.Dataset(data=val_files, transform=validate_transforms)
    val_loader = ThreadDataLoader(
        val_ds,
        num_workers=2,
        batch_size=1,
        shuffle=False)

    device = torch.device("cuda:0")

    model = parser.get_parsed_content("training_network#network")
    model = model.to(device)

    pretrained_ckpt = torch.load(ckpt_name, map_location=device)
    model.load_state_dict(pretrained_ckpt)
    logger.debug(f"[info] checkpoint {ckpt_name:s} loaded")

    if softmax:
        post_pred = transforms.Compose(
            [transforms.EnsureType(), transforms.AsDiscrete(to_onehot=None)])
    else:
        post_pred = transforms.Compose([transforms.EnsureType()])

    post_transforms = [
        transforms.Invertd(
            keys="pred",
            transform=validate_transforms,
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
        post_transforms += [transforms.AsDiscreted(keys="pred", threshold=0.5)]

    if save_mask:
        post_transforms += [
            transforms.SaveImaged(
                keys="pred",
                meta_keys="pred_meta_dict",
                output_dir=output_path,
                output_postfix="seg",
                resample=False,
                data_root_dir=data_file_base_dir)]

    post_transforms = transforms.Compose(post_transforms)

    metric_dim = output_classes - 1 if softmax else output_classes
    metric_sum = 0.0
    metric_count = 0
    metric_mat = []

    row = ["case_name"]
    for _i in range(metric_dim):
        row.append("class_" + str(_i + 1))

    with open(os.path.join(output_path, "raw.csv"), "w", encoding="UTF8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    model.eval()
    with torch.no_grad():
        metric = torch.zeros(metric_dim * 2, dtype=torch.float)

        _index = 0
        for val_data in val_loader:
            torch.cuda.empty_cache()

            val_images = None
            val_labels = None
            val_outputs = None
            finished = None

            device_list_input = [device, device, "cpu"]
            device_list_output = [device, "cpu", "cpu"]

            for _device_in, _device_out in zip(
                    device_list_input, device_list_output):
                try:
                    val_images = val_data["image"].to(_device_in)
                    val_labels = val_data["label"].to(_device_out)

                    if _device_in != device or _device_out != device:
                        model = model.cpu()
                        torch.cuda.empty_cache()
                        model = model.to(device)

                    with torch.cuda.amp.autocast(enabled=amp):
                        val_data["pred"] = sliding_window_inference(
                            inputs=val_images,
                            roi_size=patch_size_valid,
                            sw_batch_size=num_sw_batch_size,
                            predictor=model,
                            mode="gaussian",
                            overlap=overlap_ratio,
                            sw_device=device,
                            device=_device_out)

                    finished = True

                except BaseException:
                    finished = False

                if finished:
                    break

            del val_images
            val_data["image"] = val_data["image"].cpu()
            val_data["label"] = val_data["label"].cpu()
            val_data["pred"] = val_data["pred"].cpu()
            val_labels = val_labels.cpu()
            torch.cuda.empty_cache()

            val_data = [
                post_transforms(i) for i in
                monai.data.decollate_batch(val_data)]

            val_outputs = post_pred(val_data[0]["pred"])
            val_outputs = val_outputs[None, ...]

            if softmax:
                val_labels = val_labels.int()
                value = torch.zeros(1, metric_dim)
                for _k in range(1, metric_dim + 1):
                    value[0, _k - 1] = compute_dice(
                        y_pred=(val_outputs == _k).float(),
                        y=(val_labels == _k).float(),
                        include_background=not softmax)
            else:
                value = compute_dice(
                    y_pred=val_outputs,
                    y=val_labels,
                    include_background=not softmax)

            logger.debug(f"{_index + 1} / {len(val_loader)}: {value}")

            metric_count += len(value)
            metric_sum += value.sum().item()
            metric_vals = value.cpu().numpy()
            if len(metric_mat) == 0:
                metric_mat = metric_vals
            else:
                metric_mat = np.concatenate((metric_mat, metric_vals), axis=0)

            print_message = ""
            print_message += str(_index + 1)
            print_message += ", "
            print_message += val_data[0]["pred"].meta["filename_or_obj"]
            print_message += ", "
            for _k in range(metric_dim):
                if output_classes == 2:
                    print_message += f"{metric_vals.squeeze():.5f}"
                else:
                    print_message += f"{metric_vals.squeeze()[_k]:.5f}"
                print_message += ", "
            logger.debug(print_message)

            row = [val_data[0]["pred"].meta["filename_or_obj"]]
            for _i in range(metric_dim):
                row.append(metric_vals[0, _i])

            with open(os.path.join(output_path, "raw.csv"), "a", encoding="UTF8") as f:
                writer = csv.writer(f)
                writer.writerow(row)

            for _c in range(metric_dim):
                val0 = torch.nan_to_num(value[0, _c], nan=0.0)
                val1 = 1.0 - torch.isnan(value[0, _c]).float()
                metric[2 * _c] += val0
                metric[2 * _c + 1] += val1

            _index += 1

        metric = metric.tolist()
        for _c in range(metric_dim):
            logger.debug(
                f"evaluation metric - class {_c + 1:d}:", metric[2 * _c] / metric[2 * _c + 1])
        avg_metric = 0
        for _c in range(metric_dim):
            avg_metric += metric[2 * _c] / metric[2 * _c + 1]
        avg_metric = avg_metric / float(metric_dim)
        logger.debug(f"avg_metric, {avg_metric}")

        dict_file = {}
        dict_file["acc"] = float(avg_metric)
        for _c in range(metric_dim):
            dict_file["acc_class" +
                      str(_c + 1)] = metric[2 * _c] / metric[2 * _c + 1]

        with open(os.path.join(output_path, "summary.yaml"), "w") as out_file:
            yaml.dump(dict_file, stream=out_file)

    return


if __name__ == "__main__":
    from monai.utils import optional_import

    fire, _ = optional_import("fire")
    fire.Fire()
