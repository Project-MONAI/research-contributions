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
from monai.auto3dseg.utils import datafold_read
from monai.bundle import ConfigParser
from monai.bundle.scripts import _pop_args, _update_args
from monai.data import ThreadDataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import compute_dice

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
if __package__ in (None, ""):
    from train import CONFIG, pre_operation
else:
    from .train import CONFIG, pre_operation


def run(config_file: Optional[Union[str, Sequence[str]]] = None, **override):
    pre_operation(config_file, **override)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    _args = _update_args(config_file=config_file, **override)
    config_file_ = _pop_args(_args, "config_file")[0]

    parser = ConfigParser()
    parser.read_config(config_file_)
    parser.update(pairs=_args)

    data_file_base_dir = parser.get_parsed_content("data_file_base_dir")
    data_list_file_path = parser.get_parsed_content("data_list_file_path")
    amp = parser.get_parsed_content("amp")
    fold = parser.get_parsed_content("fold")
    num_sw_batch_size = parser.get_parsed_content("num_sw_batch_size")
    output_classes = parser.get_parsed_content("output_classes")
    overlap_ratio_final = parser.get_parsed_content("overlap_ratio_final")
    patch_size_valid = parser.get_parsed_content("patch_size_valid")
    softmax = parser.get_parsed_content("softmax")

    ckpt_name = parser.get_parsed_content("validate")["ckpt_name"]
    output_path = parser.get_parsed_content("validate")["output_path"]
    save_mask = parser.get_parsed_content("validate")["save_mask"]

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    CONFIG["handlers"]["file"]["filename"] = parser.get_parsed_content("validate")["log_output_file"]
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

    if "class_names" in parser and isinstance(parser["class_names"], list) and "index" in parser["class_names"][0]:
        class_index = [x["index"] for x in parser["class_names"]]

        validate_transforms = transforms.Compose(
            [
                validate_transforms,
                transforms.Lambdad(
                    keys="label",
                    func=lambda x: torch.cat([sum([x == i for i in c]) for c in class_index], dim=0).to(dtype=x.dtype),
                ),
            ]
        )

    _, val_files = datafold_read(datalist=data_list_file_path, basedir=data_file_base_dir, fold=fold)

    val_ds = monai.data.Dataset(data=val_files, transform=validate_transforms)
    val_loader = ThreadDataLoader(val_ds, num_workers=2, batch_size=1, shuffle=False)

    device = torch.device("cuda:0")

    model = parser.get_parsed_content("network")
    model = model.to(device)

    pretrained_ckpt = torch.load(ckpt_name, map_location=device)
    model.load_state_dict(pretrained_ckpt)
    logger.debug(f"Checkpoint {ckpt_name:s} loaded")

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
        post_transforms += [
            transforms.Activationsd(keys="pred", sigmoid=True),
            transforms.AsDiscreted(keys="pred", threshold=0.5),
        ]

    if save_mask:
        post_transforms += [
            transforms.SaveImaged(
                keys="pred",
                meta_keys="pred_meta_dict",
                output_dir=output_path,
                output_postfix="seg",
                data_root_dir=data_file_base_dir,
                resample=False,
            )
        ]

    post_transforms = transforms.Compose(post_transforms)

    metric_dim = output_classes - 1 if softmax else output_classes

    model.eval()
    with torch.no_grad():
        metric = torch.zeros(metric_dim * 2, dtype=torch.float)

        _index = 0
        for val_data in val_loader:
            try:
                val_filename = val_data["image_meta_dict"]["filename_or_obj"][0]
            except BaseException:
                val_filename = val_data["image"].meta["filename_or_obj"][0]
            torch.cuda.empty_cache()
            device_list_input = [device, device, "cpu"]
            device_list_output = [device, "cpu", "cpu"]
            for _device_in, _device_out in zip(device_list_input, device_list_output):
                try:
                    with torch.cuda.amp.autocast(enabled=amp):
                        val_data["pred"] = sliding_window_inference(
                            inputs=val_data["image"].to(_device_in),
                            roi_size=patch_size_valid,
                            sw_batch_size=num_sw_batch_size,
                            predictor=model,
                            mode="gaussian",
                            overlap=overlap_ratio_final,
                            sw_device=device,
                            device=_device_out,
                        )
                    try:
                        val_data = [post_transforms(i) for i in decollate_batch(val_data)]
                    except BaseException:
                        val_data["pred"] = val_data["pred"].to("cpu")
                        val_data = [post_transforms(i) for i in decollate_batch(val_data)]
                    finished = True
                except RuntimeError as e:
                    if not any(x in str(e).lower() for x in ("memory", "cuda", "cudnn")):
                        raise e
                    finished = False
                if finished:
                    break
            if not finished:
                raise RuntimeError("Validate not finishing due to OOM.")

            value = compute_dice(
                y_pred=val_data[0]["pred"][None, ...],
                y=val_data[0]["label"][None, ...].to(val_data[0]["pred"].device),
                include_background=not softmax,
                num_classes=output_classes,
            ).to("cpu")
            logger.debug(f"{_index + 1} / {len(val_loader)}/ {val_filename}: {value}")

            for _c in range(metric_dim):
                val0 = torch.nan_to_num(value[0, _c], nan=0.0)
                val1 = 1.0 - torch.isnan(value[0, _c]).float()
                metric[2 * _c] += val0
                metric[2 * _c + 1] += val1

            _index += 1

        metric = metric.tolist()
        for _c in range(metric_dim):
            logger.debug(f"Evaluation metric - class {_c + 1:d}: {metric[2 * _c] / metric[2 * _c + 1]}")
        avg_metric = 0
        for _c in range(metric_dim):
            avg_metric += metric[2 * _c] / metric[2 * _c + 1]
        avg_metric = avg_metric / float(metric_dim)
        logger.debug(f"Avg_metric: {avg_metric}")

        dict_file = {}
        dict_file["acc"] = float(avg_metric)
        for _c in range(metric_dim):
            dict_file["acc_class" + str(_c + 1)] = metric[2 * _c] / metric[2 * _c + 1]

        with open(os.path.join(output_path, "summary.yaml"), "w") as out_file:
            yaml.dump(dict_file, stream=out_file)

    return


if __name__ == "__main__":
    from monai.utils import optional_import

    fire, _ = optional_import("fire")
    fire.Fire()
