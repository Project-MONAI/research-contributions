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
import os
import sys
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.distributed as dist

import monai
from monai import transforms
from monai.apps.auto3dseg.auto_runner import logger
from monai.apps.utils import DEFAULT_FMT
from monai.auto3dseg.utils import datafold_read
from monai.bundle import ConfigParser
from monai.bundle.scripts import _pop_args, _update_args
from monai.data import ThreadDataLoader, decollate_batch, list_data_collate
from monai.inferers import sliding_window_inference

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


class InferClass:
    def __init__(self, config_file: Optional[Union[str, Sequence[str]]] = None, **override):
        pre_operation(config_file, **override)

        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

        _args = _update_args(config_file=config_file, **override)
        config_file_ = _pop_args(_args, "config_file")[0]

        parser = ConfigParser()
        parser.read_config(config_file_)
        parser.update(pairs=_args)

        self.amp = parser.get_parsed_content("training#amp")
        data_file_base_dir = parser.get_parsed_content("data_file_base_dir")
        data_list_file_path = parser.get_parsed_content("data_list_file_path")
        self.fast = parser.get_parsed_content("infer")["fast"]
        log_output_file = parser.get_parsed_content("infer#log_output_file")
        self.num_patches_per_iter = parser.get_parsed_content("training#num_patches_per_iter")
        self.num_sw_batch_size = parser.get_parsed_content("training#num_sw_batch_size")
        self.overlap_ratio = parser.get_parsed_content("training#overlap_ratio")
        self.patch_size_valid = parser.get_parsed_content("training#patch_size_valid")
        softmax = parser.get_parsed_content("training#softmax")
        self.sw_input_on_cpu = parser.get_parsed_content("training#sw_input_on_cpu")

        ckpt_name = parser.get_parsed_content("infer")["ckpt_name"]
        data_list_key = parser.get_parsed_content("infer")["data_list_key"]
        output_path = parser.get_parsed_content("infer")["output_path"]

        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        CONFIG["handlers"]["file"]["filename"] = log_output_file
        logging.config.dictConfig(CONFIG)

        self.infer_transforms = parser.get_parsed_content("transforms_infer")

        testing_files, _ = datafold_read(
            datalist=data_list_file_path, basedir=data_file_base_dir, fold=-1, key=data_list_key
        )
        self.infer_files = testing_files

        self.infer_loader = None
        if self.fast:
            infer_ds = monai.data.Dataset(data=self.infer_files, transform=self.infer_transforms)
            self.infer_loader = ThreadDataLoader(infer_ds, num_workers=8, batch_size=1, shuffle=False)

        try:
            device = f"cuda:{dist.get_rank()}"
        except BaseException:
            device = f"cuda:0"
        self.device = device

        self.model = parser.get_parsed_content("training_network#network")
        self.model = self.model.to(self.device)

        pretrained_ckpt = torch.load(ckpt_name, map_location=self.device)
        self.model.load_state_dict(pretrained_ckpt)
        logger.debug(f"checkpoint {ckpt_name:s} loaded")

        post_transforms = [
            transforms.Invertd(
                keys="pred",
                transform=self.infer_transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            ),
            transforms.Activationsd(keys="pred", softmax=softmax, sigmoid=not softmax),
        ]
        self.post_transforms_prob = transforms.Compose(post_transforms)

        if softmax:
            post_transforms += [transforms.AsDiscreted(keys="pred", argmax=True)]
        else:
            post_transforms += [transforms.AsDiscreted(keys="pred", threshold=0.5 + np.finfo(np.float32).eps)]

        post_transforms += [
            transforms.SaveImaged(
                keys="pred",
                meta_keys="pred_meta_dict",
                output_dir=output_path,
                output_postfix="seg",
                resample=False,
                print_log=False,
                data_root_dir=data_file_base_dir,
            )
        ]
        self.post_transforms = transforms.Compose(post_transforms)

        return

    @torch.no_grad()
    def infer(self, image_file, save_mask=False):
        self.model.eval()

        batch_data = self.infer_transforms(image_file)
        batch_data = list_data_collate([batch_data])

        finished = None
        device_list_input = None
        device_list_output = None

        if self.sw_input_on_cpu:
            device_list_input = ["cpu"]
            device_list_output = ["cpu"]
        else:
            device_list_input = [self.device, self.device, "cpu"]
            device_list_output = [self.device, "cpu", "cpu"]

        for _device_in, _device_out in zip(device_list_input, device_list_output):
            try:
                infer_images = batch_data["image"].to(_device_in)

                if self.num_sw_batch_size is None:
                    sw_batch_size = self.num_patches_per_iter * 12 if _device_out == "cpu" else 1
                else:
                    sw_batch_size = self.num_sw_batch_size

                with torch.cuda.amp.autocast(enabled=self.amp):
                    batch_data["pred"] = sliding_window_inference(
                        inputs=infer_images,
                        roi_size=self.patch_size_valid,
                        sw_batch_size=sw_batch_size,
                        predictor=self.model,
                        mode="gaussian",
                        overlap=self.overlap_ratio,
                        sw_device=self.device,
                        device=_device_out,
                    )

                finished = True

            except BaseException:
                finished = False

            if finished:
                break

        del infer_images
        batch_data["image"] = batch_data["image"].cpu()
        batch_data["pred"] = batch_data["pred"].cpu()
        torch.cuda.empty_cache()

        if save_mask:
            batch_data = [self.post_transforms(i) for i in decollate_batch(batch_data)]
        else:
            batch_data = [self.post_transforms_prob(i) for i in decollate_batch(batch_data)]

        return batch_data[0]["pred"]

    def infer_all(self):
        for _i in range(len(self.infer_files)):
            infer_filename = self.infer_files[_i]
            _ = self.infer(infer_filename, save_mask=True)

        return

    @torch.no_grad()
    def batch_infer(self):
        self.model.eval()
        with torch.no_grad():
            for infer_data in self.infer_loader:
                torch.cuda.empty_cache()

                finished = None
                device_list_input = None
                device_list_output = None

                if self.sw_input_on_cpu:
                    device_list_input = ["cpu"]
                    device_list_output = ["cpu"]
                else:
                    device_list_input = [self.device, self.device, "cpu"]
                    device_list_output = [self.device, "cpu", "cpu"]

                for _device_in, _device_out in zip(device_list_input, device_list_output):
                    try:
                        infer_images = infer_data["image"].to(_device_in)

                        if self.num_sw_batch_size is None:
                            sw_batch_size = self.num_patches_per_iter * 12 if _device_out == "cpu" else 1
                        else:
                            sw_batch_size = self.num_sw_batch_size

                        with torch.cuda.amp.autocast(enabled=self.amp):
                            infer_data["pred"] = sliding_window_inference(
                                inputs=infer_images,
                                roi_size=self.patch_size_valid,
                                sw_batch_size=sw_batch_size,
                                predictor=self.model,
                                mode="gaussian",
                                overlap=self.overlap_ratio,
                                sw_device=self.device,
                                device=_device_out,
                            )

                        finished = True

                    except BaseException:
                        finished = False

                    if finished:
                        break

                del infer_images
                infer_data["image"] = infer_data["image"].cpu()
                infer_data["pred"] = infer_data["pred"].cpu()
                torch.cuda.empty_cache()

                infer_data = [self.post_transforms(i) for i in decollate_batch(infer_data)]

        return


def run(config_file: Optional[Union[str, Sequence[str]]] = None, **override):
    infer_instance = InferClass(config_file, **override)
    if infer_instance.fast:
        logger.debug("fast mode")
        infer_instance.batch_infer()
    else:
        logger.debug("slow mode")
        infer_instance.infer_all()

    return


if __name__ == "__main__":
    from monai.utils import optional_import

    fire, _ = optional_import("fire")
    fire.Fire()
