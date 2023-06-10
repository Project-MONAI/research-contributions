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

import torch
import torch.distributed as dist

import monai
from monai import transforms
from monai.apps.auto3dseg.auto_runner import logger
from monai.auto3dseg.utils import datafold_read
from monai.bundle import ConfigParser
from monai.bundle.scripts import _pop_args, _update_args
from monai.data import ThreadDataLoader, decollate_batch, list_data_collate
from monai.inferers import sliding_window_inference

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
if __package__ in (None, ""):
    from train import CONFIG, pre_operation
else:
    from .train import CONFIG, pre_operation


class InferClass:
    def __init__(self, config_file: Optional[Union[str, Sequence[str]]] = None, **override):
        pre_operation(config_file, **override)
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

        _args = _update_args(config_file=config_file, **override)
        config_file_ = _pop_args(_args, "config_file")[0]

        parser = ConfigParser()
        parser.read_config(config_file_)
        parser.update(pairs=_args)

        data_file_base_dir = parser.get_parsed_content("data_file_base_dir")
        data_list_file_path = parser.get_parsed_content("data_list_file_path")
        self.amp = parser.get_parsed_content("amp")
        self.fast = parser.get_parsed_content("infer")["fast"]
        self.num_sw_batch_size = parser.get_parsed_content("num_sw_batch_size")
        self.overlap_ratio_final = parser.get_parsed_content("overlap_ratio_final")
        self.patch_size_valid = parser.get_parsed_content("patch_size_valid")
        softmax = parser.get_parsed_content("softmax")

        ckpt_name = parser.get_parsed_content("infer")["ckpt_name"]
        data_list_key = parser.get_parsed_content("infer")["data_list_key"]
        output_path = parser.get_parsed_content("infer")["output_path"]

        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        CONFIG["handlers"]["file"]["filename"] = parser.get_parsed_content("infer")["log_output_file"]
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

        self.model = parser.get_parsed_content("network")
        self.model = self.model.to(self.device)

        pretrained_ckpt = torch.load(ckpt_name, map_location=self.device)
        self.model.load_state_dict(pretrained_ckpt)
        logger.debug(f"Checkpoint {ckpt_name:s} loaded.")

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
        # return pred probs
        self.post_transforms_prob = transforms.Compose(post_transforms)
        if softmax:
            post_transforms += [transforms.AsDiscreted(keys="pred", argmax=True)]
        else:
            post_transforms += [transforms.AsDiscreted(keys="pred", threshold=0.5)]

        post_transforms += [
            transforms.SaveImaged(
                keys="pred",
                meta_keys="pred_meta_dict",
                output_dir=output_path,
                output_postfix="seg",
                resample=False,
                data_root_dir=data_file_base_dir,
                print_log=False,
            )
        ]
        self.post_transforms = transforms.Compose(post_transforms)

        return

    @torch.no_grad()
    def infer(self, image_file, save_mask=False):
        """Infer a single image_file. If save_mask is true, save the argmax prediction to disk. If false,
        do not save and return the probability maps (usually used by autorunner emsembler).
        """
        self.model.eval()
        batch_data = self.infer_transforms(image_file)
        batch_data = list_data_collate([batch_data])
        device_list_input = [self.device, self.device, "cpu"]
        device_list_output = [self.device, "cpu", "cpu"]
        for _device_in, _device_out in zip(device_list_input, device_list_output):
            try:
                logger.debug(f"Working on {image_file} on device {_device_in}/{_device_out} in/out.")
                with torch.cuda.amp.autocast(enabled=self.amp):
                    batch_data["pred"] = sliding_window_inference(
                        inputs=batch_data["image"].to(_device_in),
                        roi_size=self.patch_size_valid,
                        sw_batch_size=self.num_sw_batch_size,
                        predictor=self.model,
                        mode="gaussian",
                        overlap=self.overlap_ratio_final,
                        sw_device=self.device,
                        device=_device_out,
                    )
                if save_mask:
                    batch_data = [self.post_transforms(i) for i in decollate_batch(batch_data)]
                else:
                    batch_data = [self.post_transforms_prob(i) for i in decollate_batch(batch_data)]
                finished = True
            except RuntimeError as e:
                if not any(x in str(e).lower() for x in ("memory", "cuda", "cudnn")):
                    raise e
                finished = False
            if finished:
                break
        if not finished:
            raise RuntimeError("Infer not finished due to OOM.")
        logger.debug(f"{image_file} fininshed.")
        return batch_data[0]["pred"]

    @torch.no_grad()
    def infer_all(self, save_mask=True):
        for _i in range(len(self.infer_files)):
            infer_filename = self.infer_files[_i]
            _ = self.infer(infer_filename, save_mask)
        return

    @torch.no_grad()
    def batch_infer(self):
        self.model.eval()
        with torch.no_grad():
            for d in self.infer_loader:
                torch.cuda.empty_cache()
                device_list_input = [self.device, self.device, "cpu"]
                device_list_output = [self.device, "cpu", "cpu"]
                for _device_in, _device_out in zip(device_list_input, device_list_output):
                    try:
                        infer_images = d["image"].to(_device_in)
                        with torch.cuda.amp.autocast(enabled=self.amp):
                            d["pred"] = sliding_window_inference(
                                inputs=infer_images,
                                roi_size=self.patch_size_valid,
                                sw_batch_size=self.num_sw_batch_size,
                                predictor=self.model,
                                mode="gaussian",
                                overlap=self.overlap_ratio_final,
                                sw_device=self.device,
                                device=_device_out,
                            )
                        d = [self.post_transforms(i) for i in decollate_batch(d)]
                        finished = True
                    except RuntimeError as e:
                        if not any(x in str(e).lower() for x in ("memory", "cuda", "cudnn")):
                            raise e
                        finished = False
                    if finished:
                        break
                if not finished:
                    raise RuntimeError("Batch infer not finished due to OOM.")
        return


def run(config_file: Optional[Union[str, Sequence[str]]] = None, **override):
    infer_instance = InferClass(config_file, **override)
    if infer_instance.fast:
        logger.debug("Using fast mode.")
        infer_instance.batch_infer()
    else:
        logger.debug("Using slow mode.")
        infer_instance.infer_all()
    return


if __name__ == "__main__":
    from monai.utils import optional_import

    fire, _ = optional_import("fire")
    fire.Fire()
