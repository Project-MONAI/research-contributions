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

import copy
import os
import warnings
from typing import Optional

import fire
import numpy as np
import shutil

from monai.apps.auto3dseg import BundleAlgo
from monai.bundle import ConfigParser


def roi_ensure_divisible(roi_size, levels):
    """
    Calculate the cropping size (roi_size) that is evenly divisible by 2 for the given number of hierarchical levels
    e.g. for for the network of 5 levels (4 downsamplings), roi should be divisible by 2^(5-1)=16
    """

    multiplier = pow(2, levels - 1)
    roi_size2 = []
    for r in roi_size:
        if r % multiplier != 0:
            p = multiplier * max(
                2, int(r / float(multiplier))
            )  # divisible by levels, but not smaller then 2 at final level
            roi_size2.append(p)
        else:
            roi_size2.append(r)

    return roi_size2


def roi_ensure_levels(levels, roi_size, image_size):
    """
    In case the image (at least one axis) is smaller then roi, reduce the roi and number of levels
    """

    while all([r > 1.5 * i for r, i in zip(roi_size, image_size)]) and levels > 1:
        levels = levels - 1
        roi_size = [r // 2 for r in roi_size]
    return levels, roi_size


class SegresnetAlgo(BundleAlgo):
    def fill_template_config(self, data_stats_file: Optional[str] = None, output_path: Optional[str] = None, **kwargs):
        """
        Fill the freshly copied config templates

        Args:
            data_stats_file: the stats report from DataAnalyzer in yaml format
            output_path: the root folder to scripts/configs directories.
            kwargs: parameters to override the config writing and ``fill_with_datastats``
                a on/off switch to either use the data_stats_file to fill the template or
                load it directly from the self.fill_records
        """

        if output_path is None:
            raise ValueError("output_path is not provided")

        if kwargs.pop("fill_with_datastats", True):

            config = {"bundle_root": output_path}

            if data_stats_file is None or not os.path.exists(data_stats_file):
                raise ValueError("data_stats_file unable to read: " + str(data_stats_file))

            data_stats = ConfigParser(globals=False)
            data_stats.read_config(str(data_stats_file))

            if self.data_list_file is not None and os.path.exists(str(self.data_list_file)):
                input_config = ConfigParser.load_config_file(self.data_list_file)
                print("Loaded self.data_list_file", self.data_list_file)
            else:
                print("Unable to load self.data_list_file", self.data_list_file)

            config["data_file_base_dir"] = os.path.abspath(input_config.pop("dataroot"))
            config["data_list_file_path"] = os.path.abspath(input_config.pop("datalist"))

            ##########
            if "modality" in input_config:
                modality = input_config.pop("modality").lower().strip()
            else:
                warnings.warn("Config modality is not specified, assuming CT image")
                modality = "ct"

            if modality not in ["ct", "mri"]:
                raise ValueError("Modality must be either CT or MRI, but got" + str(modality))
            config["modality"] = modality

            input_channels = data_stats["stats_summary#image_stats#channels#max"] + len(
                input_config.get("extra_modalities", {})
            )
            output_classes = len(data_stats["stats_summary#label_stats#labels"])

            config["input_channels"] = input_channels
            config["output_classes"] = output_classes

            # update config config
            roi_size = [224, 224, 144]  # default roi
            levels = 5  # default number of hierarchical levels
            image_size = [int(i) for i in data_stats["stats_summary#image_stats#shape#percentile_99_5"]]
            config["image_size"] = image_size

            ###########################################
            # adjust to image size
            # min for each of spatial dims
            roi_size = [min(r, i) for r, i in zip(roi_size, image_size)]
            roi_size = roi_ensure_divisible(roi_size, levels=levels)
            # reduce number of levels to smaller then 5 (default) if image is too small
            levels, roi_size = roi_ensure_levels(levels, roi_size, image_size)
            config["roi_size"] = roi_size

            ###########################################
            n_cases = len(data_stats["stats_by_cases"])
            max_epochs = int(np.clip(np.ceil(80000.0 / n_cases), a_min=300, a_max=1250))
            config["num_epochs"] = max_epochs
            config["warmup_epochs"] = int(np.ceil(0.01 * max_epochs))

            ###########################################
            sigmoid = input_config.pop("sigmoid", False)
            class_names = input_config.pop("class_names", None)
            class_index = input_config.pop("class_index", None)

            if class_names is None:
                class_names = class_index = None
            elif not isinstance(class_names, list):
                warnings.warn("class_names must be a list")
                class_names = class_index = None
            elif isinstance(class_names, list) and isinstance(class_names[0], dict):
                class_index = [x["index"] for x in class_names]
                class_names = [x["name"] for x in class_names]

                # check for overlap
                all_ind = []
                for a in class_index:
                    if bool(set(all_ind) & set(a)):  # overlap found
                        sigmoid = True
                        break
                    all_ind = all_ind + a

            config["class_names"] = class_names
            config["class_index"] = class_index
            config["sigmoid"] = sigmoid

            if sigmoid and class_index is not None:
                config["output_classes"] = len(class_index)

            ###########################################

            intensity_lower_bound = float(data_stats["stats_summary#image_foreground_stats#intensity#percentile_00_5"])
            intensity_upper_bound = float(data_stats["stats_summary#image_foreground_stats#intensity#percentile_99_5"])
            config["intensity_bounds"] = [intensity_lower_bound, intensity_upper_bound]

            spacing = data_stats["stats_summary#image_stats#spacing#median"]

            if "ct" in modality:
                config["normalize_mode"] = "range"
                if not config.get("anisotropic_scales", False):
                    spacing = [1.0, 1.0, 1.0]

            elif "mr" in modality:
                config["normalize_mode"] = "meanstd"

            config["resample_resolution"] = spacing

            ###########################################
            spacing_lower_bound = np.array(data_stats["stats_summary#image_stats#spacing#percentile_00_5"])
            spacing_upper_bound = np.array(data_stats["stats_summary#image_stats#spacing#percentile_99_5"])
            config["spacing_lower"] = spacing_lower_bound.tolist()
            config["spacing_upper"] = spacing_upper_bound.tolist()


            ###########################################
            if np.any(spacing_lower_bound / np.array(spacing) < 0.5) or np.any(
                spacing_upper_bound / np.array(spacing) > 1.5
            ):
                config["resample"] = True
            else:
                config["resample"] = False

            ###########################################
            # cropping mode
            should_crop_based_on_foreground = any(
                [r < 0.5 * i for r, i in zip(roi_size, image_size)]
            )  # if any roi_size less tehn 0.5*image size
            if should_crop_based_on_foreground:
                config["crop_mode"] = "ratio"
            else:
                config["crop_mode"] = "rand"

            ###########################################
            # update network config
            blocks_down = None
            if levels >= 5:  # default
                blocks_down = [1, 2, 2, 4, 4]
            elif levels == 4:
                blocks_down = [1, 2, 2, 4]
            elif levels == 3:
                blocks_down = [1, 3, 4]
            elif levels == 2:
                blocks_down = [2, 6]
            elif levels == 1:
                blocks_down = [8]

            if blocks_down is not None:
                config["network#blocks_down"] = blocks_down

            config.update(input_config)  # override if any additional inputs provided

            fill_records = {"hyper_parameters.yaml": config}
        else:
            fill_records = self.fill_records

        for yaml_file, yaml_contents in fill_records.items():
            file_path = os.path.join(output_path, "configs", yaml_file)

            parser = ConfigParser(globals=False)
            parser.read_config(file_path)
            for k, v in yaml_contents.items():
                if k in kwargs:
                    parser[k] = kwargs.pop(k)
                else:
                    parser[k] = copy.deepcopy(v)  # some values are dicts
                yaml_contents[k] = copy.deepcopy(parser[k])

            for k, v in kwargs.items():  # override new params not in fill_records
                if parser.get(k, None) is not None:
                    parser[k] = copy.deepcopy(v)
                    yaml_contents.update({k: parser[k]})

            ConfigParser.export_config_file(
                parser.get(), file_path, fmt="yaml", default_flow_style=None, sort_keys=False
            )

        return fill_records

    def export_to_disk(self, output_path: str, algo_name: str, **kwargs):
        super().export_to_disk(output_path=output_path, algo_name=algo_name, **kwargs)

        output_path =os.path.join(output_path, algo_name)
        config = ConfigParser.load_config_file(os.path.join(output_path, "configs/hyper_parameters.yaml"))

        for c in config.get('custom_data_transforms',[]):
            if "transform" in c and "_target_" in c["transform"]:
                target = c["transform"]["_target_"]
                target = "/".join(target.split(".")[:-1]) + ".py"
                print("Copying custom transform file", target, "into", output_path)
                shutil.copy(target, output_path)
            else:
                raise ValueError("Malformed custom_data_transforms parameter!"+str(c))

if __name__ == "__main__":
    fire.Fire({"SegresnetAlgo": SegresnetAlgo})
