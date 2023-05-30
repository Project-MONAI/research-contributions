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
import shutil
import warnings
from typing import Optional

import fire
import numpy as np
import yaml

from monai.apps.auto3dseg import BundleAlgo
from monai.apps.auto3dseg.auto_runner import logger
from monai.bundle import ConfigParser

print = logger.debug

if __package__ in (None, ""):
    from utils import auto_adjust_network_settings, logger_configure
else:
    from .utils import auto_adjust_network_settings, logger_configure


class Segresnet2dAlgo(BundleAlgo):
    def pre_check_skip_algo(self, skip_bundlegen: bool = False, skip_info: str = ""):
        """
        Precheck if the algorithm needs to be skipped.
        If the median spacing of the dataset is not highly anisotropic (res_z < 3*(res_x + rex_y)/2),
        the 2D segresnet will be skipped by setting self.skip_bundlegen=True.
        """
        if self.data_stats_files is None or bool(os.environ.get("SEGRESNET2D_ALWAYS", False)):
            return skip_bundlegen, skip_info

        data_stats = ConfigParser(globals=False)
        if os.path.exists(str(self.data_stats_files)):
            data_stats.read_config(str(self.data_stats_files))
        else:
            data_stats.update(self.data_stats_files)
        spacing = data_stats["stats_summary#image_stats#spacing#median"]
        if len(spacing) > 2:
            if spacing[-1] < 3 * (spacing[0] + spacing[1]) / 2:
                skip_bundlegen = True
                skip_info = f"SegresNet2D is skipped due to median spacing of {spacing},  which means the dataset is not highly anisotropic, e.g. spacing[2] < 3*(spacing[0] + spacing[1])/2) ."

        return skip_bundlegen, skip_info

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

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048"

        if kwargs.pop("fill_with_datastats", True):
            config = {"bundle_root": output_path}

            if self.data_list_file is None or not os.path.exists(str(self.data_list_file)):
                raise ValueError(f"Unable to load self.data_list_file {self.data_list_file}")

            if data_stats_file is None or not os.path.exists(data_stats_file):
                raise ValueError("data_stats_file unable to read: " + str(data_stats_file))

            input_config = ConfigParser.load_config_file(self.data_list_file)
            logger_configure(debug=input_config.get("debug", False))
            print(f"Loaded self.data_list_file {self.data_list_file}")

            data_stats = ConfigParser(globals=False)
            data_stats.read_config(data_stats_file)

            config["data_file_base_dir"] = os.path.abspath(input_config.pop("dataroot"))
            config["data_list_file_path"] = os.path.abspath(input_config.pop("datalist"))

            # data_list_file_path = os.path.abspath(input_config.pop("datalist")) #TODO, consider relative path
            # if output_path.startswith(os.path.dirname(data_list_file_path)):
            #     config["data_list_file_path"] = os.path.relpath(data_list_file_path, output_path) #use relative path to json
            #     # config["data_list_file_path"] = f"$@bundle_root + '/' + '{os.path.relpath(data_list_file_path, output_path)}'" #use relative path to json
            # else:
            #     config["data_list_file_path"] = data_list_file_path

            ##########
            if "modality" in input_config:
                modality = input_config.pop("modality").lower().strip()
            else:
                warnings.warn("Config modality is not specified, assuming CT image")
                modality = "ct"

            if modality not in ["ct", "mri"]:
                raise ValueError("Modality must be either CT or MRI, but got" + str(modality))
            config["modality"] = modality

            input_channels = int(data_stats["stats_summary#image_stats#channels#max"]) + len(
                input_config.get("extra_modalities", {})
            )
            output_classes = len(data_stats["stats_summary#label_stats#labels"])

            config["input_channels"] = input_channels
            config["output_classes"] = output_classes

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

            spacing_median = copy.deepcopy(data_stats["stats_summary#image_stats#spacing#median"])
            spacing_10 = copy.deepcopy(data_stats["stats_summary#image_stats#spacing#percentile_10_0"])

            if "ct" in modality:
                config["normalize_mode"] = "range"
            elif "mr" in modality:
                config["normalize_mode"] = "meanstd"

            spacing = input_config.pop("resample_resolution", None)
            resample_mode = input_config.pop("resample_mode", None)
            if resample_mode is None:
                resample_mode = "auto"

            if spacing is not None:
                # if working resolution is provided manually
                pass
            elif resample_mode == "median":
                spacing = spacing_median
            elif resample_mode == "median10":
                spacing = [spacing_median[0], spacing_median[1], spacing_10[2]]
            elif resample_mode == "ones":
                spacing = [1.0, 1.0, 1.0]
            elif resample_mode == "auto" or resample_mode is None:
                spacing = [
                    spacing_median[0],
                    spacing_median[1],
                    max(0.5 * (spacing_median[0] + spacing_median[1]), float(spacing_10[2])),
                ]
            else:
                raise ValueError("Unsupported resample_mode" + str(resample_mode))

            config["resample_resolution"] = spacing

            config["anisotropic_scales"] = input_config.pop("anisotropic_scales", None)
            # if config["anisotropic_scales"]  is None:
            #     config["anisotropic_scales"] =  not (0.75 <= (0.5*(spacing[0]+spacing[1]) / spacing[2]) <= 1.25)
            config["anisotropic_scales"] = False

            ###########################################
            spacing_lower_bound = np.array(data_stats["stats_summary#image_stats#spacing#percentile_00_5"])
            spacing_upper_bound = np.array(data_stats["stats_summary#image_stats#spacing#percentile_99_5"])
            config["spacing_median"] = list(data_stats["stats_summary#image_stats#spacing#median"])
            config["spacing_lower"] = spacing_lower_bound.tolist()
            config["spacing_upper"] = spacing_upper_bound.tolist()

            ###########################################
            resample = input_config.pop("resample", None)
            # if resample is None:
            #     if np.any(spacing_lower_bound / np.array(spacing) < 0.5) or np.any(
            #         spacing_upper_bound / np.array(spacing) > 1.5
            #     ):
            #         resample = True
            #     else:
            #         resample = False
            # config["resample"] = resample

            config["resample"] = False

            print(
                f"Resampling params: \n"
                f"resample {config['resample']} \n"
                f"resolution {config['resample_resolution']} \n"
                f"resample_mode {resample_mode} \n"
                f"anisotropic_scales {config['anisotropic_scales']} \n"
                f"res bounds {config['spacing_lower']} {config['spacing_upper']} \n"
                f"modality {modality} \n"
            )

            ###########################################

            n_cases = data_stats["stats_summary#n_cases"]

            image_size_mm_90 = data_stats["stats_summary#image_stats#sizemm#percentile_90_0"]
            image_size_mm_median = data_stats["stats_summary#image_stats#sizemm#median"]
            image_size_90 = (np.array(image_size_mm_90) / np.array(spacing)).astype(np.int32).tolist()

            print(
                f"Found sizemm in new datastats median {image_size_mm_median} per90 {image_size_mm_90} n_cases  {n_cases}"
            )
            print(f"Using avg image size 90 {image_size_90} for resample res {spacing} n_cases {n_cases}")

            config["image_size_mm_median"] = image_size_mm_median
            config["image_size_mm_90"] = image_size_mm_90

            image_size = input_config.pop("image_size", image_size_90)
            config["image_size"] = image_size

            max_epochs = int(np.clip(np.ceil(80000.0 / n_cases), a_min=300, a_max=1250))
            config["num_epochs"] = max_epochs

            ###########################################

            roi_size, levels, init_filters, batch_size = auto_adjust_network_settings(
                auto_scale_batch=input_config.get("auto_scale_batch", False),
                auto_scale_roi=input_config.get("auto_scale_roi", False),
                auto_scale_filters=input_config.get("auto_scale_filters", False),
                image_size_mm=config["image_size_mm_median"],
                spacing=config["resample_resolution"],
                anisotropic_scales=config["anisotropic_scales"],
                output_classes=config["output_classes"],
            )

            if input_config.get("roi_size", None):
                roi_size = input_config.get("roi_size", None)
            if input_config.get("batch_size", None):
                batch_size = input_config.get("batch_size", None)

            config["roi_size"] = roi_size
            config["batch_size"] = batch_size

            print(f"Updating roi_size (divisible) final {roi_size} levels {levels}")

            ###########################################
            # update network config
            blocks_down = [1, 2, 2, 4, 4]
            if levels >= 5:  # default
                blocks_down = [1, 2, 2, 4, 4]
            elif levels == 4:
                blocks_down = [1, 2, 2, 4]
            elif levels == 3:
                blocks_down = [1, 2, 4]
            elif levels == 2:
                blocks_down = [1, 3]
            elif levels == 1:
                blocks_down = [2]

            config["network#blocks_down"] = blocks_down
            config["network#init_filters"] = init_filters

            ###########################################
            # cropping mode, if any roi_size less than 0.7*image size
            if any([r < 0.8 * i for r, i in zip(roi_size, image_size)]):
                config["crop_mode"] = "ratio"
            else:
                config["crop_mode"] = "rand"

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

        output_path = os.path.join(output_path, algo_name)
        config = ConfigParser.load_config_file(os.path.join(output_path, "configs/hyper_parameters.yaml"))

        for c in config.get("custom_data_transforms", []):
            if "transform" in c and "_target_" in c["transform"]:
                target = c["transform"]["_target_"]
                target = "/".join(target.split(".")[:-1]) + ".py"
                print(f"Copying custom transform file {target} into {output_path}")
                shutil.copy(target, output_path)
            else:
                raise ValueError("Malformed custom_data_transforms parameter!" + str(c))


if __name__ == "__main__":
    fire.Fire({"Segresnet2dAlgo": Segresnet2dAlgo})
