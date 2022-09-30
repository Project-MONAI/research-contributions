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

import os
from copy import deepcopy

import numpy as np
import torch
from segresnet.scripts import roi_ensure_divisible, roi_ensure_levels

from monai.apps.auto3dseg import BundleAlgo
from monai.bundle import ConfigParser
from monai.bundle.scripts import _update_args
from monai.utils import optional_import


class SegresnetAlgo(BundleAlgo):
    def fill_template_config(self, data_stats_file, output_path, **kwargs):
        """
        Fill the freshly copied config templates

        Args:
            data_stats_file: the stats report from DataAnalyzer in yaml format
            output_path: the root folder to scripts/configs directories.
            kwargs: parameters to override the config writing and ``fill_with_datastats``
                a on/off switch to either use the data_stats_file to fill the template or
                load it directly from the self.fill_records
        """
        if kwargs.pop("fill_with_datastats", True):
            if data_stats_file is None:
                return
            data_stats = ConfigParser(globals=False)
            if os.path.exists(str(data_stats_file)):
                data_stats.read_config(str(data_stats_file))
            else:
                data_stats.update(data_stats_file)
            data_src_cfg = ConfigParser(globals=False)
            if self.data_list_file is not None and os.path.exists(str(self.data_list_file)):
                data_src_cfg.read_config(self.data_list_file)

            hyper_parameters = {"bundle_root": output_path}
            network = {}
            transforms_train = {}
            transforms_validate = {}
            transforms_infer = {}

            # update hyper_parameters config
            patch_size = [224, 224, 144]  # default roi
            levels = 5  # default number of hierarchical levels
            image_size = [int(i) for i in data_stats["stats_summary#image_stats#shape#percentile_99_5"]]

            # adjust to image size
            # min for each of spatial dims
            patch_size = [min(r, i) for r, i in zip(patch_size, image_size)]
            patch_size = roi_ensure_divisible(patch_size, levels=levels)
            # reduce number of levels to smaller then 5 (default) if image is too small
            levels, patch_size = roi_ensure_levels(levels, patch_size, image_size)

            input_channels = data_stats["stats_summary#image_stats#channels#max"]
            output_classes = len(data_stats["stats_summary#label_stats#labels"])

            if "class_names" in data_src_cfg and isinstance(data_src_cfg["class_names"], list):
                if isinstance(data_src_cfg["class_names"][0], str):
                    class_names = data_src_cfg["class_names"]
                    class_index = None
                else:
                    class_names = [x["name"] for x in data_src_cfg["class_names"]]
                    class_index = [x["index"] for x in data_src_cfg["class_names"]]
                    # check for overlap
                    all_ind = []
                    for a in class_index:
                        if bool(set(all_ind) & set(a)):  # overlap found
                            hyper_parameters.update({"softmax": "false"})
                            break
                        all_ind = all_ind + a
                hyper_parameters.update({"class_names": class_names})
                hyper_parameters.update({"class_index": class_index})
            else:
                class_names = class_index = None

            hyper_parameters.update({"patch_size": patch_size})
            hyper_parameters.update({"patch_size_valid": patch_size})
            hyper_parameters.update({"data_file_base_dir": os.path.abspath(data_src_cfg["dataroot"])})
            hyper_parameters.update({"data_list_file_path": os.path.abspath(data_src_cfg["datalist"])})
            hyper_parameters.update({"input_channels": input_channels})
            hyper_parameters.update({"output_classes": output_classes})

            resample = False

            modality = data_src_cfg.get("modality", "ct").lower()

            full_range = [
                data_stats["stats_summary#image_stats#intensity#percentile_00_5"],
                data_stats["stats_summary#image_stats#intensity#percentile_99_5"],
            ]
            intensity_lower_bound = float(data_stats["stats_summary#image_foreground_stats#intensity#percentile_00_5"])
            intensity_upper_bound = float(data_stats["stats_summary#image_foreground_stats#intensity#percentile_99_5"])
            spacing_lower_bound = np.array(data_stats["stats_summary#image_stats#spacing#percentile_00_5"])
            spacing_upper_bound = np.array(data_stats["stats_summary#image_stats#spacing#percentile_99_5"])


            # update network config
            if levels >= 5:  # default
                num_blocks = [1, 2, 2, 4, 4]
            elif levels == 4:
                num_blocks = [1, 2, 2, 4]
            elif levels == 3:
                num_blocks = [1, 3, 4]
            elif levels == 2:
                num_blocks = [2, 6]
            elif levels == 1:
                num_blocks = [8]
            else:
                raise ValueError("Strange number of levels" + str(levels))

            network.update({"network#blocks_down": num_blocks})
            network.update({"network#blocks_up": [1] * (len(num_blocks) - 1)}) # [1,1,1,1..]
            if "multigpu" in data_src_cfg and data_src_cfg["multigpu"]:
                multigpu = True
            elif torch.cuda.device_count() > 1:
                multigpu = True
            else:
                multigpu = False
            if multigpu:
                network.update({"network#norm": ["BATCH", {"affine": True}]}) # use batchnorm with multi gpu
                # set act to be not in-place with multi gpu
                network.update({"network#act": ["RELU", {"inplace": False}]})
            else:
                network.update({"network#norm": ["INSTANCE", {"affine": True}]}) # use instancenorm with single gpu

            if "ct" in modality:
                spacing = [1.0, 1.0, 1.0]

                # make sure intensity range is a valid CT range
                is_valid_for_ct = full_range[0] < -300 and full_range[1] > 300
                if is_valid_for_ct:
                    lrange = intensity_upper_bound - intensity_lower_bound
                    if lrange < 500:  # make sure range is at least 500 points
                        intensity_lower_bound -= (500 - lrange) // 2
                        intensity_upper_bound += (500 - lrange) // 2
                    intensity_lower_bound = max(intensity_lower_bound, -1250)  # limit to -1250..1500
                    intensity_upper_bound = min(intensity_upper_bound, 1500)

            elif "mr" in modality:
                spacing = data_stats["stats_summary#image_stats#spacing#median"]

            # resample on the fly to this spacing
            hyper_parameters.update({"intensity_bounds": [intensity_lower_bound, intensity_upper_bound]})

            if np.any(spacing_lower_bound / np.array(spacing) < 0.5) or np.any(
                spacing_upper_bound / np.array(spacing) > 1.5
            ):
                # Resampling recommended to median spacing
                resample = True
                hyper_parameters.update({"resample": resample})

            n_cases = len(data_stats["stats_by_cases"])
            max_epochs = int(np.clip(np.ceil(80000.0 / n_cases), a_min=300, a_max=1250))
            warmup_epochs = int(np.ceil(0.01 * max_epochs))

            hyper_parameters.update({"num_epochs": max_epochs})
            hyper_parameters.update({"lr_scheduler#warmup_epochs": warmup_epochs})

            # update transform config
            transform_train_path = os.path.join(output_path, 'configs', 'transforms_train.yaml')
            _template_transform_train = ConfigParser(globals=False)
            _template_transform_train.read_config(transform_train_path)
            template_transform_train = _template_transform_train.get("transforms_train#transforms")

            transforms_validate_path = os.path.join(output_path, 'configs', 'transforms_validate.yaml')
            _template_transforms_validate = ConfigParser(globals=False)
            _template_transforms_validate.read_config(transforms_validate_path)
            template_transform_validate = _template_transforms_validate.get("transforms_validate#transforms")

            transform_infer_path = os.path.join(output_path, 'configs', 'transforms_infer.yaml')
            _template_transform_infer = ConfigParser(globals=False)
            _template_transform_infer.read_config(transform_infer_path)
            template_transform_infer = _template_transform_infer.get("transforms_infer#transforms")

            if resample:
                transforms_train.update({'transforms_train#transforms#4#pixdim': spacing})
                transforms_validate.update({'transforms_validate#transforms#4#pixdim': spacing})
                transforms_infer.update({'transforms_infer#transforms#4#pixdim': spacing})
                i = 0
            else:
                template_transform_train.pop(4)
                template_transform_validate.pop(4)
                template_transform_infer.pop(4)
                transforms_train.update({'transforms_train#transforms': template_transform_train})
                transforms_validate.update({'transforms_validate#transforms': template_transform_validate})
                transforms_infer.update({'transforms_infer#transforms': template_transform_infer})
                i = - 1

            if isinstance(class_index, list):
                labelmap = {"_target_": "LabelMapping", "keys": "@label_key", "class_index": class_index}
                transforms_train.update({'transforms_train#transforms': template_transform_train.append(labelmap)})
                transforms_validate.update({'transforms_validate#transforms': template_transform_train.append(labelmap)})

            # get crop transform
            should_crop_based_on_foreground = any(
                [r < 0.5 * i for r, i in zip(patch_size, image_size)]
            )  # if any patch_size less tehn 0.5*image size
            if should_crop_based_on_foreground:
                # Image is much larger then patch_size, using foreground cropping
                ratios = None  # equal sampling
                crop_transform = {
                        "_target_": "RandCropByLabelClassesd",
                        "keys": ["@image_key", "@label_key"],
                        "label_key": "@label_key",
                        "spatial_size": deepcopy(patch_size),
                        "num_classes": output_classes,
                        "num_samples": 1,
                        "ratios": ratios,
                    }
            else:
                # Image size is only slightly larger then patch_size, using random cropping
                crop_transform = {
                        "_target_": "RandSpatialCropd",
                        "keys": ["@image_key", "@label_key"],
                        "roi_size": deepcopy(patch_size),
                        "random_size": False,
                    }

            crop_i = 7 + i
            transforms_train.update({f"transforms_train#transforms#{crop_i}": crop_transform})

            # for key in ["transforms_infer", "transforms_train", "transforms_validate"]:
            # get intensity transform
            ct_intensity_xform_train_valid = {
                "_target_": "Compose",
                "transforms": [
                    {
                        "_target_": "ScaleIntensityRanged",
                        "keys": "@image_key",
                        "a_min": intensity_lower_bound,
                        "a_max": intensity_upper_bound,
                        "b_min": 0.0,
                        "b_max": 1.0,
                        "clip": True,
                    },
                    {"_target_": "CropForegroundd", "keys": ["@image_key", "@label_key"], "source_key": "@image_key"},
                ],
            }

            ct_intensity_xform_infer = {
                "_target_": "Compose",
                "transforms": [
                    {
                        "_target_": "ScaleIntensityRanged",
                        "keys": "@image_key",
                        "a_min": intensity_lower_bound,
                        "a_max": intensity_upper_bound,
                        "b_min": 0.0,
                        "b_max": 1.0,
                        "clip": True,
                    },
                    {"_target_": "CropForegroundd", "keys": "@image_key", "source_key": "@image_key"},
                ],
            }


            # elif "mr" in modality:
            mr_intensity_transform = {
                "_target_": "NormalizeIntensityd",
                "keys": "@image_key",
                "nonzero": True,
                "channel_wise": True
                }

            intensity_i = 5 + i
            if modality.startswith("ct"):
                transforms_train.update({f"transforms_train#transforms#{intensity_i}": ct_intensity_xform_train_valid})
                transforms_validate.update({f"transforms_validate#transforms#{intensity_i}": ct_intensity_xform_train_valid})
                transforms_infer.update({f"transforms_infer#transforms#{intensity_i}": ct_intensity_xform_infer})
            else:
                transforms_train.update({f'transforms_train#transforms#{intensity_i}': mr_intensity_transform})
                transforms_validate.update({f'transforms_validate#transforms#{intensity_i}': mr_intensity_transform})
                transforms_infer.update({f'transforms_infer#transforms#{intensity_i}': mr_intensity_transform})


            fill_records = {
                'hyper_parameters.yaml': hyper_parameters,
                'network.yaml': network,
                'transforms_train.yaml': transforms_train,
                'transforms_validate.yaml': transforms_validate,
                'transforms_infer.yaml': transforms_infer
                }
        else:
            fill_records = self.fill_records

        for yaml_file, yaml_contents in fill_records.items():
            file_path = os.path.join(output_path, 'configs', yaml_file)

            parser = ConfigParser(globals=False)
            parser.read_config(file_path)
            for k, v in yaml_contents.items():
                if k in kwargs:
                    parser[k] = kwargs.pop(k)
                else:
                    parser[k] = deepcopy(v)  # some values are dicts
                yaml_contents[k] = deepcopy(parser[k])

            for k, v in kwargs.items():  # override new params not in fill_records
                if (parser.get(k, None) is not None):
                    parser[k] = deepcopy(v)
                    yaml_contents.update({k: parser[k]})

            ConfigParser.export_config_file(parser.get(), file_path, fmt="yaml", default_flow_style=None)

        return fill_records


if __name__ == "__main__":
    from monai.utils import optional_import

    fire, _ = optional_import("fire")
    fire.Fire({"SegresnetAlgo": SegresnetAlgo})
