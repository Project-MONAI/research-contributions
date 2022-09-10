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

from monai.apps.auto3dseg import BundleAlgo
from monai.bundle import ConfigParser

class SwinunetrAlgo(BundleAlgo):
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
            network = {}  # no change on network.yaml in segresnet2d
            transforms_train = {}
            transforms_validate = {}
            transforms_infer = {}

            patch_size = [96, 96, 96]
            max_shape = data_stats["stats_summary#image_stats#shape#max"]
            patch_size = [
                max(64, shape_k // 64 * 64) if shape_k < p_k else p_k for p_k, shape_k in zip(patch_size, max_shape)
            ]

            input_channels = data_stats["stats_summary#image_stats#channels#max"]
            output_classes = len(data_stats["stats_summary#label_stats#labels"])

            hyper_parameters.update({"patch_size": patch_size})
            hyper_parameters.update({"patch_size_valid": patch_size})
            hyper_parameters.update({"data_file_base_dir": os.path.abspath(data_src_cfg["dataroot"])})
            hyper_parameters.update({"data_list_file_path": os.path.abspath(data_src_cfg["datalist"])})
            hyper_parameters.update({"input_channels": input_channels})
            hyper_parameters.update({"output_classes": output_classes})

            modality = data_src_cfg.get("modality", "ct").lower()
            spacing = data_stats["stats_summary#image_stats#spacing#median"]

            intensity_upper_bound = float(data_stats["stats_summary#image_foreground_stats#intensity#percentile_99_5"])
            intensity_lower_bound = float(data_stats["stats_summary#image_foreground_stats#intensity#percentile_00_5"])

            ct_intensity_xform = {
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

            mr_intensity_transform = {
                "_target_": "NormalizeIntensityd",
                "keys": "@image_key",
                "nonzero": True,
                "channel_wise": True,
            }

            transforms_train.update({'transforms_train#transforms#3#pixdim': spacing})
            transforms_validate.update({'transforms_validate#transforms#3#pixdim': spacing})
            transforms_infer.update({'transforms_infer#transforms#3#pixdim': spacing})

            if modality.startswith("ct"):
                transforms_train.update({"transforms_train#transforms#5": ct_intensity_xform})
                transforms_validate.update({"transforms_validate#transforms#5": ct_intensity_xform})
                transforms_infer.update({"transforms_infer#transforms#5": ct_intensity_xform})
            else:
                transforms_train.update({'transforms_train#transforms#5': mr_intensity_transform})
                transforms_validate.update({'transforms_validate#transforms#5': mr_intensity_transform})
                transforms_infer.update({'transforms_infer#transforms#5': mr_intensity_transform})


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
    fire.Fire({"SwinunetrAlgo": SwinunetrAlgo})
