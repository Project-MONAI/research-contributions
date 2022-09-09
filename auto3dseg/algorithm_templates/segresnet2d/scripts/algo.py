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
from monai.bundle.scripts import _update_args


class Segresnet2dAlgo(BundleAlgo):
    def fill_template_config(self, data_stats_file, output_path, **kwargs):
        """
        Fill the freshly copied config templates

        Args:
            data_stats_file: the stats report from DataAnalyzer in yaml format
            output_path: the root folder to scripts/configs directories.
        """
        if kwargs.pop('fill_without_datastats', True):
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

            hyper_parameters = {'_config_file_': 'hyper_parameters.yaml'}
            hyper_parameters.update({"bundle_root": output_path})

            patch_size = [320, 320]
            max_shape = data_stats["stats_summary#image_stats#shape#max"]
            patch_size = [
                max(32, shape_k // 32 * 32) if shape_k < p_k else p_k for p_k, shape_k in zip(patch_size, max_shape)
            ]

            input_channels = data_stats["stats_summary#image_stats#channels#max"]
            output_classes = len(data_stats["stats_summary#label_stats#labels"])

            hyper_parameters.update({"patch_size#0": patch_size[0]})
            hyper_parameters.update({"patch_size#1": patch_size[1]})
            hyper_parameters.update({"patch_size_valid#0": patch_size[0]})
            hyper_parameters.update({"patch_size_valid#1": patch_size[1]})
            hyper_parameters.update({"data_file_base_dir": os.path.abspath(data_src_cfg["dataroot"])})
            hyper_parameters.update({"data_list_file_path": os.path.abspath(data_src_cfg["datalist"])})
            hyper_parameters.update({"input_channels": input_channels})
            hyper_parameters.update({"output_classes": output_classes})

            modality = data_src_cfg.get("modality", "ct").lower()
            spacing = data_stats["stats_summary#image_stats#spacing#median"]
            spacing[-1] = -1.0

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

            transforms_train = {'_config_file_': 'transforms_train.yaml'}
            transforms_validate = {'_config_file_': 'transforms_validate.yaml'}
            transforms_infer = {'_config_file_': 'transforms_infer.yaml'}

            transforms_train.update({'transforms_train#transforms#3#pixdim': spacing})
            transforms_validate.update({'transforms_validate#transforms#3#pixdim': spacing})
            transforms_infer.update({'transforms_infer#transforms#3#pixdim': spacing})

            if modality.startswith("ct"):
                transforms_train.update({'transforms_train#transforms#5': ct_intensity_xform})
                transforms_validate.update({'transforms_validate#transforms#5': ct_intensity_xform})
                transforms_infer.update({'transforms_infer#transforms#5': ct_intensity_xform})
            else:
                transforms_train.update({'transforms_train#transforms#5': mr_intensity_transform})
                transforms_validate.update({'transforms_validate#transforms#5': mr_intensity_transform})
                transforms_infer.update({'transforms_infer#transforms#5': mr_intensity_transform})

            network = {'_config_file_': 'network.yaml'}

            self.fill_records = [hyper_parameters, network, transforms_train, transforms_validate, transforms_infer]

        for config in self.fill_records:
            config_cp = deepcopy(config)
            file = os.path.join(output_path, 'configs', config_cp.pop('_config_file_'))

            parser = ConfigParser(globals=False)
            parser.read_config(file)
            for k, v in config_cp.items():
                if k in kwargs:
                    parser[k] = kwargs[k]
                else:
                    parser[k] = deepcopy(v)  # some values are dicts
            ConfigParser.export_config_file(parser.get(), file, fmt="yaml", default_flow_style=None)


if __name__ == "__main__":
    from monai.utils import optional_import

    fire, _ = optional_import("fire")
    fire.Fire({"Segresnet2dAlgo": Segresnet2dAlgo})
