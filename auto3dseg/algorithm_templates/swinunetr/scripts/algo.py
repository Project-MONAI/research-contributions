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

import numpy as np
import os
import subprocess
import sys
import yaml

from copy import deepcopy
from monai.apps.auto3dseg import BundleAlgo
from monai.bundle import ConfigParser


def get_gpu_available_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = (
        subprocess.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    )
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


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
            if self.data_list_file is not None and os.path.exists(
                str(self.data_list_file)
            ):
                data_src_cfg.read_config(self.data_list_file)

            hyper_parameters = {"bundle_root": output_path}
            network = {}
            transforms_train = {}
            transforms_validate = {}
            transforms_infer = {}

            patch_size = [96, 96, 96]
            max_shape = data_stats["stats_summary#image_stats#shape#max"]
            patch_size = [
                max(64, shape_k // 64 * 64) if shape_k < p_k else p_k
                for p_k, shape_k in zip(patch_size, max_shape)
            ]

            input_channels = data_stats["stats_summary#image_stats#channels#max"]
            output_classes = len(data_stats["stats_summary#label_stats#labels"])

            hyper_parameters.update({"patch_size": patch_size})
            hyper_parameters.update({"patch_size_valid": patch_size})
            hyper_parameters.update(
                {"data_file_base_dir": os.path.abspath(data_src_cfg["dataroot"])}
            )
            hyper_parameters.update(
                {"data_list_file_path": os.path.abspath(data_src_cfg["datalist"])}
            )
            hyper_parameters.update({"input_channels": input_channels})
            hyper_parameters.update({"output_classes": output_classes})

            modality = data_src_cfg.get("modality", "ct").lower()
            spacing = data_stats["stats_summary#image_stats#spacing#median"]

            epsilon = sys.float_info.epsilon
            if max(spacing) > (1.0 + epsilon) and min(spacing) < (1.0 - epsilon):
                spacing = [1.0, 1.0, 1.0]

            min_shape = data_stats["stats_summary#image_stats#shape#min"]
            # reflection-mode padding requires a minimum image for a given patch size
            spacing = [
                min(s / int(patch_size[i] / 3 + 1), spacing[i])
                for i, s in enumerate(min_shape)
            ]

            hyper_parameters.update({"resample_to_spacing": spacing})

            intensity_upper_bound = float(
                data_stats[
                    "stats_summary#image_foreground_stats#intensity#percentile_99_5"
                ]
            )
            intensity_lower_bound = float(
                data_stats[
                    "stats_summary#image_foreground_stats#intensity#percentile_00_5"
                ]
            )

            ct_intensity_xform = {
                "_target_": "ScaleIntensityRanged",
                "keys": "@image_key",
                "a_min": intensity_lower_bound,
                "a_max": intensity_upper_bound,
                "b_min": 0.0,
                "b_max": 1.0,
                "clip": True,
            }

            mr_intensity_transform = {
                "_target_": "NormalizeIntensityd",
                "keys": "@image_key",
                "nonzero": True,
                "channel_wise": True,
            }

            if modality.startswith("ct"):
                transforms_train.update(
                    {"transforms_train#transforms#5": ct_intensity_xform}
                )
                transforms_validate.update(
                    {"transforms_validate#transforms#5": ct_intensity_xform}
                )
                transforms_infer.update(
                    {"transforms_infer#transforms#5": ct_intensity_xform}
                )
            else:
                transforms_train.update(
                    {"transforms_train#transforms#5": mr_intensity_transform}
                )
                transforms_validate.update(
                    {"transforms_validate#transforms#5": mr_intensity_transform}
                )
                transforms_infer.update(
                    {"transforms_infer#transforms#5": mr_intensity_transform}
                )

            fill_records = {
                "hyper_parameters.yaml": hyper_parameters,
                "network.yaml": network,
                "transforms_train.yaml": transforms_train,
                "transforms_validate.yaml": transforms_validate,
                "transforms_infer.yaml": transforms_infer,
            }

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
                    parser[k] = deepcopy(v)  # some values are dicts
                yaml_contents[k] = deepcopy(parser[k])

            for k, v in kwargs.items():  # override new params not in fill_records
                if parser.get(k, None) is not None:
                    parser[k] = deepcopy(v)
                    yaml_contents.update({k: parser[k]})

            ConfigParser.export_config_file(
                parser.get(), file_path, fmt="yaml", default_flow_style=None
            )

        # customize parameters for gpu
        if kwargs.pop("gpu_customization", False):
            gpu_customization_specs = kwargs.pop("gpu_customization_specs", {})
            fill_records = self.customize_param_for_gpu(
                output_path,
                data_stats_file,
                fill_records,
                gpu_customization_specs,
            )

        return fill_records

    def customize_param_for_gpu(
        self, output_path, data_stats_file, fill_records, gpu_customization_specs
    ):
        # optimize batch size for model training
        import optuna

        # default range
        num_trials = 60
        range_num_images_per_batch = [1, 20]
        range_num_sw_batch_size = [1, 40]

        # load customized range
        if (
            "swunetr" in gpu_customization_specs
            or "universal" in gpu_customization_specs
        ):
            specs_section = (
                "swunetr" if "swunetr" in gpu_customization_specs else "universal"
            )
            specs = gpu_customization_specs[specs_section]

            if "num_trials" in specs:
                num_trials = specs["num_trials"]

            if "range_num_images_per_batch" in specs:
                range_num_images_per_batch = specs["range_num_images_per_batch"]

            if "range_num_sw_batch_size" in specs:
                range_num_sw_batch_size = specs["range_num_sw_batch_size"]

        mem = get_gpu_available_memory()
        device_id = np.argmin(mem) if type(mem) is list else 0
        print(f"[info] gpu device {device_id} with minimum memory")

        mem = min(mem) if type(mem) is list else mem
        mem = round(float(mem) / 1024.0)

        def objective(trial):
            num_images_per_batch = trial.suggest_int(
                "num_images_per_batch",
                range_num_images_per_batch[0],
                range_num_images_per_batch[1],
            )
            num_sw_batch_size = trial.suggest_int(
                "num_sw_batch_size",
                range_num_sw_batch_size[0],
                range_num_sw_batch_size[1],
            )
            validation_data_device = trial.suggest_categorical(
                "validation_data_device", ["cpu", "gpu"]
            )
            device_factor = 2.0 if validation_data_device == "gpu" else 1.0

            try:
                cmd = "python {0:s}dummy_runner.py ".format(
                    os.path.join(output_path, "scripts") + os.sep
                )
                cmd += "--output_path {0:s} ".format(output_path)
                cmd += "--data_stats_file {0:s} ".format(data_stats_file)
                cmd += "--device_id {0:d} ".format(device_id)
                cmd += "run "
                cmd += f"--num_images_per_batch {num_images_per_batch} "
                cmd += f"--num_sw_batch_size {num_sw_batch_size} "
                cmd += f"--validation_data_device {validation_data_device}"
                _ = subprocess.run(cmd.split(), check=True)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    return (
                        float(num_images_per_batch)
                        * float(num_sw_batch_size)
                        * device_factor
                    )
                else:
                    raise(e)

            value = (
                -1.0
                * float(num_images_per_batch)
                * float(num_sw_batch_size)
                * device_factor
            )

            return value

        opt_result_file = os.path.join(output_path, "..", f"gpu_opt_{mem}gb.yaml")
        if os.path.exists(opt_result_file):
            with open(opt_result_file) as in_file:
                best_trial = yaml.full_load(in_file)

        if not os.path.exists(opt_result_file) or "swunetr" not in best_trial:
            study = optuna.create_study()
            study.optimize(objective, n_trials=num_trials)
            trial = study.best_trial
            best_trial = {}
            best_trial["num_images_per_batch"] = max(
                int(trial.params["num_images_per_batch"]) - 1, 1
            )
            best_trial["num_sw_batch_size"] = max(
                int(trial.params["num_sw_batch_size"]) - 1, 1
            )
            best_trial["validation_data_device"] = trial.params[
                "validation_data_device"
            ]
            best_trial["value"] = int(trial.value)
            with open(opt_result_file, "a") as out_file:
                yaml.dump({"swunetr": best_trial}, stream=out_file)

            print("\n-----  Finished Optimization  -----")
            print("Optimal value: {}".format(best_trial["value"]))
            print("Best hyperparameters: {}".format(best_trial))
        else:
            with open(opt_result_file) as in_file:
                best_trial = yaml.full_load(in_file)
            best_trial = best_trial["swunetr"]

        if best_trial["value"] < 0:
            fill_records["hyper_parameters.yaml"].update(
                {"num_images_per_batch": best_trial["num_images_per_batch"]}
            )
            fill_records["hyper_parameters.yaml"].update(
                {"num_sw_batch_size": best_trial["num_sw_batch_size"]}
            )
            if best_trial["validation_data_device"] == "cpu":
                fill_records["hyper_parameters.yaml"].update({"sw_input_on_cpu": True})
            else:
                fill_records["hyper_parameters.yaml"].update({"sw_input_on_cpu": False})

            for yaml_file, yaml_contents in fill_records.items():
                if "hyper_parameters" in yaml_file:
                    file_path = os.path.join(output_path, "configs", yaml_file)

                    parser = ConfigParser(globals=False)
                    parser.read_config(file_path)
                    for k, v in yaml_contents.items():
                        parser[k] = deepcopy(v)
                        yaml_contents[k] = deepcopy(parser[k])

                    ConfigParser.export_config_file(
                        parser.get(), file_path, fmt="yaml", default_flow_style=None
                    )

        return fill_records


if __name__ == "__main__":
    from monai.utils import optional_import

    fire, _ = optional_import("fire")
    fire.Fire({"SwinunetrAlgo": SwinunetrAlgo})
