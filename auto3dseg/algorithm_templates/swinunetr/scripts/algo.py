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
import subprocess
import sys
from copy import deepcopy

import numpy as np
import torch
import yaml

from monai.apps.auto3dseg import BundleAlgo
from monai.apps.utils import get_logger
from monai.bundle import ConfigParser

logger = get_logger(module_name=__name__)


def modify_hierarchical_dict(hierarchical_dict, keys, value):
    if len(keys) == 1:
        hierarchical_dict[keys[0]] = value
    else:
        if keys[0] not in hierarchical_dict:
            hierarchical_dict[keys[0]] = {}
        modify_hierarchical_dict(hierarchical_dict[keys[0]], keys[1:], value)


def get_mem_from_visible_gpus():
    available_mem_visible_gpus = []
    for d in range(torch.cuda.device_count()):
        available_mem_visible_gpus.append(torch.cuda.mem_get_info(device=d)[0])
    return available_mem_visible_gpus


def auto_scale(output_classes, n_cases, max_epoch=1000):
    """Scale batch size based on gpu memory and output class. Includes heuristics."""
    mem = get_mem_from_visible_gpus()
    mem = min(mem) if isinstance(mem, list) else mem
    mem = float(mem) / (1024.0**3)
    mem = max(1.0, mem - 1.0)
    # heuristics copied from dints template
    mem_bs2 = 6.0 + (20.0 - 6.0) * (output_classes - 2) / (105 - 2)
    mem_bs9 = 24.0 + (74.0 - 24.0) * (output_classes - 2) / (105 - 2)
    # heuristic scaling for swinunetr
    mem_bs2 = 12 / 6 * mem_bs2
    mem_bs9 = 12 / 6 * mem_bs9
    batch_size = 2 + (9 - 2) * (mem - mem_bs2) / (mem_bs9 - mem_bs2)
    batch_size = max(int(batch_size), 1)

    # fixed two iters per whole image, each iter with num_patches_per_iter
    num_patches_per_iter = batch_size
    num_crops_per_image = batch_size * 2
    # heuristics for 800k patch iteration. epoch * n_cases * num_crops_per_image = total 400k patch
    num_epochs = min(max_epoch, int(800000 / n_cases / num_crops_per_image))
    return {
        "num_patches_per_iter": num_patches_per_iter,
        "num_crops_per_image": num_crops_per_image,
        "num_epochs": num_epochs,
    }


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
            network = {}
            transforms_train = {}
            transforms_validate = {}
            transforms_infer = {}

            roi_size = [96, 96, 96]
            max_shape = data_stats["stats_summary#image_stats#shape#max"]
            roi_size = [
                max(64, shape_k // 64 * 64) if shape_k < p_k else p_k for p_k, shape_k in zip(roi_size, max_shape)
            ]

            try:
                if isinstance(data_src_cfg["class_names"], list):
                    hyper_parameters.update({"class_names": data_src_cfg["class_names"]})
            except BaseException:
                pass

            try:
                if isinstance(data_src_cfg["sigmoid"], bool) and data_src_cfg["sigmoid"]:
                    hyper_parameters.update({"softmax": False})
            except BaseException:
                pass

            input_channels = data_stats["stats_summary#image_stats#channels#max"]
            output_classes = len(data_stats["stats_summary#label_stats#labels"])
            n_cases = data_stats["stats_summary#n_cases"]

            hyper_parameters.update({"data_file_base_dir": os.path.abspath(data_src_cfg["dataroot"])})
            hyper_parameters.update({"data_list_file_path": os.path.abspath(data_src_cfg["datalist"])})

            hyper_parameters.update({"roi_size": roi_size})
            hyper_parameters.update({"roi_size_valid": roi_size})
            hyper_parameters.update({"input_channels": input_channels})
            hyper_parameters.update({"output_classes": output_classes})
            hyper_parameters.update({"n_cases": n_cases})

            if hasattr(self, "mlflow_tracking_uri") and self.mlflow_tracking_uri is not None:
                hyper_parameters.update({"mlflow_tracking_uri": self.mlflow_tracking_uri})
            if hasattr(self, "mlflow_experiment_name") and self.mlflow_experiment_name is not None:
                hyper_parameters.update({"mlflow_experiment_name": self.mlflow_experiment_name})

            modality = data_src_cfg.get("modality", "ct").lower()
            spacing = data_stats["stats_summary#image_stats#spacing#median"]

            epsilon = sys.float_info.epsilon
            if max(spacing) > (1.0 + epsilon) and min(spacing) < (1.0 - epsilon):
                spacing = [1.0, 1.0, 1.0]

            hyper_parameters.update({"resample_resolution": spacing})

            scaled = auto_scale(output_classes, n_cases, max_epoch=1000)
            hyper_parameters.update({"num_patches_per_iter": scaled["num_patches_per_iter"]})
            hyper_parameters.update({"num_crops_per_image": scaled["num_crops_per_image"]})
            hyper_parameters.update({"num_epochs": scaled["num_epochs"]})

            intensity_upper_bound = float(data_stats["stats_summary#image_foreground_stats#intensity#percentile_99_5"])
            intensity_lower_bound = float(data_stats["stats_summary#image_foreground_stats#intensity#percentile_00_5"])
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
                    {
                        "_target_": "CropForegroundd",
                        "keys": ["@image_key", "@label_key"],
                        "source_key": "@image_key",
                        "start_coord_key": None,
                        "end_coord_key": None,
                        "allow_smaller": True,
                    },
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
                    {"_target_": "CropForegroundd", "keys": "@image_key", "source_key": "@image_key", "allow_smaller": True},
                ],
            }
            mr_intensity_transform = {
                "_target_": "NormalizeIntensityd",
                "keys": "@image_key",
                "nonzero": True,
                "channel_wise": True,
            }

            if modality.startswith("ct"):
                transforms_train.update({"transforms_train#transforms#2": ct_intensity_xform_train_valid})
                transforms_validate.update({"transforms_validate#transforms#2": ct_intensity_xform_train_valid})
                transforms_infer.update({"transforms_infer#transforms#2": ct_intensity_xform_infer})
            else:
                transforms_train.update({"transforms_train#transforms#2": mr_intensity_transform})
                transforms_validate.update({"transforms_validate#transforms#2": mr_intensity_transform})
                transforms_infer.update({"transforms_infer#transforms#2": mr_intensity_transform})

            if (
                "class_names" in data_src_cfg
                and isinstance(data_src_cfg["class_names"], list)
                and "index" in data_src_cfg["class_names"][0]
            ):
                class_index = [x["index"] for x in data_src_cfg["class_names"]]

                pt_type_transform_train = {
                    "_target_": "CastToTyped",
                    "keys": ["@image_key", "@label_key"],
                    "dtype": ["$torch.float32", "$torch.uint8"],
                }

                pt_type_transform_valid = {
                    "_target_": "CastToTyped",
                    "keys": ["@image_key", "@label_key"],
                    "dtype": ["$torch.float32", "$torch.uint8"],
                }

                label_conversion_transforms_train = {
                    "_target_": "Compose",
                    "transforms": [
                        pt_type_transform_train,
                        {
                            "_target_": "Lambdad",
                            "keys": "@label_key",
                            "func": f"$lambda x: torch.cat([sum([x == i for i in c]) for c in {class_index}], dim=0).to(dtype=x.dtype)",
                        },
                    ],
                }

                label_conversion_transforms_valid = {
                    "_target_": "Compose",
                    "transforms": [
                        pt_type_transform_valid,
                        {
                            "_target_": "Lambdad",
                            "keys": "@label_key",
                            "func": f"$lambda x: torch.cat([sum([x == i for i in c]) for c in {class_index}], dim=0).to(dtype=x.dtype)",
                        },
                    ],
                }

                transforms_train.update({"transforms_train#transforms#5": label_conversion_transforms_train})
                transforms_validate.update({"transforms_validate#transforms#5": label_conversion_transforms_valid})

                if "sigmoid" in data_src_cfg and isinstance(data_src_cfg["sigmoid"], bool) and data_src_cfg["sigmoid"]:
                    hyper_parameters.update({"output_classes": len(data_src_cfg["class_names"])})
                    new_crop_transforms = {
                        "_target_": "Compose",
                        "transforms": [
                            {"_target_": "CopyItemsd", "keys": "@label_key", "times": 1, "names": "crop_label"},
                            {
                                "_target_": "Lambdad",
                                "keys": "crop_label",
                                "func": f"$lambda x: torch.cat([(torch.sum(x, dim=0, keepdim=True) < 1).to(dtype=x.dtype), x], dim=0)",
                            },
                            {
                                "_target_": "RandCropByLabelClassesd",
                                "keys": ["@image_key", "@label_key"],
                                "label_key": "crop_label",
                                "num_classes": None,
                                "spatial_size": "@patch_size",
                                "num_samples": "@num_patches_per_image",
                                "warn": False,
                            },
                            {"_target_": "Lambdad", "keys": "crop_label", "func": f"$lambda x: 0"},
                        ],
                    }
                    transforms_train.update({"transforms_train#transforms#9": new_crop_transforms})
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

            ConfigParser.export_config_file(parser.get(), file_path, fmt="yaml", default_flow_style=None)

        # customize parameters for gpu
        if kwargs.pop("gpu_customization", False):
            gpu_customization_specs = kwargs.pop("gpu_customization_specs", {})
            fill_records = self.customize_param_for_gpu(
                output_path, data_stats_file, fill_records, gpu_customization_specs
            )

        return fill_records

    def customize_param_for_gpu(self, output_path, data_stats_file, fill_records, gpu_customization_specs):
        # optimize batch size for model training
        import optuna

        # default range
        num_trials = 60
        range_num_images_per_batch = [1, 20]
        range_num_sw_batch_size = [1, 40]

        # load customized range
        if "swunetr" in gpu_customization_specs or "universal" in gpu_customization_specs:
            specs_section = "swunetr" if "swunetr" in gpu_customization_specs else "universal"
            specs = gpu_customization_specs[specs_section]

            if "num_trials" in specs:
                num_trials = specs["num_trials"]

            if "range_num_images_per_batch" in specs:
                range_num_images_per_batch = specs["range_num_images_per_batch"]

            if "range_num_sw_batch_size" in specs:
                range_num_sw_batch_size = specs["range_num_sw_batch_size"]

        mem = get_mem_from_visible_gpus()
        device_id = np.argmin(mem)
        print(f"[debug] device {device_id} in visible GPU list has the minimum memory.")

        mem = min(mem) if isinstance(mem, list) else mem
        mem = round(float(mem) / 1024.0)

        def objective(trial):
            num_images_per_batch = trial.suggest_int(
                "num_images_per_batch", range_num_images_per_batch[0], range_num_images_per_batch[1]
            )
            num_sw_batch_size = trial.suggest_int(
                "num_sw_batch_size", range_num_sw_batch_size[0], range_num_sw_batch_size[1]
            )
            validation_data_device = trial.suggest_categorical("validation_data_device", ["cpu", "gpu"])
            device_factor = 2.0 if validation_data_device == "gpu" else 1.0
            ps_environ = os.environ.copy()

            try:
                cmd = "python {0:s}dummy_runner.py ".format(os.path.join(output_path, "scripts") + os.sep)
                cmd += "--output_path {0:s} ".format(output_path)
                cmd += "--data_stats_file {0:s} ".format(data_stats_file)
                cmd += "--device_id {0:d} ".format(device_id)
                cmd += "run "
                cmd += f"--num_images_per_batch {num_images_per_batch} "
                cmd += f"--num_sw_batch_size {num_sw_batch_size} "
                cmd += f"--validation_data_device {validation_data_device}"
                _ = subprocess.run(cmd.split(), check=True, env=ps_environ)
            except RuntimeError as e:
                if not any(x in str(e).lower() for x in ("memory", "cuda", "cudnn")):
                    raise e
                print("[error] OOM")
                return float(num_images_per_batch) * float(num_sw_batch_size) * device_factor

            value = -1.0 * float(num_images_per_batch) * float(num_sw_batch_size) * device_factor

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
            best_trial["num_images_per_batch"] = max(int(trial.params["num_images_per_batch"]) - 1, 1)
            best_trial["num_sw_batch_size"] = max(int(trial.params["num_sw_batch_size"]) - 1, 1)
            best_trial["validation_data_device"] = trial.params["validation_data_device"]
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
            fill_records["hyper_parameters.yaml"].update({"num_images_per_batch": best_trial["num_images_per_batch"]})
            fill_records["hyper_parameters.yaml"].update({"num_sw_batch_size": best_trial["num_sw_batch_size"]})
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

                    ConfigParser.export_config_file(parser.get(), file_path, fmt="yaml", default_flow_style=None)

        return fill_records


if __name__ == "__main__":
    from monai.utils import optional_import

    fire, _ = optional_import("fire")
    fire.Fire({"SwinunetrAlgo": SwinunetrAlgo})
