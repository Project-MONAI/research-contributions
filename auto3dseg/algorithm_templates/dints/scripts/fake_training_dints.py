#!/usr/bin/env python
# pip install nvidia-ml-py3

import optuna
import os
import subprocess
import yaml


def objective(trial):
    num_images_per_batch = trial.suggest_int("num_images_per_batch", 1, 20)
    num_patches_per_image = trial.suggest_int("num_patches_per_image", 1, 20)
    num_sw_batch_size = trial.suggest_int("num_sw_batch_size", 1, 40)
    validation_data_device = trial.suggest_categorical(
        "validation_data_device", ["cpu", "gpu"]
    )

    device_factor = 2.0 if validation_data_device == "gpu" else 1.0

    try:
        cmd = f"bash fake_training_script_dints.sh {num_images_per_batch} {num_patches_per_image} {num_sw_batch_size} {validation_data_device}"
        _ = subprocess.run(cmd.split(), check=True)
    except:
        print("[error] OOM")
        return (
            float(num_images_per_batch)
            * float(num_patches_per_image)
            * float(num_sw_batch_size)
            * device_factor
        )

    value = (
        -1.0
        * float(num_images_per_batch)
        * float(num_patches_per_image)
        * float(num_sw_batch_size)
        * device_factor
    )

    dict_file = {}
    dict_file["num_images_per_batch"] = int(num_images_per_batch)
    dict_file["num_patches_per_image"] = int(num_patches_per_image)
    dict_file["num_sw_batch_size"] = int(num_sw_batch_size)
    dict_file["validation_data_device"] = str(validation_data_device)
    dict_file["value"] = int(value)

    with open("hyper_param_dints.yaml", "a") as out_file:
        yaml.dump([dict_file], stream=out_file)

    return value


os.system("clear")

study = optuna.create_study()
study.optimize(objective, n_trials=100)
print(study.best_value)
