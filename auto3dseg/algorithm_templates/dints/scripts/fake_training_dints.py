#!/usr/bin/env python

import optuna
import os
import subprocess


def objective(trial):
    num_images_per_batch = trial.suggest_int("num_images_per_batch", 1, 10)
    num_patches_per_image = trial.suggest_int("num_patches_per_image", 1, 10)
    num_sw_batch_size = trial.suggest_int("num_sw_batch_size", 1, 20)
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

    return (
        -1.0
        * float(num_images_per_batch)
        * float(num_patches_per_image)
        * float(num_sw_batch_size)
        * device_factor
    )


os.system("clear")

study = optuna.create_study()
study.optimize(objective, n_trials=100)
print(study.best_value)
