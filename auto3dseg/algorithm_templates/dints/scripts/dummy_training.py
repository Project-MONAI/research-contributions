#!/usr/bin/env python

import optuna
import os
import subprocess
import yaml


def objective(trial):
    num_images_per_batch = trial.suggest_int("num_images_per_batch", 1, 20)
    num_sw_batch_size = trial.suggest_int("num_sw_batch_size", 1, 40)
    validation_data_device = trial.suggest_categorical(
        "validation_data_device", ["cpu", "gpu"]
    )

    device_factor = 2.0 if validation_data_device == "gpu" else 1.0

    try:
        cmd = f"python dummy_training_script.py run --num_images_per_batch {num_images_per_batch} --num_sw_batch_size {num_sw_batch_size} --validation_data_device {validation_data_device}"
        _ = subprocess.run(cmd.split(), check=True)
    except:
        print("[error] OOM")
        return (
            float(num_images_per_batch)
            * float(num_sw_batch_size)
            * device_factor
        )

    value = (
        -1.0
        * float(num_images_per_batch)
        * float(num_sw_batch_size)
        * device_factor
    )

    return value


os.system("clear")

study = optuna.create_study()
study.optimize(objective, n_trials=100)
print(study.best_value)
