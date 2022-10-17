#!/usr/bin/env python

import optuna
import os
import subprocess


def objective(trial):
    num_images_per_batch = trial.suggest_int("num_images_per_batch", 1, 20)
    num_patches_per_image = trial.suggest_int("num_patches_per_image", 1, 20)
    num_sw_batch_size = trial.suggest_int("num_sw_batch_size", 1, 20)

    try:
        cmd = f"bash fake_training_script.sh {num_images_per_batch} {num_patches_per_image} {num_sw_batch_size}"
        _ = subprocess.run(cmd.split(), check=True)
    except:
        print("[error] OOM")
        return float(num_images_per_batch) * float(num_patches_per_image) * float(num_sw_batch_size)

    return -1.0 * float(num_images_per_batch) * float(num_patches_per_image) * float(num_sw_batch_size)


os.system("clear")

study = optuna.create_study()
study.optimize(objective, n_trials=100)
print(study.best_value)
