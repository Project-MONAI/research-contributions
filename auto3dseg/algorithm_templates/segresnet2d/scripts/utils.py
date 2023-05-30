import logging
import os

import numpy as np
import torch
import torch.distributed as dist

from monai.apps.auto3dseg.auto_runner import logger

print = logger.debug
roi_size_default = [384, 384, 32]


def logger_configure(log_output_file: str = None, debug=False, global_rank=0) -> None:
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"monai_default": {"format": "%(message)s"}},
        "loggers": {
            "monai.apps.auto3dseg.auto_runner": {"handlers": ["console", "file"], "level": "DEBUG", "propagate": False}
        },
        # "filters": {"rank_filter": {"{}": "__main__.RankFilter"}},
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "monai_default",
                # "filters": ["rank_filter"],
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": "runner.log",
                "mode": "a",
                "level": "DEBUG",
                "formatter": "monai_default",
                # "filters": ["rank_filter"],
            },
        },
    }

    if log_output_file is not None:
        log_config["handlers"]["file"]["filename"] = log_output_file
        log_config["handlers"]["file"]["level"] = "DEBUG"
    else:
        log_config["handlers"]["file"]["level"] = "CRITICAL"

    if debug or bool(os.environ.get("SEGRESNET_DEBUG", False)):
        log_config["handlers"]["console"]["level"] = "DEBUG"

    logging.config.dictConfig(log_config)
    # if global_rank!=0:
    #      logger.addFilter(lambda x: False)


def get_gpu_mem_size():
    gpu_mem = 0
    n_gpus = torch.cuda.device_count()
    if n_gpus > 0:
        gpu_mem = min([torch.cuda.get_device_properties(i).total_memory for i in range(n_gpus)])
        gpu_mem = gpu_mem / 1024**3
    else:
        gpu_mem = 16

    return gpu_mem


def auto_adjust_network_settings(
    auto_scale_roi=False,
    auto_scale_batch=False,
    auto_scale_filters=False,
    image_size_mm=None,
    spacing=None,
    output_classes=None,
    levels=None,
    anisotropic_scales=False,
    levels_limit=5,
    gpu_mem=None,
):
    global_rank = 0
    if dist.is_available() and dist.is_initialized():
        global_rank = dist.get_rank()

    batch_size_default = 1
    init_filters_default = 32

    roi_size = np.array(roi_size_default)
    base_numel = roi_size.prod()
    gpu_factor = 1

    if gpu_mem is None:
        gpu_mem = get_gpu_mem_size()
    if global_rank == 0:
        print(f"GPU device memory min: {gpu_mem}")

    # adapting
    if auto_scale_batch or auto_scale_roi or auto_scale_filters:
        gpu_factor_init = gpu_factor = max(1, gpu_mem / 16)
        if anisotropic_scales:
            gpu_factor = max(1, 0.8 * gpu_factor)
        if global_rank == 0:
            print(f"base_numel {base_numel} gpu_factor {gpu_factor} gpu_factor_init {gpu_factor_init}")
    else:
        gpu_mem = 16
        gpu_factor = gpu_factor_init = 1

    # account for output_classes
    output_classes_thresh = 20
    if output_classes is not None and output_classes > output_classes_thresh:
        base_adjust = gpu_mem / (output_classes * 0.2 + 12)
        if gpu_mem < 17:
            base_adjust /= 2

        if global_rank == 0:
            print(f"base_adjust {base_adjust} since output_classes {output_classes} > {output_classes_thresh}")
        if base_adjust < 0.95:  # reduce roi
            base_numel *= base_adjust
            r = int(base_numel ** (1 / 3) / 2**4)
            if r == 0 and global_rank == 0:
                print(f"Warning: given output_classes {output_classes}, unable to fit any ROI on the gpu {gpu_mem} Gb!")
            roi_size = np.array([max(1, r) * 2**4] * 3)
            gpu_factor = gpu_factor_init = 1
            auto_scale_roi = False
        else:
            gpu_factor_init = gpu_factor = base_adjust

        if global_rank == 0:
            print(f"base_numel {base_numel} roi_size {roi_size} gpu_factor {gpu_factor}")

    if image_size_mm is not None and spacing is not None:
        image_size = np.floor(np.array(image_size_mm) / np.array(spacing))
        if global_rank == 0:
            print(f"input roi {roi_size} image_size {image_size} numel  {roi_size.prod()}")
        roi_size = np.minimum(roi_size, image_size)
    else:
        raise ValueError("image_size_mm or spacing is not provided, network params may be inaccuracy")

    # adjust ROI
    max_numel = base_numel * gpu_factor if auto_scale_roi else base_numel
    while roi_size.prod() < max_numel:
        old_numel = roi_size.prod()
        roi_size = np.minimum(roi_size * 1.15, image_size)
        if global_rank == 0:
            print(f"increasing roi step {roi_size}")
        if roi_size.prod() == old_numel:
            break
        if global_rank == 0:
            print(f"increasing roi result 1 {roi_size}")

    # adjust number of network downsize levels
    if not anisotropic_scales:
        if levels is None:
            levels = np.floor(np.log2(roi_size))
            if global_rank == 0:
                print(f"levels 1 {levels}")
            levels = min(min(levels), levels_limit)  # limit to 5
            if global_rank == 0:
                print(f"levels 2' {levels}")

        factor = 2 ** (levels - 1)
        roi_size = factor * np.maximum(2, np.floor(roi_size / factor))
        if global_rank == 0:
            print(f"roi_size factored {roi_size}")

    else:
        extra_levels = np.floor(np.log2(np.max(spacing) / spacing))
        extra_levels = max(extra_levels) - extra_levels

        if levels is None:
            # calc levels
            levels = np.floor(np.log2(roi_size))
            if global_rank == 0:
                print(f"levels 1 aniso {levels} extra_levels {extra_levels}")
            levels = min(min(levels + extra_levels), levels_limit)  # limit to 5
            if global_rank == 0:
                print(f"levels 2 {levels}")

        factor = 2 ** (np.maximum(1, levels - extra_levels) - 1)
        roi_size = factor * np.maximum(2, np.floor(roi_size / factor))
        if global_rank == 0:
            print(f"roi_size factored {roi_size} factor {factor} extra_levels {extra_levels}")

    # optionally adjust initial filters (above 32)
    if auto_scale_filters and roi_size.prod() < base_numel * gpu_factor:
        init_filters = int(max(32, np.floor(4 * (base_numel / roi_size.prod())) * 8))
        if global_rank == 0:
            print(f"checking to increase init_filters {init_filters}")
        gpu_factor_init *= init_filters / 32
        gpu_factor *= init_filters / 32
    else:
        if global_rank == 0:
            print(f"kept filters the same base_numel {base_numel},  gpu_factor {gpu_factor}")

        init_filters = init_filters_default

    # finally scale batch
    if auto_scale_batch and roi_size.prod() < base_numel * gpu_factor_init:
        batch_size = int(1.1 * gpu_factor_init)
        if global_rank == 0:
            print(
                f"increased batch_size {batch_size} base_numel {base_numel},  gpu_factor {gpu_factor},  gpu_factor_init {gpu_factor_init}"
            )

    else:
        batch_size = batch_size_default
        if global_rank == 0:
            print(
                f"kept batch the same base_numel {base_numel},  gpu_factor {gpu_factor},  gpu_factor_init {gpu_factor_init}"
            )

    levels = int(levels)
    roi_size = roi_size.astype(int).tolist()

    if global_rank == 0:
        print(
            f"Suggested network parameters: \n"
            f"Batch size {batch_size_default} => {batch_size} \n"
            f"ROI size {roi_size_default} => {roi_size} \n"
            f"init_filters {init_filters_default} => {init_filters} \n"
            f"aniso: {anisotropic_scales} image_size_mm: {image_size_mm} spacing: {spacing} levels: {levels} \n"
        )

    return roi_size, levels, init_filters, batch_size
