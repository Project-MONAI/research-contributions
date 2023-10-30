from typing import Tuple

import numpy as np
import torch


def mask_rand_patch(
    window_sizes: Tuple[int, int, int], input_sizes: Tuple[int, int, int], mask_ratio: float, samples: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Patch-wise random masking."""
    if len(window_sizes) != len(input_sizes) or any(
        [input_size % window_size != 0 for window_size, input_size in zip(window_sizes, input_sizes)]
    ):
        raise ValueError(f"{window_sizes} & {input_sizes} is not compatible.")

    mask_shape = [input_size // window_size for input_size, window_size in zip(input_sizes, window_sizes)]
    num_patches = np.prod(mask_shape).item()
    mask = np.ones(num_patches, dtype=bool)
    indices = np.random.choice(num_patches, round(num_patches * mask_ratio), replace=False)
    mask[indices] = False
    mask = mask.reshape(mask_shape)
    wh, ww, wd = window_sizes
    mask = np.logical_or(mask[:, None, :, None, :, None], np.zeros([1, wh, 1, ww, 1, wd], dtype=bool)).reshape(
        input_sizes
    )
    mask = torch.from_numpy(mask).to(samples.device)

    res = samples.detach().clone()
    res[:, :, mask] = 0
    return res, mask
