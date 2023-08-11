"""View operations."""

from typing import Tuple

import numpy as np
import torch
from utils import view_transforms


def rot_rand(xs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    img_n = xs.size()[0]
    x_aug = xs.detach().clone()
    x_rot = torch.zeros(img_n, dtype=torch.int64, device=xs.device)
    for i in range(img_n):
        orientation = np.random.randint(0, 4)
        x_aug[i] = view_transforms.rotation_transforms[orientation](xs[i].unsqueeze(0))
        x_rot[i] = orientation
    return x_aug, x_rot
