"""View operations."""

from typing import Sequence, Tuple

import torch
import numpy as np

from utils import view_transforms

PermuteType = view_transforms.PermuteType
TransformFuncType = view_transforms.TransformFuncType


def get_permute_transform(view_src: PermuteType,
                          view_dst: PermuteType) -> TransformFuncType:
    """Gets transform function from view src to view dst."""

    def transform(x: torch.Tensor) -> torch.Tensor:
        x_view_0 = view_transforms.permutation_inverse_transforms[view_src](x)
        return view_transforms.permutation_transforms[view_dst](
            x_view_0).contiguous()

    return transform


def permute_inverse(xs: Sequence[torch.Tensor],
                    views: Sequence[PermuteType]) -> Sequence[torch.Tensor]:
    """Transforms data back to origin view."""
    return [get_permute_transform(view, 0)(x) for x, view in zip(xs, views)]


def permute_rand(
    x: torch.Tensor,
    num_samples: int = 2
) -> Tuple[Sequence[torch.Tensor], Sequence[PermuteType]]:
    """Samples different transforms of data."""
    num_permutes = len(view_transforms.permutation_transforms)
    if num_samples > num_permutes:
        raise ValueError('Duplicate samples.')
    view_dsts = np.random.permutation(num_permutes)[:num_samples].tolist()
    return [get_permute_transform(0, view)(x) for view in view_dsts], view_dsts
