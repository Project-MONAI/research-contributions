"""Unit test for view transforms."""

import itertools
import unittest

import numpy as np
import torch
from utils import view_transforms


class ViewTransformsTest(unittest.TestCase):
    def test_len(self):
        self.assertTrue(len(view_transforms.all_forward_transforms), len(view_transforms.all_backward_transforms))

    def test_inverse_transforms(self):
        x = np.random.uniform(size=(2, 6, 3, 4, 5))
        x_torch = torch.from_numpy(x)
        for group_name, transforms in view_transforms.all_forward_transforms.items():
            inverse_transforms = view_transforms.all_backward_transforms[group_name]
            self.assertEqual(len(transforms), len(inverse_transforms))
            for key in transforms:
                x_recon = inverse_transforms[key](transforms[key](x_torch)).numpy()
                np.testing.assert_allclose(x, x_recon)

    def test_get_transforms_func(self):
        x = np.random.uniform(size=(2, 6, 3, 4, 5))
        x_torch = torch.from_numpy(x)

        for order in [view_transforms.DEFAULT_ORDER, view_transforms.DEFAULT_ORDER[::-1]]:
            views_all = itertools.product(*[view_transforms.all_forward_transforms[gn].keys() for gn in order])
            for views in views_all:
                func, inv_func = [
                    view_transforms.get_transforms_func(views, order, inverse) for inverse in [False, True]
                ]
                np.testing.assert_allclose(x, inv_func(func(x_torch)).numpy())


if __name__ == "__main__":
    unittest.main()
