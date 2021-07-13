# --------------------------------------------------------
# Copyright (C) 2021 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# UNETR source code Ali Hatamizadeh
# --------------------------------------------------------

import unittest

import numpy as np
import torch
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.blocks.mlp import MLPBlock

TEST_CASE_MLP = []
for dropout_rate in np.linspace(0, 1, 4):
    for hidden_size in [128, 256, 512, 768]:
        for mlp_dim in [512, 1028, 2048, 3072]:

            test_case = [
                {
                    "hidden_size": hidden_size,
                    "mlp_dim": mlp_dim,
                    "dropout_rate": dropout_rate,
                },
                (2, 512, hidden_size),
                (2, 512, hidden_size),
            ]
            TEST_CASE_MLP.append(test_case)


class TestMLPBlock(unittest.TestCase):
    @parameterized.expand(TEST_CASE_MLP)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = MLPBlock(**input_param)
        with eval_mode(net):
            result = net(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)

    def test_ill_arg(self):
        with self.assertRaises(AssertionError):
            MLPBlock(hidden_size=128, mlp_dim=512, dropout_rate=5.0)


if __name__ == "__main__":
    unittest.main()
