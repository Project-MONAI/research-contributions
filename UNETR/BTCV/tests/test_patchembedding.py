# --------------------------------------------------------
# Copyright (C) 2021 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# UNETR source code Ali Hatamizadeh
# --------------------------------------------------------

import unittest
from unittest import skipUnless

import numpy as np
import torch
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.utils import optional_import

einops, has_einops = optional_import("einops")

TEST_CASE_PATCHEMBEDDINGBLOCK = []
for dropout_rate in np.linspace(0, 1, 2):
    for in_channels in [1, 4]:
        for hidden_size in [360, 768]:
            for img_size in [96, 128]:
                for patch_size in [8, 16]:
                    for num_heads in [8, 12]:
                        for pos_embed in ["conv", "perceptron"]:
                            for classification in ["False", "True"]:
                                if classification:
                                    out = (2, (img_size // patch_size) ** 3 + 1, hidden_size)
                                else:
                                    out = (2, (img_size // patch_size) ** 3, hidden_size)
                                test_case = [
                                    {
                                        "in_channels": in_channels,
                                        "img_size": (img_size, img_size, img_size),
                                        "patch_size": (patch_size, patch_size, patch_size),
                                        "hidden_size": hidden_size,
                                        "num_heads": num_heads,
                                        "pos_embed": pos_embed,
                                        "classification": classification,
                                        "dropout_rate": dropout_rate,
                                    },
                                    (2, in_channels, img_size, *([img_size] * 2)),
                                    (2, (img_size // patch_size) ** 3, hidden_size),
                                ]
                                TEST_CASE_PATCHEMBEDDINGBLOCK.append(test_case)


class TestPatchEmbeddingBlock(unittest.TestCase):
    @parameterized.expand(TEST_CASE_PATCHEMBEDDINGBLOCK)
    @skipUnless(has_einops, "Requires einops")
    def test_shape(self, input_param, input_shape, expected_shape):
        net = PatchEmbeddingBlock(**input_param)
        with eval_mode(net):
            result = net(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)

    def test_ill_arg(self):
        with self.assertRaises(AssertionError):
            PatchEmbeddingBlock(
                in_channels=1,
                img_size=(128, 128, 128),
                patch_size=(16, 16, 16),
                hidden_size=128,
                num_heads=12,
                pos_embed="conv",
                classification=False,
                dropout_rate=5.0,
            )

        with self.assertRaises(AssertionError):
            PatchEmbeddingBlock(
                in_channels=1,
                img_size=(32, 32, 32),
                patch_size=(64, 64, 64),
                hidden_size=512,
                num_heads=8,
                pos_embed="perceptron",
                classification=False,
                dropout_rate=0.3,
            )

        with self.assertRaises(AssertionError):
            PatchEmbeddingBlock(
                in_channels=1,
                img_size=(96, 96, 96),
                patch_size=(8, 8, 8),
                hidden_size=512,
                num_heads=14,
                pos_embed="conv",
                classification=False,
                dropout_rate=0.3,
            )

        with self.assertRaises(AssertionError):
            PatchEmbeddingBlock(
                in_channels=1,
                img_size=(97, 97, 97),
                patch_size=(4, 4, 4),
                hidden_size=768,
                num_heads=8,
                pos_embed="perceptron",
                classification=False,
                dropout_rate=0.3,
            )

        with self.assertRaises(KeyError):
            PatchEmbeddingBlock(
                in_channels=4,
                img_size=(96, 96, 96),
                patch_size=(16, 16, 16),
                hidden_size=768,
                num_heads=12,
                pos_embed="perc",
                classification=False,
                dropout_rate=0.3,
            )


if __name__ == "__main__":
    unittest.main()
