# Copyright 2020 - 2022 -> (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import pdb
from typing import Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from mlp_new import MLPBlock as Mlp
from patchembedding import PatchEmbed
from torch.nn import LayerNorm

from monai.networks.blocks import UnetBasicBlock, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.utils import ensure_tuple_rep, optional_import

rearrange, _ = optional_import("einops", name="rearrange")


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """Tensor initialization with truncated normal distribution.
    Based on:
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    https://github.com/rwightman/pytorch-image-models
    Args:
       tensor: an n-dimensional `torch.Tensor`.
       mean: the mean of the normal distribution.
       std: the standard deviation of the normal distribution.
       a: the minimum cutoff value.
       b: the maximum cutoff value.
    """

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Tensor initialization with truncated normal distribution.
    Based on:
    https://github.com/rwightman/pytorch-image-models
    Args:
       tensor: an n-dimensional `torch.Tensor`
       mean: the mean of the normal distribution
       std: the standard deviation of the normal distribution
       a: the minimum cutoff value
       b: the maximum cutoff value
    """

    if not std > 0:
        raise ValueError("the standard deviation should be greater than zero.")

    if a >= b:
        raise ValueError("minimum cutoff value (a) should be smaller than maximum cutoff value (b).")

    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class DropPath(nn.Module):
    """Stochastic drop paths per sample for residual blocks.
    Based on:
    https://github.com/rwightman/pytorch-image-models
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True) -> None:
        """
        Args:
            drop_prob: drop path probability.
            scale_by_keep: scaling by non-dropped probability.
        """
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

        if not (0 <= drop_prob <= 1):
            raise ValueError("Drop path prob should be between 0 and 1.")

    def drop_path(self, x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
        if drop_prob == 0.0 or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class SwinUNETR(nn.Module):
    """
    Swin UNETR based on: "Hatamizadeh et al.,
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <https://arxiv.org/abs/2201.01266>"
    """

    def __init__(
        self,
        img_size: Union[Sequence[int], int],
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: Union[Tuple, str] = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            img_size: dimension of input image.
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.
        Examples::
            # for 3D single channel input with size (96,96,96), 4-channel output and feature size of 48.
            >>> net = SwinUNETR(img_size=(96,96,96), in_channels=1, out_channels=4, feature_size=48)
            # for 3D 4-channel input with size (128,128,128), 3-channel output and (2,4,2,2) layers in each stage.
            >>> net = SwinUNETR(img_size=(128,128,128), in_channels=4, out_channels=3, depths=(2,4,2,2))
            # for 2D single channel input with size (96,96), 2-channel output and gradient checkpointing.
            >>> net = SwinUNETR(img_size=(96,96), in_channels=3, out_channels=2, use_checkpoint=True, spatial_dims=2)
        """

        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)

        if not (spatial_dims == 2 or spatial_dims == 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        for m, p in zip(img_size, patch_size):
            for i in range(5):
                if m % np.power(p, i + 1) != 0:
                    raise ValueError("input image size (img_size) should be divisible by stage-wise image resolution.")

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

        self.normalize = normalize

        self.swinViT = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.out = UnetOutBlock(
            spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels
        )  # type: ignore

    def load_from(self, weights, finetune_choice):
        if finetune_choice == "both":
            enc = 1
            dec = 1
        elif finetune_choice == "encoder":
            enc = 1
            dec = 0
        elif finetune_choice == "decoder":
            enc = 0
            dec = 1

        with torch.no_grad():
            # pdb.set_trace()

            if enc == 1:
                self.swinViT.patch_embed.proj.conv1.conv.weight.copy_(
                    weights["model"]["encoder.patch_embed.proj.conv1.conv.weight"]
                )
                self.swinViT.patch_embed.proj.conv2.conv.weight.copy_(
                    weights["model"]["encoder.patch_embed.proj.conv2.conv.weight"]
                )
                self.swinViT.patch_embed.proj.norm1.weight.copy_(
                    weights["model"]["encoder.patch_embed.proj.norm1.weight"]
                )
                self.swinViT.patch_embed.proj.norm1.bias.copy_(weights["model"]["encoder.patch_embed.proj.norm1.bias"])
                self.swinViT.patch_embed.proj.norm2.weight.copy_(
                    weights["model"]["encoder.patch_embed.proj.norm2.weight"]
                )
                self.swinViT.patch_embed.proj.norm2.bias.copy_(weights["model"]["encoder.patch_embed.proj.norm2.bias"])

                for bname, block in self.swinViT.layers1[0].blocks.named_children():
                    block.load_from(weights, n_block=bname, layer="layers1")
                self.swinViT.layers1[0].downsample.reduction.weight.copy_(
                    weights["model"]["encoder.layers1.0.0.downsample.reduction.weight"]
                )
                self.swinViT.layers1[0].downsample.norm.weight.copy_(
                    weights["model"]["encoder.layers1.0.0.downsample.norm.weight"]
                )
                self.swinViT.layers1[0].downsample.norm.bias.copy_(
                    weights["model"]["encoder.layers1.0.0.downsample.norm.bias"]
                )
                for bname, block in self.swinViT.layers2[0].blocks.named_children():
                    block.load_from(weights, n_block=bname, layer="layers2")
                self.swinViT.layers2[0].downsample.reduction.weight.copy_(
                    weights["model"]["encoder.layers2.0.0.downsample.reduction.weight"]
                )
                self.swinViT.layers2[0].downsample.norm.weight.copy_(
                    weights["model"]["encoder.layers2.0.0.downsample.norm.weight"]
                )
                self.swinViT.layers2[0].downsample.norm.bias.copy_(
                    weights["model"]["encoder.layers2.0.0.downsample.norm.bias"]
                )
                for bname, block in self.swinViT.layers3[0].blocks.named_children():
                    block.load_from(weights, n_block=bname, layer="layers3")
                self.swinViT.layers3[0].downsample.reduction.weight.copy_(
                    weights["model"]["encoder.layers3.0.0.downsample.reduction.weight"]
                )
                self.swinViT.layers3[0].downsample.norm.weight.copy_(
                    weights["model"]["encoder.layers3.0.0.downsample.norm.weight"]
                )
                self.swinViT.layers3[0].downsample.norm.bias.copy_(
                    weights["model"]["encoder.layers3.0.0.downsample.norm.bias"]
                )
                for bname, block in self.swinViT.layers4[0].blocks.named_children():
                    block.load_from(weights, n_block=bname, layer="layers4")
                self.swinViT.layers4[0].downsample.reduction.weight.copy_(
                    weights["model"]["encoder.layers4.0.0.downsample.reduction.weight"]
                )
                self.swinViT.layers4[0].downsample.norm.weight.copy_(
                    weights["model"]["encoder.layers4.0.0.downsample.norm.weight"]
                )
                self.swinViT.layers4[0].downsample.norm.bias.copy_(
                    weights["model"]["encoder.layers4.0.0.downsample.norm.bias"]
                )
            #  'encoder1.layer.conv1.conv.weight', 'encoder1.layer.conv2.conv.weight', 'encoder1.layer.conv3.conv.weight'
            # pdb.set_trace()
            if dec == 1:
                self.encoder1.layer.conv1.conv.weight.copy_(weights["model"]["encoder1.layer.conv1.conv.weight"])
                self.encoder1.layer.conv2.conv.weight.copy_(weights["model"]["encoder1.layer.conv2.conv.weight"])
                self.encoder1.layer.conv3.conv.weight.copy_(weights["model"]["encoder1.layer.conv3.conv.weight"])

                self.encoder2.layer.conv1.conv.weight.copy_(weights["model"]["encoder2.layer.conv1.conv.weight"])
                self.encoder2.layer.conv2.conv.weight.copy_(weights["model"]["encoder2.layer.conv2.conv.weight"])
                self.encoder2.layer.conv3.conv.weight.copy_(weights["model"]["encoder2.layer.conv3.conv.weight"])

                self.encoder3.layer.conv1.conv.weight.copy_(weights["model"]["encoder3.layer.conv1.conv.weight"])
                self.encoder3.layer.conv2.conv.weight.copy_(weights["model"]["encoder3.layer.conv2.conv.weight"])
                self.encoder3.layer.conv3.conv.weight.copy_(weights["model"]["encoder3.layer.conv3.conv.weight"])

                self.encoder4.layer.conv1.conv.weight.copy_(weights["model"]["encoder4.layer.conv1.conv.weight"])
                self.encoder4.layer.conv2.conv.weight.copy_(weights["model"]["encoder4.layer.conv2.conv.weight"])
                self.encoder4.layer.conv3.conv.weight.copy_(weights["model"]["encoder4.layer.conv3.conv.weight"])

                self.encoder10.layer.conv1.conv.weight.copy_(weights["model"]["encoder10.layer.conv1.conv.weight"])
                self.encoder10.layer.conv2.conv.weight.copy_(weights["model"]["encoder10.layer.conv2.conv.weight"])
                self.encoder10.layer.conv3.conv.weight.copy_(weights["model"]["encoder10.layer.conv3.conv.weight"])

                # self.encoder1.layer.conv1.conv.bias.copy_(weights["model"]["encoder1.layer.conv1.conv.bias"])
                # self.encoder1.layer.conv2.conv.bias.copy_(weights["model"]["encoder1.layer.conv2.conv.bias"])
                # self.encoder1.layer.conv3.conv.bias.copy_(weights["model"]["encoder1.layer.conv3.conv.bias"])

                # self.encoder2.layer.conv1.conv.bias.copy_(weights["model"]["encoder2.layer.conv1.conv.bias"])
                # self.encoder2.layer.conv2.conv.bias.copy_(weights["model"]["encoder2.layer.conv2.conv.bias"])
                # self.encoder2.layer.conv3.conv.bias.copy_(weights["model"]["encoder2.layer.conv3.conv.bias"])

                # self.encoder3.layer.conv1.conv.bias.copy_(weights["model"]["encoder3.layer.conv1.conv.bias"])
                # self.encoder3.layer.conv2.conv.bias.copy_(weights["model"]["encoder3.layer.conv2.conv.bias"])
                # self.encoder3.layer.conv3.conv.bias.copy_(weights["model"]["encoder3.layer.conv3.conv.bias"])

                # self.encoder4.layer.conv1.conv.bias.copy_(weights["model"]["encoder4.layer.conv1.conv.bias"])
                # self.encoder4.layer.conv2.conv.bias.copy_(weights["model"]["encoder4.layer.conv2.conv.bias"])
                # self.encoder4.layer.conv3.conv.bias.copy_(weights["model"]["encoder4.layer.conv3.conv.bias"])

                # self.encoder10.layer.conv1.conv.bias.copy_(weights["model"]["encoder10.layer.conv1.conv.bias"])
                # self.encoder10.layer.conv2.conv.bias.copy_(weights["model"]["encoder10.layer.conv2.conv.bias"])
                # self.encoder10.layer.conv3.conv.bias.copy_(weights["model"]["encoder10.layer.conv3.conv.bias"])

                self.decoder1.transp_conv.conv.weight.copy_(weights["model"]["decoder1.transp_conv.conv.weight"])
                self.decoder1.conv_block.conv1.conv.weight.copy_(
                    weights["model"]["decoder1.conv_block.conv1.conv.weight"]
                )
                self.decoder1.conv_block.conv2.conv.weight.copy_(
                    weights["model"]["decoder1.conv_block.conv2.conv.weight"]
                )
                self.decoder1.conv_block.conv3.conv.weight.copy_(
                    weights["model"]["decoder1.conv_block.conv3.conv.weight"]
                )

                self.decoder2.transp_conv.conv.weight.copy_(weights["model"]["decoder2.transp_conv.conv.weight"])
                self.decoder2.conv_block.conv1.conv.weight.copy_(
                    weights["model"]["decoder2.conv_block.conv1.conv.weight"]
                )
                self.decoder2.conv_block.conv2.conv.weight.copy_(
                    weights["model"]["decoder2.conv_block.conv2.conv.weight"]
                )
                self.decoder2.conv_block.conv3.conv.weight.copy_(
                    weights["model"]["decoder2.conv_block.conv3.conv.weight"]
                )

                self.decoder3.transp_conv.conv.weight.copy_(weights["model"]["decoder3.transp_conv.conv.weight"])
                self.decoder3.conv_block.conv1.conv.weight.copy_(
                    weights["model"]["decoder3.conv_block.conv1.conv.weight"]
                )
                self.decoder3.conv_block.conv2.conv.weight.copy_(
                    weights["model"]["decoder3.conv_block.conv2.conv.weight"]
                )
                self.decoder3.conv_block.conv3.conv.weight.copy_(
                    weights["model"]["decoder3.conv_block.conv3.conv.weight"]
                )

                self.decoder4.transp_conv.conv.weight.copy_(weights["model"]["decoder4.transp_conv.conv.weight"])
                self.decoder4.conv_block.conv1.conv.weight.copy_(
                    weights["model"]["decoder4.conv_block.conv1.conv.weight"]
                )
                self.decoder4.conv_block.conv2.conv.weight.copy_(
                    weights["model"]["decoder4.conv_block.conv2.conv.weight"]
                )
                self.decoder4.conv_block.conv3.conv.weight.copy_(
                    weights["model"]["decoder4.conv_block.conv3.conv.weight"]
                )

                self.decoder5.transp_conv.conv.weight.copy_(weights["model"]["decoder5.transp_conv.conv.weight"])
                self.decoder5.conv_block.conv1.conv.weight.copy_(
                    weights["model"]["decoder5.conv_block.conv1.conv.weight"]
                )
                self.decoder5.conv_block.conv2.conv.weight.copy_(
                    weights["model"]["decoder5.conv_block.conv2.conv.weight"]
                )
                self.decoder5.conv_block.conv3.conv.weight.copy_(
                    weights["model"]["decoder5.conv_block.conv3.conv.weight"]
                )

                # self.decoder1.layer.conv1.conv.bias.copy_(weights["model"]["decoder1.layer.conv1.conv.bias"])
                # self.decoder1.layer.conv2.conv.bias.copy_(weights["model"]["decoder1.layer.conv2.conv.bias"])
                # self.decoder1.layer.conv3.conv.bias.copy_(weights["model"]["decoder1.layer.conv3.conv.bias"])

                # self.decoder2.layer.conv1.conv.bias.copy_(weights["model"]["decoder2.layer.conv1.conv.bias"])
                # self.decoder2.layer.conv2.conv.bias.copy_(weights["model"]["decoder2.layer.conv2.conv.bias"])
                # self.decoder2.layer.conv3.conv.bias.copy_(weights["model"]["decoder2.layer.conv3.conv.bias"])

                # self.decoder3.layer.conv1.conv.bias.copy_(weights["model"]["decoder3.layer.conv1.conv.bias"])
                # self.decoder3.layer.conv2.conv.bias.copy_(weights["model"]["decoder3.layer.conv2.conv.bias"])
                # self.decoder3.layer.conv3.conv.bias.copy_(weights["model"]["decoder3.layer.conv3.conv.bias"])

                # self.decoder4.layer.conv1.conv.bias.copy_(weights["model"]["decoder4.layer.conv1.conv.bias"])
                # self.decoder4.layer.conv2.conv.bias.copy_(weights["model"]["decoder4.layer.conv2.conv.bias"])
                # self.decoder4.layer.conv3.conv.bias.copy_(weights["model"]["decoder4.layer.conv3.conv.bias"])

                # self.decoder5.layer.conv1.conv.bias.copy_(weights["model"]["decoder5.layer.conv1.conv.bias"])
                # self.decoder5.layer.conv2.conv.bias.copy_(weights["model"]["decoder5.layer.conv2.conv.bias"])
                # self.decoder5.layer.conv3.conv.bias.copy_(weights["model"]["decoder5.layer.conv3.conv.bias"])

    def forward(self, x_in):
        hidden_states_out = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        # import pdb; pdb.set_trace()
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)
        return logits


class SwinUNETR_OG(nn.Module):
    """
    Swin UNETR based on: "Hatamizadeh et al.,
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <https://arxiv.org/abs/2201.01266>"
    """

    def __init__(
        self,
        img_size: Union[Sequence[int], int],
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: Union[Tuple, str] = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            img_size: dimension of input image.
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.
        Examples::
            # for 3D single channel input with size (96,96,96), 4-channel output and feature size of 48.
            >>> net = SwinUNETR(img_size=(96,96,96), in_channels=1, out_channels=4, feature_size=48)
            # for 3D 4-channel input with size (128,128,128), 3-channel output and (2,4,2,2) layers in each stage.
            >>> net = SwinUNETR(img_size=(128,128,128), in_channels=4, out_channels=3, depths=(2,4,2,2))
            # for 2D single channel input with size (96,96), 2-channel output and gradient checkpointing.
            >>> net = SwinUNETR(img_size=(96,96), in_channels=3, out_channels=2, use_checkpoint=True, spatial_dims=2)
        """

        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)

        if not (spatial_dims == 2 or spatial_dims == 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        for m, p in zip(img_size, patch_size):
            for i in range(5):
                if m % np.power(p, i + 1) != 0:
                    raise ValueError("input image size (img_size) should be divisible by stage-wise image resolution.")

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

        self.normalize = normalize

        self.swinViT = SwinTransformerOG(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.out = UnetOutBlock(
            spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels
        )  # type: ignore

    def load_from(self, weights, finetune_choice):
        if finetune_choice == "both":
            enc = 1
            dec = 1
        elif finetune_choice == "encoder":
            enc = 1
            dec = 0
        elif finetune_choice == "decoder":
            enc = 0
            dec = 1

        with torch.no_grad():
            # pdb.set_trace()

            if enc == 1:
                self.swinViT.patch_embed.proj.conv1.conv.weight.copy_(
                    weights["model"]["encoder.patch_embed.proj.conv1.conv.weight"]
                )
                self.swinViT.patch_embed.proj.conv2.conv.weight.copy_(
                    weights["model"]["encoder.patch_embed.proj.conv2.conv.weight"]
                )
                self.swinViT.patch_embed.proj.norm1.weight.copy_(
                    weights["model"]["encoder.patch_embed.proj.norm1.weight"]
                )
                self.swinViT.patch_embed.proj.norm1.bias.copy_(weights["model"]["encoder.patch_embed.proj.norm1.bias"])
                self.swinViT.patch_embed.proj.norm2.weight.copy_(
                    weights["model"]["encoder.patch_embed.proj.norm2.weight"]
                )
                self.swinViT.patch_embed.proj.norm2.bias.copy_(weights["model"]["encoder.patch_embed.proj.norm2.bias"])

                for bname, block in self.swinViT.layers1[0].blocks.named_children():
                    block.load_from(weights, n_block=bname, layer="layers1")
                self.swinViT.layers1[0].downsample.reduction.weight.copy_(
                    weights["model"]["encoder.layers1.0.0.downsample.reduction.weight"]
                )
                self.swinViT.layers1[0].downsample.norm.weight.copy_(
                    weights["model"]["encoder.layers1.0.0.downsample.norm.weight"]
                )
                self.swinViT.layers1[0].downsample.norm.bias.copy_(
                    weights["model"]["encoder.layers1.0.0.downsample.norm.bias"]
                )
                for bname, block in self.swinViT.layers2[0].blocks.named_children():
                    block.load_from(weights, n_block=bname, layer="layers2")
                self.swinViT.layers2[0].downsample.reduction.weight.copy_(
                    weights["model"]["encoder.layers2.0.0.downsample.reduction.weight"]
                )
                self.swinViT.layers2[0].downsample.norm.weight.copy_(
                    weights["model"]["encoder.layers2.0.0.downsample.norm.weight"]
                )
                self.swinViT.layers2[0].downsample.norm.bias.copy_(
                    weights["model"]["encoder.layers2.0.0.downsample.norm.bias"]
                )
                for bname, block in self.swinViT.layers3[0].blocks.named_children():
                    block.load_from(weights, n_block=bname, layer="layers3")
                self.swinViT.layers3[0].downsample.reduction.weight.copy_(
                    weights["model"]["encoder.layers3.0.0.downsample.reduction.weight"]
                )
                self.swinViT.layers3[0].downsample.norm.weight.copy_(
                    weights["model"]["encoder.layers3.0.0.downsample.norm.weight"]
                )
                self.swinViT.layers3[0].downsample.norm.bias.copy_(
                    weights["model"]["encoder.layers3.0.0.downsample.norm.bias"]
                )
                for bname, block in self.swinViT.layers4[0].blocks.named_children():
                    block.load_from(weights, n_block=bname, layer="layers4")
                self.swinViT.layers4[0].downsample.reduction.weight.copy_(
                    weights["model"]["encoder.layers4.0.0.downsample.reduction.weight"]
                )
                self.swinViT.layers4[0].downsample.norm.weight.copy_(
                    weights["model"]["encoder.layers4.0.0.downsample.norm.weight"]
                )
                self.swinViT.layers4[0].downsample.norm.bias.copy_(
                    weights["model"]["encoder.layers4.0.0.downsample.norm.bias"]
                )
            #  'encoder1.layer.conv1.conv.weight', 'encoder1.layer.conv2.conv.weight', 'encoder1.layer.conv3.conv.weight'
            # pdb.set_trace()
            if dec == 1:
                self.encoder1.layer.conv1.conv.weight.copy_(weights["model"]["encoder1.layer.conv1.conv.weight"])
                self.encoder1.layer.conv2.conv.weight.copy_(weights["model"]["encoder1.layer.conv2.conv.weight"])
                self.encoder1.layer.conv3.conv.weight.copy_(weights["model"]["encoder1.layer.conv3.conv.weight"])

                self.encoder2.layer.conv1.conv.weight.copy_(weights["model"]["encoder2.layer.conv1.conv.weight"])
                self.encoder2.layer.conv2.conv.weight.copy_(weights["model"]["encoder2.layer.conv2.conv.weight"])
                self.encoder2.layer.conv3.conv.weight.copy_(weights["model"]["encoder2.layer.conv3.conv.weight"])

                self.encoder3.layer.conv1.conv.weight.copy_(weights["model"]["encoder3.layer.conv1.conv.weight"])
                self.encoder3.layer.conv2.conv.weight.copy_(weights["model"]["encoder3.layer.conv2.conv.weight"])
                self.encoder3.layer.conv3.conv.weight.copy_(weights["model"]["encoder3.layer.conv3.conv.weight"])

                self.encoder4.layer.conv1.conv.weight.copy_(weights["model"]["encoder4.layer.conv1.conv.weight"])
                self.encoder4.layer.conv2.conv.weight.copy_(weights["model"]["encoder4.layer.conv2.conv.weight"])
                self.encoder4.layer.conv3.conv.weight.copy_(weights["model"]["encoder4.layer.conv3.conv.weight"])

                self.encoder10.layer.conv1.conv.weight.copy_(weights["model"]["encoder10.layer.conv1.conv.weight"])
                self.encoder10.layer.conv2.conv.weight.copy_(weights["model"]["encoder10.layer.conv2.conv.weight"])
                self.encoder10.layer.conv3.conv.weight.copy_(weights["model"]["encoder10.layer.conv3.conv.weight"])

                # self.encoder1.layer.conv1.conv.bias.copy_(weights["model"]["encoder1.layer.conv1.conv.bias"])
                # self.encoder1.layer.conv2.conv.bias.copy_(weights["model"]["encoder1.layer.conv2.conv.bias"])
                # self.encoder1.layer.conv3.conv.bias.copy_(weights["model"]["encoder1.layer.conv3.conv.bias"])

                # self.encoder2.layer.conv1.conv.bias.copy_(weights["model"]["encoder2.layer.conv1.conv.bias"])
                # self.encoder2.layer.conv2.conv.bias.copy_(weights["model"]["encoder2.layer.conv2.conv.bias"])
                # self.encoder2.layer.conv3.conv.bias.copy_(weights["model"]["encoder2.layer.conv3.conv.bias"])

                # self.encoder3.layer.conv1.conv.bias.copy_(weights["model"]["encoder3.layer.conv1.conv.bias"])
                # self.encoder3.layer.conv2.conv.bias.copy_(weights["model"]["encoder3.layer.conv2.conv.bias"])
                # self.encoder3.layer.conv3.conv.bias.copy_(weights["model"]["encoder3.layer.conv3.conv.bias"])

                # self.encoder4.layer.conv1.conv.bias.copy_(weights["model"]["encoder4.layer.conv1.conv.bias"])
                # self.encoder4.layer.conv2.conv.bias.copy_(weights["model"]["encoder4.layer.conv2.conv.bias"])
                # self.encoder4.layer.conv3.conv.bias.copy_(weights["model"]["encoder4.layer.conv3.conv.bias"])

                # self.encoder10.layer.conv1.conv.bias.copy_(weights["model"]["encoder10.layer.conv1.conv.bias"])
                # self.encoder10.layer.conv2.conv.bias.copy_(weights["model"]["encoder10.layer.conv2.conv.bias"])
                # self.encoder10.layer.conv3.conv.bias.copy_(weights["model"]["encoder10.layer.conv3.conv.bias"])

                self.decoder1.transp_conv.conv.weight.copy_(weights["model"]["decoder1.transp_conv.conv.weight"])
                self.decoder1.conv_block.conv1.conv.weight.copy_(
                    weights["model"]["decoder1.conv_block.conv1.conv.weight"]
                )
                self.decoder1.conv_block.conv2.conv.weight.copy_(
                    weights["model"]["decoder1.conv_block.conv2.conv.weight"]
                )
                self.decoder1.conv_block.conv3.conv.weight.copy_(
                    weights["model"]["decoder1.conv_block.conv3.conv.weight"]
                )

                self.decoder2.transp_conv.conv.weight.copy_(weights["model"]["decoder2.transp_conv.conv.weight"])
                self.decoder2.conv_block.conv1.conv.weight.copy_(
                    weights["model"]["decoder2.conv_block.conv1.conv.weight"]
                )
                self.decoder2.conv_block.conv2.conv.weight.copy_(
                    weights["model"]["decoder2.conv_block.conv2.conv.weight"]
                )
                self.decoder2.conv_block.conv3.conv.weight.copy_(
                    weights["model"]["decoder2.conv_block.conv3.conv.weight"]
                )

                self.decoder3.transp_conv.conv.weight.copy_(weights["model"]["decoder3.transp_conv.conv.weight"])
                self.decoder3.conv_block.conv1.conv.weight.copy_(
                    weights["model"]["decoder3.conv_block.conv1.conv.weight"]
                )
                self.decoder3.conv_block.conv2.conv.weight.copy_(
                    weights["model"]["decoder3.conv_block.conv2.conv.weight"]
                )
                self.decoder3.conv_block.conv3.conv.weight.copy_(
                    weights["model"]["decoder3.conv_block.conv3.conv.weight"]
                )

                self.decoder4.transp_conv.conv.weight.copy_(weights["model"]["decoder4.transp_conv.conv.weight"])
                self.decoder4.conv_block.conv1.conv.weight.copy_(
                    weights["model"]["decoder4.conv_block.conv1.conv.weight"]
                )
                self.decoder4.conv_block.conv2.conv.weight.copy_(
                    weights["model"]["decoder4.conv_block.conv2.conv.weight"]
                )
                self.decoder4.conv_block.conv3.conv.weight.copy_(
                    weights["model"]["decoder4.conv_block.conv3.conv.weight"]
                )

                self.decoder5.transp_conv.conv.weight.copy_(weights["model"]["decoder5.transp_conv.conv.weight"])
                self.decoder5.conv_block.conv1.conv.weight.copy_(
                    weights["model"]["decoder5.conv_block.conv1.conv.weight"]
                )
                self.decoder5.conv_block.conv2.conv.weight.copy_(
                    weights["model"]["decoder5.conv_block.conv2.conv.weight"]
                )
                self.decoder5.conv_block.conv3.conv.weight.copy_(
                    weights["model"]["decoder5.conv_block.conv3.conv.weight"]
                )

                # self.decoder1.layer.conv1.conv.bias.copy_(weights["model"]["decoder1.layer.conv1.conv.bias"])
                # self.decoder1.layer.conv2.conv.bias.copy_(weights["model"]["decoder1.layer.conv2.conv.bias"])
                # self.decoder1.layer.conv3.conv.bias.copy_(weights["model"]["decoder1.layer.conv3.conv.bias"])

                # self.decoder2.layer.conv1.conv.bias.copy_(weights["model"]["decoder2.layer.conv1.conv.bias"])
                # self.decoder2.layer.conv2.conv.bias.copy_(weights["model"]["decoder2.layer.conv2.conv.bias"])
                # self.decoder2.layer.conv3.conv.bias.copy_(weights["model"]["decoder2.layer.conv3.conv.bias"])

                # self.decoder3.layer.conv1.conv.bias.copy_(weights["model"]["decoder3.layer.conv1.conv.bias"])
                # self.decoder3.layer.conv2.conv.bias.copy_(weights["model"]["decoder3.layer.conv2.conv.bias"])
                # self.decoder3.layer.conv3.conv.bias.copy_(weights["model"]["decoder3.layer.conv3.conv.bias"])

                # self.decoder4.layer.conv1.conv.bias.copy_(weights["model"]["decoder4.layer.conv1.conv.bias"])
                # self.decoder4.layer.conv2.conv.bias.copy_(weights["model"]["decoder4.layer.conv2.conv.bias"])
                # self.decoder4.layer.conv3.conv.bias.copy_(weights["model"]["decoder4.layer.conv3.conv.bias"])

                # self.decoder5.layer.conv1.conv.bias.copy_(weights["model"]["decoder5.layer.conv1.conv.bias"])
                # self.decoder5.layer.conv2.conv.bias.copy_(weights["model"]["decoder5.layer.conv2.conv.bias"])
                # self.decoder5.layer.conv3.conv.bias.copy_(weights["model"]["decoder5.layer.conv3.conv.bias"])

    def forward(self, x_in):
        hidden_states_out = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        # import pdb; pdb.set_trace()
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)
        return logits


def window_partition(x, window_size):
    """window partition operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
     Args:
        x: input tensor.
        window_size: local window size.
    """
    x_shape = x.size()
    if len(x_shape) == 5:
        b, d, h, w, c = x_shape
        x = x.view(
            b,
            d // window_size[0],
            window_size[0],
            h // window_size[1],
            window_size[1],
            w // window_size[2],
            window_size[2],
            c,
        )
        windows = (
            x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2], c)
        )
    elif len(x_shape) == 4:
        b, h, w, c = x.shape
        x = x.view(b, h // window_size[0], window_size[0], w // window_size[1], window_size[1], c)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0] * window_size[1], c)
    return windows


def window_reverse(windows, window_size, dims):
    """window reverse operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
     Args:
        windows: windows tensor.
        window_size: local window size.
        dims: dimension values.
    """
    if len(dims) == 4:
        b, d, h, w = dims
        x = windows.view(
            b,
            d // window_size[0],
            h // window_size[1],
            w // window_size[2],
            window_size[0],
            window_size[1],
            window_size[2],
            -1,
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)

    elif len(dims) == 3:
        b, h, w = dims
        x = windows.view(b, h // window_size[0], w // window_size[0], window_size[0], window_size[1], -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    """Computing window size based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
     Args:
        x_size: input size.
        window_size: local window size.
        shift_size: window shifting size.
    """

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention(nn.Module):
    """
    Window based multi-head self attention module with relative position bias based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            qkv_bias: add a learnable bias to query, key, value.
            attn_drop: attention dropout rate.
            proj_drop: dropout rate of output.
        """

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        mesh_args = torch.meshgrid.__kwdefaults__

        if len(self.window_size) == 3:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1),
                    num_heads,
                )
            )
            coords_d = torch.arange(self.window_size[0])
            coords_h = torch.arange(self.window_size[1])
            coords_w = torch.arange(self.window_size[2])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        elif len(self.window_size) == 2:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
            )
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:n, :n].reshape(-1)
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer block based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        shift_size: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: str = "GELU",
        norm_layer: Type[LayerNorm] = nn.LayerNorm,  # type: ignore
        use_checkpoint: bool = False,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            shift_size: window shift size.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: stochastic depth rate.
            act_layer: activation layer.
            norm_layer: normalization layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(hidden_size=dim, mlp_dim=mlp_hidden_dim, act=act_layer, dropout_rate=drop, dropout_mode="swin")

    def forward_part1(self, x, mask_matrix):
        x_shape = x.size()
        x = self.norm1(x)
        # print(x_shape)
        if len(x_shape) == 5:
            b, d, h, w, c = x.shape
            window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
            pad_l = pad_t = pad_d0 = 0
            pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
            pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
            pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
            _, dp, hp, wp, _ = x.shape
            dims = [b, dp, hp, wp]

        elif len(x_shape) == 4:
            b, h, w, c = x.shape
            window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
            pad_l = pad_t = 0
            pad_r = (window_size[0] - h % window_size[0]) % window_size[0]
            pad_b = (window_size[1] - w % window_size[1]) % window_size[1]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, hp, wp, _ = x.shape
            dims = [b, hp, wp]

        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            elif len(x_shape) == 4:
                shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)
        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
            elif len(x_shape) == 4:
                x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        if len(x_shape) == 5:
            if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
                x = x[:, :d, :h, :w, :].contiguous()
        elif len(x_shape) == 4:
            if pad_r > 0 or pad_b > 0:
                x = x[:, :h, :w, :].contiguous()

        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def load_from(self, weights, n_block, layer):
        root = f"encoder.{layer}.0.0.blocks.{n_block}."
        block_names = [
            "norm1.weight",
            "norm1.bias",
            "attn.relative_position_bias_table",
            "attn.relative_position_index",
            "attn.qkv.weight",
            "attn.qkv.bias",
            "attn.proj.weight",
            "attn.proj.bias",
            "norm2.weight",
            "norm2.bias",
            "mlp.fc1.weight",
            "mlp.fc1.bias",
            "mlp.fc2.weight",
            "mlp.fc2.bias",
        ]
        with torch.no_grad():
            self.norm1.weight.copy_(weights["model"][root + block_names[0]])
            self.norm1.bias.copy_(weights["model"][root + block_names[1]])
            self.attn.relative_position_bias_table.copy_(weights["model"][root + block_names[2]])
            self.attn.relative_position_index.copy_(weights["model"][root + block_names[3]])
            self.attn.qkv.weight.copy_(weights["model"][root + block_names[4]])
            self.attn.qkv.bias.copy_(weights["model"][root + block_names[5]])
            self.attn.proj.weight.copy_(weights["model"][root + block_names[6]])
            self.attn.proj.bias.copy_(weights["model"][root + block_names[7]])
            self.norm2.weight.copy_(weights["model"][root + block_names[8]])
            self.norm2.bias.copy_(weights["model"][root + block_names[9]])
            self.mlp.linear1.weight.copy_(weights["model"][root + block_names[10]])
            self.mlp.linear1.bias.copy_(weights["model"][root + block_names[11]])
            self.mlp.linear2.weight.copy_(weights["model"][root + block_names[12]])
            self.mlp.linear2.bias.copy_(weights["model"][root + block_names[13]])

    def forward(self, x, mask_matrix):
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)
        return x


class PatchMerging(nn.Module):
    """
    Patch merging layer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self, dim: int, norm_layer: Type[LayerNorm] = nn.LayerNorm, spatial_dims: int = 3
    ) -> None:  # type: ignore
        """
        Args:
            dim: number of feature channels.
            norm_layer: normalization layer.
            spatial_dims: number of spatial dims.
        """

        super().__init__()
        self.dim = dim
        if spatial_dims == 3:
            self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(8 * dim)
        elif spatial_dims == 2:
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(4 * dim)

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 5:
            b, d, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, d % 2, 0, w % 2, 0, h % 2))
            x0 = x[:, 0::2, 0::2, 0::2, :]
            x1 = x[:, 1::2, 0::2, 0::2, :]
            x2 = x[:, 0::2, 1::2, 0::2, :]
            x3 = x[:, 0::2, 0::2, 1::2, :]
            x4 = x[:, 1::2, 0::2, 1::2, :]
            x5 = x[:, 0::2, 1::2, 0::2, :]
            x6 = x[:, 0::2, 0::2, 1::2, :]
            x7 = x[:, 1::2, 1::2, 1::2, :]
            x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)

        elif len(x_shape) == 4:
            b, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
            x0 = x[:, 0::2, 0::2, :]
            x1 = x[:, 1::2, 0::2, :]
            x2 = x[:, 0::2, 1::2, :]
            x3 = x[:, 1::2, 1::2, :]
            x = torch.cat([x0, x1, x2, x3], -1)

        x = self.norm(x)
        # print(x.shape)
        x = self.reduction(x)
        # print(x.shape)
        return x


def compute_mask(dims, window_size, shift_size, device):
    """Computing region masks based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
     Args:
        dims: dimension values.
        window_size: local window size.
        shift_size: shift size.
        device: device.
    """

    cnt = 0

    if len(dims) == 3:
        d, h, w = dims
        img_mask = torch.zeros((1, d, h, w, 1), device=device)
        for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1

    elif len(dims) == 2:
        h, w = dims
        img_mask = torch.zeros((1, h, w, 1), device=device)
        for h in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for w in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                img_mask[:, h, w, :] = cnt
                cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


class BasicLayer(nn.Module):
    """
    Basic Swin Transformer layer in one stage based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Sequence[int],
        drop_path: list,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,  # type: ignore
        downsample: isinstance = None,  # type: ignore
        use_checkpoint: bool = False,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            window_size: local window size.
            drop_path: stochastic depth rate.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            norm_layer: normalization layer.
            downsample: downsample layer at the end of the layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, spatial_dims=len(self.window_size))

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 5:
            b, c, d, h, w = x_shape
            # print(x_shape)
            window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
            x = rearrange(x, "b c d h w -> b d h w c")
            dp = int(np.ceil(d / window_size[0])) * window_size[0]
            hp = int(np.ceil(h / window_size[1])) * window_size[1]
            wp = int(np.ceil(w / window_size[2])) * window_size[2]
            attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x.device)
            for blk in self.blocks:
                x = blk(x, attn_mask)
            x = x.view(b, d, h, w, -1)
            if self.downsample is not None:
                x = self.downsample(x)
            x = rearrange(x, "b d h w c -> b c d h w")

        elif len(x_shape) == 4:
            b, c, h, w = x_shape
            window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
            x = rearrange(x, "b c h w -> b h w c")
            hp = int(np.ceil(h / window_size[0])) * window_size[0]
            wp = int(np.ceil(w / window_size[1])) * window_size[1]
            attn_mask = compute_mask([hp, wp], window_size, shift_size, x.device)
            for blk in self.blocks:
                x = blk(x, attn_mask)
            x = x.view(b, h, w, -1)
            if self.downsample is not None:
                x = self.downsample(x)
            x = rearrange(x, "b h w c -> b c h w")
        return x


class PatchEmbed3D(nn.Module):
    """Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=(4, 4, 4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = UnetBasicBlock(
            spatial_dims=3,
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=3,
            stride=2,
            norm_name=("INSTANCE", {"affine": True}),
        )
        # self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """Forward function."""
        x = self.proj(x)  # B C D Wh Ww
        return x


class SwinTransformer(nn.Module):
    """
    Swin Transformer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        window_size: Sequence[int],
        patch_size: Sequence[int],
        depths: Sequence[int],
        num_heads: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,  # type: ignore
        patch_norm: bool = False,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            in_chans: dimension of input channels.
            embed_dim: number of linear projection output channels.
            window_size: local window size.
            patch_size: patch size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            drop_path_rate: stochastic depth rate.
            norm_layer: normalization layer.
            patch_norm: add normalization after patch embedding.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: spatial dimension.
        """

        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed3D(
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,  # type: ignore
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers3 = nn.ModuleList()
        self.layers4 = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                downsample=PatchMerging,
                use_checkpoint=use_checkpoint,
            )
            if i_layer == 0:
                self.layers1.append(layer)
            elif i_layer == 1:
                self.layers2.append(layer)
            elif i_layer == 2:
                self.layers3.append(layer)
            elif i_layer == 3:
                self.layers4.append(layer)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.size()
            if len(x_shape) == 5:
                n, ch, d, h, w = x_shape
                x = rearrange(x, "n c d h w -> n d h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n d h w c -> n c d h w")
            elif len(x_shape) == 4:
                n, ch, h, w = x_shape
                x = rearrange(x, "n c h w -> n h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n h w c -> n c h w")
        return x

    def forward(self, x, normalize=True):
        x0 = self.patch_embed(x)
        x0 = self.pos_drop(x0)
        x0_out = self.proj_out(x0, normalize)
        x1 = self.layers1[0](x0.contiguous())
        x1_out = self.proj_out(x1, normalize)
        x2 = self.layers2[0](x1.contiguous())
        x2_out = self.proj_out(x2, normalize)
        x3 = self.layers3[0](x2.contiguous())
        x3_out = self.proj_out(x3, normalize)
        x4 = self.layers4[0](x3.contiguous())
        x4_out = self.proj_out(x4, normalize)
        return [x0_out, x1_out, x2_out, x3_out, x4_out]


class SwinTransformerOG(nn.Module):
    """
    Swin Transformer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        window_size: Sequence[int],
        patch_size: Sequence[int],
        depths: Sequence[int],
        num_heads: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,  # type: ignore
        patch_norm: bool = False,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            in_chans: dimension of input channels.
            embed_dim: number of linear projection output channels.
            window_size: local window size.
            patch_size: patch size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            drop_path_rate: stochastic depth rate.
            norm_layer: normalization layer.
            patch_norm: add normalization after patch embedding.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: spatial dimension.
        """

        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,  # type: ignore
            spatial_dims=spatial_dims,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers3 = nn.ModuleList()
        self.layers4 = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                downsample=PatchMerging,
                use_checkpoint=use_checkpoint,
            )
            if i_layer == 0:
                self.layers1.append(layer)
            elif i_layer == 1:
                self.layers2.append(layer)
            elif i_layer == 2:
                self.layers3.append(layer)
            elif i_layer == 3:
                self.layers4.append(layer)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.size()
            if len(x_shape) == 5:
                n, ch, d, h, w = x_shape
                x = rearrange(x, "n c d h w -> n d h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n d h w c -> n c d h w")
            elif len(x_shape) == 4:
                n, ch, h, w = x_shape
                x = rearrange(x, "n c h w -> n h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n h w c -> n c h w")
        return x

    def forward(self, x, normalize=True):
        x0 = self.patch_embed(x)
        x0 = self.pos_drop(x0)
        x0_out = self.proj_out(x0, normalize)
        x1 = self.layers1[0](x0.contiguous())
        x1_out = self.proj_out(x1, normalize)
        x2 = self.layers2[0](x1.contiguous())
        x2_out = self.proj_out(x2, normalize)
        x3 = self.layers3[0](x2.contiguous())
        x3_out = self.proj_out(x3, normalize)
        x4 = self.layers4[0](x3.contiguous())
        x4_out = self.proj_out(x4, normalize)
        return [x0_out, x1_out, x2_out, x3_out, x4_out]
