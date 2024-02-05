from __future__ import print_function

import pdb
import random
from typing import Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer

from .swin_transformer_3d import SwinTransformer3D


class ConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(ConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        # pdb.set_trace()
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError("`features` needs to be [bsz, n_views, ...]," "at least 3 dimensions are required")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class PixelShuffle3D(nn.Module):
    """
    https://github.com/assassint2017/PixelShuffle3D/blob/master/PixelShuffle3D.py
    """

    def __init__(self, upscale_factor):
        super(PixelShuffle3D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, inputs):
        batch_size, channels, in_depth, in_height, in_width = inputs.size()
        channels //= self.upscale_factor**3
        out_depth = in_depth * self.upscale_factor
        out_height = in_height * self.upscale_factor
        out_width = in_width * self.upscale_factor
        input_view = inputs.contiguous().view(
            batch_size,
            channels,
            self.upscale_factor,
            self.upscale_factor,
            self.upscale_factor,
            in_depth,
            in_height,
            in_width,
        )
        shuffle_out = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        return shuffle_out.view(batch_size, channels, out_depth, out_height, out_width)


class SwinTransformerForSimMIM(SwinTransformer3D):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, mean=0.0, std=0.02)
        self.layers = nn.ModuleList([self.layers1, self.layers2, self.layers3, self.layers4])

    def forward(self, x, mask):
        _, _, D, H, W = x.size()
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        assert mask is not None
        B, L, _ = x.shape
        mask_tokens = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        x = x * (1.0 - w) + mask_tokens * w
        x = self.pos_drop(x)
        x = x.view(-1, self.embed_dim, D // self.patch_size[0], H // self.patch_size[1], W // self.patch_size[2])

        for layer in self.layers:
            x = layer[0](x)

        reduction = self.patch_size[0] * 16
        x = x.reshape(-1, (D // reduction) * (H // reduction) * (W // reduction), 2 * self.num_features)
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = x.view(-1, 2 * self.num_features, D // 32, H // 32, W // 32)

        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {"mask_token"}


class SwinTransformerSkipForSimMIM(SwinTransformer3D):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, mean=0.0, std=0.02)
        self.layers1 = nn.ModuleList([self.layers1])
        self.layers2 = nn.ModuleList([self.layers2])
        self.layers3 = nn.ModuleList([self.layers3])
        self.layers4 = nn.ModuleList([self.layers4])

    def forward(self, x, mask, choice):
        x_out = []

        _, _, D, H, W = x.size()
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        B, L, _ = x.shape
        if choice == "mae":
            assert mask is not None
            mask_tokens = self.mask_token.expand(B, L, -1)
            w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
            x = x * (1.0 - w) + mask_tokens * w

        x = self.pos_drop(x)
        x = x.view(-1, self.embed_dim, D // self.patch_size[0], H // self.patch_size[1], W // self.patch_size[2])
        # pdb.set_trace()

        if choice == "sld" or choice == "all":
            # pdb.set_trace()
            rand_choice = random.sample(range(0, x.shape[1]), int(0.6 * x.shape[1]))
            mask = torch.ones(x.shape).cuda()
            mask[:, rand_choice, :, :, :] = 0
            x = x * mask

        if choice == "sld_noise":
            # pdb.set_trace()
            rand_choice = random.sample(range(0, x.shape[1]), int(0.6 * x.shape[1]))
            mask = torch.ones(x.shape).cuda()

            B, C, H, W, Z = mask.shape
            noise = (0.1**0.5) * torch.randn(B, int(0.6 * C), H, W, Z).cuda()
            # pdb.set_trace()
            mask[:, rand_choice, :, :, :] = noise
            x = x * mask

        x_out.append(x)
        for layer in self.layers1:
            x = layer[0](x)

        # if choice=="sld":
        #     # pdb.set_trace()
        #     rand_choice = random.sample(range(0, x.shape[1]), int(0.2*x.shape[1]))
        #     mask = torch.ones(x.shape).cuda()
        #     mask[:,rand_choice,:,:,:] = 0
        #     x = x* mask

        x_out.append(x)

        for layer in self.layers2:
            x = layer[0](x)

        # if choice=="sld":
        #     # pdb.set_trace()
        #     rand_choice = random.sample(range(0, x.shape[1]), int(0.2*x.shape[1]))
        #     mask = torch.ones(x.shape).cuda()
        #     mask[:,rand_choice,:,:,:] = 0
        #     x = x* mask

        x_out.append(x)

        for layer in self.layers3:
            x = layer[0](x)

        # if choice=="sld":
        #     # pdb.set_trace()
        #     rand_choice = random.sample(range(0, x.shape[1]), int(0.2*x.shape[1]))
        #     mask = torch.ones(x.shape).cuda()
        #     mask[:,rand_choice,:,:,:] = 0
        #     x = x* mask

        x_out.append(x)

        for layer in self.layers4:
            x = layer[0](x)

        # if choice=="sld":
        #     # pdb.set_trace()
        #     rand_choice = random.sample(range(0, x.shape[1]), int(0.2*x.shape[1]))
        #     mask = torch.ones(x.shape).cuda()
        #     mask[:,rand_choice,:,:,:] = 0
        #     x = x* mask

        x_out.append(x)
        reduction = self.patch_size[0] * 16
        x = x.reshape(-1, (D // reduction) * (H // reduction) * (W // reduction), 2 * self.num_features)
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = x.view(-1, 2 * self.num_features, D // 32, H // 32, W // 32)

        return x, x_out

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {"mask_token"}


class DecoderForSIM(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.upsample1 = nn.Sequential(
            nn.Conv3d(1536, 384, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(384),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear"),
        )

        self.upsample2 = nn.Sequential(
            nn.Conv3d(768, 192, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(192),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear"),
        )

        self.upsample3 = nn.Sequential(
            nn.Conv3d(384, 96, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(96),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear"),
        )

        self.upsample4 = nn.Sequential(
            nn.Conv3d(192, 48, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(48),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear"),
        )

        self.upsample5 = nn.Sequential(
            nn.Conv3d(96, 24, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(24),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear"),
        )

        self.out = nn.Sequential(nn.Conv3d(24, 1, kernel_size=1, stride=1))

    def forward(self, z, z_out):
        z4 = self.upsample1(torch.cat((z, z_out[4]), 1))
        z3 = self.upsample2(torch.cat((z4, z_out[3]), 1))
        z2 = self.upsample3(torch.cat((z3, z_out[2]), 1))
        z1 = self.upsample4(torch.cat((z2, z_out[1]), 1))
        x_rec = self.out(self.upsample5(torch.cat((z1, z_out[0]), 1)))
        return x_rec

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {"mask_token"}


class SimMIM(nn.Module):
    def __init__(self, encoder, encoder_stride, decoder="deconv", loss="mask_only"):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride
        self.decoder = decoder
        self.loss = loss

        if decoder == "pixel_shuffle":
            self.conv1 = nn.Conv3d(
                in_channels=2 * self.encoder.num_features, out_channels=self.encoder_stride**3 * 1, kernel_size=1
            )
            self.pixel_shuffle = PixelShuffle3D(self.encoder_stride)
        elif decoder == "deconv":
            self.transp_conv1 = nn.ConvTranspose3d(768, 384, 2, stride=2)
            self.transp_conv2 = nn.ConvTranspose3d(384, 192, 2, stride=2)
            self.transp_conv3 = nn.ConvTranspose3d(192, 96, 2, stride=2)
            self.transp_conv4 = nn.ConvTranspose3d(96, 48, 2, stride=2)
            self.transp_conv5 = nn.ConvTranspose3d(48, 1, 2, stride=2)
            self.conv = nn.Conv3d(1, 1, kernel_size=1, stride=1)

        elif decoder == "upsample":
            self.conv_block1 = UnetResBlock(3, 768, 384, kernel_size=3, stride=1, norm_name="instance")
            self.conv_block2 = UnetResBlock(3, 384, 192, kernel_size=3, stride=1, norm_name="instance")
            self.conv_block3 = UnetResBlock(3, 192, 96, kernel_size=3, stride=1, norm_name="instance")
            self.conv_block4 = UnetResBlock(3, 96, 48, kernel_size=3, stride=1, norm_name="instance")
            self.conv_block5 = UnetResBlock(3, 48, 1, kernel_size=3, stride=1, norm_name="instance")
            self.conv_block6 = nn.Conv3d(1, 1, kernel_size=1, stride=1)
            self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
            # self.act = nn.ReLU()
            # self.act = nn.Sigmoid()
        elif decoder == "vae":
            self.upsample = nn.Sequential(
                nn.Conv3d(768, 384, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(384),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear"),
                nn.Conv3d(384, 192, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(192),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear"),
                nn.Conv3d(192, 96, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(96),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear"),
                nn.Conv3d(96, 48, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(48),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear"),
                nn.Conv3d(48, 1, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(48),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear"),
                nn.Conv3d(1, 1, kernel_size=1, stride=1),
                nn.Tanh(),
            )
        elif decoder == "vae2":
            self.upsample = nn.Sequential(
                nn.Conv3d(768, 384, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(384),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear"),
                nn.Conv3d(384, 192, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(192),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear"),
                nn.Conv3d(192, 96, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(96),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear"),
                nn.Conv3d(96, 48, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(48),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear"),
                nn.Conv3d(48, 24, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(24),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear"),
                nn.Conv3d(24, 1, kernel_size=1, stride=1),
            )

        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size

    def forward(self, x, mask, x_org):
        z = self.encoder(x, mask)
        if self.decoder == "pixel_shuffle":
            x_rec = self.pixel_shuffle(self.conv1(z))
        elif self.decoder == "deconv":
            x_rec = self.conv(
                self.transp_conv5(self.transp_conv4(self.transp_conv3(self.transp_conv2(self.transp_conv1(z)))))
            )
        elif self.decoder == "upsample":
            x_rec = self.upsample(
                self.conv_block5(
                    self.upsample(
                        self.conv_block4(
                            self.upsample(
                                self.conv_block3(self.upsample(self.conv_block2(self.upsample(self.conv_block1(z)))))
                            )
                        )
                    )
                )
            )
            x_rec = self.conv_block6(x_rec)
        elif self.decoder == "vae":
            x_rec = self.upsample(z)
        elif self.decoder == "vae2":
            x_rec = self.upsample(z)

        mask = (
            mask.repeat_interleave(self.patch_size[0], 1)
            .repeat_interleave(self.patch_size[1], 2)
            .repeat_interleave(self.patch_size[2], 3)
            .unsqueeze(1)
            .contiguous()
        )
        loss_recon = F.l1_loss(x_org, x_rec, reduction="none")
        if self.loss == "mask_only":
            loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        elif self.loss == "all_img":
            loss = loss_recon

        return loss, x_rec, mask

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, "no_weight_decay"):
            return {"encoder." + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, "no_weight_decay_keywords"):
            return {"encoder." + i for i in self.encoder.no_weight_decay_keywords()}
        return {}


class SimMIMSkip(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        encoder,
        encoder_stride,
        img_size: Union[Sequence[int], int],
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
        decoder="deconv",
        loss="mask_only",
        choice="mae",
        inf="notsim",
        temperature=0.07,
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride
        self.decoder = decoder
        self.loss = loss

        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size

        # add UNETR blocks

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

        self.choice = choice
        self.inf = inf
        self.temp = temperature
        self.contrast = ConLoss(temperature=self.temp)
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels
        )  # type: ignore

    def get_image_prior_losses(self, inputs_jit):
        # COMPUTE total variation regularization loss
        diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
        diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
        diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
        diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

        loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
        loss_var_l1 = (
            (diff1.abs() / 255.0).mean()
            + (diff2.abs() / 255.0).mean()
            + (diff3.abs() / 255.0).mean()
            + (diff4.abs() / 255.0).mean()
        )
        loss_var_l1 = loss_var_l1 * 255.0
        return loss_var_l1, loss_var_l2

    def forward(self, x, mask, x_org, cl_type):
        # pdb.set_trace()
        choice = self.choice
        inf = self.inf

        cont_loss = self.contrast

        z, hidden_states_out = self.encoder(x, mask, choice)
        # if inf=="sim":
        #     return hidden_states_out
        mask = (
            mask.repeat_interleave(self.patch_size[0], 1)
            .repeat_interleave(self.patch_size[1], 2)
            .repeat_interleave(self.patch_size[2], 3)
            .unsqueeze(1)
            .contiguous()
        )
        if choice == "mae":
            x = x * mask
        enc0 = self.encoder1(x)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        # pdb.set_trace()
        out = self.decoder1(dec0, enc0)
        x_rec = self.out(out)

        # mask = mask.repeat_interleave(self.patch_size[0], 1).repeat_interleave(self.patch_size[1], 2).repeat_interleave(self.patch_size[2], 3).unsqueeze(1).contiguous()
        # pdb.set_trace()

        # Implement MM Cont Loss

        mask_tmp = []
        for i in range(len(cl_type)):
            tmp_val = cl_type[i]
            if tmp_val == "ct":
                mask_tmp.append(0)
            elif tmp_val == "t1":
                mask_tmp.append(1)
            elif tmp_val == "t1ce":
                mask_tmp.append(2)
            elif tmp_val == "t2":
                mask_tmp.append(3)
            elif tmp_val == "flair":
                mask_tmp.append(4)

        _, b = np.unique(np.array(mask_tmp), return_inverse=True)

        label_tmp = torch.tensor(b).cuda()

        bsz = hidden_states_out[4].shape[0]
        # pdb.set_trace()
        logits_con = torch.einsum(
            "i d, j d -> i j", hidden_states_out[4].reshape(bsz, -1), hidden_states_out[4].reshape(bsz, -1)
        ) * torch.exp(torch.tensor(self.temp))
        loss_t = F.cross_entropy(logits_con, label_tmp)
        loss_i = F.cross_entropy(logits_con.T, label_tmp)

        loss_cont = (loss_t + loss_i) / 2

        # End MM Cont Loss

        if self.loss == "mask_only":
            loss_recon = F.l1_loss(x_org, x_rec, reduction="none")
            loss_recon = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
            # _, loss_var_l2 = self.get_image_prior_losses(x_rec)
            # bs = x.shape[0]
            # img_l2_loss = torch.norm(x_rec.view(bs, -1), dim=1).mean()
            # loss = loss + 0.0001*loss_var_l2+1e-5*img_l2_loss

        elif self.loss == "all_img":
            loss_recon = F.l1_loss(x_org, x_rec, reduction="mean")
            # loss = loss_recon

        elif self.loss == "l2":
            loss = F.mse_loss(x_org, x_rec, reduction="mean")
            _, loss_var_l2 = self.get_image_prior_losses(x_rec)
            bs = x.shape[0]
            img_l2_loss = torch.norm(x_rec.view(bs, -1), dim=1).mean()
            loss = loss + 0.0001 * loss_var_l2 + 1e-5 * img_l2_loss

        # print(loss_recon,loss_cont)
        return loss_recon, loss_cont, x_rec, mask

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, "no_weight_decay"):
            return {"encoder." + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, "no_weight_decay_keywords"):
            return {"encoder." + i for i in self.encoder.no_weight_decay_keywords()}
        return {}


class SimMIMSkip_light(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        encoder,
        encoder_stride,
        img_size: Union[Sequence[int], int],
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
        decoder="deconv",
        loss="mask_only",
        choice="mae",
        inf="sim",
        temperature=0.07,
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride
        self.decoder = decoder
        self.loss = loss

        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size

        # add UNETR blocks

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

        if decoder == "pixel_shuffle":
            self.conv1 = nn.Conv3d(
                in_channels=2 * self.encoder.num_features, out_channels=self.encoder_stride**3 * 1, kernel_size=1
            )
            self.pixel_shuffle = PixelShuffle3D(self.encoder_stride)
        elif decoder == "deconv":
            self.transp_conv1 = nn.ConvTranspose3d(768, 384, 2, stride=2)
            self.transp_conv2 = nn.ConvTranspose3d(384, 192, 2, stride=2)
            self.transp_conv3 = nn.ConvTranspose3d(192, 96, 2, stride=2)
            self.transp_conv4 = nn.ConvTranspose3d(96, 48, 2, stride=2)
            self.transp_conv5 = nn.ConvTranspose3d(48, 1, 2, stride=2)
            self.conv = nn.Conv3d(1, 1, kernel_size=1, stride=1)

        elif decoder == "upsample":
            self.conv_block1 = UnetResBlock(3, 768, 384, kernel_size=3, stride=1, norm_name="instance")
            self.conv_block2 = UnetResBlock(3, 384, 192, kernel_size=3, stride=1, norm_name="instance")
            self.conv_block3 = UnetResBlock(3, 192, 96, kernel_size=3, stride=1, norm_name="instance")
            self.conv_block4 = UnetResBlock(3, 96, 48, kernel_size=3, stride=1, norm_name="instance")
            self.conv_block5 = UnetResBlock(3, 48, 1, kernel_size=3, stride=1, norm_name="instance")
            self.conv_block6 = nn.Conv3d(1, 1, kernel_size=1, stride=1)
            self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
            # self.act = nn.ReLU()
            # self.act = nn.Sigmoid()
        elif decoder == "vae":
            self.upsample = nn.Sequential(
                nn.Conv3d(768, 384, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(384),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear"),
                nn.Conv3d(384, 192, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(192),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear"),
                nn.Conv3d(192, 96, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(96),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear"),
                nn.Conv3d(96, 48, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(48),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear"),
                nn.Conv3d(48, 1, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(48),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear"),
                nn.Conv3d(1, 1, kernel_size=1, stride=1),
                nn.Tanh(),
            )
        elif decoder == "vae2":
            self.upsample1 = nn.Sequential(
                nn.Conv3d(1536, 384, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(384),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear"),
            )

            self.upsample2 = nn.Sequential(
                nn.Conv3d(768, 192, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(192),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear"),
            )

            self.upsample3 = nn.Sequential(
                nn.Conv3d(384, 96, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(96),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear"),
            )

            self.upsample4 = nn.Sequential(
                nn.Conv3d(192, 48, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(48),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear"),
            )

            self.upsample5 = nn.Sequential(
                nn.Conv3d(96, 24, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(24),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear"),
            )

            self.out = nn.Sequential(nn.Conv3d(24, 1, kernel_size=1, stride=1))

        self.choice = choice
        self.inf = inf
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels
        )  # type: ignore

    def get_image_prior_losses(self, inputs_jit):
        # COMPUTE total variation regularization loss
        diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
        diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
        diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
        diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

        loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
        loss_var_l1 = (
            (diff1.abs() / 255.0).mean()
            + (diff2.abs() / 255.0).mean()
            + (diff3.abs() / 255.0).mean()
            + (diff4.abs() / 255.0).mean()
        )
        loss_var_l1 = loss_var_l1 * 255.0
        return loss_var_l1, loss_var_l2

    def forward(self, x, mask, x_org):
        # pdb.set_trace()
        choice = self.choice
        inf = self.inf

        z, hidden_states_out = self.encoder(x, mask, choice)

        if inf == "sim":
            return hidden_states_out

        if self.decoder == "pixel_shuffle":
            x_rec = self.pixel_shuffle(self.conv1(z))
        elif self.decoder == "deconv":
            x_rec = self.conv(
                self.transp_conv5(self.transp_conv4(self.transp_conv3(self.transp_conv2(self.transp_conv1(z)))))
            )
        elif self.decoder == "upsample":
            x_rec = self.upsample(
                self.conv_block5(
                    self.upsample(
                        self.conv_block4(
                            self.upsample(
                                self.conv_block3(self.upsample(self.conv_block2(self.upsample(self.conv_block1(z)))))
                            )
                        )
                    )
                )
            )
            x_rec = self.conv_block6(x_rec)
        elif self.decoder == "vae":
            x_rec = self.upsample(z)
        elif self.decoder == "vae2":
            x_rec = self.upsample(z)

        mask = (
            mask.repeat_interleave(self.patch_size[0], 1)
            .repeat_interleave(self.patch_size[1], 2)
            .repeat_interleave(self.patch_size[2], 3)
            .unsqueeze(1)
            .contiguous()
        )

        mask_tmp = []
        for i in range(len(cl_type)):
            tmp_val = cl_type[i]
            if tmp_val == "ct":
                mask_tmp.append(0)
            elif tmp_val == "t1":
                mask_tmp.append(1)
            elif tmp_val == "t1ce":
                mask_tmp.append(2)
            elif tmp_val == "t2":
                mask_tmp.append(3)
            elif tmp_val == "flair":
                mask_tmp.append(4)

        _, b = np.unique(np.array(mask_tmp), return_inverse=True)

        label_tmp = torch.tensor(b).cuda()

        bsz = hidden_states_out[4].shape[0]
        # pdb.set_trace()
        logits_con = (
            torch.einsum(
                "i d, j d -> i j", hidden_states_out[4].reshape(bsz, -1), hidden_states_out[4].reshape(bsz, -1)
            )
            * self.temp
        )
        loss_t = F.cross_entropy(logits_con, label_tmp)
        loss_i = F.cross_entropy(logits_con.T, label_tmp)

        loss_cont = (loss_t + loss_i) / 2

        if self.loss == "mask_only":
            loss_recon = F.l1_loss(x_org, x_rec, reduction="none")
            loss_recon = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
            # _, loss_var_l2 = self.get_image_prior_losses(x_rec)
            # bs = x.shape[0]
            # img_l2_loss = torch.norm(x_rec.view(bs, -1), dim=1).mean()
            # loss = loss + 0.0001*loss_var_l2+1e-5*img_l2_loss

        elif self.loss == "all_img":
            loss_recon = F.l1_loss(x_org, x_rec, reduction="mean")
            # loss = loss_recon

        elif self.loss == "l2":
            loss = F.mse_loss(x_org, x_rec, reduction="mean")
            _, loss_var_l2 = self.get_image_prior_losses(x_rec)
            bs = x.shape[0]
            img_l2_loss = torch.norm(x_rec.view(bs, -1), dim=1).mean()
            loss = loss + 0.0001 * loss_var_l2 + 1e-5 * img_l2_loss

        return loss_recon, loss_cont, x_rec, mask

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, "no_weight_decay"):
            return {"encoder." + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, "no_weight_decay_keywords"):
            return {"encoder." + i for i in self.encoder.no_weight_decay_keywords()}
        return {}


class UnetUpBlock(nn.Module):
    """
    An upsampling module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.
    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        upsample_kernel_size: convolution kernel size for transposed convolution layers.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.
        trans_bias: transposed convolution bias.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
        trans_bias: bool = False,
    ):
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            dropout=dropout,
            bias=trans_bias,
            act=None,
            norm=None,
            conv_only=False,
            is_transposed=True,
        )
        self.conv_block = UnetBasicBlock(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            norm_name=norm_name,
            act_name=act_name,
        )

    def forward(self, inp):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        # out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out


class SimMIMSkip2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        encoder,
        encoder_stride,
        img_size: Union[Sequence[int], int],
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
        decoder="deconv",
        loss="mask_only",
        choice="mae",
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride
        self.decoder = decoder
        self.loss = loss

        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size

        # add UNETR blocks

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

        self.decoder1 = UnetUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            stride=1,
            # res_block=True,
        )

        self.choice = choice

        self.out = UnetOutBlock(
            spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels
        )  # type: ignore

    def get_image_prior_losses(self, inputs_jit):
        # COMPUTE total variation regularization loss
        diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
        diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
        diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
        diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

        loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
        loss_var_l1 = (
            (diff1.abs() / 255.0).mean()
            + (diff2.abs() / 255.0).mean()
            + (diff3.abs() / 255.0).mean()
            + (diff4.abs() / 255.0).mean()
        )
        loss_var_l1 = loss_var_l1 * 255.0
        return loss_var_l1, loss_var_l2

    def forward(self, x, mask, x_org):
        # pdb.set_trace()
        choice = self.choice
        z, hidden_states_out = self.encoder(x, mask, choice)
        mask = (
            mask.repeat_interleave(self.patch_size[0], 1)
            .repeat_interleave(self.patch_size[1], 2)
            .repeat_interleave(self.patch_size[2], 3)
            .unsqueeze(1)
            .contiguous()
        )
        if choice == "mae":
            x = x * mask
        enc0 = self.encoder1(x)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        # pdb.set_trace()
        out = self.decoder1(dec0)
        x_rec = self.out(out)

        # mask = mask.repeat_interleave(self.patch_size[0], 1).repeat_interleave(self.patch_size[1], 2).repeat_interleave(self.patch_size[2], 3).unsqueeze(1).contiguous()

        if self.loss == "mask_only":
            loss_recon = F.l1_loss(x_org, x_rec, reduction="none")
            loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
            # _, loss_var_l2 = self.get_image_prior_losses(x_rec)
            # bs = x.shape[0]
            # img_l2_loss = torch.norm(x_rec.view(bs, -1), dim=1).mean()
            # loss = loss + 0.0001*loss_var_l2+1e-5*img_l2_loss

        elif self.loss == "all_img":
            loss_recon = F.l1_loss(x_org, x_rec, reduction="mean")
            loss = loss_recon

        elif self.loss == "l2":
            loss = F.mse_loss(x_org, x_rec, reduction="mean")
            _, loss_var_l2 = self.get_image_prior_losses(x_rec)
            bs = x.shape[0]
            img_l2_loss = torch.norm(x_rec.view(bs, -1), dim=1).mean()
            loss = loss + 0.0001 * loss_var_l2 + 1e-5 * img_l2_loss

        return loss, x_rec, mask

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, "no_weight_decay"):
            return {"encoder." + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, "no_weight_decay_keywords"):
            return {"encoder." + i for i in self.encoder.no_weight_decay_keywords()}
        return {}


def build_simmim(args):
    model_type = args.model_type
    if model_type == "swin":
        encoder = SwinTransformerForSimMIM(
            num_classes=args.num_classes,
            img_size=args.img_size,
            patch_size=args.patch_size,
            in_chans=args.in_channels,
            embed_dim=args.embed_dim,
            depths=args.depths,
            num_heads=args.num_heads,
            window_size=args.window_size,
            mlp_ratio=args.mlp_ratio,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=args.drop_rate,
            drop_path_rate=args.drop_path_rate,
            use_checkpoint=args.use_grad_checkpoint,
            patch_norm=True,
        )
        encoder_stride = 32
        model = SimMIM(encoder=encoder, encoder_stride=encoder_stride, decoder=args.decoder, loss=args.loss_type)

    elif model_type == "swin_skip":
        encoder = SwinTransformerSkipForSimMIM(
            num_classes=args.num_classes,
            img_size=args.img_size,
            patch_size=args.patch_size,
            in_chans=args.in_channels,
            embed_dim=args.embed_dim,
            depths=args.depths,
            num_heads=args.num_heads,
            window_size=args.window_size,
            mlp_ratio=args.mlp_ratio,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=args.drop_rate,
            drop_path_rate=args.drop_path_rate,
            use_checkpoint=args.use_grad_checkpoint,
            patch_norm=True,
        )
        encoder_stride = 32
        model = SimMIMSkip(
            encoder=encoder,
            encoder_stride=encoder_stride,
            loss=args.loss_type,
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=args.dropout_path_rate,
            use_checkpoint=args.use_checkpoint,
            choice=args.choice,
            temperature=args.temperature,
        )
    elif model_type == "swin_skip2":
        encoder = SwinTransformerSkipForSimMIM(
            num_classes=args.num_classes,
            img_size=args.img_size,
            patch_size=args.patch_size,
            in_chans=args.in_channels,
            embed_dim=args.embed_dim,
            depths=args.depths,
            num_heads=args.num_heads,
            window_size=args.window_size,
            mlp_ratio=args.mlp_ratio,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=args.drop_rate,
            drop_path_rate=args.drop_path_rate,
            use_checkpoint=args.use_grad_checkpoint,
            patch_norm=True,
        )
        encoder_stride = 32
        model = SimMIMSkip_light(
            encoder=encoder,
            encoder_stride=encoder_stride,
            loss=args.loss_type,
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=args.dropout_path_rate,
            use_checkpoint=args.use_checkpoint,
            choice=args.choice,
            temperature=args.temperature,
        )
    else:
        raise NotImplementedError(f"Unknown pre-train model: {model_type}")

    return model
