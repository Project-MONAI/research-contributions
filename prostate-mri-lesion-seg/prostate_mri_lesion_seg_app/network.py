#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "RRUNet3D",
    "UNet3D",
    "UNet3D_BatchNorm",
    "UNet3D_BN_Multiple",
    "UNet3D_Multiple",
    "UNet3D_Multiple_Cascade",
    "UNet3D_Multiple_Pool4",
    "UNet3D_Multiple_PPM",
    "UNetRecon3D",
]


class UNetRecon3D(nn.Module):
    def __init__(self, in_channels, out_channels, num_ops):
        super(UNetRecon3D, self).__init__()
        self.encoder_cell1 = self.conv_cell(in_channels, 16, num_ops)
        self.encoder_cell2 = self.conv_cell(16, 32, num_ops)
        self.encoder_cell3 = self.conv_cell(32, 64, num_ops)
        self.encoder_cell4 = self.conv_cell(64, 128, num_ops)

        self.decoder_cell3 = self.conv_cell(192, 64, num_ops)
        self.decoder_cell2 = self.conv_cell(96, 32, num_ops)
        self.decoder_cell1 = self.conv_cell(48, 16, num_ops)

        self.recon_decoder_cell3 = self.conv_cell(128, 64, num_ops)
        self.recon_decoder_cell2 = self.conv_cell(64, 32, num_ops)
        self.recon_decoder_cell1 = self.conv_cell(32, 16, num_ops)

        self.output_conv = nn.Conv3d(16, out_channels, 1, bias=False)
        self.recon_output_conv = nn.Conv3d(16, in_channels, 1, bias=False)

        self.output_activation = nn.Softmax(dim=1)
        self.pool = nn.MaxPool3d(2, 2)
        self.unpool = nn.Upsample(scale_factor=2)

    def conv_cell(self, in_channels, out_channels, num_ops):
        cell = []
        for j in range(num_ops):
            if j == 0:
                cell.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
            else:
                cell.append(nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1))
            cell.append(nn.InstanceNorm3d(out_channels))
            cell.append(nn.ReLU(inplace=True))

        return nn.Sequential(*cell)

    def forward(self, x):
        e1 = self.encoder_cell1(x)
        x = self.pool(e1)
        e2 = self.encoder_cell2(x)
        x = self.pool(e2)
        e3 = self.encoder_cell3(x)
        x = self.pool(e3)
        e4 = self.encoder_cell4(x)

        x = self.unpool(e4)
        d3 = self.decoder_cell3(torch.cat([x, e3], dim=1))
        x = self.unpool(d3)
        d2 = self.decoder_cell2(torch.cat([x, e2], dim=1))
        x = self.unpool(d2)
        d1 = self.decoder_cell1(torch.cat([x, e1], dim=1))
        x = self.output_conv(d1)
        x = self.output_activation(x)

        x0 = self.unpool(e4)
        recon_d3 = self.recon_decoder_cell3(x0)
        x0 = self.unpool(recon_d3)
        recon_d2 = self.recon_decoder_cell2(x0)
        x0 = self.unpool(recon_d2)
        recon_d1 = self.recon_decoder_cell1(x0)
        x0 = self.recon_output_conv(recon_d1)

        return [x, x0]


class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, num_ops):
        super(UNet3D, self).__init__()
        self.encoder_cell1 = self.conv_cell(in_channels, 16, num_ops)
        self.encoder_cell2 = self.conv_cell(16, 32, num_ops)
        self.encoder_cell3 = self.conv_cell(32, 64, num_ops)
        self.encoder_cell4 = self.conv_cell(64, 128, num_ops)

        self.decoder_cell3 = self.conv_cell(192, 64, num_ops)
        self.decoder_cell2 = self.conv_cell(96, 32, num_ops)
        self.decoder_cell1 = self.conv_cell(48, 16, num_ops)

        self.output_conv = nn.Conv3d(16, out_channels, 1, bias=False)

        self.output_activation = nn.Softmax(dim=1)
        self.pool = nn.MaxPool3d(2, 2)
        self.unpool = nn.Upsample(scale_factor=2)

    def conv_cell(self, in_channels, out_channels, num_ops):
        cell = []
        for j in range(num_ops):
            if j == 0:
                cell.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
            else:
                cell.append(nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1))
            cell.append(nn.InstanceNorm3d(out_channels))
            cell.append(nn.ReLU(inplace=True))

        return nn.Sequential(*cell)

    def forward(self, x):
        e1 = self.encoder_cell1(x)
        x = self.pool(e1)
        e2 = self.encoder_cell2(x)
        x = self.pool(e2)
        e3 = self.encoder_cell3(x)
        x = self.pool(e3)
        e4 = self.encoder_cell4(x)

        x = self.unpool(e4)
        d3 = self.decoder_cell3(torch.cat([x, e3], dim=1))
        x = self.unpool(d3)
        d2 = self.decoder_cell2(torch.cat([x, e2], dim=1))
        x = self.unpool(d2)
        d1 = self.decoder_cell1(torch.cat([x, e1], dim=1))
        x = self.output_conv(d1)
        x = self.output_activation(x)

        return x


class UNet3D_BatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, num_ops):
        super(UNet3D_BatchNorm, self).__init__()
        self.encoder_cell1 = self.conv_cell(in_channels, 16, num_ops)
        self.encoder_cell2 = self.conv_cell(16, 32, num_ops)
        self.encoder_cell3 = self.conv_cell(32, 64, num_ops)
        self.encoder_cell4 = self.conv_cell(64, 128, num_ops)

        self.decoder_cell3 = self.conv_cell(192, 64, num_ops)
        self.decoder_cell2 = self.conv_cell(96, 32, num_ops)
        self.decoder_cell1 = self.conv_cell(48, 16, num_ops)

        self.output_conv = nn.Conv3d(16, out_channels, 1, bias=False)

        self.output_activation = nn.Softmax(dim=1)
        self.pool = nn.MaxPool3d(2, 2)
        self.unpool = nn.Upsample(scale_factor=2)

    def conv_cell(self, in_channels, out_channels, num_ops):
        cell = []
        for j in range(num_ops):
            if j == 0:
                cell.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
            else:
                cell.append(nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1))
            cell.append(nn.BatchNorm3d(out_channels))
            cell.append(nn.ReLU(inplace=True))

        return nn.Sequential(*cell)

    def forward(self, x):
        e1 = self.encoder_cell1(x)
        x = self.pool(e1)
        e2 = self.encoder_cell2(x)
        x = self.pool(e2)
        e3 = self.encoder_cell3(x)
        x = self.pool(e3)
        e4 = self.encoder_cell4(x)

        x = self.unpool(e4)
        d3 = self.decoder_cell3(torch.cat([x, e3], dim=1))
        x = self.unpool(d3)
        d2 = self.decoder_cell2(torch.cat([x, e2], dim=1))
        x = self.unpool(d2)
        d1 = self.decoder_cell1(torch.cat([x, e1], dim=1))
        x = self.output_conv(d1)
        x = self.output_activation(x)

        return x


class UNet3D_Multiple(nn.Module):
    def __init__(self, in_channels, out_channels, num_ops, multiple=16):
        super(UNet3D_Multiple, self).__init__()
        self.encoder_cell1 = self.conv_cell(in_channels, multiple * 1, num_ops)
        self.encoder_cell2 = self.conv_cell(multiple * 1, multiple * 2, num_ops)
        self.encoder_cell3 = self.conv_cell(multiple * 2, multiple * 4, num_ops)
        self.encoder_cell4 = self.conv_cell(multiple * 4, multiple * 8, num_ops)

        self.decoder_cell3 = self.conv_cell(multiple * 12, multiple * 4, num_ops)
        self.decoder_cell2 = self.conv_cell(multiple * 6, multiple * 2, num_ops)
        self.decoder_cell1 = self.conv_cell(multiple * 3, multiple * 1, num_ops)

        self.output_conv = nn.Conv3d(multiple * 1, out_channels, 1, bias=False)

        self.output_activation = nn.Softmax(dim=1)
        self.pool = nn.MaxPool3d(2, 2)
        self.unpool = nn.Upsample(scale_factor=2)

    def conv_cell(self, in_channels, out_channels, num_ops):
        cell = []
        for j in range(num_ops):
            if j == 0:
                cell.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
            else:
                cell.append(nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1))
            cell.append(nn.InstanceNorm3d(out_channels))
            cell.append(nn.ReLU(inplace=True))

        return nn.Sequential(*cell)

    def forward(self, x):
        e1 = self.encoder_cell1(x)
        x = self.pool(e1)
        e2 = self.encoder_cell2(x)
        x = self.pool(e2)
        e3 = self.encoder_cell3(x)
        x = self.pool(e3)
        e4 = self.encoder_cell4(x)

        x = self.unpool(e4)
        d3 = self.decoder_cell3(torch.cat([x, e3], dim=1))
        x = self.unpool(d3)
        d2 = self.decoder_cell2(torch.cat([x, e2], dim=1))
        x = self.unpool(d2)
        d1 = self.decoder_cell1(torch.cat([x, e1], dim=1))
        x = self.output_conv(d1)
        x = self.output_activation(x)

        return x


class UNet3D_BN_Multiple(nn.Module):
    def __init__(self, in_channels, out_channels, num_ops, multiple=16):
        super(UNet3D_BN_Multiple, self).__init__()
        self.encoder_cell1 = self.conv_cell(in_channels, multiple * 1, num_ops)
        self.encoder_cell2 = self.conv_cell(multiple * 1, multiple * 2, num_ops)
        self.encoder_cell3 = self.conv_cell(multiple * 2, multiple * 4, num_ops)
        self.encoder_cell4 = self.conv_cell(multiple * 4, multiple * 8, num_ops)

        self.decoder_cell3 = self.conv_cell(multiple * 12, multiple * 4, num_ops)
        self.decoder_cell2 = self.conv_cell(multiple * 6, multiple * 2, num_ops)
        self.decoder_cell1 = self.conv_cell(multiple * 3, multiple * 1, num_ops)

        self.output_conv = nn.Conv3d(multiple * 1, out_channels, 1, bias=False)

        self.output_activation = nn.Softmax(dim=1)
        self.pool = nn.MaxPool3d(2, 2)
        self.unpool = nn.Upsample(scale_factor=2)

    def conv_cell(self, in_channels, out_channels, num_ops):
        cell = []
        for j in range(num_ops):
            if j == 0:
                cell.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
            else:
                cell.append(nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1))
            cell.append(nn.BatchNorm3d(out_channels))
            cell.append(nn.ReLU(inplace=True))

        return nn.Sequential(*cell)

    def forward(self, x):
        e1 = self.encoder_cell1(x)
        x = self.pool(e1)
        e2 = self.encoder_cell2(x)
        x = self.pool(e2)
        e3 = self.encoder_cell3(x)
        x = self.pool(e3)
        e4 = self.encoder_cell4(x)

        x = self.unpool(e4)
        d3 = self.decoder_cell3(torch.cat([x, e3], dim=1))
        x = self.unpool(d3)
        d2 = self.decoder_cell2(torch.cat([x, e2], dim=1))
        x = self.unpool(d2)
        d1 = self.decoder_cell1(torch.cat([x, e1], dim=1))
        x = self.output_conv(d1)
        x = self.output_activation(x)

        return x


class UNet3D_Multiple_Pool4(nn.Module):
    def __init__(self, in_channels, out_channels, num_ops, multiple=16):
        super(UNet3D_Multiple_Pool4, self).__init__()
        self.encoder_cell1 = self.conv_cell(in_channels, multiple * 1, num_ops)
        self.encoder_cell2 = self.conv_cell(multiple * 1, multiple * 2, num_ops)
        self.encoder_cell3 = self.conv_cell(multiple * 2, multiple * 4, num_ops)
        self.encoder_cell4 = self.conv_cell(multiple * 4, multiple * 8, num_ops)
        self.encoder_cell5 = self.conv_cell(multiple * 8, multiple * 16, num_ops)

        self.decoder_cell4 = self.conv_cell(multiple * 24, multiple * 8, num_ops)
        self.decoder_cell3 = self.conv_cell(multiple * 12, multiple * 4, num_ops)
        self.decoder_cell2 = self.conv_cell(multiple * 6, multiple * 2, num_ops)
        self.decoder_cell1 = self.conv_cell(multiple * 3, multiple * 1, num_ops)

        self.output_conv = nn.Conv3d(multiple * 1, out_channels, 1, bias=False)

        self.output_activation = nn.Softmax(dim=1)
        self.pool = nn.MaxPool3d(2, 2)
        self.unpool = nn.Upsample(scale_factor=2)

    def conv_cell(self, in_channels, out_channels, num_ops):
        cell = []
        for j in range(num_ops):
            if j == 0:
                cell.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
            else:
                cell.append(nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1))
            cell.append(nn.InstanceNorm3d(out_channels))
            cell.append(nn.ReLU(inplace=True))

        return nn.Sequential(*cell)

    def forward(self, x):
        e1 = self.encoder_cell1(x)
        x = self.pool(e1)
        e2 = self.encoder_cell2(x)
        x = self.pool(e2)
        e3 = self.encoder_cell3(x)
        x = self.pool(e3)
        e4 = self.encoder_cell4(x)
        x = self.pool(e4)
        e5 = self.encoder_cell5(x)

        x = self.unpool(e5)
        d4 = self.decoder_cell4(torch.cat([x, e4], dim=1))
        x = self.unpool(d4)
        d3 = self.decoder_cell3(torch.cat([x, e3], dim=1))
        x = self.unpool(d3)
        d2 = self.decoder_cell2(torch.cat([x, e2], dim=1))
        x = self.unpool(d2)
        d1 = self.decoder_cell1(torch.cat([x, e1], dim=1))
        x = self.output_conv(d1)
        x = self.output_activation(x)

        return x


class PPM_3D(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins, norm_layer):
        super(PPM_3D, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool3d(bin),
                    nn.Conv3d(in_dim, reduction_dim, kernel_size=1, bias=False),
                    norm_layer(reduction_dim),
                    nn.ReLU(inplace=True),
                )
            )
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode="trilinear", align_corners=True))
        return torch.cat(out, 1)


class UNet3D_Multiple_PPM(nn.Module):
    def __init__(self, in_channels, out_channels, num_ops, bins, multiple=16):
        super(UNet3D_Multiple_PPM, self).__init__()
        self.encoder_cell1 = self.conv_cell(in_channels, multiple * 1, num_ops)
        self.encoder_cell2 = self.conv_cell(multiple * 1, multiple * 2, num_ops)
        self.encoder_cell3 = self.conv_cell(multiple * 2, multiple * 4, num_ops)
        self.encoder_cell4 = self.conv_cell(multiple * 4, multiple * 8, num_ops)

        self.decoder_cell3 = self.conv_cell(multiple * 12, multiple * 4, num_ops)
        self.decoder_cell2 = self.conv_cell(multiple * 6, multiple * 2, num_ops)
        self.decoder_cell1 = self.conv_cell(multiple * 3, multiple * 1, num_ops)

        # define PPM Model
        fea_dim = multiple * 1
        self.ppm = PPM_3D(fea_dim, int(fea_dim / len(bins)), bins, nn.InstanceNorm3d)

        self.output_conv = nn.Conv3d(fea_dim * 2, out_channels, 1, bias=False)

        self.output_activation = nn.Softmax(dim=1)
        self.pool = nn.MaxPool3d(2, 2)
        self.unpool = nn.Upsample(scale_factor=2)

    def conv_cell(self, in_channels, out_channels, num_ops):
        cell = []
        for j in range(num_ops):
            if j == 0:
                cell.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
            else:
                cell.append(nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1))
            cell.append(nn.InstanceNorm3d(out_channels))
            cell.append(nn.ReLU(inplace=True))

        return nn.Sequential(*cell)

    def forward(self, x):
        e1 = self.encoder_cell1(x)
        x = self.pool(e1)
        e2 = self.encoder_cell2(x)
        x = self.pool(e2)
        e3 = self.encoder_cell3(x)
        x = self.pool(e3)
        e4 = self.encoder_cell4(x)

        x = self.unpool(e4)
        d3 = self.decoder_cell3(torch.cat([x, e3], dim=1))
        x = self.unpool(d3)
        d2 = self.decoder_cell2(torch.cat([x, e2], dim=1))
        x = self.unpool(d2)
        d1 = self.decoder_cell1(torch.cat([x, e1], dim=1))
        x = d1
        x = self.ppm(x)
        x = self.output_conv(x)
        x = self.output_activation(x)

        return x


class UNet3D_Multiple_Cascade(nn.Module):
    def __init__(self, in_channels, out_channels, num_ops, multiple=16, num_unets=1):
        super(UNet3D_Multiple_Cascade, self).__init__()

        # initialization
        self.num_unets = num_unets
        self.first_unet = None
        self.rest_unets = []

        # module construction
        self.first_unet = UNet3D_Multiple(in_channels, out_channels, num_ops=num_ops, multiple=multiple)
        if num_unets > 1:
            for _ in range(self.num_unets - 1):
                self.rest_unets.append(
                    UNet3D_Multiple(in_channels + out_channels, out_channels, num_ops=num_ops, multiple=multiple)
                )
        self.rest_unets = nn.ModuleList(self.rest_unets)

    def forward(self, x):
        x_img = x
        output = []

        for _i in range(self.num_unets):
            if _i == 0:
                x = self.first_unet(x_img)
            else:
                x = torch.cat([x_img, x], 1)
                x = self.rest_unets[_i - 1](x)
            output.append(x)

        return output


class RecurrentBlock(nn.Module):
    def __init__(self, out_channels, t=2):
        super(RecurrentBlock, self).__init__()
        self.t = t
        self.conv = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x + x1)
        return x1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_ops: int):
        super(ConvBlock, self).__init__()

        self.ops = []

        for _i in range(num_ops):
            if _i == 0:
                self.ops.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
            else:
                self.ops.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
            self.ops.append(nn.BatchNorm3d(out_channels))
            self.ops.append(nn.ReLU(inplace=True))

        self.ops = nn.ModuleList(self.ops)

    def forward(self, x):
        for _j, _layer in enumerate(self.ops):
            x = _layer(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True), nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True), nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True), nn.BatchNorm3d(1), nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, recurrent: bool, residual: bool, num_ops: int):
        super(ResidualBlock, self).__init__()

        # initialization
        self.in_c = in_channels
        self.out_c = out_channels
        self.num_ops = num_ops
        self.recurrent = recurrent
        self.residual = residual

        self.CONV = self._make_layer_conv(recurrent)
        self.RCNN = self._make_layer_rcnn(recurrent, num_ops)

    def _make_layer_conv(self, recurrent: bool):
        return ConvBlock(self.in_c, self.out_c, num_ops=1)

    def _make_layer_rcnn(self, recurrent: bool, num_ops: int):
        if recurrent is False:
            if num_ops == 1:
                return
            return ConvBlock(self.out_c, self.out_c, num_ops=num_ops - 1)

        self.modules = []
        for _ in range(num_ops):
            self.modules.append(RecurrentBlock(self.out_c, t=2))
        return nn.Sequential(*self.modules)

    def forward(self, x):
        x = self.CONV(x)

        if self.num_ops == 1 and self.recurrent is False:
            return x

        out = self.RCNN(x)

        if self.residual is False:
            return out

        return x + out


class SEBlock3D(nn.Module):
    def __init__(self, num_channels, reduction_ratio=2):
        super(SEBlock3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, num_channels, D, H, W = x.size()

        # Average along each channel
        squeeze_x = self.avg_pool(x)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_x.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        out = torch.mul(x, fc_out_2.view(batch_size, num_channels, 1, 1, 1))

        return out


class RRUNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        blocks_down: str,
        blocks_up: str,
        num_init_kernels: int,
        recurrent=True,
        residual=True,
        attention=True,
        se=False,
        debug=False,
    ):
        super(RRUNet3D, self).__init__()

        self.attention = attention
        self.se = se
        self.debug = debug

        self.pool = nn.MaxPool3d(2, 2)
        self.unpool = nn.Upsample(scale_factor=2)
        self.input_conv = nn.Conv3d(in_channels, num_init_kernels, 3, stride=2, padding=1)
        self.output_conv = nn.Conv3d(num_init_kernels, out_channels, 1, stride=1, padding=0, bias=False)
        self.output_activation = nn.Softmax(dim=1)

        self.blocks_down = list(map(int, blocks_down.split(",")))
        self.blocks_up = list(map(int, blocks_up.split(",")))
        self.blocks_up = self.blocks_up[::-1]
        if self.debug:
            print("blocks_down", self.blocks_down)
            print("blocks_up", self.blocks_up)
        assert len(self.blocks_down) - 1 == len(
            self.blocks_up
        ), "blocks_down and blocks_up are not matching (one dimension difference)!"

        if self.se is True:
            self.encoders_se = []
            self.decoders_se = []

        # define modules
        self.levels_down = len(self.blocks_down)
        self.encoders = []
        for _i in range(self.levels_down):
            in_c = num_init_kernels * 2**_i if _i == 0 else num_init_kernels * 2 ** (_i - 1)
            out_c = num_init_kernels * 2**_i
            # if self.debug:
            #     print("in_c, out_c, blocks_down", in_c, out_c, self.blocks_down[_i])
            self.encoders.append(
                ResidualBlock(
                    in_channels=in_c,
                    out_channels=out_c,
                    recurrent=recurrent,
                    residual=residual,
                    num_ops=self.blocks_down[_i],
                )
            )
            if self.se is True:
                self.encoders_se.append(SEBlock3D(num_channels=out_c))

        self.levels_up = len(self.blocks_up)
        self.decoders = []
        for _i in range(self.levels_up):
            in_c = num_init_kernels * 2 ** (_i + 1)
            out_c = num_init_kernels * 2**_i
            # if self.debug:
            #     print("in_c, out_c, blocks_down", in_c, out_c, self.blocks_up[_i])
            self.decoders.append(
                ResidualBlock(
                    in_channels=in_c // 2 * 3,
                    out_channels=out_c,
                    recurrent=recurrent,
                    residual=residual,
                    num_ops=self.blocks_up[_i],
                )
            )
            if self.se is True:
                self.decoders_se.append(SEBlock3D(num_channels=out_c))

        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

        if self.se is True:
            if self.debug:
                print("use se")
            self.encoders_se = nn.ModuleList(self.encoders_se)
            self.decoders_se = nn.ModuleList(self.decoders_se)

        if self.attention is True:
            self.attention_blocks = []
            for _i in range(self.levels_up):
                F_g = num_init_kernels * 2 ** (_i + 1)
                F_l = num_init_kernels * 2**_i
                F_int = num_init_kernels * 2**_i
                self.attention_blocks.append(AttentionBlock(F_g=F_g, F_l=F_l, F_int=F_int))
            self.attention_blocks = nn.ModuleList(self.attention_blocks)

    def forward(self, x):
        # input
        x = self.input_conv(x)

        skip_connection = []
        for _i in range(len(self.encoders)):
            # if self.debug:
            #     print("encoder", x.shape)
            x = self.encoders[_i](x)
            if self.se is True:
                x = self.encoders_se[_i](x)

            if _i < len(self.encoders) - 1:
                skip_connection.append(x)
                x = self.pool(x)

        for _j in range(len(self.decoders) - 1, -1, -1):
            x = self.unpool(x)
            if self.attention is True:
                atten = self.attention_blocks[_j](g=x, x=skip_connection[_j])
                x = torch.cat((x, atten), 1)
            else:
                x = torch.cat((x, skip_connection[_j]), 1)
            # if self.debug:
            #     print("decoder", x.shape)
            x = self.decoders[_j](x)
            if self.se is True:
                x = self.decoders_se[_j](x)

        # output
        x = self.unpool(x)
        out = self.output_conv(x)
        out = self.output_activation(out)
        return out
