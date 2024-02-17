import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from icon_registration import config


class ConvNet(nn.Module):
    def __init__(self, dimension=2, output_dim=100):
        super().__init__()
        self.dimension = dimension

        if dimension == 2:
            self.Conv = nn.Conv2d
            self.avg_pool = F.avg_pool2d
        else:
            self.Conv = nn.Conv3d
            self.avg_pool = F.avg_pool3d

        self.features = [2, 16, 32, 64, 128, 128, 256]
        self.convs = nn.ModuleList([])
        for depth in range(len(self.features) - 1):
            self.convs.append(
                self.Conv(
                    self.features[depth],
                    self.features[depth + 1],
                    kernel_size=3,
                    padding=1,
                )
            )
        self.dense2 = nn.Linear(256, 300)
        self.dense3 = nn.Linear(300, output_dim)

    def forward(self, x):
        for depth in range(len(self.features) - 1):
            x = F.relu(x)
            x = self.convs[depth](x)
            x = self.avg_pool(x, 2, ceil_mode=True)
        x = self.avg_pool(x, x.shape[2:], ceil_mode=True)
        x = torch.reshape(x, (-1, 256))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x

def pad_or_crop(x, shape, dimension):
    y = x[:, : shape[1]]
    if x.size()[1] < shape[1]:
        if dimension == 3:
            y = F.pad(y, (0, 0, 0, 0, 0, 0, shape[1] - x.size()[1], 0))
        else:
            y = F.pad(y, (0, 0, 0, 0, shape[1] - x.size()[1], 0))
    assert y.size()[1] == shape[1]

    return y


class UNet2(nn.Module):
    def __init__(self, num_layers, channels, dimension):
        super().__init__()
        self.dimension = dimension
        if dimension == 2:
            self.BatchNorm = nn.BatchNorm2d
            self.Conv = nn.Conv2d
            self.ConvTranspose = nn.ConvTranspose2d
            self.avg_pool = F.avg_pool2d
            self.interpolate_mode = "bilinear"
        else:
            self.BatchNorm = nn.BatchNorm3d
            self.Conv = nn.Conv3d
            self.ConvTranspose = nn.ConvTranspose3d
            self.avg_pool = F.avg_pool3d
            self.interpolate_mode = "trilinear"
        self.num_layers = num_layers
        down_channels = np.array(channels[0])
        up_channels_out = np.array(channels[1])
        up_channels_in = down_channels[1:] + np.concatenate([up_channels_out[1:], [0]])
        self.downConvs = nn.ModuleList([])
        self.upConvs = nn.ModuleList([])
        self.batchNorms = nn.ModuleList(
            [
                self.BatchNorm(num_features=up_channels_out[_])
                for _ in range(self.num_layers)
            ]
        )
        for depth in range(self.num_layers):
            self.downConvs.append(
                self.Conv(
                    down_channels[depth],
                    down_channels[depth + 1],
                    kernel_size=3,
                    padding=1,
                    stride=2,
                )
            )
            self.upConvs.append(
                self.ConvTranspose(
                    up_channels_in[depth],
                    up_channels_out[depth],
                    kernel_size=4,
                    padding=1,
                    stride=2,
                )
            )
        self.lastConv = self.Conv(
            down_channels[0] + up_channels_out[0], dimension, kernel_size=3, padding=1
        )
        torch.nn.init.zeros_(self.lastConv.weight)
        torch.nn.init.zeros_(self.lastConv.bias)

    def forward(self, x):
        skips = []
        for depth in range(self.num_layers):
            skips.append(x)
            y = self.downConvs[depth](F.leaky_relu(x))
            x = y + pad_or_crop(
                self.avg_pool(x, 2, ceil_mode=True), y.size(), self.dimension
            )

        for depth in reversed(range(self.num_layers)):
            y = self.upConvs[depth](F.leaky_relu(x))
            x = y + F.interpolate(
                pad_or_crop(x, y.size(), self.dimension),
                scale_factor=2,
                mode=self.interpolate_mode,
                align_corners=False,
            )
            x = self.batchNorms[depth](x)
            if self.dimension == 2:
                x = x[:, :, : skips[depth].size()[2], : skips[depth].size()[3]]
            else:
                x = x[
                    :,
                    :,
                    : skips[depth].size()[2],
                    : skips[depth].size()[3],
                    : skips[depth].size()[4],
                ]
            x = torch.cat([x, skips[depth]], 1)
        x = self.lastConv(x)
        return x * 10


class UNetDenseMiddle(nn.Module):
    def __init__(self, num_layers, channels, dimension):
        super().__init__()
        self.dimension = dimension
        if dimension == 2:
            self.BatchNorm = nn.BatchNorm2d
            self.Conv = nn.Conv2d
            self.ConvTranspose = nn.ConvTranspose2d
            self.avg_pool = F.avg_pool2d
            self.interpolate_mode = "bilinear"
        else:
            self.BatchNorm = nn.BatchNorm3d
            self.Conv = nn.Conv3d
            self.ConvTranspose = nn.ConvTranspose3d
            self.avg_pool = F.avg_pool3d
            self.interpolate_mode = "trilinear"
        self.num_layers = num_layers
        down_channels = np.array(channels[0])
        up_channels_out = np.array(channels[1])
        up_channels_in = down_channels[1:] + np.concatenate([up_channels_out[1:], [0]])
        self.downConvs = nn.ModuleList([])
        self.upConvs = nn.ModuleList([])
        self.batchNorms = nn.ModuleList(
            [
                self.BatchNorm(num_features=up_channels_out[_])
                for _ in range(self.num_layers)
            ]
        )
        for depth in range(self.num_layers):
            self.downConvs.append(
                self.Conv(
                    down_channels[depth],
                    down_channels[depth + 1],
                    kernel_size=3,
                    padding=1,
                    stride=2,
                )
            )
            self.upConvs.append(
                self.ConvTranspose(
                    up_channels_in[depth],
                    up_channels_out[depth],
                    kernel_size=4,
                    padding=1,
                    stride=2,
                )
            )
        self.lastConv = self.Conv(18, dimension, kernel_size=3, padding=1)
        torch.nn.init.zeros_(self.lastConv.weight)

        self.middle_dense = nn.ModuleList(
            [
                torch.nn.Linear(512 * 2 * 3 * 3, 128 * 2 * 3 * 3),
                torch.nn.Linear(128 * 2 * 3 * 3, 512 * 2 * 3 * 3),
            ]
        )

    def forward(self, x):
        skips = []
        for depth in range(self.num_layers):
            skips.append(x)
            y = self.downConvs[depth](F.leaky_relu(x))
            x = y + pad_or_crop(
                self.avg_pool(x, 2, ceil_mode=True), y.size(), self.dimension
            )
            y = F.layer_norm

        x = torch.reshape(x, (-1, 512 * 2 * 3 * 3))
        x = self.middle_dense[1](F.leaky_relu(self.middle_dense[0](x)))
        x = torch.reshape(x, (-1, 512, 2, 3, 3))
        for depth in reversed(range(self.num_layers)):
            y = self.upConvs[depth](F.leaky_relu(x))
            x = y + F.interpolate(
                pad_or_crop(x, y.size(), self.dimension),
                scale_factor=2,
                mode=self.interpolate_mode,
                align_corners=False,
            )
            x = self.batchNorms[depth](x)

            x = x[:, :, : skips[depth].size()[2], : skips[depth].size()[3]]
            x = torch.cat([x, skips[depth]], 1)
        x = self.lastConv(x)
        return x / 10




def tallUNet2(dimension=2, input_channels=1):
    return UNet2(
        5,
        [[input_channels * 2, 16, 32, 64, 256, 512], [16, 32, 64, 128, 256]],
        dimension,
    )


class FCNet1D(nn.Module):
    def __init__(self, size=28):
        super().__init__()
        self.size = size
        self.dense1 = nn.Linear(size * 2, 8000)
        self.dense2 = nn.Linear(8000, 3000)
        self.dense3 = nn.Linear(3000, size)
        torch.nn.init.zeros_(self.dense3.weight)

    def forward(self, x):
        x = torch.reshape(x, (-1, 2 * self.size))
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        x = torch.reshape(x, (-1, 1, self.size))
        return x


class FCNet(nn.Module):
    def __init__(self, size=28):
        super().__init__()
        self.size = size
        self.dense1 = nn.Linear(size * size * 2, 8000)
        self.dense2 = nn.Linear(8000, 3000)
        self.dense3 = nn.Linear(3000, size * size * 2)
        torch.nn.init.zeros_(self.dense3.weight)

    def forward(self, x):
        x = torch.reshape(x, (-1, 2 * self.size * self.size))
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        x = torch.reshape(x, (-1, 2, self.size, self.size))
        return x

class FCNet3D(nn.Module):
    def __init__(self, shape, bottleneck=128):
        super().__init__()
        self.shape = shape.copy()
        self.shape[1] = 3
        self.bottleneck = bottleneck
        self.dense1 = nn.Linear(2 * np.product(self.shape[2:]), self.bottleneck)
        self.dense2 = nn.Linear(self.bottleneck, 8000)
        self.dense3 = nn.Linear(8000, self.bottleneck)
        self.dense4 = nn.Linear(self.bottleneck, np.product(self.shape[1:]))
        torch.nn.init.zeros_(self.dense4.weight)
        torch.nn.init.zeros_(self.dense4.bias)

    def forward(self, x):
        x = torch.reshape(x, (-1, 2 * np.product(self.shape[2:])))
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        x = self.dense4(x)
        x = torch.reshape(x, tuple(self.shape))
        return x


class ConvolutionalMatrixNet(nn.Module):
    def __init__(self, dimension=2):
        super().__init__()
        self.dimension = dimension

        if dimension == 2:
            self.Conv = nn.Conv2d
            self.avg_pool = F.avg_pool2d
        else:
            self.Conv = nn.Conv3d
            self.avg_pool = F.avg_pool3d

        self.features = [2, 16, 32, 64, 128, 256, 512]
        self.convs = nn.ModuleList([])
        for depth in range(len(self.features) - 1):
            self.convs.append(
                self.Conv(
                    self.features[depth],
                    self.features[depth + 1],
                    kernel_size=3,
                    padding=1,
                )
            )
        self.dense2 = nn.Linear(512, 300)
        self.dense3 = nn.Linear(300, 6 if self.dimension == 2 else 12)
        torch.nn.init.zeros_(self.dense3.weight)
        torch.nn.init.zeros_(self.dense3.bias)

    def forward(self, x):
        for depth in range(len(self.features) - 1):
            x = F.relu(x)
            x = self.convs[depth](x)
            x = self.avg_pool(x, 2, ceil_mode=True)
        x = self.avg_pool(x, x.shape[2:], ceil_mode=True)
        x = torch.reshape(x, (-1, 512))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        if self.dimension == 3:
            x = torch.reshape(x, (-1, 3, 4))
            x = torch.cat(
                [
                    x,
                    torch.Tensor([[[0, 0, 0, 1]]])
                    .to(x.device)
                    .expand(x.shape[0], -1, -1),
                ],
                1,
            )
            x = x + torch.Tensor(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]
            ).to(x.device)
            x = torch.matmul(
                torch.Tensor(
                    [[1, 0, 0, 0.5], [0, 1, 0, 0.5], [0, 0, 1, 0.5], [0, 0, 0, 1]]
                ).to(x.device),
                x,
            )
            x = torch.matmul(
                x,
                torch.Tensor(
                    [[1, 0, 0, -0.5], [0, 1, 0, -0.5], [0, 0, 1, -0.5], [0, 0, 0, 1]]
                ).to(x.device),
            )
        elif self.dimension == 2:
            x = torch.reshape(x, (-1, 2, 3))
            x = torch.cat(
                [
                    x,
                    torch.Tensor([[[0, 0, 1]]]).to(x.device).expand(x.shape[0], -1, -1),
                ],
                1,
            )
            x = x + torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 0]]).to(x.device)
            x = torch.matmul(
                torch.Tensor([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]).to(x.device), x
            )
            x = torch.matmul(
                x,
                torch.Tensor([[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1]]).to(x.device),
            )
        else:
            raise ArgumentError()
        return x
