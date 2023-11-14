"""
Prostate-MRI_Lesion_Detection, v2.0 (Release date: August 2, 2023)
DEFINITIONS: AUTHOR(S) NVIDIA Corp. and National Cancer Institute, NIH

PROVIDER: the National Cancer Institute (NCI), a participating institute of the
National Institutes of Health (NIH), and an agency of the United States Government.

SOFTWARE: the machine readable, binary, object code form,
and the related documentation for the modules of the Prostate-MRI_Lesion_Detection, v2.0
software package, which is a collection of operators which accept (T2, ADC, and High
b-value DICOM images) and produce prostate organ and lesion segmentation files

RECIPIENT: the party that downloads the software.

By downloading or otherwise receiving the SOFTWARE, RECIPIENT may
use and/or redistribute the SOFTWARE, with or without modification,
subject to RECIPIENT’s agreement to the following terms:

1. THE SOFTWARE SHALL NOT BE USED IN THE TREATMENT OR DIAGNOSIS
OF HUMAN SUBJECTS.  RECIPIENT is responsible for
compliance with all laws and regulations applicable to the use
of the SOFTWARE.

2. THE SOFTWARE is distributed for NON-COMMERCIAL RESEARCH PURPOSES ONLY. RECIPIENT is
responsible for appropriate-use compliance.

3.	RECIPIENT agrees to acknowledge PROVIDER’s contribution and
the name of the author of the SOFTWARE in all written publications
containing any data or information regarding or resulting from use
of the SOFTWARE.

4.	THE SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT
ARE DISCLAIMED. IN NO EVENT SHALL THE PROVIDER OR THE INDIVIDUAL DEVELOPERS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.

5.	RECIPIENT agrees not to use any trademarks, service marks, trade names,
logos or product names of NVIDIA, NCI or NIH to endorse or promote products derived
from the SOFTWARE without specific, prior and written permission.

6.	For sake of clarity, and not by way of limitation, RECIPIENT may add its
own copyright statement to its modifications or derivative works of the SOFTWARE
and may provide additional or different license terms and conditions in its
sublicenses of modifications or derivative works of the SOFTWARE provided that
RECIPIENT’s use, reproduction, and distribution of the SOFTWARE otherwise complies
with the conditions stated in this Agreement. Whenever Recipient distributes or
redistributes the SOFTWARE, a copy of this Agreement must be included with
each copy of the SOFTWARE."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["RRUNet3D"]


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
