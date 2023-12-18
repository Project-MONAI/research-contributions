import torch
from icon_registration.losses import AdaptiveNCC, normalize
import math
import numbers
import torch
from torch import nn

import unittest

class TestLosses(unittest.TestCase):
    def test_adaptive_ncc(self):
        a = torch.rand(1,1,64,64,64)
        b = torch.rand(1,1,64,64,64)

        sim = AdaptiveNCC()
        l = sim(a, b)
        sim_origin = adaptive_ncc()
        l_origin = sim_origin(a, b)

        self.assertAlmostEqual(l.item(), l_origin.item(), places=5)

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / std) ** 2) / 2)
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = nn.functional.conv1d
        elif dim == 2:
            self.conv = nn.functional.conv2d
        elif dim == 3:
            self.conv = nn.functional.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(
            input,
            weight=self.weight,
            groups=self.groups,
            padding=int(self.weight.shape[2] / 2),
        )

def adaptive_ncc(
    level=4, threshold=0.1, gamma=1.5, smoother=GaussianSmoothing(1, 5, 2, 3)):
    def _nccBeforeMean(image_A, image_B):
        A = normalize(image_A[:, :1])
        B = normalize(image_B)
        res = torch.mean(A * B, dim=(1, 2, 3, 4))
        return 1 - res

    def _sim(x, y):
        sims = [_nccBeforeMean(x, y)]
        for i in range(level):
            if i == 0:
                sims.append(_nccBeforeMean(smoother(x), smoother(y)))
            else:
                sims.append(
                    _nccBeforeMean(
                        smoother(nn.functional.avg_pool3d(x, 2**i)),
                        smoother(nn.functional.avg_pool3d(y, 2**i)),
                    )
                )

        sim_loss = sims[0] + 0
        lamb_ = 1.0
        for i in range(1, len(sims)):
            lamb = torch.clamp(
                sims[i].detach() / (threshold / (gamma ** (len(sims) - i))), 0, 1
            )
            sim_loss = lamb * sims[i] + (1 - lamb) * sim_loss
            lamb_ *= 1 - lamb

        return torch.mean(sim_loss)

    return _sim

    