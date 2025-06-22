import torch

def normalize(image):
    dimension = len(image.shape) - 2
    if dimension == 2:
        dim_reduce = [2, 3]
    elif dimension == 3:
        dim_reduce = [2, 3, 4]
    image_centered = image - torch.mean(image, dim_reduce, keepdim=True)
    stddev = torch.sqrt(torch.mean(image_centered**2, dim_reduce, keepdim=True))
    return image_centered / stddev


class SimilarityBase:
    def __init__(self, isInterpolated=False):
        self.isInterpolated = isInterpolated


class NCC(SimilarityBase):
    def __init__(self):
        super().__init__(isInterpolated=False)

    def __call__(self, image_A, image_B):
        assert (
            image_A.shape == image_B.shape
        ), "The shape of image_A and image_B sould be the same."
        A = normalize(image_A)
        B = normalize(image_B)
        res = torch.mean(A * B)
        return 1 - res


# torch removed this function from torchvision.functional_tensor, so we are vendoring it.
def _get_gaussian_kernel1d(kernel_size, sigma):
    ksize_half = (kernel_size - 1) * 0.5
    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()
    return kernel1d


def gaussian_blur(tensor, kernel_size, sigma, padding="same"):
    kernel1d = _get_gaussian_kernel1d(kernel_size=kernel_size, sigma=sigma).to(
        tensor.device, dtype=tensor.dtype
    )
    out = tensor
    group = tensor.shape[1]

    if len(tensor.shape) - 2 == 1:
        out = torch.conv1d(
            out,
            kernel1d[None, None, :].expand(group, -1, -1),
            padding="same",
            groups=group,
        )
    elif len(tensor.shape) - 2 == 2:
        out = torch.conv2d(
            out,
            kernel1d[None, None, :, None].expand(group, -1, -1, -1),
            padding="same",
            groups=group,
        )
        out = torch.conv2d(
            out,
            kernel1d[None, None, None, :].expand(group, -1, -1, -1),
            padding="same",
            groups=group,
        )
    elif len(tensor.shape) - 2 == 3:
        out = torch.conv3d(
            out,
            kernel1d[None, None, :, None, None].expand(group, -1, -1, -1, -1),
            padding="same",
            groups=group,
        )
        out = torch.conv3d(
            out,
            kernel1d[None, None, None, :, None].expand(group, -1, -1, -1, -1),
            padding="same",
            groups=group,
        )
        out = torch.conv3d(
            out,
            kernel1d[None, None, None, None, :].expand(group, -1, -1, -1, -1),
            padding="same",
            groups=group,
        )

    return out


class LNCC(SimilarityBase):
    def __init__(self, sigma):
        super().__init__(isInterpolated=False)
        self.sigma = sigma

    def blur(self, tensor):
        return gaussian_blur(tensor, self.sigma * 4 + 1, self.sigma)

    def __call__(self, image_A, image_B):
        I = image_A
        J = image_B
        assert I.shape == J.shape, "The shape of image I and J sould be the same."

        return torch.mean(
            1
            - (self.blur(I * J) - (self.blur(I) * self.blur(J)))
            / torch.sqrt(
                (self.blur(I * I) - self.blur(I) ** 2 + 0.00001)
                * (self.blur(J * J) - self.blur(J) ** 2 + 0.00001)
            )
        )


class LNCCOnlyInterpolated(SimilarityBase):
    def __init__(self, sigma):
        super().__init__(isInterpolated=True)
        self.sigma = sigma

    def blur(self, tensor):
        return gaussian_blur(tensor, self.sigma * 4 + 1, self.sigma)

    def __call__(self, image_A, image_B):
        I = image_A[:, :-1]
        J = image_B

        assert I.shape == J.shape, "The shape of image I and J sould be the same."
        lncc_everywhere = 1 - (
            self.blur(I * J) - (self.blur(I) * self.blur(J))
        ) / torch.sqrt(
            (self.blur(I * I) - self.blur(I) ** 2 + 0.00001)
            * (self.blur(J * J) - self.blur(J) ** 2 + 0.00001)
        )

        with torch.no_grad():
            A_inbounds = image_A[:, -1:]

            inbounds_mask = self.blur(A_inbounds) > 0.999

        if len(image_A.shape) - 2 == 3:
            dimensions_to_sum_over = [2, 3, 4]
        elif len(image_A.shape) - 2 == 2:
            dimensions_to_sum_over = [2, 3]
        elif len(image_A.shape) - 2 == 1:
            dimensions_to_sum_over = [2]

        lncc_loss = torch.sum(
            inbounds_mask * lncc_everywhere, dimensions_to_sum_over
        ) / torch.sum(inbounds_mask, dimensions_to_sum_over)

        return torch.mean(lncc_loss)


class BlurredSSD(SimilarityBase):
    def __init__(self, sigma):
        super().__init__(isInterpolated=False)
        self.sigma = sigma

    def blur(self, tensor):
        return gaussian_blur(tensor, self.sigma * 4 + 1, self.sigma)

    def __call__(self, image_A, image_B):
        assert (
            image_A.shape == image_B.shape
        ), "The shape of image_A and image_B sould be the same."
        return torch.mean((self.blur(image_A) - self.blur(image_B)) ** 2)


class AdaptiveNCC(SimilarityBase):
    def __init__(self, level=4, threshold=0.1, gamma=1.5, sigma=2):
        super().__init__(isInterpolated=False)
        self.level = level
        self.threshold = threshold
        self.gamma = gamma
        self.sigma = sigma

    def blur(self, tensor):
        return gaussian_blur(tensor, self.sigma * 2 + 1, self.sigma)

    def __call__(self, image_A, image_B):
        assert (
            image_A.shape == image_B.shape
        ), "The shape of image_A and image_B sould be the same."

        def _nccBeforeMean(image_A, image_B):
            A = normalize(image_A)
            B = normalize(image_B)
            res = torch.mean(A * B, dim=(1, 2, 3, 4))
            return 1 - res

        sims = [_nccBeforeMean(image_A, image_B)]
        for i in range(self.level):
            if i == 0:
                sims.append(_nccBeforeMean(self.blur(image_A), self.blur(image_B)))
            else:
                sims.append(
                    _nccBeforeMean(
                        self.blur(F.avg_pool3d(image_A, 2**i)),
                        self.blur(F.avg_pool3d(image_B, 2**i)),
                    )
                )

        sim_loss = sims[0] + 0
        lamb_ = 1.0
        for i in range(1, len(sims)):
            lamb = torch.clamp(
                sims[i].detach() / (self.threshold / (self.gamma ** (len(sims) - i))),
                0,
                1,
            )
            sim_loss = lamb * sims[i] + (1 - lamb) * sim_loss
            lamb_ *= 1 - lamb

        return torch.mean(sim_loss)


class SSD(SimilarityBase):
    def __init__(self):
        super().__init__(isInterpolated=False)

    def __call__(self, image_A, image_B):
        assert (
            image_A.shape == image_B.shape
        ), "The shape of image_A and image_B sould be the same."
        return torch.mean((image_A - image_B) ** 2)


class SSDOnlyInterpolated(SimilarityBase):
    def __init__(self):
        super().__init__(isInterpolated=True)

    def __call__(self, image_A, image_B):
        if len(image_A.shape) - 2 == 3:
            dimensions_to_sum_over = [2, 3, 4]
        elif len(image_A.shape) - 2 == 2:
            dimensions_to_sum_over = [2, 3]
        elif len(image_A.shape) - 2 == 1:
            dimensions_to_sum_over = [2]

        inbounds_mask = image_A[:, -1:]
        image_A = image_A[:, :-1]
        assert (
            image_A.shape == image_B.shape
        ), "The shape of image_A and image_B sould be the same."

        inbounds_squared_distance = inbounds_mask * (image_A - image_B) ** 2
        sum_squared_distance = torch.sum(
            inbounds_squared_distance, dimensions_to_sum_over
        )
        divisor = torch.sum(inbounds_mask, dimensions_to_sum_over)
        ssds = sum_squared_distance / divisor
        return torch.mean(ssds)


def flips(phi, in_percentage=False):
    if len(phi.size()) == 5:
        a = (phi[:, :, 1:, 1:, 1:] - phi[:, :, :-1, 1:, 1:]).detach()
        b = (phi[:, :, 1:, 1:, 1:] - phi[:, :, 1:, :-1, 1:]).detach()
        c = (phi[:, :, 1:, 1:, 1:] - phi[:, :, 1:, 1:, :-1]).detach()

        dV = torch.sum(torch.cross(a, b, 1) * c, axis=1, keepdims=True)
        if in_percentage:
            return torch.mean((dV < 0).float()) * 100.0
        else:
            return torch.sum(dV < 0) / phi.shape[0]
    elif len(phi.size()) == 4:
        du = (phi[:, :, 1:, :-1] - phi[:, :, :-1, :-1]).detach()
        dv = (phi[:, :, :-1, 1:] - phi[:, :, :-1, :-1]).detach()
        dA = du[:, 0] * dv[:, 1] - du[:, 1] * dv[:, 0]
        if in_percentage:
            return torch.mean((dA < 0).float()) * 100.0
        else:
            return torch.sum(dA < 0) / phi.shape[0]
    elif len(phi.size()) == 3:
        du = (phi[:, :, 1:] - phi[:, :, :-1]).detach()
        if in_percentage:
            return torch.mean((du < 0).float()) * 100.0
        else:
            return torch.sum(du < 0) / phi.shape[0]
    else:
        raise ValueError()
