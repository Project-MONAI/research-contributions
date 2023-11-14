# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch.nn import functional as F


class ContrastLoss(torch.nn.Module):
    def __init__(self, args, batch_size, temperature=0.5):
        super().__init__()
        device = torch.device(f"cuda:{args.local_rank}")
        self.batch_size = batch_size
        self.register_buffer("temp", torch.tensor(temperature).to(torch.device(f"cuda:{args.local_rank}")))
        self.register_buffer("neg_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())

    def forward(self, x_i, x_j):
        z_i = F.normalize(x_i, dim=1)
        z_j = F.normalize(x_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        sim_ij = torch.diag(sim, self.batch_size)
        sim_ji = torch.diag(sim, -self.batch_size)
        pos = torch.cat([sim_ij, sim_ji], dim=0)
        nom = torch.exp(pos / self.temp)
        denom = self.neg_mask * torch.exp(sim / self.temp)
        return torch.sum(-torch.log(nom / torch.sum(denom, dim=1))) / (2 * self.batch_size)


class MutualLoss(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.alpha = 1.0
        self.mask_ratio = args.mask_ratio
        self.recon_loss_2 = torch.nn.MSELoss().cuda()

    def __call__(self, rec1, rec2, mask):
        mask = mask.to(dtype=rec1.dtype)
        rec1, rec2 = [val * mask for val in [rec1, rec2]]

        recon_loss = self.recon_loss_2(rec1, rec2) / self.mask_ratio
        return self.alpha * recon_loss


class Loss(torch.nn.Module):
    def __init__(self, batch_size, args):
        super().__init__()
        self.rot_loss = torch.nn.CrossEntropyLoss().cuda()
        self.recon_loss = torch.nn.L1Loss().cuda()
        self.recon_loss_2 = torch.nn.MSELoss().cuda()
        self.contrast_loss = ContrastLoss(args, batch_size).cuda()
        self.alpha1 = 1.0
        self.alpha2 = 1.0
        self.alpha3 = 1.0
        self.norm_pix_loss = args.norm_pix_loss
        self.mask_ratio = args.mask_ratio

    def __call__(
        self,
        output_rot,
        target_rot,
        output_contrastive,
        target_contrastive,
        output_recons,
        target_recons,
        mask,
        only_mae=False,
    ):
        B, C, H, W, D = output_recons.shape
        target_recons = target_recons.reshape(B, C, -1)

        if self.norm_pix_loss:
            mean = target_recons.mean(dim=-1, keepdim=True)
            var = target_recons.var(dim=-1, keepdim=True)
            target_recons = (target_recons - mean) / (var + 1.0e-6) ** 0.5
        target_recons = target_recons.reshape(B, C, H, W, D)
        # masked voxels.
        mask = mask.to(dtype=target_recons.dtype)[None, ...]
        target_recons, output_recons = [val * mask for val in [target_recons, output_recons]]
        recon_loss = self.recon_loss_2(output_recons, target_recons) / self.mask_ratio
        recon_loss = self.alpha3 * recon_loss
        if only_mae:
            return recon_loss
        contrast_loss = self.alpha2 * self.contrast_loss(output_contrastive, target_contrastive)
        rot_loss = self.alpha1 * self.rot_loss(output_rot, target_rot)
        total_loss = rot_loss + contrast_loss + recon_loss

        return total_loss, (rot_loss, contrast_loss, recon_loss)
