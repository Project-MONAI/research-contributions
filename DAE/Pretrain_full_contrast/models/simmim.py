import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer

from .swin_transformer_3d import SwinTransformer3D


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

    def forward(self, x, mask):
        x_out = []
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
        x_out.append(x)
        for layer in self.layers1:
            x = layer[0](x)
        x_out.append(x)

        for layer in self.layers2:
            x = layer[0](x)
        x_out.append(x)

        for layer in self.layers3:
            x = layer[0](x)
        x_out.append(x)

        for layer in self.layers4:
            x = layer[0](x)
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

    def forward(self, x, mask):
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
        loss_recon = F.l1_loss(x, x_rec, reduction="none")
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
    def __init__(self, encoder, encoder_stride, decoder="deconv", loss="mask_only"):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride
        self.decoder = decoder
        self.loss = loss

        # if decoder == 'pixel_shuffle':
        #     self.conv1 = nn.Conv3d(in_channels=2*self.encoder.num_features,out_channels=self.encoder_stride ** 3 * 1,
        #                            kernel_size=1)
        #     self.pixel_shuffle = PixelShuffle3D(self.encoder_stride)
        # elif decoder == 'deconv':
        #     self.transp_conv1 = nn.ConvTranspose3d(768, 384, 2, stride=2)
        #     self.transp_conv2 = nn.ConvTranspose3d(384, 192, 2, stride=2)
        #     self.transp_conv3 = nn.ConvTranspose3d(192, 96, 2, stride=2)
        #     self.transp_conv4 = nn.ConvTranspose3d(96, 48, 2, stride=2)
        #     self.transp_conv5 = nn.ConvTranspose3d(48, 1, 2, stride=2)
        #     self.conv = nn.Conv3d(1, 1, kernel_size=1, stride=1)

        # elif decoder == 'upsample':
        #     self.conv_block1 = UnetResBlock(3, 768, 384, kernel_size=3, stride=1, norm_name='instance')
        #     self.conv_block2 = UnetResBlock(3, 384, 192, kernel_size=3, stride=1, norm_name='instance')
        #     self.conv_block3 = UnetResBlock(3, 192, 96, kernel_size=3, stride=1, norm_name='instance')
        #     self.conv_block4 = UnetResBlock(3, 96, 48, kernel_size=3, stride=1, norm_name='instance')
        #     self.conv_block5 = UnetResBlock(3, 48, 1, kernel_size=3, stride=1, norm_name='instance')
        #     self.conv_block6 = nn.Conv3d(1, 1, kernel_size=1, stride=1)
        #     self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        #     #self.act = nn.ReLU()
        #     # self.act = nn.Sigmoid()
        # elif decoder =='vae':
        #     self.upsample = nn.Sequential(
        #         nn.Conv3d(768, 384, kernel_size=3, stride=1, padding=1),
        #         nn.InstanceNorm3d(384),
        #         nn.LeakyReLU(),
        #         nn.Upsample(scale_factor=2, mode='trilinear'),
        #         nn.Conv3d(384, 192, kernel_size=3, stride=1, padding=1),
        #         nn.InstanceNorm3d(192),
        #         nn.LeakyReLU(),
        #         nn.Upsample(scale_factor=2, mode='trilinear'),
        #         nn.Conv3d(192, 96, kernel_size=3, stride=1, padding=1),
        #         nn.InstanceNorm3d(96),
        #         nn.LeakyReLU(),
        #         nn.Upsample(scale_factor=2, mode='trilinear'),
        #         nn.Conv3d(96, 48, kernel_size=3, stride=1, padding=1),
        #         nn.InstanceNorm3d(48),
        #         nn.LeakyReLU(),
        #         nn.Upsample(scale_factor=2, mode='trilinear'),
        #         nn.Conv3d(48, 1, kernel_size=3, stride=1, padding=1),
        #         nn.InstanceNorm3d(48),
        #         nn.LeakyReLU(),
        #         nn.Upsample(scale_factor=2, mode='trilinear'),
        #         nn.Conv3d(1, 1, kernel_size=1, stride=1),
        #         nn.Tanh()
        #     )
        # elif decoder =='vae2':

        #     self.upsample1 = nn.Sequential(
        #         nn.Conv3d(1536, 384, kernel_size=3, stride=1, padding=1),
        #         nn.InstanceNorm3d(384),
        #         nn.LeakyReLU(),
        #         nn.Upsample(scale_factor=2, mode='trilinear')
        #     )

        #     self.upsample2 = nn.Sequential(
        #         nn.Conv3d(768, 192, kernel_size=3, stride=1, padding=1),
        #         nn.InstanceNorm3d(192),
        #         nn.LeakyReLU(),
        #         nn.Upsample(scale_factor=2, mode='trilinear')
        #     )

        #     self.upsample3 = nn.Sequential(
        #         nn.Conv3d(384, 96, kernel_size=3, stride=1, padding=1),
        #         nn.InstanceNorm3d(96),
        #         nn.LeakyReLU(),
        #         nn.Upsample(scale_factor=2, mode='trilinear')
        #     )

        #     self.upsample4 = nn.Sequential(
        #         nn.Conv3d(192, 48, kernel_size=3, stride=1, padding=1),
        #         nn.InstanceNorm3d(48),
        #         nn.LeakyReLU(),
        #         nn.Upsample(scale_factor=2, mode='trilinear')
        #     )

        #     self.upsample5 = nn.Sequential(
        #         nn.Conv3d(96, 24, kernel_size=3, stride=1, padding=1),
        #         nn.InstanceNorm3d(24),
        #         nn.LeakyReLU(),
        #         nn.Upsample(scale_factor=2, mode='trilinear')
        #     )

        #     self.out = nn.Sequential(
        #         nn.Conv3d(24, 1, kernel_size=1, stride=1),
        #     )

        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size

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

    def forward(self, x, mask):
        z, z_out = self.encoder(x, mask)

        x_rec = self.decoder(z, z_out)

        mask = (
            mask.repeat_interleave(self.patch_size[0], 1)
            .repeat_interleave(self.patch_size[1], 2)
            .repeat_interleave(self.patch_size[2], 3)
            .unsqueeze(1)
            .contiguous()
        )

        if self.loss == "mask_only":
            loss_recon = F.l1_loss(x, x_rec, reduction="none")
            loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
            # _, loss_var_l2 = self.get_image_prior_losses(x_rec)
            # bs = x.shape[0]
            # img_l2_loss = torch.norm(x_rec.view(bs, -1), dim=1).mean()
            # loss = loss + 0.0001*loss_var_l2+1e-5*img_l2_loss

        elif self.loss == "all_img":
            loss_recon = F.l1_loss(x, x_rec, reduction="mean")
            loss = loss_recon

        elif self.loss == "l2":
            loss = F.mse_loss(x, x_rec, reduction="mean")
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
        decoder = DecoderForSIM()
        model = SimMIMSkip(encoder=encoder, encoder_stride=encoder_stride, decoder=decoder, loss=args.loss_type)
    else:
        raise NotImplementedError(f"Unknown pre-train model: {model_type}")

    return model
