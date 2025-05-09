
import math

import torch
import numpy as np
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial

from torch.utils import data
from pathlib import Path
from torchvision import transforms as T
from PIL import Image

from tqdm import tqdm
from einops import rearrange
from dataloader import cache_transformed_train_data
import glob, os
from einops_exts import check_shape, rearrange_many

from text import tokenize, bert_embed, BERT_MODEL_DIM

from accelerate import Accelerator
from accelerate.utils import set_seed

import xformers, xformers.ops

from utils import *

# relative positional bias

class RelativePositionBias(nn.Module):
    def __init__(
            self,
            heads=8,
            num_buckets=16,
            max_distance=128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
                torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets=self.num_buckets,
                                                   max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')


# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


def Downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding=(0, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)


class Block3d(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (3, 3, 3), padding=(1, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)


class ResnetBlock3d(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block3d(dim, dim_out, groups=groups)
        self.block2 = Block3d(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)

class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = rearrange_many(qkv, 'b (h c) x y -> (b h) (x y) c', h=self.heads)

        query = q.contiguous()
        key = k.contiguous()
        value = v.contiguous()
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=None)

        out = rearrange(hidden_states, '(b h) (x y) c -> b (h c) x y', h=self.heads, x=h, y=w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b=b)

class CrossAttention(nn.Module):
    def __init__(self, dim, dim_con, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_q = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.to_kv = nn.Linear(dim_con, hidden_dim*2, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x, kv=None):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        self.to_kv(kv)
        kv = torch.cat([kv.unsqueeze(dim=1)]*f, dim=1)
        kv = rearrange(kv, 'b f h c -> (b f) h c')
        k, v = self.to_kv(kv).chunk(2, dim=-1)
        k = rearrange(k, 'b d (h c) -> (b h) d c', h=self.heads)
        v = rearrange(v, 'b d (h c) -> (b h) d c', h=self.heads)

        q = self.to_q(x)
        q = rearrange(q, 'b (h c) x y -> (b h) (x y) c', h=self.heads)

        query = q.contiguous()
        key = k.contiguous()
        value = v.contiguous()
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=None)

        out = rearrange(hidden_states, '(b h) (x y) c -> b (h c) x y', h=self.heads, x=h, y=w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b=b)


# attention along space and time

class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads=4,
            dim_head=32,
            rotary_emb=None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(
            self,
            x,
            pos_bias=None,
            focus_present_mask=None
    ):
        n, device = x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        if exists(focus_present_mask) and focus_present_mask.all():
            # if all batch samples are focusing on present
            # it would be equivalent to passing that token's values through to the output
            values = qkv[-1]
            return self.to_out(values)

        # split out heads

        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h=self.heads)

        # scale

        q = q * self.scale

        # rotate positions into queries and keys for time attention

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity

        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        # relative positional bias

        if exists(pos_bias):
            sim = sim + pos_bias

        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones((n, n), device=device, dtype=torch.bool)
            attend_self_mask = torch.eye(n, device=device, dtype=torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # numerical stability

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)


# model

class Unet3D(nn.Module):
    def __init__(
            self,
            dim,
            cond_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            attn_heads=8,
            attn_dim_head=32,
            total_slices=256,
            use_bert_text_cond=False,
            init_dim=None,
            init_kernel_size=7,
            use_sparse_linear_attn=True,
            block_type='resnet',
            resnet_groups=8
    ):
        super().__init__()

        self.channels = channels
        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)

        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(channels, init_dim, (init_kernel_size, init_kernel_size, init_kernel_size),
                                   padding=(init_padding, init_padding, init_padding))

        # dimensions

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time conditioning

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # text conditioning

        self.has_cond = exists(cond_dim) or use_bert_text_cond

        self.null_cond_emb = nn.Parameter(torch.randn(1, 192, cond_dim)) if self.has_cond else None

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        # block type

        block_klass = partial(ResnetBlock, groups=resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim=time_dim)

        block_klass3d = partial(ResnetBlock3d, groups=resnet_groups)
        block_klass_cond3d = partial(block_klass3d, time_emb_dim=time_dim)

        # modules for all layers

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                block_klass_cond(dim_out, dim_out),
                block_klass_cond3d(dim_out, dim_out),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)

        spatial_attn = EinopsToAndFrom('b c f h w', 'b f (h w) c', Attention(mid_dim, heads=attn_heads))

        self.mid_spatial_attn1 = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_cross_attn1 = Residual(PreNorm(mid_dim, CrossAttention(mid_dim, heads=attn_heads, dim_con=cond_dim)))
        self.mid_temporal_attn1 = block_klass_cond3d(mid_dim, mid_dim)
        ###
        self.mid_spatial_attn2 = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_cross_attn2 = Residual(PreNorm(mid_dim, CrossAttention(mid_dim, heads=attn_heads, dim_con=cond_dim)))
        self.mid_temporal_attn2 = block_klass_cond3d(mid_dim, mid_dim)
        ###
        self.mid_spatial_attn3 = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_cross_attn3 = Residual(PreNorm(mid_dim, CrossAttention(mid_dim, heads=attn_heads, dim_con=cond_dim)))
        self.mid_temporal_attn3 = block_klass_cond3d(mid_dim, mid_dim)
        ###
        self.mid_spatial_attn4 = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_cross_attn4 = Residual(PreNorm(mid_dim, CrossAttention(mid_dim, heads=attn_heads, dim_con=cond_dim)))
        self.mid_temporal_attn4 = block_klass_cond3d(mid_dim, mid_dim)

        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_in),
                block_klass_cond(dim_in, dim_in),
                block_klass_cond3d(dim_in, dim_in),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, channels, 1)
        )

    def forward_with_cond_scale(
            self,
            x, t, indexes=None, cond=None, cond_scale=None
    ):
        # cond_scale = 1
        if cond_scale == 1 or not self.has_cond:
            logits = self.forward(x, t, cond=cond, null_cond_prob=0.)
            return logits
        elif cond_scale == -1:
            batch, device = x.shape[0], x.device
            null_cond_emb = torch.cat([self.null_cond_emb] * batch, dim=0)
            logits = self.forward(x, t, cond=null_cond_emb, null_cond_prob=0.)
            return logits
        else:
            batch, device = x.shape[0], x.device
            x = torch.cat([x]*2, dim=0)
            t = torch.cat([t]*2, dim=0)
            null_cond_emb = torch.cat([self.null_cond_emb] * batch, dim=0)
            cond = torch.cat([cond, null_cond_emb], dim=0)
            logits, null_logits = self.forward(x, t, cond=cond, null_cond_prob=0.).chunk(2, dim=0)
        return [logits, null_logits]

    def forward(
            self,
            x,
            time,
            indexes=None,
            cond=None,
            null_cond_prob=0.1,
            focus_present_mask=None,
            prob_focus_present=0.
            # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely arrested attention across time)
    ):
        assert not (self.has_cond and not exists(cond)), 'cond must be passed in if cond_dim specified'

        x = self.init_conv(x)

        r = x.clone()
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        for idx,(block1, block2, temporal_block, downsample) in enumerate(self.downs):
            x = block1(x, t)
            x = block2(x, t)
            h.append(x)
            x = downsample(x)
            x = temporal_block(x, t)

        x = self.mid_block1(x, t)
        ###
        x = self.mid_spatial_attn1(x)
        x = self.mid_cross_attn1(x, kv=cond)
        x = self.mid_temporal_attn1(x, t)
        ###
        x = self.mid_spatial_attn2(x)
        x = self.mid_cross_attn2(x, kv=cond)
        x = self.mid_temporal_attn2(x, t)
        ###
        x = self.mid_spatial_attn3(x)
        x = self.mid_cross_attn3(x, kv=cond)
        x = self.mid_temporal_attn3(x, t)
        ###
        x = self.mid_spatial_attn4(x)
        x = self.mid_cross_attn4(x, kv=cond)
        x = self.mid_temporal_attn4(x, t)
        ###
        x = self.mid_block2(x, t)

        for block1, block2, temporal_block, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = upsample(x)
            x = block2(x, t)
            x = temporal_block(x, t)

        x = torch.cat((x, r), dim=1)
        return self.final_conv(x)


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)


class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            denoise_fn,
            *,
            image_size,
            num_frames,
            text_use_bert_cls=False,
            channels=3,
            timesteps=1000,
            loss_type='l1',
            use_dynamic_thres=False,  # from the Imagen paper
            dynamic_thres_percentile=0.9,
            volume_depth=128,
            ddim_timesteps=50,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn
        self.volume_depth = volume_depth

        betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        self.ddim_timesteps = ddim_timesteps

        ddim_timesteps = make_ddim_timesteps(ddim_discr_method="uniform", num_ddim_timesteps=200,
                                                  num_ddpm_timesteps=timesteps, verbose=True)

        # register buffer helper function that casts float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        ddim_eta = 0

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=ddim_timesteps,
                                                                                   eta=ddim_eta, verbose=True)

        ddim_alphas_prev = torch.from_numpy(ddim_alphas_prev)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                    1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

        # text conditioning parameters

        self.text_use_bert_cls = text_use_bert_cls

        # dynamic thresholding when sampling

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        x_t = rearrange(x_t, "b c f h w -> (b f) c h w")
        noise = rearrange(noise, "b c f h w -> (b f) c h w")
        t = rearrange(t, "b f -> (b f)")

        out = (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
               extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise)

        out = rearrange(out, "(b f) c h w-> b c f h w ", f=self.num_frames)
        return out

    def q_posterior(self, x_start, x_t, t):
        x_t = rearrange(x_t, "b c f h w -> (b f) c h w")
        x_start = rearrange(x_start, "b c f h w -> (b f) c h w")
        t = rearrange(t, "b f -> (b f)")

        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)

        posterior_mean = rearrange(posterior_mean, "(b f) c h w-> b c f h w ", f=self.num_frames)
        posterior_variance = rearrange(posterior_variance, "(b f) c h w-> b c f h w ", f=self.num_frames)
        posterior_log_variance_clipped = rearrange(posterior_log_variance_clipped, "(b f) c h w-> b c f h w ",
                                                   f=self.num_frames)

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, indexes=None, cond=None, cond_scale=1.):

        x_recon = self.denoise_fn.forward_with_cond_scale(x, t, indexes=indexes, cond=cond, cond_scale=cond_scale)
            # self.predict_start_from_noise(x, t=t, noise=self.denoise_fn.forward_with_cond_scale(x, t, indexes=indexes, cond=cond, cond_scale=cond_scale))

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )

                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            # clip by threshold, depending on whether static or dynamic
            x_recon = x_recon.clamp(-s, s) / s

        # model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        model_mean, posterior_variance = get_z_t_via_z_tp1(x_recon, x, (t - 1) * 1.0 / (self.num_timesteps - 1.0),
                                                           (t * 1.0) / (self.num_timesteps - 1.0))
        return model_mean, posterior_variance


    @torch.inference_mode()
    def p_sample(self, x, t, indexes=None, cond=None, cond_scale=1., clip_denoised=True):
        b, *_, device = *x.shape, x.device

        model_mean, model_variance = self.p_mean_variance(x=x, t=t, indexes=indexes, clip_denoised=clip_denoised,
                                                                 cond=cond,
                                                                 cond_scale=cond_scale)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, 1, self.num_frames, 1, 1)
        return model_mean + nonzero_mask * (model_variance**0.5) * noise

    @torch.inference_mode()
    def p_sample_ddim(self, x, t, t_minus, image=None, modal=None,  indexes=None, cond=None, cond_scale=1., clip_denoised=True):
        b, *_, device = *x.shape, x.device

        x_recon = self.denoise_fn.forward_with_cond_scale(x, t, indexes=indexes, cond=cond, cond_scale=cond_scale)

        if cond_scale > 1:
            x_recon, x_recon_null = x_recon
            eps = get_eps_x_t(x_recon, x, t)
            eps_null = get_eps_x_t(x_recon_null, x, t)
            final_eps = eps_null + (eps - eps_null) * cond_scale
            x_recon = get_x0_x_t(final_eps, x, t)

        if image is not None and modal is not None:
            for md in modal:
                x_recon[:, md, ...] = image[:, md, ...]

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )

                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            # clip by threshold, depending on whether static or dynamic
            x_recon = x_recon.clamp(-s, s) / s
        if t[0]<int(self.num_timesteps / self.ddim_timesteps):
            x = x_recon
        else:
            t_minus = torch.clip(t_minus, min=0.0)
            x = ddim_sample(x_recon, x, (t_minus * 1.0) / (self.num_timesteps), (t * 1.0) / (self.num_timesteps))
        return x

    @torch.inference_mode()
    def p_sample_loop(self, shape, image=None, modal=None, cond=None, cond_scale=1., use_ddim=True):
        device = self.betas.device

        bsz = shape[0]

        if use_ddim:
            time_steps = range(0, self.num_timesteps+1, int(self.num_timesteps/self.ddim_timesteps))
        else:
            time_steps = range(0, self.num_timesteps)

        img = torch.randn(shape, device=device)
        indexes = []
        for b in range(bsz):
            index = np.arange(self.num_frames)
            indexes.append(torch.from_numpy(index))
        indexes = torch.stack(indexes, dim=0).long().to(device)
        for i, t in enumerate(tqdm(reversed(time_steps), desc='sampling loop time step',
                                   total=len(time_steps))):
            time = torch.full((bsz,), t, device=device, dtype=torch.float32)

            if use_ddim:
                time_minus = time - int(self.num_timesteps / self.ddim_timesteps)
                img = self.p_sample_ddim(img, time, time_minus, image=image, modal=modal, indexes=indexes, cond=cond,
                                         cond_scale=cond_scale)
            else:
                img = self.p_sample(img, time, indexes=indexes, cond=cond,
                                    cond_scale=cond_scale)
        return unnormalize_img(img)

    @torch.inference_mode()
    def sample(self, image=None, modal=None, cond=None, cond_scale=2., batch_size=16, DDIM=True):
        device = next(self.denoise_fn.parameters()).device

        if is_list_str(cond):
            cond = bert_embed(tokenize(cond)).to(device)

        batch_size = cond.shape[0] if exists(cond) else batch_size
        image_size = self.image_size
        channels = self.channels
        num_frames = self.num_frames
        return self.p_sample_loop((batch_size, channels, num_frames, image_size, image_size), image=image, modal=modal,
                                  cond=cond, cond_scale=cond_scale, use_ddim=DDIM)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, indexes=None, cond=None, noise=None, **kwargs):
        b, c, f, h, w, device = *x_start.shape, x_start.device

        x_noisy, noise = get_z_t(x_start, t)

        if is_list_str(cond):
            cond = bert_embed(tokenize(cond), return_cls_repr=self.text_use_bert_cls)
            cond = cond.to(device)

        x_recon = self.denoise_fn(x_noisy, t*(self.num_timesteps-1), indexes=indexes, cond=cond, **kwargs)

        if self.loss_type == 'l1':
            loss = F.l1_loss(x_start, x_recon)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(x_start, x_recon)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, *args, **kwargs):
        b, device, img_size, = x.shape[0], x.device, self.image_size
        check_shape(x, 'b c f h w', c=self.channels, f=self.num_frames, h=img_size, w=img_size)
        t = torch.rand((b), device=device).float()
        return self.p_losses(x, t, *args, **kwargs)


# trainer class

CHANNELS_TO_MODE = {
    1: 'L',
    3: 'RGB',
    4: 'RGBA'
}


def seek_all_images(img, channels=3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1


# tensor of shape (channels, frames, height, width) -> gif

def video_tensor_to_gif(tensor, path, duration=120, loop=0, optimize=True):
    images = map(T.ToPILImage(), tensor.unbind(dim=1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all=True, append_images=rest_imgs, duration=duration, loop=loop, optimize=optimize)
    return images


# gif -> (channels, frame, height, width) tensor

def gif_to_tensor(path, channels=3, transform=T.ToTensor()):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels=channels)))
    return torch.stack(tensors, dim=1)


def identity(t, *args, **kwargs):
    return t


def normalize_img(t):
    return t * 2 - 1


def unnormalize_img(x_recon):
    x_recon = x_recon.clamp(-1, 1)
    return (x_recon + 1) * 0.5


def cast_num_frames(t, *, frames):
    f = t.shape[1]

    if f == frames:
        return t

    if f > frames:
        return t[:, :frames]

    return F.pad(t, (0, 0, 0, 0, 0, frames - f))


class Dataset(data.Dataset):
    def __init__(
            self,
            folder,
            image_size,
            channels=3,
            num_frames=16,
            horizontal_flip=False,
            force_num_frames=True,
            exts=['gif']
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.channels = channels
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.cast_num_frames_fn = partial(cast_num_frames, frames=num_frames) if force_num_frames else identity

        self.transform = T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        tensor = gif_to_tensor(path, self.channels, transform=self.transform)
        return self.cast_num_frames_fn(tensor)


# trainer class

class Trainer(object):
    def __init__(
            self,
            diffusion_model,
            folder,
            *,
            ema_decay=0.995,
            num_frames=16,
            train_batch_size=32,
            train_lr=1e-4,
            train_num_steps=100000,
            gradient_accumulate_every=2,
            amp=False,
            step_start_ema=2000,
            update_ema_every=10,
            save_and_sample_every=1000,
            results_folder='./results',
            num_sample_rows=4,
            max_grad_norm=None
    ):
        super().__init__()
        self.ema_model = diffusion_model
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        image_size = diffusion_model.image_size
        self.num_frames = diffusion_model.num_frames

        train_files = []

        for img_dir in os.listdir(folder):
            train_files.append({"image": os.path.join(folder, img_dir),
                                'text': os.path.join("dir of prompt features",
                                                     img_dir)})

        self.ds = cache_transformed_train_data(shape=[self.num_frames, image_size, image_size],
                                                   train_files=train_files)

        print(f'found {len(self.ds)} videos as gif files at {folder}')
        assert len(self.ds) > 0, 'need to have at least 1 video to start training (although 1 is not great, try 100k)'

        self.dl = data.DataLoader(self.ds, batch_size=train_batch_size, shuffle=False, pin_memory=True)

        self.step = 0

        self.amp = amp
        self.max_grad_norm = max_grad_norm

        self.num_sample_rows = num_sample_rows
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)

        if amp:
            mixed_precision = "fp16"
        else:
            mixed_precision = "no"

        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulate_every,
            mixed_precision=mixed_precision,
        )

        self.ema_model, self.dl = self.accelerator.prepare(self.ema_model, self.dl)

    def save(self, milestone):
        self.accelerator.save_state(str(self.results_folder / f'{milestone}_ckpt'))

    def load(self, milestone, **kwargs):
        if milestone == -1:
            dirs = os.listdir(self.results_folder)
            dirs = [d for d in dirs if d.endswith("ckpt")]
            dirs = sorted(dirs, key=lambda x: int(x.split("_")[0]))
            path = dirs[-1]

        self.step = int(path.split("_")[0]) * self.save_and_sample_every + 1

        self.accelerator.load_state(os.path.join(self.results_folder, path), strict=False)

    def train(
            self,
            prob_focus_present=0.,
            focus_present_mask=None,
            log_fn=noop,
            seed=1
    ):
        assert callable(log_fn)
        set_seed(seed, device_specific=True)

        modal = []
        results_folder = os.path.join(str(self.results_folder), f"diseases_cfg3.0_seed{str(seed)}_ddim100_cfg_eval_marginalize_"+str(modal))
        if not os.path.exists(results_folder) and self.accelerator.is_main_process:
            os.mkdir(results_folder)
        self.ema_model.eval()
        for i, data in enumerate(self.dl):

            image = data["image"]
            text = data["text"]
            text = text.to(self.accelerator.device)
            image = image.to(self.accelerator.device).squeeze(dim=1)

            with torch.no_grad():

                file_name = data['text_meta_dict']['filename_or_obj'][0].split('/')[-1]

                num_samples = self.num_sample_rows ** 2
                batches = num_to_groups(num_samples, self.batch_size)
                all_videos_list = list(
                    map(lambda n: self.ema_model.module.sample(image=image, cond_scale=3.0, modal=modal, batch_size=n, cond=text), batches))
                all_videos_list = torch.cat(all_videos_list, dim=0)
                np.save(os.path.join(results_folder, str(f'{file_name}')),
                        all_videos_list.cpu().numpy())
                all_videos_list, all_videos_list_lobe, all_videos_list_airway, all_videos_list_vessel = all_videos_list.chunk(
                    4, dim=1)
                all_videos_list = torch.cat(
                    [all_videos_list, all_videos_list_lobe, all_videos_list_airway, all_videos_list_vessel], dim=0)

                all_videos_list = F.pad(all_videos_list, (2, 2, 2, 2))

                one_gif = rearrange(all_videos_list, '(i j) c f h w -> c f (i h) (j w)',
                                    i=self.num_sample_rows)
                video_path = os.path.join(results_folder, str(f'{file_name}_modal_{str(modal)}.gif')).replace(".npy", "")
                video_tensor_to_gif(one_gif, video_path)

                #####
                image = (image+1.0)/2.0
                all_videos_list, all_videos_list_lobe, all_videos_list_airway, all_videos_list_vessel = image.chunk(
                    4, dim=1)
                all_videos_list = torch.cat(
                    [all_videos_list, all_videos_list_lobe, all_videos_list_airway, all_videos_list_vessel], dim=0)

                all_videos_list = F.pad(all_videos_list, (2, 2, 2, 2))

                one_gif = rearrange(all_videos_list, '(i j) c f h w -> c f (i h) (j w)',
                                    i=self.num_sample_rows)
                video_path = os.path.join(results_folder, str(f'{file_name}_real.gif')).replace(".npy", "")
                video_tensor_to_gif(one_gif, video_path)
            break

model = Unet3D(
    dim=160,
    cond_dim=768,
    dim_mults=(1, 2, 4, 8),
    channels=4,
    attn_heads=8,
    attn_dim_head=32,
    use_bert_text_cond=False,
    init_dim=None,
    init_kernel_size=7,
    use_sparse_linear_attn=True,
    block_type='resnet',
    resnet_groups=8
)

diffusion_model = GaussianDiffusion(
    denoise_fn=model,
    image_size=64,
    num_frames=64,
    text_use_bert_cls=False,
    channels=4,
    timesteps=1000,
    loss_type='l2',
    use_dynamic_thres=False,  # from the Imagen paper
    dynamic_thres_percentile=0.995,
    volume_depth=64,
    ddim_timesteps=100,
)

trainer = Trainer(diffusion_model=diffusion_model,
                  folder="your data",
                  ema_decay=0.995,
                  num_frames=64,
                  train_batch_size=1,
                  train_lr=1e-4,
                  train_num_steps=1000000,
                  gradient_accumulate_every=4,
                  amp=False,
                  step_start_ema=10000,
                  update_ema_every=10,
                  save_and_sample_every=1000,
                  results_folder='dir of saving generated data',
                  num_sample_rows=1,
                  max_grad_norm=1.0)
trainer.load(-1)
trainer.train(seed=1)
trainer.train(seed=2)
trainer.train(seed=3)
