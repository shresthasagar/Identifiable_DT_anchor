"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelOnlyNorm(nn.Module):
    """Normalizes across channels only (not spatial), with learnable affine params."""
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            # per-channel gamma/beta (broadcast over H,W)
            self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def forward(self, x):  # x: (N,C,H,W)
        mean = x.mean(dim=1, keepdim=True)                 # (N,1,H,W)
        var = x.var(dim=1, unbiased=False, keepdim=True)   # (N,1,H,W)
        xhat = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            xhat = xhat * self.gamma + self.beta
        return xhat


class IdentityNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def get_norm_layer(norm_type, num_features, affine=True):
    """
    Returns a normalization layer based on norm_type.
    
    Args:
        norm_type: 'instance' for InstanceNorm2d, 'channel_only' for ChannelOnlyNorm
        num_features: number of channels
        affine: whether to use learnable affine parameters
    """
    if norm_type == 'channel_only':
        return ChannelOnlyNorm(num_features, affine=affine)
    elif norm_type == 'identity':
        return IdentityNorm()
    else:  # default to instance norm
        return nn.InstanceNorm2d(num_features, affine=affine)


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False, norm_type='instance'):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self.norm_type = norm_type
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = get_norm_layer(self.norm_type, dim_in, affine=True)
            self.norm2 = get_norm_layer(self.norm_type, dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features, norm_type='instance'):
        super().__init__()
        # AdaIN uses non-affine norm (gamma/beta come from style)
        self.norm = get_norm_layer(norm_type, num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample=False, norm_type='instance'):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self.norm_type = norm_type
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in, norm_type=self.norm_type)
        self.norm2 = AdaIN(style_dim, dim_out, norm_type=self.norm_type)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class InResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample=False, norm_type='instance'):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self.norm_type = norm_type
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = get_norm_layer(self.norm_type, dim_in, affine=True)
        self.norm2 = get_norm_layer(self.norm_type, dim_out, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class HighPass(nn.Module):
    def __init__(self, w_hpf, device):
        super(HighPass, self).__init__()
        self.register_buffer('filter',
                             torch.tensor([[-1, -1, -1],
                                           [-1, 8., -1],
                                           [-1, -1, -1]]) / w_hpf)

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))


class Generator(nn.Module):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=1, use_adain=True, num_downsample=None, in_channels=3, norm_type='instance'):
        super().__init__()
        dim_in = 2**14 // img_size
        self.img_size = img_size
        self.in_channels = in_channels
        self.norm_type = norm_type
        self.from_rgb = nn.Conv2d(in_channels, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_rgb = nn.Sequential(
            get_norm_layer(norm_type, dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, in_channels, 1, 1, 0))

        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size)) - 4 if num_downsample is None else num_downsample
        if w_hpf > 0:
            repeat_num += 1
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=True, norm_type=norm_type))
            if use_adain:
                self.decode.insert(
                    0, AdainResBlk(dim_out, dim_in, style_dim,
                                w_hpf=w_hpf, upsample=True, norm_type=norm_type))  # stack-like
            else:
                self.decode.insert(
                    0, InResBlk(dim_out, dim_in, style_dim,
                                w_hpf=w_hpf, upsample=True, norm_type=norm_type))
            dim_in = dim_out

        # bottleneck blocks
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True, norm_type=norm_type))
            if use_adain:
                self.decode.insert(
                    0, AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf, norm_type=norm_type))
            else:
                self.decode.insert(
                    0, InResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf, norm_type=norm_type))

        if w_hpf > 0:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            self.hpf = HighPass(w_hpf, device)

    def forward(self, x, s=None, masks=None):
        x = self.from_rgb(x)
        cache = {}
        for block in self.encode:
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                cache[x.size(2)] = x
            x = block(x)
        for block in self.decode:
            x = block(x, s)
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                mask = masks[0] if x.size(2) in [32] else masks[1]
                mask = F.interpolate(mask, size=x.size(2), mode='bilinear')
                x = x + self.hpf(mask * cache[x.size(2)])
        return self.to_rgb(x)


class Discriminator(nn.Module):
    def __init__(self, img_size=256, num_domains=2, max_conv_dim=512, in_channels=3, norm_type='instance'):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(in_channels, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True, norm_type=norm_type)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def forward(self, x, y):
        out = self.main(x)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        out_rel = torch.Tensor().to(y.device)
        cond_ids = []
        for i in range(y.size(1)):
            id = (y[:, i] == 1)
            out_rel = torch.cat((out_rel, out[id, i]))
            cond_ids.extend([i]*id.sum().item())
        return out_rel, cond_ids


class ConstantInput(nn.Module):
    def __init__(self, shape):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, *shape))
        self.ones = [1]*len(shape)
        
    def forward(self, x):
        batch = x.shape[0]
        out = self.input.repeat(batch, *self.ones)

        return out


class Translator(nn.Module):
    def __init__(self, img_size=256, style_dim=64, use_adain=True, w_hpf=1, num_downsample=None, in_channels=3, norm_type='instance'):
        super().__init__()
        self.use_adain = use_adain
        self.generator = Generator(img_size=img_size, style_dim=style_dim, use_adain=self.use_adain, w_hpf=w_hpf, num_downsample=num_downsample, in_channels=in_channels, norm_type=norm_type)
        if self.use_adain:
            self.const_style_encoder = ConstantInput(shape=(style_dim,))
        
    def forward(self, x):
        if self.use_adain:
            style = self.const_style_encoder(x)
            return self.generator(x, style)
        else:
            return self.generator(x)


class FullyConnectedTranslator(nn.Module):
    def __init__(self, img_size=32, in_channels=3, hidden_dim=1024):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.input_dim = in_channels * img_size * img_size
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.input_dim),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.net(x)
        return out.view(x.size(0), self.in_channels, self.img_size, self.img_size)


class FullyConnectedDiscriminator(nn.Module):
    def __init__(self, img_size=32, num_domains=2, in_channels=3, hidden_dim=1024):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.input_dim = in_channels * img_size * img_size
        self.num_domains = num_domains
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, num_domains)
        )

    def forward(self, x, y):
        out = self.net(x)
        out_rel = torch.Tensor().to(y.device)
        cond_ids = []
        for i in range(y.size(1)):
            id = (y[:, i] == 1)
            out_rel = torch.cat((out_rel, out[id, i]))
            cond_ids.extend([i] * id.sum().item())
        return out_rel, cond_ids
