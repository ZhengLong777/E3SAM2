# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Some utilities for backbones, in particular for windowing"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

def window_partition(x, window_size):
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.
    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    """
    Window unpartition into original sequences and removing padding.
    Args:
        x (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.
    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(
        B, Hp // window_size, Wp // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, ...] = (7, 7),
        stride: Tuple[int, ...] = (4, 4),
        padding: Tuple[int, ...] = (3, 3),
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
        """
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x



# -----------------------------
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        # x: [B, C, H, W] → permute to [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x

# 1. LoG Filter
# -----------------------------
class LoGFilter(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, sigma, norm_layer, act_layer):
        super().__init__()
        self.conv_init = nn.Conv2d(in_c, out_c, kernel_size=7, stride=1, padding=3)
        ax = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = (xx**2 + yy**2 - 2 * sigma**2) / (2 * math.pi * sigma**4) * torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel - kernel.mean()
        kernel = kernel / kernel.sum()
        log_kernel = kernel.unsqueeze(0).unsqueeze(0)
        self.LoG = nn.Conv2d(out_c, out_c, kernel_size=kernel_size, stride=1, padding=kernel_size//2,
                             groups=out_c, bias=False)
        self.LoG.weight.data = log_kernel.repeat(out_c, 1, 1, 1)
        self.norm1 = norm_layer(out_c)
        self.norm2 = norm_layer(out_c)
        self.act = act_layer()

    def forward(self, x):
        x = self.conv_init(x)
        log_out = self.LoG(x)
        log_edge = self.act(self.norm1(log_out))
        x = self.norm2(x + log_edge)
        return x

# -----------------------------
# 2. Gaussian Module
# -----------------------------
class Gaussian(nn.Module):
    def __init__(self, dim, size, sigma, norm_layer, act_layer, feature_extra=False):
        super().__init__()
        self.feature_extra = feature_extra
        ax = torch.arange(-(size // 2), (size // 2) + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, size, size)
        self.gaussian = nn.Conv2d(dim, dim, kernel_size=size, padding=size//2, groups=dim, bias=False)
        self.gaussian.weight.data = kernel.repeat(dim, 1, 1, 1)
        self.norm = norm_layer(dim)
        self.act = act_layer()
        if feature_extra:
            self.extra = nn.Sequential(
                nn.Conv2d(dim, dim, 1), act_layer(),
                nn.Conv2d(dim, dim, 3, padding=1), act_layer(),
                nn.Conv2d(dim, dim, 1)
            )
        else:
            self.extra = None

    def forward(self, x):
        x_out = self.gaussian(x)
        x_out = self.act(self.norm(x_out))
        if self.feature_extra:
            x_out = self.extra(x + x_out)
        return x_out

# -----------------------------
# 3. DRFD Module
# -----------------------------
class DRFD(nn.Module):
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim * 2, 3, padding=1, groups=dim)
        self.conv_c = nn.Conv2d(dim * 2, dim * 2, 3, stride=2, padding=1, groups=dim * 2)
        self.act_c = act_layer()
        self.norm_c = norm_layer(dim * 2)
        self.max_m = nn.MaxPool2d(3, stride=2, padding=1)
        self.norm_m = norm_layer(dim * 2)
        self.fusion = nn.Conv2d(dim * 4, dim * 2, 1)
        self.gaussian = Gaussian(dim * 2, 5, 0.5, norm_layer, act_layer, feature_extra=False)
        self.norm_g = norm_layer(dim * 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm_g(x + self.gaussian(x))
        max_m = self.norm_m(self.max_m(x))
        conv_c = self.norm_c(self.act_c(self.conv_c(x)))
        x = torch.cat([conv_c, max_m], dim=1)
        x = self.fusion(x)
        return x

# -----------------------------
# 4. StructureAwarePatchEmbed
# -----------------------------
class StructureAwarePatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=768, act_layer=nn.ReLU, norm_layer=LayerNorm2d):
        super().__init__()
        dim_c1 = embed_dim // 4
        dim_c2 = embed_dim // 2
        self.log_filter = LoGFilter(in_chans, dim_c1, 7, 1.0, norm_layer, act_layer)
        self.downsample = nn.Sequential(
            nn.Conv2d(dim_c1, dim_c2, 3, padding=1, stride=1, groups=dim_c1),
            nn.Conv2d(dim_c2, dim_c2, 3, padding=1, stride=2, groups=dim_c2),
            norm_layer(dim_c2)
        )
        self.gaussian = Gaussian(dim_c2, 9, 0.5, norm_layer, act_layer)

        self.proj = nn.Sequential(
            nn.Conv2d(dim_c2, embed_dim, kernel_size=3, stride=2, padding=1),
            norm_layer(embed_dim)
        )
        # self.drfd = DRFD(dim_c2, norm_layer, act_layer)
        # self.proj = nn.Sequential(
        #     nn.Conv2d(dim_c2 * 2, embed_dim, kernel_size=3, stride=2, padding=1),
        #     norm_layer(embed_dim)
        # )

    def forward(self, x):
        x = self.log_filter(x)
        x = self.downsample(x)
        x = x + self.gaussian(x)
        # x = self.drfd(x)
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)  # B C H W → B H W C
        return x

