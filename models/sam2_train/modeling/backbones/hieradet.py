
from functools import partial
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sam2_train.modeling.backbones.utils import PatchEmbed, window_partition, window_unpartition
from models.sam2_train.modeling.sam2_utils import DropPath, MLP


def do_pool(x: torch.Tensor, pool: nn.Module | None) -> torch.Tensor:
    if pool is None:
        return x
    return pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class MultiScaleAttention(nn.Module):
    def __init__(self, dim: int, dim_out: int, num_heads: int, q_pool: nn.Module | None = None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.scale = (dim_out // num_heads) ** -0.5
        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, height, width, _ = x.shape
        qkv = self.qkv(x).reshape(batch, height * width, 3, self.num_heads, -1)
        q, k, v = torch.unbind(qkv, 2)
        if self.q_pool:
            q = do_pool(q.reshape(batch, height, width, -1), self.q_pool)
            height, width = q.shape[1:3]
            q = q.reshape(batch, height * width, self.num_heads, -1)
        x = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.proj(x.transpose(1, 2).reshape(batch, height, width, -1))


class EntropyAttention(nn.Module):
    global_attention_maps: list[tuple[torch.Tensor, torch.Tensor]] = []

    def __init__(self, channels: int, eps: float = 1e-8, score_mode: str = "power",
                 gamma: float = 4.0, gate_mode: str = "attention"):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.score_mode = score_mode
        self.gamma = gamma
        self.gate_mode = gate_mode
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm([channels])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        flat = x.view(batch, channels, height * width).permute(0, 2, 1)
        query, key, value = self.query(flat), self.key(flat), self.value(flat)
        attention = F.softmax(query @ key.transpose(-2, -1) / channels**0.5, dim=-1)
        entropy = -(attention * torch.log(attention + self.eps)).sum(dim=-1)
        maximum = torch.log(torch.tensor(attention.size(-1), device=x.device, dtype=entropy.dtype))
        entropy = entropy / (maximum + self.eps)
        if self.training:
            self.global_attention_maps.append((attention, entropy))
        gate = (1.0 - entropy).clamp(min=1e-3) ** self.gamma
        attention = attention * gate.unsqueeze(-1)
        attention = attention / (attention.sum(dim=-1, keepdim=True) + self.eps)
        output = self.norm(attention @ value + flat)
        return output.permute(0, 2, 1).view(batch, channels, height, width)


class Local(nn.Module):
    def __init__(self, dim: int, bias: bool, growth_rate: float = 2.0):
        super().__init__()
        hidden = int(dim * growth_rate)
        self.conv_1_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv3x3 = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2, bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden * 2, hidden * 2, 5, padding=2, groups=hidden * 2, bias=bias)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.conv_1_out = nn.Conv2d(hidden * 4, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1_in(x)
        return self.conv_1_out(torch.cat([self.relu3(self.dwconv3x3(x)), self.relu5(self.dwconv5x5(x))], dim=1))


class LocalGlobalFusion(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, bias: bool = False):
        super().__init__()
        self.local_branch = Local(dim, bias)
        self.global_branch = EntropyAttention(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.local_branch(x) + self.global_branch(x)


class Adapter_G_MSB(nn.Module):
    def __init__(self, blk: nn.Module):
        super().__init__()
        self.block = blk
        self.glf = LocalGlobalFusion(blk.attn.qkv.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prompt = self.glf(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return self.block(prompt)


class MultiScaleBlock(nn.Module):
    def __init__(self, dim: int, dim_out: int, num_heads: int, mlp_ratio: float = 4.0,
                 drop_path: float = 0.0, norm_layer: Union[nn.Module, str] = "LayerNorm",
                 q_stride: Tuple[int, int] | None = None, act_layer: nn.Module = nn.GELU,
                 window_size: int = 0):
        super().__init__()
        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        self.window_size = window_size
        self.pool = nn.MaxPool2d(q_stride, stride=q_stride, ceil_mode=False) if q_stride else None
        self.q_stride = q_stride
        self.attn = MultiScaleAttention(dim, dim_out, num_heads, self.pool)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim_out)
        self.mlp = MLP(dim_out, int(dim_out * mlp_ratio), dim_out, num_layers=2, activation=act_layer)
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)
        window_size = self.window_size
        if window_size > 0:
            height, width = x.shape[1:3]
            x, padded_hw = window_partition(x, window_size)
        x = self.attn(x)
        if self.q_stride:
            window_size = self.window_size // self.q_stride[0]
            height, width = shortcut.shape[1:3]
            padded_hw = (height + (window_size - height % window_size) % window_size,
                         width + (window_size - width % window_size) % window_size)
        if self.window_size > 0:
            x = window_unpartition(x, window_size, padded_hw, (height, width))
        x = shortcut + self.drop_path(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class Hiera(nn.Module):
    def __init__(self, embed_dim: int = 96, num_heads: int = 1, drop_path_rate: float = 0.0,
                 q_pool: int = 3, q_stride: Tuple[int, int] = (2, 2),
                 stages: Tuple[int, ...] = (1, 2, 11, 2), dim_mul: float = 2.0,
                 head_mul: float = 2.0, window_pos_embed_bkg_spatial_size: Tuple[int, int] = (7, 7),
                 window_spec: Tuple[int, ...] = (8, 4, 14, 7),
                 global_att_blocks: Tuple[int, ...] = (7, 10, 13), return_interm_layers: bool = True):
        super().__init__()
        self.window_spec = window_spec
        depth = sum(stages)
        self.q_stride = q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers
        self.patch_embed = PatchEmbed(embed_dim=embed_dim)
        self.global_att_blocks = global_att_blocks
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, *window_pos_embed_bkg_spatial_size))
        self.pos_embed_window = nn.Parameter(torch.zeros(1, embed_dim, window_spec[0], window_spec[0]))
        drop_rates = torch.linspace(0, drop_path_rate, depth).tolist()
        stage = 1
        blocks = []
        for index in range(depth):
            dim_out = embed_dim
            window_size = 0 if index in global_att_blocks else window_spec[stage - 1]
            if index - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                stage += 1
            blocks.append(MultiScaleBlock(embed_dim, dim_out, num_heads, drop_path=drop_rates[index],
                                          q_stride=q_stride if index in self.q_pool_blocks else None,
                                          window_size=window_size))
            embed_dim = dim_out
        self.blocks = nn.ModuleList(blocks)
        self.channel_list = [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
        adapter_indices = [end + 1 for end in self.stage_ends]
        self.blocks = nn.ModuleList([Adapter_G_MSB(block) if i in adapter_indices else block
                                     for i, block in enumerate(self.blocks)])
        for block in self.blocks:
            if not isinstance(block, Adapter_G_MSB):
                for parameter in block.parameters():
                    parameter.requires_grad = False

    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        pos = F.interpolate(self.pos_embed, size=hw, mode="bicubic")
        pos = pos + self.pos_embed_window.tile([x // y for x, y in zip(pos.shape, self.pos_embed_window.shape)])
        return pos.permute(0, 2, 3, 1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.patch_embed(x)
        x = x + self._get_pos_embed(x.shape[1:3])
        outputs = []
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index == self.stage_ends[-1] or (index in self.stage_ends and self.return_interm_layers):
                outputs.append(x.permute(0, 3, 1, 2))
        return outputs

