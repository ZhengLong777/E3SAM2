"""Final two-scale Sobel edge supervision head."""

import numpy as np
import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


def get_sobel(in_channels: int, out_channels: int):
    filter_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    filter_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    filter_x = np.repeat(np.repeat(filter_x.reshape(1, 1, 3, 3), in_channels, 1), out_channels, 0)
    filter_y = np.repeat(np.repeat(filter_y.reshape(1, 1, 3, 3), in_channels, 1), out_channels, 0)
    conv_x = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
    conv_y = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
    conv_x.weight = nn.Parameter(torch.from_numpy(filter_x), requires_grad=False)
    conv_y.weight = nn.Parameter(torch.from_numpy(filter_y), requires_grad=False)
    return nn.Sequential(conv_x, nn.BatchNorm2d(out_channels)), nn.Sequential(conv_y, nn.BatchNorm2d(out_channels))


def run_sobel(conv_x: nn.Module, conv_y: nn.Module, x: torch.Tensor) -> torch.Tensor:
    gradient = torch.sqrt(torch.pow(conv_x(x), 2) + torch.pow(conv_y(x), 2))
    return torch.sigmoid(gradient) * x


class EdgeEncoder1(nn.Module):
    def __init__(self, in_chanls=(32, 64, 256), out_chanls=64):
        super().__init__()
        self.sobel_x1, self.sobel_y1 = get_sobel(in_chanls[0], 1)
        self.sobel_x3, self.sobel_y3 = get_sobel(in_chanls[2], 1)
        self.conv1_c1 = nn.Conv2d(in_chanls[0], out_chanls, 1)
        self.conv1_c3 = nn.Conv2d(in_chanls[2], out_chanls, 1)
        self.upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.conv3_GE_LN = nn.Sequential(
            nn.Conv2d(out_chanls * 2, out_chanls, 3, padding=1),
            nn.GELU(),
            LayerNorm2d(out_chanls),
            nn.Conv2d(out_chanls, out_chanls, 3, padding=1),
            nn.GELU(),
            LayerNorm2d(out_chanls),
        )
        self.lastconv = nn.Conv2d(out_chanls, 1, 1)

    def forward(self, feature_map):
        c1, _, c3 = feature_map
        c1 = self.conv1_c1(run_sobel(self.sobel_x1, self.sobel_y1, c1))
        c3 = self.upsample(self.conv1_c3(run_sobel(self.sobel_x3, self.sobel_y3, c3)))
        return self.lastconv(self.conv3_GE_LN(torch.cat((c1, c3), dim=1)))

