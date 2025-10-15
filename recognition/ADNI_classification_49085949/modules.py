# Reference:
#   GitHub: https://github.com/facebookresearch/ConvNeXt
#   Paper: "A ConvNet for the 2020s" - https://arxiv.org/pdf/2201.03545.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, DropPath
# from timm.models.registry import register_model

class Block(nn.Module):
    """
    ConvNeXt Block: depthwise conv + pointwise conv + residual + LayerNorm + optional LayerScale
    Implementation (channels_last) for speed.

    Args:
        dim (int): Number of input channels
        drop_path (float): Stochastic depth rate
        layer_scale_init_value (float): Init value for LayerScale parameter gamma
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)  # LayerNorm in channels_last format
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise conv (1x1) as Linear
        self.act = nn.GELU()  # GELU activation
        self.pwconv2 = nn.Linear(4 * dim, dim)  # project back to original channels
        # Layer scale gamma (optional)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) \
                     if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # stochastic depth

    def forward(self, x):
        input = x
        x = self.dwconv(x)  # depthwise conv
        x = x.permute(0, 2, 3, 1)  # (N,C,H,W) -> (N,H,W,C) for LayerNorm & Linear
        x = self.norm(x)  # layer normalization
        x = self.pwconv1(x)  # expand channels
        x = self.act(x)  # GELU
        x = self.pwconv2(x)  # project back
        if self.gamma is not None:
            x = self.gamma * x  # apply LayerScale
        x = x.permute(0, 3, 1, 2)  # (N,H,W,C) -> (N,C,H,W)
        x = input + self.drop_path(x)  # residual connection with optional stochastic depth
        return x

class ConvNeXt(nn.Module):
    """
    ConvNeXt model
    Args:
        in_chans (int): input image channels
        num_classes (int): classification head output
        depths (list[int]): number of blocks per stage
        dims (list[int]): channel dimensions per stage
        drop_path_rate (float): stochastic depth rate
        layer_scale_init_value (float): init value for LayerScale gamma
        head_init_scale (float): init scaling for classification head
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.):
        super().__init__()

        # downsample layers (stem + 3 stages)
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),  # stem conv
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")  # norm
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),  # norm
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),  # downsample
            )
            self.downsample_layers.append(downsample_layer)

        # stages: stack of blocks per stage
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth schedule
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur+j], layer_scale_init_value=layer_scale_init_value)
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # final norm and classifier head
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final LayerNorm
        self.head = nn.Linear(dims[-1], num_classes)  # classification head

        # initialize weights
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)  # truncated normal init
            nn.init.constant_(m.bias, 0)  # bias init to 0

    def forward_features(self, x):
        # compute features stage by stage
        for i in range(4):
            x = self.downsample_layers[i](x)  # downsample
            x = self.stages[i](x)  # block stack
        x = x.mean([-2, -1])  # global average pooling
        x = self.norm(x)  # final norm
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)  # classification head
        return x

class LayerNorm(nn.Module):
    """
    Custom LayerNorm supporting channels_first or channels_last
    Args:
        normalized_shape (int)
        eps (float)
        data_format (str): "channels_last" or "channels_first"
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:  # channels_first
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

