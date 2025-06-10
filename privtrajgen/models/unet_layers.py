#!/usr/bin/env python3
"""UNet Layers. Based on the DiffTraj implementation."""
import logging
from typing import Optional

import torch
from torch import nn, Tensor

log = logging.getLogger()


def nonlinearity(x: torch.Tensor, activation: str = "swish") -> torch.Tensor:
    if activation == "swish":
        return x * torch.sigmoid(x)
    elif activation == "relu":
        return torch.relu(x)
    elif activation == "gelu":
        return torch.nn.functional.gelu(x)
    elif activation == "tanh":
        return torch.tanh(x)
    elif activation == "sigmoid":
        return torch.sigmoid(x)
    elif activation == "leaky_relu":
        return torch.nn.functional.leaky_relu(x, negative_slope=0.2)
    else:
        raise NotImplementedError(f"Activation '{activation}' not implemented.")

def Normalize(in_channels: int, num_groups: int = 32) -> torch.nn.GroupNorm:
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, with_conv:bool = True, target_length: Optional[int] = None):
        super().__init__()
        self.target_length = target_length
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.target_length is None:
            # Double the length
            x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        else:
            # Upsample to target length
            x = torch.nn.functional.interpolate(x, size=self.target_length, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool = True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """For input x of shape (batch, channels, length), output shape is (batch, channels, length//2)."""
        if self.with_conv:
            pad = (1, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool1d(x, kernel_size=2, stride=2)
        return x

class ResnetBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: Optional[int] = None,
                 conv_shortcut: bool = False,
                 dropout: float = 0.1,
                 temb_channels: Optional[int] = None,
                 activation: str = "swish"
                 ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.activation = activation

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels is not None and temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv1d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv1d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x: Tensor, temb: Optional[Tensor] = None) -> Tensor:
        h = x
        h = self.norm1(h)
        h = nonlinearity(h, activation=self.activation)
        h = self.conv1(h)
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb, activation=self.activation))[:, :, None]
        h = self.norm2(h)
        h = nonlinearity(h, activation=self.activation)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h

class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, w = q.shape
        q = q.permute(0, 2, 1)  # b,hw,c
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)
        # attend to values
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, w)

        h_ = self.proj_out(h_)

        return x + h_

def make_attn(in_channels: int, attn_type: str = "vanilla") -> nn.Module:
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    log.info(f"Making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        raise NotImplementedError(f'attn_type {attn_type} not implemented')
        # return LinAttnBlock(in_channels)
