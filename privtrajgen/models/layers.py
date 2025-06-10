#!/usr/bin/env python3
"""Helpful layers for the models"""
import logging
from typing import Any

import torch
from torch import nn

from privtrajgen.config import NormOption, ActivationOption

log = logging.getLogger()

ACTIVATION_MAP = {
    ActivationOption.RELU: nn.ReLU,
    ActivationOption.LEAKY_RELU: nn.LeakyReLU,
    ActivationOption.TANH: nn.Tanh,
    ActivationOption.SIGMOID: nn.Sigmoid,
}


class Norm(nn.Module):
    """Abstraction Layer to quickly switch between different normalization layers"""

    def __init__(self, channels: int, norm_type: NormOption, channels_last: bool = False):
        """

        :param channels: Number of channels
        :param norm_type: see NORM_OPTIONS
        :param channels_last: If false, input expected to be (N, C, L) or (N, C), if true (N, L, C)
            Of course, in the case of (N, C), the channels are last, but in this case the option is ignored.
        """
        super().__init__()
        self.channels = channels
        self.norm_type = norm_type
        self.channels_last = channels_last
        if norm_type == NormOption.BATCH_1D or norm_type == NormOption.BATCH:
            self.norm: nn.Module = nn.BatchNorm1d(channels)
        elif norm_type == NormOption.LAYER:
            self.norm = nn.LayerNorm(channels)
        elif norm_type == NormOption.NONE:
            self.norm = nn.Identity()
        elif norm_type == NormOption.DROPOUT:
            self.norm = nn.Dropout(0.5)
        elif norm_type == NormOption.GROUP:
            # Based on DiffTraj
            self.norm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        log.debug(f"Norm: x.shape = {x.shape}")
        swap = self.norm_type == 'layer' and len(x.shape) > 2 and not self.channels_last
        if swap:
            # x might either have shape (N, C) or (N, C, L), but for layer norm, C needs to be in the end!
            x = x.transpose(2, 1)
        y = self.norm(x)
        if swap:
            y = y.transpose(2, 1)
        return y


class Activation(nn.Module):
    """Abstraction Layer to quickly switch between different activation functions"""

    def __init__(self, mode: ActivationOption):
        super().__init__()
        if mode == ActivationOption.LEAKY_RELU:
            self.activation = nn.LeakyReLU(0.2)
        elif mode in ACTIVATION_MAP:
            self.activation = ACTIVATION_MAP[mode]()
        else:
            raise ValueError(f"Unknown activation: {mode}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x)

    def __str__(self) -> str:
        return str(self.activation)


class CustomLinear(nn.Module):
    """
    Abstraction Layer to quickly switch between different linear and equivalent layers (conv1d).
    This was just an experiment, but it does not make much sense to use this as a Linear Layer
    and Conv1D layer with kernel size 1 are equivalent in terms of output.
    Of course, this only true for batches of shape (N, C).
    """

    def __init__(self, in_features: int, out_features: int, *args: Any, mode: str = 'linear'):
        super().__init__()
        self.mode = mode
        self.in_features = in_features
        self.out_features = out_features
        if mode == 'linear':
            self.layer = nn.Linear(in_features, out_features, *args)
        elif mode == 'conv1d':
            self.layer = nn.Conv1d(in_features, out_features, 1, *args)
        else:
            raise ValueError(f"Unknown Layer type: {mode}.")
        self.weight = self.layer.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transpose = False
        if self.mode == 'conv1d' and x.dim() < 3:
            x = x.view(-1, self.in_features, 1)
            transpose = True
        x = self.layer(x)
        if transpose:
            x = x.view(-1, self.out_features)
        return x


class STN_Seq(nn.Module):
    def __init__(self, k: int = 2, apply_transform: bool = True, depth: int = 5, use_batch_norm: bool = True):
        """
        Modified Spatial Transformer Network for sequential point data.
        This is based on the Large Spatial Transformer Network (STN) module from GeoPointGAN.
        However, we modify it to support sequences of 2D points instead of batches of 2D points.
        In order to do so, we flatten the input from (batch_size, sequence_length, k) to (batch_size * sequence_length, k).
        Moreover, we replace all Conv1d layers with Linear layers. However, as the original implementation uses Conv1D layers with kernel size 1, this should not make a (huge)difference, as Conv1D with kernel size 1 are mostly identical to Linear layers, except for initialisation and implementation details.

        :param k: Number of dimensions for each point in the sequence
        :param apply_transform: If True, the transformation is applied to the input. If False, the transformation matrix is returned.
        :param depth: Number of layers in the STN (Will be doubled, as we move up and down in the number of features)
        :return: If apply_transform is True, the transformed input is returned. Otherwise, the transformation matrix is returned.
        """
        super().__init__()
        self.k = k  # Number of features for each point in the sequence is k
        self.apply_transform = apply_transform
        self.depth = depth
        self.use_batch_norm = use_batch_norm

        if depth < 3:
            raise ValueError(f"Depth must be at least 3, but got {depth}.")

        # First is 32, then double up to 32 * 2 ** (depth - 1)
        feature_dims = [32 * 2 ** i for i in range(0, depth)]
        # Add k to the start of the list
        feature_dims.insert(0, k)
        # Move down until 128 (=32 * 2 ** 2)
        for i in range(depth - 2, 0, -1):
            feature_dims.append(32 * 2 ** i)
        # Add k*k as final layer
        feature_dims.append(k * k)

        assert len(feature_dims) == 2 * depth, \
            f"Expected {2 * depth} feature dimensions, but got {len(feature_dims)} dimensions:{feature_dims}."

        # Create the layers
        self.fc_layers = nn.ModuleList()
        for i in range(len(feature_dims) - 1):
            self.fc_layers.append(nn.Linear(feature_dims[i], feature_dims[i + 1]))

        assert len(self.fc_layers) == 2 * depth - 1, f"Expected {2 * depth - 2} layers, but got {len(self.fc_layers)} layers."

        # Create BatchNorm1d layers
        self.norm_layers = nn.ModuleList()
        if use_batch_norm:
            for i in range(1, len(self.fc_layers)):
                self.norm_layers.append(nn.BatchNorm1d(feature_dims[i]))

            assert len(self.norm_layers) == len(self.fc_layers) - 1, f"Expected {len(self.fc_layers) - 1} BatchNorm1d layers, but got {len(self.norm_layers)} layers."

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the STN_Seq module.
        :param x: Input tensor of shape (batch_size, sequence_length, k)
        :return: Transformed input tensor of shape (batch_size, sequence_length, k) if apply_transform is True,
                 otherwise the transformation matrix of shape (batch_size, sequence_length, k, k)
        """
        assert x.dim() == 3, f"Input tensor must have 3 dimensions, but has {x.dim()} dimensions."
        batchsize, sequence_length, features = x.size()
        assert features == self.k, f"Input tensor has {features} features, but expected {self.k} features."

        # Process each point in the sequence with the same set of linear transformations
        # Reshape x to treat each point independently but apply the same transformations
        trans = x.view(batchsize * sequence_length, -1)

        for i in range(len(self.fc_layers)):
            # Linear Layer
            trans = self.fc_layers[i](trans)
            # Norm Layer
            if self.use_batch_norm and i < len(self.norm_layers):
                # No Norm for the last layer
                trans = self.norm_layers[i](trans)
            # Activation (except for the last layer)
            if i < len(self.fc_layers) - 1:
                trans = trans.relu()

        iden = torch.eye(self.k, dtype=torch.float32, device=x.device).flatten().view(1, self.k * self.k).repeat(
            batchsize * sequence_length, 1)

        trans = trans + iden
        trans = trans.view(batchsize * sequence_length, self.k, self.k)
        # trans has shape (batchsize * sequence_length, k, k)

        if self.apply_transform:
            # x has shape (batchsize, sequence_length, k)
            # Reshape x to (batchsize * sequence_length, 1, k)
            x = x.view(batchsize * sequence_length, 1, self.k)
            # Apply transformation to each point in the sequence
            x = torch.bmm(x, trans).transpose(2, 1)
            # Reshape back to (batchsize, sequence_length, k)
            x = x.view(batchsize, sequence_length, self.k)
        else:
            x = trans

        return x
