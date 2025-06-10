#!/usr/bin/env python3
"""Implements the Clipping Methods used for the conditioanl information."""
import logging
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Any

import torch
from pydantic import validate_call
from torch import nn
from torch.nn import functional as F

from privtrajgen.config import ConditionalInfoClippingType, NoiseType
from privtrajgen.evaluation.config import ConditionalInfoConfig

log = logging.getLogger(__name__)


class Clipping(ABC, nn.Module):
    """Base class for clipping methods."""

    @validate_call
    def __init__(self, max_norm: float, norm_type: float, *args: Any, **kwargs: Any):
        """Enforce that these parameters are set correctly."""
        super().__init__()
        self.max_norm: float = max_norm
        self.norm_type: float = norm_type

    @staticmethod
    def check_input(data: torch.Tensor) -> None:
        """Verify that the input tensor is 2-dimensional."""
        if data.dim() != 2:
            raise ValueError(f"Input tensor must be 2D, but got shape {data.shape}")

    @abstractmethod
    def clip(self, data: torch.Tensor) -> torch.Tensor:
        """
        Clips the norm of a tensor to ensure its norm does not exceed `max_norm`.

        :param data: The input tensor to be clipped. Shape: (batch_size, embedding_dim)
        :return: The clipped tensor with a per sample norm not exceeding `max_norm`.
        """
        raise NotImplementedError

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.clip(data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_norm={self.max_norm}, norm_type={self.norm_type})"

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return self.clip(data)

    @property
    @abstractmethod
    def apply_activation(self) -> bool:
        """Return True if this module internally applies an activation function."""
        raise NotImplementedError


class NormClipping(Clipping):
    """
        Clips the norm of a tensor to ensure its norm does not exceed `max_norm`.
        This method scales the entire tensor to keep the specified norm constraint,
        similar to PyTorch's `clip_grad_norm_`.

        :param max_norm: The maximum allowed norm for the tensor.
        :param norm_type: The type of norm to use for clipping (e.g., L1, L2).
        :param scale_up: Whether to scale up the tensor if the norm is below `max_norm`.
            Usually, we don't want this (DP-SGD does not) but the VMF mechanism requires it:
            The VMF mechanism requires scaling not clipping! (see https://arxiv.org/abs/2211.04686)
    """

    @validate_call
    def __init__(self, max_norm: float, norm_type: float, scale_up: bool = False):
        super().__init__(max_norm, norm_type)
        self.scale_up = scale_up
        self.eps = 1e-8  # Small value to prevent division by zero
        # If we scale up, our max_clamp should be infinity, otherwise it should be 1.0.
        if self.scale_up:
            self.max_clamp = torch.inf
            self.eps = 0.0  # We need to make sure all vectors are unit vectors
            log.info("Using scale up for norm clipping.")
        else:
            self.max_clamp = 1.0


    @property
    def apply_activation(self) -> bool:
        return False

    def clip(self, data: torch.Tensor) -> torch.Tensor:
        """
        Clips the norm of a tensor to ensure its norm does not exceed `max_norm`.

        :param data: The input tensor to be clipped. Shape: (batch_size, embedding_dim)

        :return: The clipped tensor with a per sample norm not exceeding `max_norm`.
        """
        # Check input
        self.check_input(data)

        # Calculate the norm for each sample in the batch
        total_norm = data.norm(p=self.norm_type, dim=1, keepdim=True)

        # Calculate the clipping coefficient  (We cannot use eps for VMF).
        clip_coefficient = self.max_norm / (total_norm + self.eps)

        # Clamp the coefficient to a maximum of 1 to prevent scaling up
        clip_coefficient = torch.clamp(clip_coefficient, max=self.max_clamp)

        # Multiply by the clamped coefficient (avoiding conditionals for GPU compatibility)
        data = data * clip_coefficient

        return data


class MaxClipping(Clipping):
    """
        Scales the input tensor to ensure that the total norm is bounded by `max_norm`.

        $$ d_{scaled} = \\frac{C}{M} \\cdot d $$
        where $M$ is the maximal possible norm of the tensor.
        M is calculated as the norm of the tensor if all values are at their maximal sensitivity.

        :param max_norm: The maximum allowed norm for the tensor.
        :param norm_type: The type of norm to use for clipping (e.g., L1, L2).
        :param apply_sigmoid: Whether to apply a sigmoid activation to the embedding before clipping.
        :param feature_sensitivity: The sensitivity of each feature in the tensor. (e.g., 1 for sigmoid activation,
            2 for tanh activation, etc.). Mandatory if apply_sigmoid is False.
        :param check: Whether to check if the feature sensitivity is exceeded. Disable for performance.

        :return: The clipped tensor with a per sample norm not exceeding `max_norm`.
    """

    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(self, max_norm: float, norm_type: float, apply_sigmoid: bool, ndim: int,
                 feature_sensitivity: Optional[Union[float, List[float]]] = None,
                 check: bool = True, device: Optional[torch.device] = None):
        super().__init__(max_norm, norm_type)
        self.apply_sigmoid = apply_sigmoid
        if self.apply_sigmoid:
            feature_sensitivity = 1.0
        elif feature_sensitivity is None:
                raise ValueError("Feature sensitivity must be provided if apply_sigmoid is False.")
        self.check = check

        # Calculate the maximal possible norm (M) based on feature sensitivity
        if isinstance(feature_sensitivity, float):
            self.sensitivities = torch.full((ndim,), feature_sensitivity)
        else:
            assert len(feature_sensitivity) == ndim, "Feature sensitivity must have the same length as the embedding."
            self.sensitivities = torch.tensor(feature_sensitivity)

        assert self.sensitivities.shape == (ndim,), "Shape of sensitivities must match the embedding dimension."

        # Compute the maximal possible norm (M) based on feature sensitivity
        self.max_possible_norm = self.sensitivities.norm(p=self.norm_type, dim=0, keepdim=False)

        # Calculate the clipping coefficient
        self.clip_coefficient = self.max_norm / self.max_possible_norm

        # Verify that the clipping coefficient is a scalar
        assert self.clip_coefficient.dim() == 0, "Clipping coefficient must be a scalar."

        # Move the clipping coefficient to the device
        if device is not None:
            self.clip_coefficient = self.clip_coefficient.to(device)

    @property
    def apply_activation(self) -> bool:
        return self.apply_sigmoid

    def clip(self, data: torch.Tensor) -> torch.Tensor:
        """
            Clips the norm of a tensor to ensure its norm does not exceed `max_norm`.

            :param data: The input tensor to be clipped. Shape: (batch_size, embedding_dim)

            :return: The clipped tensor with a per sample norm not exceeding `max_norm`.
        """
        # Verify that tensor is 2-dimensional
        self.check_input(data)

        # Apply sigmoid activation to the embedding to ensure all values are in [0, 1]
        if self.apply_sigmoid:
            data = F.sigmoid(data)
        else:
            if self.check:
                # Assert that no feature exceeds the feature sensitivity
                # Performance overhead -> skip for sigmoid activation

                # Compute the per-feature range (maximum - minimum) across the batch
                data_min, _ = data.min(dim=0)
                data_max, _ = data.max(dim=0)
                data_range = data_max - data_min
                # Check if any feature's range exceeds the feature sensitivity
                if torch.any(data_range > self.sensitivities):
                    raise ValueError(
                        "Data range exceeds feature sensitivity. Please ensure that data is scaled properly.")

        # Scale data by the clipped coefficient
        data = data * self.clip_coefficient

        return data


def get_clipping_method(config: ConditionalInfoConfig) -> Clipping:
    """
    Get the clipping method based on the configuration.

    :param config: The configuration containing the clipping information.

    :return: The clipping method.
    """
    if config.clipping_type == ConditionalInfoClippingType.NORM_CLIPPING:
        scale_up = (config.noise == NoiseType.VMF)
        return NormClipping(max_norm=config.max_norm, norm_type=config.norm_type, scale_up=scale_up)
    elif config.clipping_type == ConditionalInfoClippingType.MAX_CLIPPING:
        return MaxClipping(max_norm=config.max_norm, norm_type=config.norm_type,
                           apply_sigmoid=True, ndim=config.embedding_dim)
    else:
        raise ValueError(f"Unknown clipping type: {config.clipping_type}")


# Functional API
def clip_to_norm(data: torch.Tensor, max_norm: float, norm_type: float, scale_up: bool = False) -> torch.Tensor:
    """
    Clips the norm of a tensor to ensure its norm does not exceed `max_norm`.

    :param data: The input tensor to be clipped. Shape: (batch_size, embedding_dim)
    :param max_norm: The maximum allowed norm for the tensor.
    :param norm_type: The type of norm to use for clipping (e.g., L1, L2).
    :param scale_up: Whether to scale up the tensor if the norm is below `max_norm`.

    :return: The clipped tensor with a per sample norm not exceeding `max_norm`.
    """
    return NormClipping(max_norm=max_norm, norm_type=norm_type, scale_up=scale_up)(data)


def clip_to_max(data: torch.Tensor, max_norm: float, norm_type: float, apply_sigmoid: bool,
                feature_sensitivity: Optional[Union[float, List[float]]] = None, check: bool = True) -> torch.Tensor:
    """
    Clips the norm of a tensor to ensure its norm does not exceed `max_norm`.
    This method scales the entire tensor to keep the specified norm constraint,
    similar to PyTorch's `clip_grad_norm_`.

    :param data: The input tensor to be clipped. Shape: (batch_size, embedding_dim)
    :param max_norm: The maximum allowed norm for the tensor.
    :param norm_type: The type of norm to use for clipping (e.g., L1, L2).
    :param apply_sigmoid: Whether to apply a sigmoid activation to the embedding before clipping.
    :param feature_sensitivity: The sensitivity of each feature in the tensor. (e.g., 1 for sigmoid activation,
        2 for tanh activation, etc.). Mandatory if apply_sigmoid is False.
    :param check: Whether to check if the feature sensitivity is exceeded. Disable for performance.

    :return: The clipped tensor with a per sample norm not exceeding `max_norm`.
    """
    return MaxClipping(max_norm=max_norm, norm_type=norm_type, apply_sigmoid=apply_sigmoid,
                       ndim=data.shape[1], feature_sensitivity=feature_sensitivity, check=check)(data)
