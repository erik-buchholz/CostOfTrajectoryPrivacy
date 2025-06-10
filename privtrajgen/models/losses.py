#!/usr/bin/env python3
""" """
from typing import List, Optional, Sequence

import torch
from torch import Tensor

from privtrajgen.utils.nn_utils import compute_mask_from_lengths


class MultiFeatureReconstructionLoss(torch.nn.Module):

    def __init__(self, losses: Sequence[torch.nn.Module] = (torch.nn.MSELoss(), ), weights: Optional[List[float]] = None):
        """
        Constructor for MultiFeatureReconstructionLoss.
        :param losses: List of losses to be used for each feature. (in order)
        """
        super(MultiFeatureReconstructionLoss, self).__init__()
        self.losses = losses
        if weights is None:
            self.weights = [1.0] * len(losses)
        else:
            assert len(weights) == len(losses), f"Number of weights ({len(weights)}) does not match number of losses ({len(losses)})"
            self.weights = weights

    def forward(self, generated: List[Tensor], target: List[Tensor], lengths: Optional[Sequence[int]] = None) -> Tensor:
        """
        Forward pass for MultiFeatureReconstructionLoss.

        :param generated: List of generated tensors
        :param target: List of target tensors
        :param lengths: Lengths of the input sequences for masking
        :return: Loss value
        """
        loss = 0
        for i, (gen, tar) in enumerate(zip(generated, target)):
            # This is the loss for the i-th feature
            # Compute mask
            if lengths is not None:
                mask = compute_mask_from_lengths(tar, lengths).unsqueeze(-1)
            else:
                mask = torch.ones_like(tar)
            # Compute loss
            loss += self.weights[i] * self.losses[i](gen * mask, tar * mask)
        return loss



