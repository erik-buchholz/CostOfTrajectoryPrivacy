#!/usr/bin/env python3
"""Conditional Embedding. Based on the DiffTraj implementation [1] & Generalised.

[1] https://github.com/Yasoz/DiffTraj
"""
import logging
from typing import Sequence, Optional, Union, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import validate_call
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from privtrajgen.config import ConditionalInfoSource, ConditionalInfoLocation
from privtrajgen.evaluation.config import ConditionalInfoConfig, BaseConfig
from privtrajgen.models.clipping import get_clipping_method, Clipping
from privtrajgen.privacy import get_dp_mechanism, NoiseMechanism

log = logging.getLogger()


def logc(header: str, *args: Any, column_width: int = 60) -> None:
    log.info(f"{header:<{column_width}}" + "".join([str(arg) for arg in args]))


@validate_call(config={"arbitrary_types_allowed": True})
def split_batch(batch: Sequence[Tensor], config: BaseConfig,
                device: Optional[torch.device] = None,
                dtype: Optional[torch.dtype] = None) -> Tuple[Tensor, Union[Tensor, None]]:
    """
    Splits the batch into the trajectory and the conditional information.

    :param batch: The batch tensor. Shape: (batch_size, in_channels, traj_length)
    :param config: The configuration object.
    :param device: The device to move the tensors to.
    :param dtype: The data type to convert the tensors to.
    :return: traj, attr
        traj: The trajectory tensor. Shape: (batch_size, out_channels, traj_length)
        attr: The conditional information tensor. Shape: Dependent on the configuration.
    """
    trainX = batch[0]
    assert trainX.dim() == 3, f"Expected 3D tensor, got {trainX.dim()}D tensor."
    if not config.conditional:
        attr = None
    elif config.cond.source == ConditionalInfoSource.DIFFTRAJ:
        # Shape: (batch_size, 8)
        attr = batch[1]
        assert attr.dim() == 2, f"Expected 2D tensor, got {attr.dim()}D tensor."
        assert attr.shape[1] == 8, f"Expected 8 attributes, got {attr.shape[1]}."
    elif config.cond.source == ConditionalInfoSource.SAMPLE:
        # Shape: (batch_size, in_channels, traj_length)
        attr = trainX
    else:
        raise ValueError(f"Unknown conditional information source: {config.cond.source}")
    # Move to device
    if device:
        trainX = trainX.to(device, dtype=dtype)
        if attr is not None:
            attr = attr.to(device, dtype=dtype)
    return trainX, attr


class ClipAndAddNoise(nn.Module):

    @validate_call
    def __init__(self, config: ConditionalInfoConfig, total_steps: int = 0):
        super().__init__()
        self.config: ConditionalInfoConfig = config
        # Create the clipping method
        if self.config.dp or self.config.clip_norm:
            self.clipping: Optional[Clipping] = get_clipping_method(config=config)
            logc("Cond. Info Clipping Method:", self.clipping)
        else:
            self.clipping = None
        # Create the noise mechanism
        if self.config.dp:
            if config.location == ConditionalInfoLocation.HIDDEN:
                dim = config.hidden_dim
            elif config.location == ConditionalInfoLocation.EMBEDDING:
                dim = config.embedding_dim
            else:
                raise ValueError(f"Unknown noise location: {config.location}")
            # Proof in Paper: sensitivity = 2 * max_norm
            sensitivity = 2 * config.max_norm
            self.mechanism: Optional[NoiseMechanism] = get_dp_mechanism(
                mechanism=config.noise, epsilon=config.epsilon, delta=config.delta,  # type: ignore
                sensitivity=sensitivity, dim=dim)
            logc("Cond. Info Noise Mechanism:", self.mechanism)
        else:
            self.mechanism = None
        # Create a schedule for the noise
        self.step = 0
        self.total_steps = total_steps
        if config.increase_noise:
            # We start with no noise and increase it over time throughout the training
            self.noise_schedule = torch.linspace(0, 1, total_steps)
            logc("Cond. Info Noise Schedule:", f"Increasing noise over {total_steps} steps.")
            log.warning("Noise Schedule MUST NOT be used for inference, as it represents a privacy leak!")
        elif config.noise_schedule:
            logc("Cond. Info Noise Schedule:", f"Random Noise Schedule.")
            log.warning("Noise Schedule MUST NOT be used for inference, as it represents a privacy leak!")

    @property
    def apply_activation(self) -> bool:
        """Return True if this module internally applies an activation function."""
        if self.clipping:
            return self.clipping.apply_activation
        return False

    def forward(self, embedding: Tensor) -> Tensor:
        """
        Clips the norm of the embedding and adds noise to it to comply with DP.

        :param embedding: The input embedding tensor to be clipped and noised. Shape: (batch_size, embedding_dim)
        :return: The clipped and noised embedding tensor.
        """
        # print("Initial Embedding:", embedding)

        if self.clipping:
            # Clipping required
            embedding = self.clipping(embedding)

        # print("Clipped Embedding:", embedding)

        if self.mechanism:
            # Add noise to the embedding
            noised_embedding = self.mechanism(embedding).to(dtype=embedding.dtype)

            if self.config.increase_noise and self.training:
                # Only partially use the noised_embedding in training - Deactivate for inference
                # Make sure we do not exceed the schedule --> min
                beta = self.noise_schedule[min(self.step, self.total_steps - 1)]
                embedding = (1 - beta) * embedding + beta * noised_embedding
            elif self.config.noise_schedule and self.training:
                # Draw a random factor for each element in the batch
                # 1.0 means full noise, 0.0 means no noise
                beta = torch.rand((embedding.size(0), 1), device=embedding.device, dtype=embedding.dtype)
                # Expand to the embedding size
                beta = beta.expand_as(embedding)
                embedding = (1 - beta) * embedding + beta * noised_embedding
            else:
                embedding = noised_embedding

            # print("Noised Embedding:", embedding)
            # Clip again to have the same format as w/o DP
            # embedding = self.clipping(embedding)
        # print("Final Embedding:", embedding)

        # Increase the step counter
        self.step += 1

        return embedding


class ConditionalEmbedding(nn.Module):
    """Conditional Embedding Model."""

    def __init__(
            self,
            config: BaseConfig,  # We need access to data dependent values
    ):
        """
        Initialize the ConditionalEmbedding model.
        """
        super().__init__()

        # Save the configuration
        self.config: ConditionalInfoConfig = config.cond

        # Log the configuration
        log.info('#' * 40 + " Conditional Information " + '#' * 40)

        if self.config.source == ConditionalInfoSource.DIFFTRAJ:
            # Use the DiffTraj format (This part is consistent with the original implementation that
            # was part of the DiffTraj model, called, WideAndDeep (refer. Traj_UNet.py)

            if self.config.location == ConditionalInfoLocation.EMBEDDING:
                # This is the original implementation of the model
                hidden_dim = self.config.hidden_dim
                embedding_dim = self.config.embedding_dim
            elif self.config.location == ConditionalInfoLocation.HIDDEN:
                # We need to slightly adjust the model to have a fc layer after noise addition
                # Use the default values for hidden_dim and embedding_dim, but add the compression
                # afterward so that clipping and noise addition is performed with the combined attributes.
                hidden_dim = 256
                embedding_dim = 512
            else:
                raise ValueError(f"Unknown location for conditional information: {self.config.location}")

            # Embed continuous attributes via linear layers (wide part)
            self.wide_fc = nn.Linear(len(self.config.continuous_idx), embedding_dim)

            # Embed categorical attributes via embedding + linear layers (deep part)
            self.categorical_embeddings = nn.ModuleList([
                nn.Embedding(dim, embedding_dim) for dim in self.config.categorical_dims
            ])
            self.deep_fc1 = nn.Linear(embedding_dim * len(self.config.categorical_idx),
                                      hidden_dim)
            self.deep_fc2 = nn.Linear(hidden_dim, embedding_dim)

            # Add additional layers for hidden space clipping
            if self.config.location == ConditionalInfoLocation.HIDDEN and (self.config.clip_norm or self.config.dp):
                logc("Hidden Space Clipping:", "True")
                self.compression = nn.Linear(embedding_dim, self.config.hidden_dim)
                self.decompression = nn.Linear(self.config.hidden_dim, self.config.embedding_dim)
            else:
                logc("Hidden Space Clipping:", "False")
                self.compression = None
                self.decompression = None

        elif self.config.source == ConditionalInfoSource.SAMPLE:
            # Use an entire sample as conditional information (needs embedding)
            # Start by using a simple Linear layer
            input_dim = config.data.traj_length * config.in_channels
            self.sample_fc = nn.Linear(input_dim, self.config.hidden_dim)
            self.sample_fc2 = nn.Linear(self.config.hidden_dim, self.config.embedding_dim)
        else:
            raise ValueError(f"Unknown conditional information source: {self.config.source}")

        # Create the clipping and noise adding module
        # If disabled in config, the module will not do anything
        if config.steps is None:
            raise ValueError("The total number of steps must be provided for the noise schedule.")
        self.clip_and_add_noise = ClipAndAddNoise(self.config, total_steps=config.steps)

        ##############################################################
        # Log the configuration
        ##############################################################
        logc("Conditional Information Source:", self.config.source)
        logc("Conditional Information Noise Location:", self.config.location)
        if self.config.clip_norm:
            logc("Conditional Information Clipping Type:", self.config.clipping_type)
            logc("Conditional Information Max Norm:", self.config.max_norm)
        else:
            logc("Conditional Information Clipping:", "False")
        if self.config.dp:
            logc("Conditional Information DP:",
                 f"(Epsilon: {self.config.epsilon}, Delta: {self.config.delta})")
        else:
            logc("Conditional Information DP:", "False")
        logc("Parameters of Conditional Emebedding:", f"{sum(p.numel() for p in self.parameters()):,}")
        log.info('#' * 105)

    def forward(self, attr: Tensor) -> Tensor:
        """
        :param attr:
            if source == DIFFTRAJ: Shape: (batch_size, 8)
            if source == SAMPLE: Shape: (batch_size, config.input_channels, config.data.traj_length)
        :return: Shape: (batch_size, embedding_dim)
        """
        if self.config.source == ConditionalInfoSource.DIFFTRAJ:
            # ##################################################
            # Original Embedding (DiffTraj)
            # ##################################################
            # attr.shape = (batch_size, 8)
            # Continuous attributes
            continuous_attrs = attr[:, self.config.continuous_idx].float()
            # Categorical attributes
            categorical_attrs = attr[:, self.config.categorical_idx].long()

            # Wide part
            wide_out = self.wide_fc(continuous_attrs)

            # Deep part
            embeddings = [
                emb(categorical_attrs[:, i]) for i, emb in enumerate(self.categorical_embeddings)
            ]
            tensor_embeddings = torch.cat(embeddings, dim=1)
            deep_out = F.relu(self.deep_fc1(tensor_embeddings))
            deep_out = self.deep_fc2(deep_out)

            # Combine wide and deep embeddings
            embedding = wide_out + deep_out

            # ##################################################
            # Compression -> Clip -> Noise -> Decompression
            # Additional code for hidden space clipping
            # ##################################################
            if self.config.location == ConditionalInfoLocation.HIDDEN and (self.compression is not None):
                embedding = F.relu(embedding)
                if self.compression is not None:
                    # Compress the attributes
                    embedding = self.compression(embedding)
                if not self.clip_and_add_noise.apply_activation:
                    embedding = F.relu(embedding)
                # Clip and add noise in hidden space
                embedding = self.clip_and_add_noise(embedding)
                if self.decompression is not None:
                    # Decompress the attributes
                    embedding = self.decompression(embedding)

        elif self.config.source == ConditionalInfoSource.SAMPLE:
            # attr.shape = (batch_size, config.input_channels, config.data.traj_length)
            # Make attr float
            attr = attr.float()
            # Flatten the input
            attr = attr.view(attr.size(0), -1)
            # Forward pass
            embedding = self.sample_fc(attr)

            # Sigmoid activation will be used for clipping in case of MAX_CLIPPING, in case
            # we use clipping in the hidden space and use max clipping, we should not apply RelU
            # to avoid two consecutive activations
            if (not self.clip_and_add_noise.apply_activation) or (
                    not self.config.location == ConditionalInfoLocation.HIDDEN):
                embedding = F.relu(embedding)

            # Clip and add noise in hidden space
            if self.config.location == ConditionalInfoLocation.HIDDEN:
                # The method will check if clipping is required
                embedding = self.clip_and_add_noise(embedding)

            # Second FC layer
            embedding = self.sample_fc2(embedding)
        else:
            raise ValueError(f"Unknown conditional information source: {self.config.source}")

        # Clip and add noise in embedding space (the method will check if clipping is required)
        if self.config.location == ConditionalInfoLocation.EMBEDDING:
            embedding = self.clip_and_add_noise(embedding)

        return embedding

    def track_gradients(self, writer: SummaryWriter, step: int, prefix: Optional[str] = None) -> None:
        """Track individual gradients of the model."""
        prefix = f'Grad/{prefix}/' if prefix else 'Grad/'
        if self.config.source == ConditionalInfoSource.DIFFTRAJ:
            writer.add_scalar(f"{prefix}/wide_fc", self.wide_fc.weight.grad.norm(), step)
            writer.add_scalar(f"{prefix}/deep_fc2", self.deep_fc2.weight.grad.norm(), step)
        elif self.config.source == ConditionalInfoSource.SAMPLE:
            writer.add_scalar(f"{prefix}/fc_1", self.sample_fc.weight.grad.norm(), step)
            writer.add_scalar(f"{prefix}/fc_2", self.sample_fc2.weight.grad.norm(), step)
