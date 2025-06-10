#!/usr/bin/env python3
"""A GAN based on DiffTraj's UNet architecture and
https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/model.py."""
import logging
from typing import Optional, Any

import math
import torch
from matplotlib import pyplot as plt
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from privtrajgen.config import RANDOM_SEED, NORMALISED_BBOX
from privtrajgen.evaluation.config import GANConfig, UNetGANConfig
from privtrajgen.metrics import sliced_wasserstein_distance, JS_from_points
from . import base_gan
from .base_gan import BaseGAN, BaseGenerator
from .cond_emb import ConditionalEmbedding
from .unet import UNetModel, UNetEncoder, UNetDecoder
from .unet_layers import nonlinearity

log = logging.getLogger()

def logc(header: str, *args: Any, column_width: int = 40) -> None:
    log.info(f"{header:<{column_width}}" + "".join([str(arg) for arg in args]))


class Generator(BaseGenerator):
    """The generator part of the GAN."""
    def __init__(self, config: UNetGANConfig, output_activation: Optional[nn.Module] = torch.nn.Tanh()):
        super().__init__()
        self.traj_length = config.data.traj_length
        self.input_channels = config.in_channels
        self.conditional = config.conditional

        if self.conditional:
            self.cond_emb = ConditionalEmbedding(config)

        self.unet = UNetModel(
            traj_length = config.data.traj_length,
            in_channels = config.in_channels,
            out_ch = config.in_channels,
            ch = config.hidden_dim,
            use_timestep=False,
            activation="relu",
        )
        self.output_activation = output_activation

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        if self.conditional == (context is None):
            raise ValueError("Conditional flag and attribute tensor must match.")
        if self.conditional:
            context = self.cond_emb(context)
        else:
            context = None
        x = self.unet(x, context=context)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x

    def get_noise(self, batch_size: int) -> Tensor:
        return torch.randn(batch_size, self.input_channels, self.traj_length, device=self.device)

class WeakGenerator(BaseGenerator):
    """Using the decoder portion of the UNet only."""

    def __init__(self, config: UNetGANConfig, output_activation: Optional[nn.Module] = torch.nn.Tanh()):
        super().__init__()
        self.traj_length = config.data.traj_length
        self.input_channels = config.in_channels
        self.conditional = config.conditional

        if self.conditional:
            self.cond_emb: Optional[ConditionalEmbedding] = ConditionalEmbedding(config)
        else:
            self.cond_emb = None

        self.unet = UNetDecoder(
            traj_length=config.data.traj_length,
            ch=config.hidden_dim,
            latent_dim=2,
            tanh_out=False,
            activation="relu",
            conditional=config.conditional,
        )
        self.output_activation = output_activation

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        if self.conditional == (context is None):
            raise ValueError("Conditional flag and attribute tensor must match.")
        if self.conditional:
            context = self.cond_emb(context)  # type: ignore
        else:
            context = None
        x = self.unet(x, context=context)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x

    def get_noise(self, batch_size: int) -> Tensor:
        return torch.randn(batch_size, self.input_channels, self.traj_length, device=self.device)

class Discriminator(nn.Module):
    """The discriminator part of the GAN. Uses full UNet architecture."""

    def __init__(self, config: UNetGANConfig, output_activation: Optional[nn.Module] = None):
        super().__init__()
        self.conditional = config.conditional

        if self.conditional:
            self.cond_emb = ConditionalEmbedding(config)

        self.unet = UNetModel(
            traj_length = config.data.traj_length,
            in_channels = config.in_channels,
            out_ch = 1,
            ch = config.hidden_dim,
            use_timestep=False,
            activation="leaky_relu",
            conditional=config.conditional,
        )
        self.classifier = nn.Linear(config.data.traj_length, 1)
        self.output_activation = output_activation

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        if self.conditional == (context is None):
            raise ValueError("Conditional flag and attribute tensor must match.")
        if self.conditional:
            context = self.cond_emb(context)
        x = self.unet(x, context=context)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x

class WeakDiscriminator(nn.Module):
    """Using the encoder portion of the UNet only."""

    def __init__(self, config: UNetGANConfig, output_activation: Optional[nn.Module] = None):
        super().__init__()
        self.conditional = config.conditional

        if self.conditional:
            self.cond_emb = ConditionalEmbedding(config)

        self.unet = UNetEncoder(
            traj_length = config.data.traj_length,
            in_channels = config.in_channels,
            ch = config.hidden_dim,
            vae=False,
            latent_dim=config.hidden_dim,
            activation="leaky_relu",
            conditional=config.conditional,
        )  # Output has shape (batch_size, hidden_dim, ceil(traj_length / 8))
        output_length = math.ceil(config.data.traj_length / 8)
        self.final_layer = nn.Conv1d(config.hidden_dim, 1, kernel_size=3, padding=1)
        self.classifier = nn.Linear(output_length, 1)
        self.output_activation = output_activation

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        if self.conditional == (context is None):
            raise ValueError("Conditional flag and attribute tensor must match.")
        if self.conditional:
            context = self.cond_emb(context)
        else:
            context = None
        x = self.unet(x, context=context)
        x = nonlinearity(self.final_layer(x), activation="leaky_relu")
        x = x.view(x.size(0), -1)
        x = nonlinearity(self.classifier(x), activation="leaky_relu")
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x

class UNetGAN(BaseGAN):
    def plot_gradients(self, writer: SummaryWriter, global_step: int) -> None:
        pass

    def __init__(
            self,
            config: UNetGANConfig,
            # traj_length: int,
            # name: str,
            # input_channels: int = 2,
            # hidden_dim: int = 128,
            # weak_discriminator: bool = True,
            # weak_generator: bool = True,
            # conditional: bool = False,
    ):
        super().__init__(name=config.name)

        self.conditional = config.conditional
        logc("Conditional Mode:", self.conditional)

        # Create the generator
        if config.weak_gen:
            self.gen = WeakGenerator(config=config)
        else:
            self.gen = Generator(config=config)
        logc("Weak Generator:", config.weak_gen)

        # Create the discriminator
        if config.weak_dis:
            self.dis = WeakDiscriminator(config=config)
        else:
            self.dis = Discriminator(config=config)
        logc("Weak Discriminator:", config.weak_dis)

        gen_params = sum(p.numel() for p in self.gen.parameters())
        dis_params = sum(p.numel() for p in self.dis.parameters())  # type: ignore
        logc("Generator Parameter Count:", f"{gen_params:,}")
        logc("Discriminator Parameter Count:", f"{dis_params:,}")
        logc("Total Parameter Count:", f"{gen_params + dis_params:,}")
        logc("Proportion [Dis/Gen]:", f"{dis_params / gen_params:.2f}")

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        return self.gen.forward(x, *args, **kwargs)

    def training_loop(
            self,
            dataloader: DataLoader,
            config: GANConfig
    ) -> None:
        """
        Train the GAN model.

        :param dataloader: DataLoader for the dataset
        :param config: GANConfig object
        """
        base_gan.train_gan(
            dataloader=dataloader,
            generator=self.gen,
            discriminator=self.dis,
            config=config,
            plot_fn=plot_samples,
        )

def get_unet_gan_model(config: UNetGANConfig) -> UNetGAN:
    model = UNetGAN(config=config)
    return model

def plot_samples(real_batch:torch.Tensor, fake_batch:torch.Tensor, writer: SummaryWriter, global_step: int) -> None:
    tmp = []
    for title, data in [("Original", real_batch), ("Generated", fake_batch)]:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        trajs = data.transpose(1, 2).reshape(-1, 2).detach().cpu().numpy()
        tmp.append(trajs)
        ax.scatter(trajs[:, 0], trajs[:, 1], color='blue', alpha=0.1, s=0.1)
        writer.add_figure(f"{title}", fig, global_step=global_step)
    # Add some metrics to tensorboard
    points_orig, points_syn = tmp
    swd = sliced_wasserstein_distance(points_orig, points_syn, seed=RANDOM_SEED)
    writer.add_scalar("Metrics/SWD", swd, global_step)
    jsd = JS_from_points(points_orig, points_syn, grid_size=32, bbox=NORMALISED_BBOX)
    writer.add_scalar("Metrics/JSD", jsd, global_step)