#!/usr/bin/env python3
"""GAN Configuration."""
from pydantic.dataclasses import dataclass
from simple_parsing import field

from privtrajgen.config import Models
from privtrajgen.config.enums import CustomEnum
from .baseconfig import BaseConfig, OptimizerConfig
from ...utils import enum_field


@dataclass(config=dict(validate_assignment=True))
class GANLossConfig:
    """GAN Loss-specific configuration."""
    wgan: bool
    gradient_penalty: bool
    lipschitz_penalty: bool
    name: str
    lambda_gp: float = 10.0


class GANLossOptions(CustomEnum):
    """GAN Loss Options."""
    GAN = "bce"
    WGAN = "wgan"
    iWGAN = "iwgan"
    WGAN_LP = "wgan-lp"

gan_loss_configs = {
    GANLossOptions.GAN: GANLossConfig(wgan=False, gradient_penalty=False, lipschitz_penalty=False, name="GAN"),
    GANLossOptions.WGAN: GANLossConfig(wgan=True, gradient_penalty=False, lipschitz_penalty=False, name="WGAN"),
    GANLossOptions.iWGAN: GANLossConfig(wgan=True, gradient_penalty=True, lipschitz_penalty=False, name="iWGAN"),
    GANLossOptions.WGAN_LP: GANLossConfig(wgan=True, gradient_penalty=True, lipschitz_penalty=True, name="WGAN-LP"),
}


@dataclass(config=dict(validate_assignment=True))
class GANConfig(BaseConfig):
    """GAN-specific configuration."""
    loss_option: GANLossOptions = enum_field(default=GANLossOptions.GAN)  # Loss to use
    lr_dis: float = 1e-4    # Discriminator Learning Rate
    lr_gen: float = 1e-4    # Generator Learning Rate
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(beta1=0.5, lr=None))  # Optimizer Configuration - lr overwritten by lr_dis and lr_gen
    n_critic: int = 1       # Number of Discriminator Updates per Generator Update
    dp_discriminator: bool = False  # Use Differential Privacy in the Discriminator (otherwise in the Generator)
    # Use Gradient Clipping to avoid exploding gradients (Will only be used if DP-SGD is not used)
    grad_norm_clip: float = 0.0

    @property
    def loss(self) -> GANLossConfig:
        return gan_loss_configs[self.loss_option]

    @property
    def dp_generator(self) -> bool:
        return not self.dp_discriminator

    def __post_init__(self) -> None:
        if self.conditional and self.dp_discriminator:
            raise ValueError("Conditional Models have to apply DP-SGD to the Generator!")

@dataclass(config=dict(validate_assignment=True))
class UNetGANConfig(GANConfig):
    model_type: Models = field(default=Models.UNetGAN,
                               metadata={"custom_args": {"metavar": None}}, cmd=False)  # Model to use
    strong_dis: bool = False  # Strong Discriminator
    strong_gen: bool = False  # Strong Generator
    in_channels: int = 2  # Input Channels
    hidden_dim: int = 128  # Hidden Dimension

    @property
    def weak_dis(self) -> bool:
        return not self.strong_dis

    @weak_dis.setter
    def weak_dis(self, value: bool) -> None:
        self.strong_dis = not value

    @property
    def weak_gen(self) -> bool:
        return not self.strong_gen

    @weak_gen.setter
    def weak_gen(self, value: bool) -> None:
        self.strong_gen = not value
