#!/usr/bin/env python3
""" """
from typing import Dict, Callable

from .base_model import BaseModel
from .ema import EMAHelper
from .unet_gan import UNetGAN, get_unet_gan_model
from .unet_vae import UNetVAE, get_unet_vae_model
from .diffTraj import Guide_UNet, Uncond_UNet, get_difftraj_model
from privtrajgen.config.enums import Models
from ..evaluation.config import BaseConfig

model_factories: Dict[Models, Callable[[BaseConfig], BaseModel]] = {
    Models.UNetVAE: get_unet_vae_model,  # type: ignore
    Models.UNetGAN: get_unet_gan_model,  # type: ignore
    Models.DiffTraj: get_difftraj_model,  # type: ignore
}
