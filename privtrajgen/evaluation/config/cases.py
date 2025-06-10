#!/usr/bin/env python3
"""Define concrete evaluation configurations."""

from privtrajgen.config import Datasets
from .baseconfig import UNetVAEConfig, DPConfig
from .difftraj_config import DiffTrajConfig
from .gan_config import UNetGANConfig, GANLossOptions

# DEBUG Cases: only 1000 steps, save freq 100, plot freq 100
sf, pf, debug_steps = 50, 50, 1000
difftraj_uncond_debug = DiffTrajConfig(dataset=Datasets.PORTO, conditional=False, steps=debug_steps, save_freq=sf,
                                       plot_freq=pf)
unetvae_uncond_debug = UNetVAEConfig(dataset=Datasets.PORTO, conditional=False, steps=debug_steps, save_freq=sf,
                                     plot_freq=pf)
unetgan_uncond_debug = UNetGANConfig(dataset=Datasets.PORTO, conditional=False, steps=debug_steps, save_freq=sf,
                                     plot_freq=pf)
difftraj_cond_debug = DiffTrajConfig(dataset=Datasets.PORTO, conditional=True, steps=debug_steps, save_freq=sf,
                                     plot_freq=pf, k_fold=1, num_runs=1)
unetvae_cond_debug = UNetVAEConfig(dataset=Datasets.PORTO, conditional=True, steps=debug_steps, save_freq=sf,
                                   plot_freq=pf)
unetgan_cond_debug = UNetGANConfig(dataset=Datasets.PORTO, conditional=True, steps=debug_steps, save_freq=sf,
                                   plot_freq=pf)
DEBUG = [
    difftraj_uncond_debug,
    unetvae_uncond_debug,
    unetgan_uncond_debug,
    difftraj_cond_debug,
    unetvae_cond_debug,
    unetgan_cond_debug
]
DEBUG_COND = [
    difftraj_cond_debug,
    unetvae_cond_debug,
    unetgan_cond_debug
]
difftraj_uncond_dp_debug = DiffTrajConfig(dataset=Datasets.PORTO, conditional=False, steps=debug_steps, save_freq=sf,
                                          plot_freq=pf, dp=DPConfig(dp=True))
unetvae_uncond_dp_debug = UNetVAEConfig(dataset=Datasets.PORTO, conditional=False, steps=debug_steps, save_freq=sf,
                                        plot_freq=pf, dp=DPConfig(dp=True))
unetgan_uncond_dp_debug = UNetGANConfig(dataset=Datasets.PORTO, conditional=False, steps=debug_steps, save_freq=sf,
                                        plot_freq=pf, dp=DPConfig(dp=True), loss_option=GANLossOptions.GAN)
difftraj_cond_dp_debug = DiffTrajConfig(dataset=Datasets.PORTO, conditional=True, steps=debug_steps, save_freq=sf,
                                        plot_freq=pf, dp=DPConfig(dp=True))
unetvae_cond_dp_debug = UNetVAEConfig(dataset=Datasets.PORTO, conditional=True, steps=debug_steps, save_freq=sf,
                                      plot_freq=pf, dp=DPConfig(dp=True))
unetgan_cond_dp_debug = UNetGANConfig(dataset=Datasets.PORTO, conditional=True, steps=debug_steps, save_freq=sf,
                                      plot_freq=pf, dp=DPConfig(dp=True))
DEBUG_DP = [
    difftraj_uncond_dp_debug,
    unetvae_uncond_dp_debug,
    unetgan_uncond_dp_debug,
    difftraj_cond_dp_debug,
    unetvae_cond_dp_debug,
    unetgan_cond_dp_debug
]

# Porto Cases: 20k steps, save_freq 5k, plot 1k
difftraj_uncond_porto = DiffTrajConfig(dataset=Datasets.PORTO, conditional=False, steps=20000, save_freq=5000,
                                       plot_freq=1000)
unetvae_uncond_porto = UNetVAEConfig(dataset=Datasets.PORTO, conditional=False, steps=20000, save_freq=5000,
                                     plot_freq=1000)
unetgan_uncond_porto = UNetGANConfig(dataset=Datasets.PORTO, conditional=False, steps=20000, save_freq=5000,
                                     plot_freq=1000)
difftraj_cond_porto = DiffTrajConfig(dataset=Datasets.PORTO, conditional=True, steps=20000, save_freq=5000,
                                     plot_freq=1000)
unetvae_cond_porto = UNetVAEConfig(dataset=Datasets.PORTO, conditional=True, steps=20000, save_freq=5000,
                                   plot_freq=1000)
unetgan_cond_porto = UNetGANConfig(dataset=Datasets.PORTO, conditional=True, steps=20000, save_freq=5000,
                                   plot_freq=1000)
PORTO_20k = [difftraj_uncond_porto,
             unetvae_uncond_porto,
             # unetgan_uncond_porto,  # The unconditional GAN is not working, so we skip it
             difftraj_cond_porto,
             unetvae_cond_porto,
             unetgan_cond_porto]

# 100k steps, save_freq 25k, plot 5k
PORTO_100k = [config.copy() for config in PORTO_20k]
for config in PORTO_100k:
    config.steps = 100000
    config.save_freq = 25000
    config.plot_freq = 5000

PORTO_100k_difftraj = [config for config in PORTO_100k if config.is_diffusion]

# GeoLife Cases: Copy all Porto cases and change dataset
# Due to the small number of samples, we should reduce the batch size to 256
GEOLIFE_20k = [config.copy() for config in PORTO_20k]
for config in GEOLIFE_20k:
    config.dataset = Datasets.GEOLIFE

# FourSquare Cases: Copy all Porto cases and change dataset
FOURSQUARE_20k = [config.copy() for config in PORTO_20k]
for config in FOURSQUARE_20k:
    config.dataset = Datasets.FSNYC
    config.batch_size = 256

# Define DP cases
difftraj_uncond_dp = DiffTrajConfig(dataset=Datasets.PORTO, conditional=False, steps=20000, save_freq=5000,
                                    plot_freq=1000, dp=DPConfig(dp=True))
unetvae_uncond_dp = UNetVAEConfig(dataset=Datasets.PORTO, conditional=False, steps=20000, save_freq=5000,
                                  plot_freq=1000, dp=DPConfig(dp=True))
unetgan_uncond_dp = UNetGANConfig(dataset=Datasets.PORTO, conditional=False, steps=20000, save_freq=5000,
                                  plot_freq=1000, dp=DPConfig(dp=True), loss_option=GANLossOptions.GAN)
difftraj_cond_dp = DiffTrajConfig(dataset=Datasets.PORTO, conditional=True, steps=20000, save_freq=5000, plot_freq=1000,
                                  dp=DPConfig(dp=True))
unetvae_cond_dp = UNetVAEConfig(dataset=Datasets.PORTO, conditional=True, steps=20000, save_freq=5000, plot_freq=1000,
                                dp=DPConfig(dp=True))
unetgan_cond_dp = UNetGANConfig(dataset=Datasets.PORTO, conditional=True, steps=20000, save_freq=5000, plot_freq=1000,
                                dp=DPConfig(dp=True))
PORTO_20k_DP = [difftraj_uncond_dp,
                unetvae_uncond_dp,
                # unetgan_uncond_dp,  # The unconditional GAN is not working, so we skip it
                difftraj_cond_dp,
                unetvae_cond_dp,
                unetgan_cond_dp]
