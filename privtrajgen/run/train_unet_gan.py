#!/usr/bin/env python3
""" """
import logging
import os

from simple_parsing import parse

from privtrajgen import config
from privtrajgen.evaluation.config import UNetGANConfig
from privtrajgen.evaluation.train import train_model_subprocess
from privtrajgen.config.enums import Models, Datasets
from privtrajgen.utils.logger import configure_root_loger

# Set the logging level for 'simple-parsing' to WARNING
logging.getLogger('simple_parsing').setLevel(logging.WARNING)

if __name__ == "__main__":
    # Configure logging
    log = configure_root_loger(logging_level=logging.INFO, file=os.path.join(config.LOG_DIR, "unet_gan.log"))

    # Default Config
    cfg = UNetGANConfig(
        model_type= Models.UNetGAN,
        conditional = False,
        dataset = Datasets.PORTO,
    )

    # Parse the config
    cfg = parse(UNetGANConfig, default=cfg, description="Train a UNet GAN model.")

    # Run
    train_model_subprocess(cfg)
