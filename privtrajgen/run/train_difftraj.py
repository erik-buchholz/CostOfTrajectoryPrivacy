#!/usr/bin/env python3
"""Train a UNet VAE model."""
import logging
import os

from simple_parsing import parse

from privtrajgen import config
from privtrajgen.evaluation.config import DiffTrajConfig
from privtrajgen.evaluation.train import train_model_subprocess
from privtrajgen.utils.logger import configure_root_loger

# Set the logging level for 'simple-parsing' to WARNING
logging.getLogger('simple_parsing').setLevel(logging.WARNING)

if __name__ == "__main__":
    # Configure logging
    log = configure_root_loger(logging_level=logging.INFO, file=os.path.join(config.LOG_DIR, "train_diffTraj.log"))

    # Parse the config
    cfg = parse(DiffTrajConfig, description="Train a DiffTraj model.")

    # Run
    train_model_subprocess(cfg)
