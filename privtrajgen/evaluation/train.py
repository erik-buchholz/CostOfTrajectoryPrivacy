#!/usr/bin/env python3
"""Run Script for Evaluation"""
import logging
import os.path
from dataclasses import asdict
from pprint import pformat

import matplotlib

from privtrajgen.config import LOG_DIR
from privtrajgen.models.naming import generate_model_name
from privtrajgen.utils import add_filehandler, set_random_seed
from .config import BaseConfig
from .factories import get_train_dataloader, get_model, train_model, execute

log = logging.getLogger()


def train_model_subprocess(config: BaseConfig) -> None:
    # Set the matplotlib backend to Agg to fix multiprocessing error on HPC
    matplotlib.use('Agg')

    # Set the random seeds
    set_random_seed(config.seed)

    # Retrieve DataLoader
    dataloader = get_train_dataloader(config)

    # Generate model name / Run ID might have changed
    config.name = generate_model_name(config)

    # Add logfile based on model name
    add_filehandler(file=os.path.join(LOG_DIR, f"{config.name}_train.log"))

    # Retrieve the model
    model = get_model(config)

    # Log the configuration
    log.info(f"Configuration: {pformat(asdict(config))}")

    # Train the model
    train_model(config, model, dataloader)


if __name__ == '__main__':
    execute(subprocess=train_model_subprocess, operation="train")
