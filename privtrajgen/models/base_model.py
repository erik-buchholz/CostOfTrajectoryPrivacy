#!/usr/bin/env python3
"""Base model for PyTorch models implementing certain features"""
import logging
import os.path
from abc import ABC, abstractmethod
from typing import Optional, Any

import torch
from torch import nn

from privtrajgen.models.save_and_load import prepare_parameter_file, save_model, load_model

log = logging.getLogger()


def logc(header: str, *args: Any, column_width: int = 40) -> None:
    log.info(f"{header:<{column_width}}" + "".join([str(arg) for arg in args]))


class BaseModel(ABC, nn.Module):

    def __init__(self,
                 name: str,
                 param_dir: Optional[str] = None
                 ):
        """
        :param name: Name of the model
        :param param_dir: Path to save/load parameters
        """
        super().__init__()
        self.name: str = name

        # Prepare parameter path
        self.param_path = prepare_parameter_file(modelname=name, param_dir=param_dir)
        logc(f"Parameter Path:", os.path.dirname(self.param_path))

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        pass

    @abstractmethod
    def training_loop(self, *args: Any, **kwargs: Any) -> None:
        pass

    @property
    def device(self) -> torch.device:
        """Assumes that all parameters are on the same device"""
        return next(self.parameters()).device

    def save_parameters(self, epoch: int, param_path: Optional[str] = None) -> None:
        param_path = param_path or self.param_path
        # Save model
        save_model(self, epoch=epoch, param_path=param_path)

    def load_parameters(self, epoch: int, param_path: Optional[str] = None) -> None:
        param_path = param_path or self.param_path
        # Load model
        load_model(self, epoch=epoch, param_path=param_path, device=self.device)
