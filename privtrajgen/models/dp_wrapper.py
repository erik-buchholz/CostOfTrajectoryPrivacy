#!/usr/bin/env python3
""" """
import json
import logging
import os
import warnings
from enum import Enum
from typing import Optional, Union, Any, cast

import torch.optim
from opacus import PrivacyEngine
from opacus.optimizers import DPOptimizer
from opacus.validators import ModuleValidator
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from privtrajgen.models.save_and_load import prepare_parameter_file, get_ema_filename, save_model, load_model
from privtrajgen.utils import dpsgd
from privtrajgen.utils.helpers import compute_delta

log = logging.getLogger()

def logc(header: str, *args: Any, column_width: int = 40) -> None:
    log.info(f"{header:<{column_width}}" + "".join([str(arg) for arg in args]))

class PARAM_MODE(Enum):
    """Enum for parameter modes."""
    MODEL = 1  # Model parameters -> These can be directly loaded into a model without the wrapper
    DP = 2  # This file stores the privacy engine's checkpoint
    EMA = 3  # EMA parameters
    DP_PARAMS = 4  # DP parameters
    DIR = 5  # Directory containing the parameters

def _get_param_path(filename: str, mode: PARAM_MODE = PARAM_MODE.MODEL) -> str:
    """Returns the correct parameter path based on the base filename."""
    if mode == PARAM_MODE.MODEL:
        return filename
    elif mode == PARAM_MODE.DP:
        if not filename.endswith('.pth'):
            return filename + "_dp.pth"
        return filename.replace(".pth", "_dp.pth")
    elif mode == PARAM_MODE.EMA:
        return get_ema_filename(filename)
    elif mode == PARAM_MODE.DP_PARAMS:
        return os.path.join(os.path.dirname(filename), 'dp_params.json')
    elif mode == PARAM_MODE.DIR:
        return os.path.dirname(filename)
    else:
        raise ValueError("Unknown parameter mode.")


class DP_Wrapper(nn.Module):

    def __init__(self,
                 name: str,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 dp: bool,
                 dp_accountant: str = 'prv',
                 param_dir: Optional[str] = None,
                 ):
        """
        :param name: Name of the model (Used for saving/loading)
        :param dp: Whether to use differential privacy (wrapper can be used without DP!)
        :param dp_accountant: Accountant to use for DP-SGD (default: 'prv')
        """
        super().__init__()
        self.name = name
        self.model = model
        self.opt = optimizer

        # Differential Privacy
        self.dp = dp
        self.accountant = dp_accountant
        self.dp_initialised = False
        if self.dp:
            with warnings.catch_warnings(record=True) as w:
                self._privacy_engine = PrivacyEngine(accountant=self.accountant)
                if len(w) > 0:
                    for warning in w:
                        log.warning(str(warning.message))

        self.param_file: str = prepare_parameter_file(modelname=name, param_dir=param_dir)
        logc(f"Saving parameters to:",f"{os.path.dirname(self.param_file)}")

    @property
    def device(self) -> torch.device:
        """Assumes that all parameters are on the same device"""
        return next(self.model.parameters()).device

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.dp and not self.dp_initialised:
            raise RuntimeError("Model has to be initialised with dp_init() before forward pass.")
        return self.model(*args, **kwargs)

    # Pass through VAE methods
    def sample(self, *args: Any, **kwargs: Any) -> Any:
        # Ensure that this does not create any gradients --> Then, it's fine to access the module directly
        # This sampling cannot be used for training, so privacy leakage
        # noinspection PyProtectedMember
        module = self.model._module if self.dp_initialised else self.model
        self.model.eval()
        with torch.no_grad():
            return module.sample(*args, **kwargs)

    # Pass through get_noise method of GAN Generator
    def get_noise(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        # It's fine to access the module directly here, as the noise generation does not affect the gradients
        # noinspection PyProtectedMember
        module = self.model._module if self.dp_initialised else self.model
        return module.get_noise(*args, **kwargs)

    def track_gradients(self, writer: SummaryWriter, step: int) -> None:
        """Track individual gradients of the model.
        This method passed through to the model, if it has a track_gradients method.

        :param writer: SummaryWriter to track the gradients
        :param step: Current step
        :return: None
        """
        if hasattr(self.model, 'track_gradients'):
            self.model.track_gradients(writer=writer, step=step)

    # ########################## DP-SGD ##########################
    def validate_dp(self) -> None:
        dpsgd.validate_model(self.model)

    def fix_model(self) -> nn.Module:
        """Returns a fixed version of the model!"""
        self.model =  ModuleValidator.fix(self.model)
        return self.model

    def get_epsilon(self, delta: Optional[float]=None) -> float:
        if delta is None:
            delta = self.delta
        try:
            epsilon: float = self._privacy_engine.get_epsilon(delta)
        except ValueError:
            epsilon = 0.0
            log.warning("Epsilon could not be computed. This is expected before the first gradient step.")
        return epsilon

    def init_dp(self,
                dataloader: DataLoader,
                epochs: int,
                max_grad_norm: float,
                noise_multiplier: Optional[float] = None,
                target_epsilon: Optional[float] = None,
                delta: Optional[float] = None,
                optimizer: Optional[torch.optim.Optimizer] = None,
                ) -> DataLoader:
        self.validate_dp()
        log.info("Initializing privacy engine, might take some time.")
        # The privacy engine raises a few warning we cannot do anything about
        warnings.simplefilter("ignore")
        self._delta = compute_delta(len(dataloader.dataset)) if delta is None else delta
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs

        # Check if optimizer is provided or self.opt is defined
        if optimizer is not None:
            self.opt = optimizer
        elif self.opt is None:
            raise ValueError("Optimizer has to be provided or defined in the model as self.opt.")

        if target_epsilon is not None:
            self._target_epsilon = target_epsilon
            logc(f"Targeting:",f"(ε = {self._target_epsilon}, δ = {self.delta})")
        elif noise_multiplier is not None:
            self.noise_multiplier = noise_multiplier
            self._target_epsilon = 0  # Important for saving checkpoints
            logc(f"Utilizing:",f"(σ = {self.noise_multiplier}, δ = {self.delta})")
        else:
            raise ValueError("Either target_epsilon or noise_multiplier have to be provided!")

        compatible = self._privacy_engine.is_compatible(
            module=self.model,
            optimizer=self.opt,
            data_loader=dataloader
        )
        if compatible:
            log.info("Model compatible with privacy settings!")
        else:
            raise RuntimeError("Model, Optimizer or dataset not compatible with DP-SGD.")

        if target_epsilon is not None:
            self.model, self.opt, dataloader = self._privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=self.opt,
                data_loader=dataloader,
                epochs=self.epochs,
                target_epsilon=self._target_epsilon,
                target_delta=self.delta,
                max_grad_norm=self.max_grad_norm,
                poisson_sampling=True,
                # grad_sample_mode="ew"  # functorch
            )
            warnings.simplefilter("default")
        else:
            self.model, self.opt, dataloader = self._privacy_engine.make_private(
                module=self.model,
                optimizer=self.opt,
                data_loader=dataloader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
                poisson_sampling=True
            )
        self.dp_initialised = True
        if not isinstance(self.opt, DPOptimizer):
            raise RuntimeError("Optimizer has to be DPOptimizer after make_private. Something went wrong.")
        self.opt = cast(DPOptimizer, self.opt)
        log.info(f"Using σ={self.opt.noise_multiplier} and C={self.max_grad_norm}")
        return dataloader

    @property
    def delta(self) -> float:
        return self._delta

    @property
    def epsilon(self) -> float:
        if self.dp:
            eps: float = self._privacy_engine.get_epsilon(self.delta)
            return eps
        else:
            raise RuntimeError("Model is trained without differential privacy.")

    def save_parameters(self,
                        param_file: Optional[str] = None,
                        epoch: Optional[int] = None,
                        ema_helper: Optional[Any] = None,
                        model_placeholder: Optional[str] = None) -> None:
        """
        Save model parameters. For DP models also save DP parameters.
        This function saves the following:
        1) Model trained without DP (self.dp=False)
            - Model parameters to {base_filename}.pth
            - EMA parameters if provided to {base_filename}_ema.pth
        2) Model trained with DP (self.dp=True)
            - Parameters of underlying non-DP model to {base_filename}.pth
            - DP parameters to param_directory/dp_params.json
            - Privacy engine's checkpoint to {base_filename}_dp.pth
            - EMA parameters if provided to {base_filename}_ema.pth

        :param param_file: Path to save the model parameters to
        :param epoch: Epoch to save the model parameters for
        :param ema_helper: EMAHelper to save the EMA parameters
        :param model_placeholder: Placeholder for the model name

        :return: None
        """
        param_file = param_file or self.param_file

        # If provided save EMA parameters, these are always non DP
        # The path will automatically be adjusted through is_ema=True
        if ema_helper is not None:
            save_model(model=ema_helper, param_path=_get_param_path(filename=param_file, mode=PARAM_MODE.EMA),
                       epoch=epoch, is_ema=True, model_placeholder=model_placeholder)

        if not self.dp:
            # Save non-DP model
            save_model(model=self.model, param_path=_get_param_path(filename=param_file, mode=PARAM_MODE.MODEL),
                       epoch=epoch, is_ema=False, model_placeholder=model_placeholder)
        else:
            # Save DP model
            if not self.dp_initialised:
                raise RuntimeError("Checkpointing only possible after dp_init() has been called.")

            # Use privacy engine's checkpointing functionality
            # 1. Save model via privacy engine
            priv_eng_path = _get_param_path(filename=param_file, mode=PARAM_MODE.DP).format(EPOCH=epoch, MODEL=model_placeholder)
            self._privacy_engine.save_checkpoint(
                path=priv_eng_path,
                module=self.model,
                optimizer=self.opt,
            )
            # 2. Save DP parameters
            self.opt = cast(DPOptimizer, self.opt)
            dp_params = {
                'delta': self.delta,
                'max_grad_norm': self.opt.max_grad_norm,
                'noise_multiplier': self.opt.noise_multiplier,
                'epochs': self.epochs,
                'target_epsilon': self._target_epsilon
            }
            # Determine the directory of the parameters
            with open(_get_param_path(param_file, mode=PARAM_MODE.DP_PARAMS), 'w') as f:
                json.dump(dp_params, f)
            # 3. Also save the model parameters for loading into a non-DP model
            non_dp_param_path = _get_param_path(param_file, mode=PARAM_MODE.MODEL)
            # noinspection PyProtectedMember
            save_model(self.model._module, epoch=epoch, param_path=non_dp_param_path,
                       verbose=False, model_placeholder=model_placeholder)
            logc(f"Saved parameters to:",f"{_get_param_path(param_file, mode=PARAM_MODE.DIR)}")


    def load_non_dp_parameters(self, epoch: Optional[int], param_path: Optional[str] = None,
                               ema: bool = False, model_placeholder: Optional[str] = None,
                               ) -> nn.Module:
        """Return the trained model only. Should only be used for evaluation not further training!"""
        log.warning("Loading non-DP parameters. Model should be used for validation only and not further training!")
        if self.dp_initialised:
            raise RuntimeError("Model has been initialised with DP, use load_parameters() instead.")
        param_file = param_path or self.param_file
        param_file = _get_param_path(param_file, mode=PARAM_MODE.EMA if ema else PARAM_MODE.MODEL)
        load_model(model=self.model, param_path=param_file, epoch=epoch, model_placeholder=model_placeholder)
        return self.model


    def load_parameters(self,
                        param_file: Optional[str] = None,
                        epoch: Optional[int] = None,
                        model_placeholder: Optional[str] = None,
                        dataloader: Optional[DataLoader] = None,
                        device: Optional[Union[str, torch.device]] = None,
                        ) -> Optional[DataLoader]:
        """
        Load model parameters and privacy engine if DP model.
        This is to continue training.

        :param param_file: Path to load the model parameters from if different from self.param_file
        :param epoch: Epoch to load from
        :param model_placeholder: Placeholder for the model name
        :param dataloader: DataLoader to load the privacy engine
        :param device: Device to load the model to. Defaults to model's device.
        :return: The modified dataloader if DP model, else None
        """
        param_file = param_file or self.param_file

        # Determine the device
        device = device or self.device

        if not self.dp:
            return load_model(self.model, epoch=epoch, param_path=_get_param_path(param_file, mode=PARAM_MODE.MODEL),
                       device=device, model_placeholder=model_placeholder, is_ema=False)
        else:
            if dataloader is None:
                raise RuntimeError("DataLoader has to be provided if loading DP model.")
            if self.opt is None:
                raise RuntimeError("wrapper.opt has to be defined if loading DP model.")

            # Load dp parameters
            if not self.dp_initialised:
                with open(_get_param_path(param_file, mode=PARAM_MODE.DP_PARAMS), 'r') as f:
                    dp_params = json.load(f)
                self._delta = dp_params['delta']
                self.epochs = dp_params['epochs']
                self._target_epsilon = dp_params['target_epsilon']
                # Load privacy engine & generator
                self.model, self.opt, dataloader = self._privacy_engine.make_private(
                    module=self.model,
                    optimizer=self.opt,
                    data_loader=dataloader,
                    noise_multiplier=dp_params['noise_multiplier'],
                    max_grad_norm=dp_params['max_grad_norm']
                )
            checkpoint_path = _get_param_path(param_file, mode=PARAM_MODE.DP).format(EPOCH=epoch, MODEL=model_placeholder)
            self._privacy_engine.load_checkpoint(
                path=checkpoint_path,
                module=self.model,
                optimizer=self.opt,
                torch_load_kwargs={'map_location': device},
            )
            self.model.to(device=device)
            self.dp_initialised = True
            logc(f"Loaded parameters from:",f"{checkpoint_path}")
            return dataloader