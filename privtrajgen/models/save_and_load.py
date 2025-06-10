#!/usr/bin/env python3
"""Helper functions for saving and loading models."""
import logging
import os
import re
from typing import Optional, Union, Any

import torch
from pydantic import validate_call

from privtrajgen.config import PARAM_PATH

# CONSTANTS - Define placeholder values
EPOCH_PLACEHOLDER = "{EPOCH}"
MODEL_PLACEHOLDER = "{MODEL}"

log = logging.getLogger(__name__)

def logc(header: str, *args: Any, column_width: int = 40) -> None:
    log.info(f"{header:<{column_width}}" + "".join([str(arg) for arg in args]))


def get_parameter_dir(modelname: str) -> str:
    """Get the parameter directory for the model."""
    return os.path.join(PARAM_PATH, modelname)

def prepare_parameter_file(
        modelname: str,
        param_dir: Optional[str] = None,
        use_epoch_placeholder: bool = True,
        use_model_placeholder: bool = False,
        model_placeholder: str = MODEL_PLACEHOLDER,
        ema: bool = False,
) -> str:
    """
    Prepares the parameter path for saving model parameters.
    If no parameter path is provided, the default path based on the model name is used.
    In either case the directory is created if it doesn't exist.

    :param modelname: Used to create the default parameter path if no param_dir is provided.
    :param param_dir: If provided, the directory to save the parameters in.
    :param use_epoch_placeholder: If True, the placeholder for the epoch is used in the filename.
    :param use_model_placeholder: If True, the placeholder for the model is used in the filename.
    :param model_placeholder: The placeholder for the model name. Defaults to `{MODEL}`. You can already
                                provide the sub-model name here if you don't want to use a placeholder.
    :param ema: If True, the parameters are EMA parameters. Defaults to False.
    :return: The prepared parameter path ending with `.pth`.
    """
    # Set default path if not provided
    param_dir = param_dir or get_parameter_dir(modelname)

    # Create the directory if it doesn't exist
    os.makedirs(param_dir, exist_ok=True)

    # Create the filename with placeholders
    if use_epoch_placeholder and use_model_placeholder:
        filename = f"{EPOCH_PLACEHOLDER}_{model_placeholder}.pth"
    elif use_epoch_placeholder:
        filename = f"{EPOCH_PLACEHOLDER}.pth"
    elif use_model_placeholder:
        filename = f"{MODEL_PLACEHOLDER}.pth"
    else:
        filename = "model.pth"

    parameter_file = os.path.join(param_dir, filename)

    # Add the EMA suffix if necessary
    if ema and not is_ema_filename(parameter_file):
        parameter_file = get_ema_filename(parameter_file)

    return parameter_file

def is_ema_filename(filename: str) -> bool:
    """Check if the filename is for an EMA model."""
    return "_ema" in filename

def get_ema_filename(filename: str) -> str:
    """Get the filename for the EMA model from the standard model filename."""
    return filename.replace(".pth", "_ema.pth")


# Add pydantic input validation
@validate_call(config={"arbitrary_types_allowed": True})
def save_model(model: Any,
               param_path: str,
               epoch: Optional[int] = None,
               model_placeholder: Optional[str] = None,
               is_ema: bool = False,
               verbose: bool = True) -> None:
    """
    Save model parameters to a file.

    This function saves the state dictionary of the given PyTorch model to the specified file path.
    It supports dynamic path formatting by replacing placeholders `{EPOCH}` and `{MODEL}` with provided values.

    :param model: The PyTorch model or EMAHelper instance to save parameters from.
    :param param_path: Path to the parameter file. Can include `{EPOCH}` and/or `{MODEL}` placeholders.
    :param epoch: Value to replace the `{EPOCH}` placeholder in `param_path`.
                  Required if `{EPOCH}` is present in `param_path`. Defaults to None.
    :param model_placeholder: Value to replace the `{MODEL}` placeholder in `param_path`.
                      Required if `{MODEL}` is present in `param_path`. Defaults to None.
    :param is_ema: If True, the parameters are EMA parameters. Defaults to False.
    :param verbose: If True, prints a confirmation message after saving. Defaults to True.

    :raises ValueError:
        - If a provided placeholder (`{EPOCH}` or `{MODEL}`) is missing from `param_path`.
        - If a required placeholder value is not provided.

    :return: None
    """
    # Assert that the path contains a placeholder for the provided values
    if epoch is not None and EPOCH_PLACEHOLDER not in param_path:
        raise ValueError(f"Parameter Path must contain placeholder for epoch: {param_path}")
    if model_placeholder is not None and MODEL_PLACEHOLDER not in param_path:
        raise ValueError(f"Parameter Path must contain placeholder for model_str: {param_path}")

    # Try to fill the placeholders
    try:
        # Extra parameters are not a problem, but missing parameters will raise an error
        param_path = param_path.format(EPOCH=epoch, MODEL=model_placeholder)
    except KeyError as e:
        raise ValueError(f"Value for placeholder {e} is missing.")

    # Add the EMA suffix if necessary
    if is_ema and not is_ema_filename(param_path):
        param_path = get_ema_filename(param_path)

    # Save the model's state dictionary
    torch.save(model.state_dict(), param_path)

    if verbose:
        logc("Saved model to:", param_path)


# Add pydantic input validation
@validate_call(config={"arbitrary_types_allowed": True})
def load_model(model: Any,
               param_path: str,
               epoch: Optional[int] = None,
               model_placeholder: Optional[str] = None,
               is_ema: bool = False,
               device: Optional[Union[str, torch.device]] = None,
               verbose: bool = True) -> None:
    """
    Load model parameters from a file into a PyTorch model.

    This function loads a state dictionary from the specified file path into the given PyTorch model.
    It supports dynamic path formatting by replacing placeholders `{EPOCH}` and `{MODEL}` with provided values.

    :param model: The PyTorch model or EMAHelper instance to load parameters into.
    :param param_path: Path to the parameter file. Can include `{EPOCH}` and/or `{MODEL}` placeholders.
    :param epoch: Value to replace the `{EPOCH}` placeholder in `param_path`.
                  Required if `{EPOCH}` is present in `param_path`. Defaults to None.
    :param model_placeholder: Value to replace the `{MODEL}` placeholder in `param_path`.
                      Required if `{MODEL}` is present in `param_path`. Defaults to None.
    :param is_ema: If True, the parameters are EMA parameters. Defaults to False.
    :param device: The device to load the model onto (e.g., 'cpu', 'cuda').
                   If None, the device is inferred from the model's existing parameters. 
                   Defaults to None.
    :param verbose: If True, prints a confirmation message after loading. Defaults to True.
    
    :raises ValueError: 
        - If a provided placeholder (`{EPOCH}` or `{MODEL}`) is missing from `param_path`.
        - If a required placeholder value is not provided.
    :raises FileNotFoundError: If the specified parameter file does not exist.
    :raises RuntimeError: If there is an error loading the state dictionary into the model.

    :return: None
    """
    # Assert that the path contains a placeholder for the provided values
    if epoch is not None and EPOCH_PLACEHOLDER not in param_path:
        raise ValueError(f"Parameter Path must contain placeholder for epoch: {param_path}")
    if model_placeholder is not None and MODEL_PLACEHOLDER not in param_path:
        raise ValueError(f"Parameter Path must contain placeholder for model_str: {param_path}")

    # Try to fill the placeholders
    try:
        # Extra parameters are not a problem, but missing parameters will raise an error
        param_path = param_path.format(EPOCH=epoch, MODEL=model_placeholder)
    except KeyError as e:
        raise ValueError(f"Value for placeholder {e} is missing.")

    # Add the EMA suffix if necessary
    if is_ema and not is_ema_filename(param_path):
        param_path = get_ema_filename(param_path)

    # Determine the device
    device = device or next(model.parameters()).device

    # Move the model to the device
    model.to(device=device)

    # Load the model
    model.load_state_dict(torch.load(param_path, map_location=device, weights_only=True))

    if verbose:
        logc("Loaded model from:", param_path)


@validate_call(config={"arbitrary_types_allowed": True})
def get_last_saved_epoch(input: Union[str, Any]) -> int:
    """
    Returns the last epoch that was saved based on parameter file names.

    :param input: Directory path or a BaseConfig object
    :return: The highest epoch number found
    """
    if not isinstance(input, str):
        directory = get_parameter_dir(input.name)
    else:
        directory = input

    max_epoch = 0
    epoch_pattern = re.compile(r"(\d+)(?:_.*)?\.pt?h$")

    for file in os.listdir(directory):
        match = epoch_pattern.match(file)
        if match:
            epoch = int(match.group(1))
            max_epoch = max(max_epoch, epoch)

    return max_epoch
