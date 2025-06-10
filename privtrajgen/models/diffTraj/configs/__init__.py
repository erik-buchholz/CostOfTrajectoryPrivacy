#!/usr/bin/env python3
"""These are the legacy configs that have been replaced by evaluation.config.difftraj_config.DiffTrajConfig."""

from .config_cond_dp import args as config_cond_dp
from .config_uncond_dp import args as config_uncond_dp
from .config_cond import args as config_cond
from .config_uncond import args as config_uncond
from .config import args as original_config

def get_config(conditional: bool, dp: bool) -> dict:
    if conditional and dp:
        return config_cond_dp
    elif conditional and not dp:
        return config_cond
    elif not conditional and dp:
        return config_uncond_dp
    else:
        return config_uncond
