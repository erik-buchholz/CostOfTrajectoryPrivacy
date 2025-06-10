#!/usr/bin/env python3
""" """
from typing import Dict

from .baseconfig import *
from .cases import *
from .difftraj_config import *
from .final import final_evaluation
from .gan_config import *
from .privtrace import PrivTraceConfig, privTrace_configs, privTrace_configs_paper

# Callable configuration
ALL_CONFIGS: Dict[str, Union[BaseConfig, List[BaseConfig]]] = {
    # Debug Cases individual (uncond. version of each model)
    "difftraj_debug": difftraj_cond_debug,
    "unetvae_debug": unetvae_uncond_debug,
    "unetgan_debug": unetgan_uncond_debug,
    # Same for DP
    "difftraj_debug_dp": difftraj_uncond_dp_debug,
    "unetvae_debug_dp": unetvae_uncond_dp_debug,
    "unetgan_debug_dp": unetgan_uncond_dp_debug,
    # Individual cases of 20k models (base configuration)
    "difftraj_uncond_20k": difftraj_uncond_porto,
    "difftraj_cond_20k": difftraj_cond_porto,
    "unetvae_uncond_20k": unetvae_uncond_porto,
    "unetvae_cond_20k": unetvae_cond_porto,
    "unetgan_uncond_20k": unetgan_uncond_porto,
    "unetgan_cond_20k": unetgan_cond_porto,
    # Base configurations for DP
    "difftraj_uncond_20k_dp": difftraj_uncond_dp,
    "difftraj_cond_20k_dp": difftraj_cond_dp,
    "unetvae_uncond_20k_dp": unetvae_uncond_dp,
    "unetvae_cond_20k_dp": unetvae_cond_dp,
    "unetgan_uncond_20k_dp": unetgan_uncond_dp,
    "unetgan_cond_20k_dp": unetgan_cond_dp,

    # List of configurations
    "debug": DEBUG,
    "debug_cond": DEBUG_COND,
    "debug_dp": DEBUG_DP,
    "porto_20k": PORTO_20k,
    "porto_100k": PORTO_100k,
    "geolife_20k": GEOLIFE_20k,
    "foursquare_20k": FOURSQUARE_20k,
    "porto_20k_dp": PORTO_20k_DP,
    # "geolife_20k_dp": GEOLIFE_20k_DP,
    # "foursquare_20k_dp": FOURSQUARE_20k_DP,
}

# Update with final evaluation
ALL_CONFIGS.update(final_evaluation)
# Add PrivTrace configurations
final_evaluation['40p'] = privTrace_configs_paper['privtrace_PORTO_e10.0']
final_evaluation['40g'] = privTrace_configs_paper['privtrace_GEOLIFE_e10.0']
