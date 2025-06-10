#!/usr/bin/env python3
"""Helper functions for PyTorch models implementing DP-SGD"""
import logging

import torch.nn

log = logging.getLogger()

def validate_model(model: torch.nn.Module) -> None:
    # Import locally such that Opacus is only imported if needed (import is slow)
    from opacus.validators import ModuleValidator
    errors = ModuleValidator.validate(model, strict=False)
    if len(errors) == 0:
        log.info("No errors found - model can be trained with DP.")
    else:
        log.error("The following errors prevent the model from DP training:")
        for e in errors:
            log.error(str(e))
            ModuleValidator.validate(model, strict=True)