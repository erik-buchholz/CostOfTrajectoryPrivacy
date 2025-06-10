"""
Exponential Moving Average (EMA) helper class.
Based on https://github.com/Yasoz/DiffTraj with very minor modifications.
"""
import logging
import torch
import torch.nn as nn
from opacus import GradSampleModule
from typing_extensions import Union, Callable

from privtrajgen.models.dp_wrapper import DP_Wrapper
from typing import Optional, Dict, Any

log = logging.getLogger(__name__)


# Disable inspection because of access to ._module of GradSampleModule
# noinspection PyProtectedMember
class EMAHelper(object):
    """
    Exponential Moving Average (EMA) Helper for maintaining and updating
    shadow parameters of a PyTorch model.

    This class is useful for stabilizing training by keeping a moving average
    of the model parameters, which can yield better evaluation performance
    compared to using the raw model parameters, especidally for Diffusion models.
    It is less commonly used for GANs and VAEs.

    Attributes:
        mu (float): EMA decay rate. (Default for GANs & VAEs is 0.999, for diffusion models is 0.9999)
            The closer to 1, the less the EMA shadow parameters are affected by new updates.
        shadow (dict): Dictionary storing the EMA of model parameters.
    """

    def __init__(self, mu: float =0.999, module: Optional[nn.Module] = None):
        """
        Initialize the EMAHelper.

        :param mu: EMA decay rate. Default for GANs & VAEs is 0.999, for diffusion models is 0.9999.
                    Type: float
        """
        self.mu = mu
        self.shadow: Dict[str, torch.Tensor] = {}
        self._registered = False
        self._updated = False
        self._optimizer_registered = False
        if module is not None:
            self.register(module)

    def register(self, module: nn.Module) -> None:
        """
        Register the parameters of a module for EMA tracking.

        This method iterates over the named parameters of the given module
        and initializes their EMA shadow copies. It handles various module
        wrappers such as `nn.DataParallel`, `DP_Wrapper`, and `GradSampleModule`
        to ensure that the underlying model parameters are registered correctly.

        :param module: The PyTorch module whose parameters are to be registered.
                       It can be wrapped in `nn.DataParallel`, `DP_Wrapper`, or `GradSampleModule`.
                       Type: torch.nn.Module
        """
        if self._registered:
            raise ValueError("A module can be registered only once.")
        if isinstance(module, nn.DataParallel):
            module = module.module
        if isinstance(module, DP_Wrapper):
            module = module.model
        if isinstance(module, GradSampleModule):
            # We don't want to register the GradSampleModule, but the underlying model
            module = module._module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        self._registered = True
        self._registered_module = module

    def update(self, module: nn.Module) -> None:
        """
        Update the EMA shadow parameters with the current parameters of the module.

        This method applies the EMA formula to update each registered parameter:
            shadow = mu * shadow + (1 - mu) * current_parameter

        It ensures that updates are applied to the correct underlying module,
        handling wrappers like `nn.DataParallel`, `DP_Wrapper`, and `GradSampleModule`.

        :param module: The PyTorch module whose parameters are to be used for updating EMA.
                       It can be wrapped in `nn.DataParallel`, `DP_Wrapper`, or `GradSampleModule`.
                       Type: torch.nn.Module
        """
        if self._optimizer_registered:
            raise ValueError("An optimizer has been registered with this EMAHelper instance. "
                             "Use the optimizer's step function to update EMA automatically.")
        self._update(module)


    def _update(self, module: nn.Module) -> None:
        """Internal method to update the EMA shadow parameters."""
        if not self._registered:
            raise ValueError("A module must be registered before updating EMA.")
        if isinstance(module, nn.DataParallel):
            module = module.module
        if isinstance(module, DP_Wrapper):
            module = module.model
        if isinstance(module, GradSampleModule):
            # We don't want to register the GradSampleModule, but the underlying model
            module = module._module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = ((1.0 - self.mu) * param.data + self.mu * self.shadow[name].data)
        self._updated = True

    def ema(self, module: Union[nn.Module, DP_Wrapper, GradSampleModule]) -> None:
        """
        Apply the EMA shadow parameters to the module.

        This method copies the EMA shadow parameters to the module's parameters,
        effectively replacing the module's current parameters with their EMA counterparts.

        :param module: The PyTorch module to which EMA parameters will be applied.
                       It can be wrapped in `nn.DataParallel`, `DP_Wrapper`, or `GradSampleModule`.
                       Type: torch.nn.Module
        """
        if not self._registered:
            raise ValueError("A module must be registered before updating EMA.")
        if isinstance(module, nn.DataParallel):
            module = module.module
        if isinstance(module, DP_Wrapper):
            module = module.model
        if isinstance(module, GradSampleModule):
            # We don't want to register the GradSampleModule, but the underlying model
            module = module._module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module: Union[nn.Module, DP_Wrapper, nn.DataParallel]) -> nn.Module:
        """
        Create a copy of the module with EMA shadow parameters applied.

        This method generates a duplicate of the given module, initializes it with the same
        configuration and state, and then applies the EMA shadow parameters to the copy.
        The copied module can be used for evaluation or checkpointing without affecting
        the original model.

        :param module: The PyTorch module to copy and apply EMA parameters to.
                       It can be wrapped in `nn.DataParallel` or be a standard module.
                       Type: torch.nn.Module

        :return: A new instance of the module with EMA parameters applied.
                 Type: torch.nn.Module

        :raises NotImplementedError: If the module is an instance of `DP_Wrapper`, as copying is not implemented.
        """
        if not self._registered:
            raise ValueError("A module must be registered before updating EMA.")
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        elif isinstance(module, DP_Wrapper):
            raise NotImplementedError("EMAHelper.ema_copy() for DP_Wrapper not implemented.")
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Retrieve the EMA shadow parameters.

        This method returns the current state of the EMA shadow parameters, allowing
        for serialization or inspection.

        :return: A dictionary containing the EMA shadow parameters, keyed by parameter names.
                 Type: dict
        """
        if not self._updated:
            log.warning("EMA shadow parameters have not been updated since registration or last saving.")
        # Reset updated flag
        self._updated = False
        return self.shadow

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """
        Load EMA shadow parameters from a state dictionary.

        This method replaces the current EMA shadow parameters with those provided
        in the `state_dict`. It is useful for restoring EMA state from a checkpoint.

        :param state_dict: A dictionary containing EMA shadow parameters, keyed by parameter names.
                           Type: dict
        """
        self.shadow = state_dict

    def __eq__(self, other: Any) -> bool:
        """
        Check equality between this EMAHelper instance and another.

        Two EMAHelper instances are considered equal if:
            1. They are instances of the EMAHelper class.
            2. Their `mu` (decay rate) attributes are equal.
            3. Their `shadow` dictionaries have the same keys and corresponding tensor values are equal.

        :param other: The other object to compare with.
                      Type: object

        :return: True if both instances are equal based on the criteria above; False otherwise.
                 Type: bool
        """
        if not isinstance(other, EMAHelper):
            return False

        if self.mu != other.mu:
            return False

        if self.shadow.keys() != other.shadow.keys():
            return False

        for key in self.shadow:
            if not torch.equal(self.shadow[key], other.shadow[key]):
                return False

        return True

    def register_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Bind the EMA update to the optimizer's step function.

        This method wraps the optimizer's `step` method so that after each
        optimizer step, the EMA is automatically updated. This removes the need
        to manually call `ema_helper.update(model)` after each optimization step.

        :param optimizer: The optimizer whose step function will be wrapped.
                          Type: torch.optim.Optimizer

        :raises ValueError:
            - If no module has been registered before registering the optimizer.
            - If an optimizer has already been registered with this EMAHelper instance.
        """
        if not self._registered:
            raise ValueError("A module must be registered before registering the optimizer.")

        if self._optimizer_registered:
            raise ValueError("An optimizer has already been registered with this EMAHelper instance.")

        # Verify that the optimizer has a step function
        if not hasattr(optimizer, "step"):
            raise ValueError("The optimizer does not have a step function.")

        # Save the original optimizer step function
        original_step: Callable = optimizer.step

        def optimizer_step_with_ema(*args: Any, **kwargs: Any) -> torch.Tensor:
            # Call the original optimizer step
            loss = original_step(*args, **kwargs)
            # Update the EMA
            self._update(self._registered_module)
            return loss

        # Replace the optimizer's step function with the new one
        optimizer.step = optimizer_step_with_ema

        # Store the optimizer and mark as registered
        self._optimizer_registered = True
        self._registered_optimizer = optimizer
