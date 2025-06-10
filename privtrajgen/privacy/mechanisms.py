#!/usr/bin/env python3
"""Implement DP Mechanisms."""
import logging
from abc import ABC, abstractmethod
from typing import Any

import torch
from diffprivlib.mechanisms import GaussianAnalytic, Laplace
from pydantic import validate_call
from torch import nn

from privtrajgen.config import NoiseType
from privtrajgen.privacy.vmf import RandVonMisesFisher

log = logging.getLogger()


def logc(header: str, *args: Any, column_width: int = 60) -> None:
    log.info(f"{header:<{column_width}}" + "".join([str(arg) for arg in args]))


class NoiseMechanism(ABC, nn.Module):
    """Base class for Noise Mechanisms.

    Attributes:
        epsilon: float - The privacy parameter epsilon.
        delta: float - The privacy parameter delta.
        sensitivity: float - The sensitivity of the function f.
        scale: Union[Tensor, float] - The scale of the noise to be added. Might have a different meaning for each
            mechanism.
    Methods:
        add_noise: Adds noise to the input tensor.
        __call__: Calls the add_noise function --> Maps to add_noise.
    """

    @validate_call
    def __init__(self, epsilon: float, delta: float, sensitivity: float, scale: float, *args: Any, **kwargs: Any):
        """Enforce that these parameters are set correctly."""
        super().__init__()
        self.epsilon: float = epsilon
        self.delta: float = delta
        self.sensitivity: float = sensitivity
        self.scale: float = scale

    @abstractmethod
    def add_noise(self, data: torch.Tensor) -> torch.Tensor:
        """Adds noise to the input tensor."""
        raise NotImplementedError

    @abstractmethod
    def get_noise(self, data: torch.Tensor) -> torch.Tensor:
        """Returns noise in the same shape as the input tensor. Not available for all mechanisms."""
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.add_noise(*args, **kwargs)

    def __repr__(self) -> str:
        delta = 0 if self.delta == 0 else f'{self.delta:.2e}'
        return (f"{self.__class__.__name__}(epsilon={self.epsilon}, delta={delta}, "
                f"sensitivity={self.sensitivity}, scale={self.scale})")

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.add_noise(*args, **kwargs)


class GaussianMechanism(NoiseMechanism):
    """Implementation of the Gaussian Mechanism for Differential Privacy on Tensors.

    We use the GaussianAnalytic class from IBM's diffprivlib under the hood.
    It is based on the Analytic Gaussian Mechanism proposed by Balle et al. [1].

    [1] B. Balle and Y.-X. Wang Improving the Gaussian Mechanism for Differential Privacy:
    Analytical Calibration and Optimal Denoising. International Conference on Machine Learning (ICML), 2018.
    https://arxiv.org/pdf/1805.06530

    :param epsilon: The privacy parameter epsilon.
    :param delta: The privacy parameter delta.
    :param sensitivity: The sensitivity of the function f.
    """

    # We want to access the private variable _scale from the Gaussian classes
    # noinspection PyProtectedMember
    @validate_call
    def __init__(self, epsilon: float, delta: float, sensitivity: float):

        # Sanity Checks
        if delta == 0:
            raise ValueError("Delta must be greater than 0.")
        if epsilon == 0:
            raise ValueError("Epsilon must be greater than 0.")

        # Standard Gaussian Mechanism -- Not used as it is not as tight as the analytic version
        # if epsilon < 1:
        #     logc("Gaussian Mechanism", "Standard Version")
        #     # Create an object from the library to compute the standard deviation and not make any mistakes
        #     gaussian = Gaussian(
        #         epsilon=epsilon,
        #         delta=delta,
        #         sensitivity=sensitivity
        #     )
        #     self.scale = gaussian._scale
        # else:

        # Always use the analytic version b/c it provides tighter bounds
        logc("Gaussian Mechanism", "Analytic Version")
        # Create an object from the library to compute the standard deviation and not make any mistakes
        gaussian = GaussianAnalytic(
            epsilon=epsilon,
            delta=delta,
            sensitivity=sensitivity
        )
        scale = gaussian._scale

        logc("Gaussian Mechanism Scale:", scale)
        # Make the scale a tensor
        scale = torch.tensor(scale)

        # Call the super constructor
        super().__init__(epsilon=epsilon, delta=delta, sensitivity=sensitivity, scale=scale)

    def get_noise(self, data: torch.Tensor) -> torch.Tensor:
        """Get the noise for the Gaussian Mechanism.
        Based on: https://github.com/pytorch/opacus/blob/main/opacus/optimizers/optimizer.py#L103
        (We only implemented the non-secure version)

        :param data: The input tensor defining shape and device of the noise tensor.

        :return: The noise tensor.

        """
        noise = torch.normal(
            mean=0,
            std=self.scale,
            size=data.shape,
            device=data.device,
        )
        return noise

    def add_noise(self, data: torch.Tensor) -> torch.Tensor:
        """
        Adds Gaussian noise to the input tensor.
        Based on: https://github.com/pytorch/opacus/blob/main/opacus/optimizers/optimizer.py#L103
        (We only implemented the non-secure version)

        Independent Gausian noise is added to each element of the input tensor.

        :param data: The input tensor to add noise to.

        :return: The noised tensor.
        """
        noise = self.get_noise(data)
        return data + noise


class LaplaceMechanism(NoiseMechanism):
    """Implementation of the Laplace Mechanism for Differential Privacy on Tensors.

    We use the standard Laplace mechanism from Dwork et al. [1] for differential privacy.

    ~~DEPRECATED: We use the scale as defined by Holohan et al. [2] (see Example 5) as it allows for delta > 0,
    but equals the original Laplace mechanism when delta = 0.~~
    Decided against it b/c the proof gets more complex and the difference in scale is negligible.

    [1] C. Dwork and A. Roth, “The algorithmic foundations of differential privacy,”
        Found. Trends Theor. Comput. Sci., vol. 9, no. 3–4, pp. 211–407, 2013, doi: https://doi.org/10.1561/0400000042.
    [2] Holohan, Naoise, Douglas J. Leith, and Oliver Mason. "Differential privacy in metric spaces: Numerical,
        categorical and functional data under the one roof." Information Sciences 305 (2015): 256-268.
        https://doi.org/10.1016/j.ins.2015.01.021

    :param epsilon: The privacy parameter epsilon.
    :param delta: The privacy parameter delta.
    :param sensitivity: The sensitivity of the function f.
    """

    @validate_call
    def __init__(self, epsilon: float, delta: float, sensitivity: float):
        # Although it does not compute the scale, we create an object form diffprivlib to benefit from their
        # sanity checks
        _ = Laplace(
            epsilon=epsilon,
            delta=delta,
            sensitivity=sensitivity
        )

        # Compute the scale as defined by Holohan et al. [2] (see Example 5)
        # scale = sensitivity / (epsilon - np.log(1 - delta))

        # Use the standard scale defined by Dwork et al. [1]
        scale = sensitivity / epsilon

        super().__init__(epsilon=epsilon, delta=delta, sensitivity=sensitivity, scale=scale)

        # Create a PyTorch Distribution object for the Laplace distribution
        self.laplace_dist = torch.distributions.Laplace(loc=0, scale=self.scale)

        logc("Laplace Mechanism Scale:", self.scale)

    def get_noise(self, data: torch.Tensor) -> torch.Tensor:
        """Get the noise for the Laplace Mechanism."""
        return self.laplace_dist.sample(data.shape).to(data.device)

    def add_noise(self, data: torch.Tensor) -> torch.Tensor:
        """
        Adds Laplace noise to the input tensor.

        :param data: The input tensor to add noise to.

        :return: The noised tensor.
        """
        noise = self.get_noise(data)
        return data + noise


@validate_call
def get_dp_mechanism(mechanism: NoiseType, epsilon: float, delta: float, sensitivity: float,
                     dim: int) -> NoiseMechanism:
    """Get the Differential Privacy Mechanism based on the NoiseType.

    :param mechanism: The NoiseType of the mechanism.
    :param epsilon: The privacy parameter epsilon.
    :param delta: The privacy parameter delta.
    :param sensitivity: The sensitivity of the function f.
    :param dim: The dimensionality of the data.

    :return: The Differential Privacy Mechanism.
    """
    if mechanism == NoiseType.GAUSSIAN:
        return GaussianMechanism(epsilon=epsilon, delta=delta, sensitivity=sensitivity)
    elif mechanism == NoiseType.LAPLACE:
        return LaplaceMechanism(epsilon=epsilon, delta=delta, sensitivity=sensitivity)
    elif mechanism == NoiseType.VMF:
        return VMFMechanism(epsilon=epsilon, delta=delta, sensitivity=sensitivity, dim=dim)
    else:
        raise ValueError(f"Mechanism {mechanism} not implemented.")


class VMFMechanism(NoiseMechanism):
    """Implementation of the Von Mises-Fisher Mechanism for Differential Privacy on Tensors.

    We use the von Mises-Fisher Mechanism proposed by Weggenmann et al. [1] for directional data and applied
    to deep learning by Faustini et al. [2]. Details on the implementation can be found in [3].

    [1] B. Weggenmann and F. Kerschbaum, “Differential privacy for directional data,”
    in Proceedings of the 2021 ACM SIGSAC Conference on Computer and Communications Security,
    Virtual Event Republic of Korea: ACM, Nov. 2021, pp. 1205–1222. doi: https://doi.org/10.1145/3460120.3484734.
    [2] P. Faustini, N. Fernandes, S. Tonni, A. McIver, and M. Dras,
    “Directional privacy for deep learning,” 2022, arXiv. doi: https://doi.org/10.48550/ARXIV.2211.04686.
    [3] D. Whittenbury, “DS-2: sampling and visualising the von mises-fisher distribution in p dimensions,”
    dlwhittenbury.com. Accessed: Nov. 18, 2024. [Online].Available:
    https://dlwhittenbury.github.io/ds-2-sampling-and-visualising-the-von-mises-fisher-distribution-in-p-dimensions.html



    :param epsilon: The privacy parameter epsilon.
    :param delta: The privacy parameter delta.
    :param sensitivity: The sensitivity of the function f.
    :param dim: The dimensionality of the data.
    """

    @validate_call
    def __init__(self, epsilon: float, delta: float, sensitivity: float, dim: int):
        if delta > 0.0:
            raise ValueError("Delta must be 0 for the Von Mises-Fisher Mechanism.")

        # Epsilon must be greater than 0
        if epsilon == 0:
            raise ValueError("Epsilon must be greater than 0.")

        # Sensitivity must be 1.0 for the VMF Mechanism (Unit Sphere)
        # However, the sensitivity is 2 * max_norm due to using the replace-one adjacency relation
        # Refer to cond_emb.py.
        if sensitivity != 2.0:
            raise ValueError("Sensitivity must be 2.0 for the Von Mises-Fisher Mechanism."
                             "I.e., each vector must be on the unit sphere.")

        # sigma = epsilon / 2 (Theorem 6.9 in the paper)
        sigma = epsilon / 2

        # Call the super constructor
        super().__init__(epsilon=epsilon, delta=delta, sensitivity=sensitivity, scale=sigma)

        # Create VMF Class
        self.vmf = RandVonMisesFisher(kappa=sigma, p=dim)

    @property
    def sigma(self) -> float:
        """Naming convention for kappa used in [2]."""
        return self.scale

    @property
    def kappa(self) -> float:
        """Original naming convention for kappa used in [1]."""
        return self.scale

    def get_noise(self, data: torch.Tensor) -> torch.Tensor:
        """Get the noise for the Laplace Mechanism."""
        raise ValueError("The VMF Mechanism does not add noise to the data.")

    def add_noise(self, data: torch.Tensor) -> torch.Tensor:
        r"""
        Applies the VMF Mechanism to the input tensor.
        Refer to Definition 3 in [2]:

        The VMF mechanism on the $K$-dimensional unit sphere is given by the density function:
        \[
        \mathcal{V}(\sigma, x)(y) = C_K(\sigma) e^{\sigma x^T y},
        \]
        where $\sigma > 0$ and $C_K(\sigma)$ is the normalisation factor.

        The sigma here is \varepsilon in Definition 3 of [2] and $\kappa$ in Definition 10 of [1].

        :param data: (batch_size, dim) The input tensor to apply the mechanism to. $x$ in the formula above.
        :return: The outputs of the VMF mechanism.
        """
        # Check that the data is on the unit sphere
        # Some vectors might be nan after scaling
        # replace nan values by random unit vector
        # ~Reconsider this later~
        # It's fine. This really only is an issue during the first few  iterations while the weight are getting
        # initialised. After that, there are no zero vectors anymore.
        if torch.isnan(data).any():
            mask = torch.isnan(data).any(dim=1)
            # log.warning(f"Mask.shape: {mask.shape}, Mask.sum: {mask.sum()}")
            data[mask] = torch.randn_like(data[mask])
            data[mask] /= data[mask].norm(dim=-1, keepdim=True)
        # Sample from the Von Mises-Fisher distribution
        noised_data = self.vmf(data)

        return noised_data
