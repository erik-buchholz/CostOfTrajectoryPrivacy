#!/usr/bin/env python3
"""
Implementation of the von Mises-Fisher distribution.

This file is based on the blog post by Whittenbury [1]. In fact, the numpy code is directly copied from
the blog post. We simply translated the code to PyTorch in order to work with tensors. Thereby,
we tried to keep the code as close as possible to the original numpy code.


[1] D. Whittenbury, “DS-2: sampling and visualising the von mises-fisher distribution in p dimensions,”
dlwhittenbury.com. Accessed: Nov. 18, 2024. [Online]. Available:
https://dlwhittenbury.github.io/ds-2-sampling-and-visualising-the-von-mises-fisher-distribution-in-p-dimensions.html
"""
import logging
from typing import Optional, Union, cast

import numpy as np
import scipy
import torch
from pydantic import validate_call
from torch import Tensor
from torch import nn

log = logging.getLogger()


####################################################################################################
# NumPyImplementation from [1]
####################################################################################################

def rand_uniform_hypersphere_np(N: int, p: int) -> np.ndarray:
    r"""
        rand_uniform_hypersphere(N,p)
        =============================

        Generate random samples from the uniform distribution on the (p-1)-dimensional
        hypersphere $\mathbb{S}^{p-1} \subset \mathbb{R}^{p}$. We use the method by
        Muller [1], see also Ref. [2] for other methods.

        INPUT:

            * N (int) - Number of samples
            * p (int) - The dimension of the generated samples on the (p-1)-dimensional hypersphere.
                - p = 2 for the unit circle $\mathbb{S}^{1}$
                - p = 3 for the unit sphere $\mathbb{S}^{2}$
            Note that the (p-1)-dimensional hypersphere $\mathbb{S}^{p-1} \subset \mathbb{R}^{p}$ and the
            samples are unit vectors in $\mathbb{R}^{p}$ that lie on the sphere $\mathbb{S}^{p-1}$.

    References:

    [1] Muller, M. E. "A Note on a Method for Generating Points Uniformly on N-Dimensional Spheres."
    Comm. Assoc. Comput. Mach. 2, 19-20, Apr. 1959.

    [2] https://mathworld.wolfram.com/SpherePointPicking.html

    """

    if (p <= 0) or (type(p) is not int):
        raise Exception("p must be a positive integer.")

    # Check N>0 and is an int
    if (N <= 0) or (type(N) is not int):
        raise Exception("N must be a non-zero positive integer.")

    v = np.random.normal(0, 1, (N, p))

    v = np.divide(v, np.linalg.norm(v, axis=1, keepdims=True))

    return v


def rand_t_marginal_np(kappa: float, p: int, N: int = 1) -> np.ndarray:
    r"""
        rand_t_marginal(kappa,p,N=1)
        ============================

        Samples the marginal distribution of t using rejection sampling of Wood [3].

        INPUT:

            * kappa (float) - concentration
            * p (int) - The dimension of the generated samples on the (p-1)-dimensional hypersphere.
                - p = 2 for the unit circle $\mathbb{S}^{1}$
                - p = 3 for the unit sphere $\mathbb{S}^{2}$
            Note that the (p-1)-dimensional hypersphere $\mathbb{S}^{p-1} \subset \mathbb{R}^{p}$ and the
            samples are unit vectors in $\mathbb{R}^{p}$ that lie on the sphere $\mathbb{S}^{p-1}$.
            * N (int) - number of samples

        OUTPUT:

            * samples (array of floats of shape (N,1)) - samples of the marginal distribution of t
    """

    # Check kappa >= 0 is numeric
    if (kappa < 0) or ((type(kappa) is not float) and (type(kappa) is not int)):
        raise Exception("kappa must be a non-negative number.")

    if (p <= 0) or (type(p) is not int):
        raise Exception("p must be a positive integer.")

    # Check N>0 and is an int
    if (N <= 0) or (type(N) is not int):
        raise Exception("N must be a non-zero positive integer.")

    # Start of algorithm
    b = (p - 1.0) / (2.0 * kappa + np.sqrt(4.0 * kappa ** 2 + (p - 1.0) ** 2))
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + (p - 1.0) * np.log(1.0 - x0 ** 2)

    samples = np.zeros((N, 1))

    # Loop over number of samples
    for i in range(N):

        # Continue until you have an acceptable sample
        while True:

            # Sample Beta distribution
            Z = np.random.beta((p - 1.0) / 2.0, (p - 1.0) / 2.0)

            # Sample Uniform distribution
            U = np.random.uniform(low=0.0, high=1.0)

            # W is essentially t
            W = (1.0 - (1.0 + b) * Z) / (1.0 - (1.0 - b) * Z)

            # Check whether to accept or reject
            if kappa * W + (p - 1.0) * np.log(1.0 - x0 * W) - c >= np.log(U):
                # Accept sample
                samples[i] = W
                break

    return samples


def rand_von_mises_fisher_np(mu: np.ndarray, kappa:float, N: int = 1) -> np.ndarray:
    r"""
        rand_von_mises_fisher(mu,kappa,N=1)
        ===================================

        Samples the von Mises-Fisher distribution with mean direction mu and concentration kappa.

        INPUT:

            * mu (array of floats of shape (p,1)) - mean direction. This should be a unit vector.
            * kappa (float) - concentration.
            * N (int) - Number of samples.

        OUTPUT:

            * samples (array of floats of shape (N,p)) - samples of the von Mises-Fisher distribution
            with mean direction mu and concentration kappa.
    """

    # Check that mu is a unit vector
    eps = 10 ** (-8)  # Precision
    norm_mu = np.linalg.norm(mu)
    if abs(norm_mu - 1.0) > eps:
        raise Exception("mu must be a unit vector.")

    # Check kappa >= 0 is numeric
    if (kappa < 0) or ((type(kappa) is not float) and (type(kappa) is not int)):
        raise Exception("kappa must be a non-negative number.")

    # Check N>0 and is an int
    if (N <= 0) or (type(N) is not int):
        raise Exception("N must be a non-zero positive integer.")

    # Dimension p
    p = len(mu)

    # Make sure that mu has a shape of px1
    mu = np.reshape(mu, (p, 1))

    # Array to store samples
    samples = np.zeros((N, p))

    #  Component in the direction of mu (Nx1)
    t = rand_t_marginal_np(kappa, p, N)

    # Component orthogonal to mu (Nx(p-1))
    xi = rand_uniform_hypersphere_np(N, p - 1)

    # von-Mises-Fisher samples Nxp

    # Component in the direction of mu (Nx1).
    # Note that here we are choosing an
    # intermediate mu = [1, 0, 0, 0, ..., 0] later
    # we rotate to the desired mu below
    samples[:, [0]] = t

    # Component orthogonal to mu (Nx(p-1))
    # New code
    samples[:, 1:] = np.tile(np.sqrt(1 - t ** 2), (1, p - 1)) * xi
    # Deprecated code
    # deprecated = matlib.repmat(np.sqrt(1 - t ** 2), 1, p - 1) * xi
    # assert deprecated.shape == samples[:, 1:].shape and np.allclose(deprecated, samples[:, 1:]), \
    #     "The deprecated and new code are not the same."

    # Rotation of samples to desired mu
    O = scipy.linalg.null_space(mu.T)
    R = np.concatenate((mu, O), axis=1)
    samples = np.dot(R, samples.T).T

    return samples


def rotate_sample_to_mu_np(samples: np.ndarray, mu: np.ndarray) -> np.ndarray:
    r"""
        Rotates the samples to the desired mean direction mu.

        :param samples: np.ndarray - Shape (N, p) - Samples to be rotated.
        :param mu: np.ndarray - Shape (p, 1) - Desired mean direction.
        :return: np.ndarray - Shape (N, p) - Rotated samples.
    """
    # Rotation of samples to desired mu
    O = scipy.linalg.null_space(mu.T)
    R = np.concatenate((mu, O), axis=1)
    samples = np.dot(R, samples.T).T

    return samples


####################################################################################################
# PyTorch Implementation
####################################################################################################

# @validate_call(validate_return=True, config={"arbitrary_types_allowed": True})
def rand_uniform_hypersphere(N: int, p: int, device: Optional[torch.device] = None) -> torch.Tensor:
    r"""
        PyTorch Implementation of rand_uniform_hypersphere(N,p).

        Generate random samples from the uniform distribution on the (p-1)-dimensional
        hypersphere $\mathbb{S}^{p-1} \subset \mathbb{R}^{p}$. We use the method by
        Muller [1], see also Ref. [2] for other methods.

        :param N: int - Number of samples
        :param p: int - The dimension of the generated samples on the (p-1)-dimensional hypersphere.
            - p = 2 for the unit circle $\mathbb{S}^{1}$
            - p = 3 for the unit sphere $\mathbb{S}^{2}$
            The (p-1)-dimensional hypersphere $\mathbb{S}^{p-1} \subset \mathbb{R}^{p}$ and the
            samples are unit vectors in $\mathbb{R}^{p}$ that lie on the sphere $\mathbb{S}^{p-1}$.
        :param device: torch.device - Device to store the samples.
        :return: torch.Tensor - Shape: (N, p) -
                 Samples of the uniform distribution on the (p-1)-dimensional hypersphere.


        References:
        [1] Muller, M. E. "A Note on a Method for Generating Points Uniformly on N-Dimensional Spheres."
        Comm. Assoc. Comput. Mach. 2, 19-20, Apr. 1959
        [2] https://mathworld.wolfram.com/SpherePointPicking.html
    """

    if (p <= 0) or not isinstance(p, int):
        raise Exception("p must be a positive integer.")

    # Check N>0 and is an int
    if (N <= 0) or not isinstance(N, int):
        raise Exception("N must be a non-zero positive integer.")

    # v = np.random.normal(0, 1, (N, p))
    v = torch.normal(mean=0, std=1, size=(N, p), device=device)

    # v = np.divide(v, np.linalg.norm(v, axis=1, keepdims=True))
    v = v / torch.norm(v, dim=1, keepdim=True)
    return v


class RandTMarginal(nn.Module):
    r"""
        PyTorch class version of rand_t_marginal(kappa,p,N=1).
        Allows re-use of the same object for multiple calls.

        Samples the marginal distribution of t using rejection sampling of Wood [3].

        :param kappa: float - Concentration
        :param p: int - The dimension of the generated samples on the (p-1)-dimensional hypersphere.
            - p = 2 for the unit circle $\mathbb{S}^{1}$
            - p = 3 for the unit sphere $\mathbb{S}^{2}$
            Note that the (p-1)-dimensional hypersphere $\mathbb{S}^{p-1} \subset \mathbb{R}^{p}$ and the
            samples are unit vectors in $\mathbb{R}^{p}$ that lie on the sphere $\mathbb{S}^{p-1}$.
    """

    @validate_call
    def __init__(self, kappa: float, p: int):
        super().__init__()

        # Check kappa >= 0 is numeric
        if (kappa < 0) or not isinstance(kappa, (float, int)):
            raise Exception("kappa must be a non-negative number.")

        if (p <= 0) or not isinstance(p, int):
            raise Exception("p must be a positive integer.")

        # Required Conversions for PyTorch
        self.kappa: torch.Tensor = torch.tensor(kappa)
        self.p: int = p

        # Start of algorithm
        self.b = (self.p - 1.0) / (2.0 * self.kappa + torch.sqrt(4.0 * self.kappa ** 2 + (self.p - 1.0) ** 2))
        self.x0 = (1.0 - self.b) / (1.0 + self.b)
        self.c = self.kappa * self.x0 + (self.p - 1.0) * torch.log(1.0 - self.x0 ** 2)

    # @validate_call
    def sample(self, N: int = 1, device: Optional[torch.device] = None) -> torch.Tensor:
        """
            Samples the marginal distribution of t using rejection sampling of Wood [3].

            :param N: int - Number of samples
            :param device: torch.device - Device to store the samples.
            :return: torch.Tensor - Shape: (N, 1) - Samples of the marginal distribution of t.
        """
        # Check N>0 and is an int
        if (N <= 0) or not isinstance(N, int):
            raise Exception("N must be a non-zero positive integer.")

        samples = torch.zeros((N, 1))

        n_accepted = 0

        while n_accepted < N:
            missing = N - n_accepted
            # Sample Beta distribution
            Z = torch.distributions.Beta((self.p - 1.0) / 2.0, (self.p - 1.0) / 2.0).sample((missing,)).to(device=device)
            # Sample Uniform distribution
            U = torch.rand((missing,), device=device)

            # Compute W for all candidates
            W = (1.0 - (1.0 + self.b) * Z) / (1.0 - (1.0 - self.b) * Z)
            accept_mask = self.kappa * W + (self.p - 1.0) * torch.log(1.0 - self.x0 * W) - self.c >= torch.log(U)

            # Filter accepted samples
            accepted_samples = W[accept_mask]
            accepted_iter = accepted_samples.shape[0]

            # Store accepted samples
            samples[n_accepted:n_accepted + accepted_iter] = accepted_samples.unsqueeze(1)
            n_accepted += accepted_iter

        return samples

    def forward(self, N: int = 1) -> torch.Tensor:
        return self.sample(N)


# @validate_call(validate_return=True, config={"arbitrary_types_allowed": True})
def rand_t_marginal(kappa: Union[float, torch.Tensor], p: int, N: int = 1, device: Optional[torch.device] = None) -> torch.Tensor:
    r"""
        PyTorch version of rand_t_marginal(kappa,p,N=1)

        Samples the marginal distribution of t using rejection sampling of Wood [3].

        :param kappa: float - Concentration
        :param p: int - The dimension of the generated samples on the (p-1)-dimensional hypersphere.
            - p = 2 for the unit circle $\mathbb{S}^{1}$
            - p = 3 for the unit sphere $\mathbb{S}^{2}$
            Note that the (p-1)-dimensional hypersphere $\mathbb{S}^{p-1} \subset \mathbb{R}^{p}$ and the
            samples are unit vectors in $\mathbb{R}^{p}$ that lie on the sphere $\mathbb{S}^{p-1}$.
        :param N: int - Number of samples
        :param device: torch.device - Device to store the samples.
        :return: torch.Tensor - Shape: (N, 1) - Samples of the marginal distribution of t.
    """

    # Check kappa >= 0 is numeric
    if (kappa < 0) or not isinstance(kappa, (float, int, Tensor)):
        raise Exception("kappa must be a non-negative number.")

    if (p <= 0) or not isinstance(p, int):
        raise Exception("p must be a positive integer.")

    # Check N>0 and is an int
    if (N <= 0) or not isinstance(N, int):
        raise Exception("N must be a non-zero positive integer.")

    # Required Conversions for PyTorch
    kappa = cast(Tensor, torch.tensor(kappa, device=device) if not isinstance(kappa, torch.Tensor) else kappa)

    # Start of algorithm
    b = (p - 1.0) / (2.0 * kappa + torch.sqrt(4.0 * kappa ** 2 + (p - 1.0) ** 2))
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + (p - 1.0) * torch.log(1.0 - x0 ** 2)

    samples = torch.zeros((N, 1), device=device)

    n_accepted = 0

    while n_accepted < N:
        missing = N - n_accepted
        # Sample Beta distribution
        Z = torch.distributions.Beta((p - 1.0) / 2.0, (p - 1.0) / 2.0).sample((missing,)).to(device=device)
        # Sample Uniform distribution
        U = torch.rand((missing,), device=device)

        # Compute W for all candidates
        W = (1.0 - (1.0 + b) * Z) / (1.0 - (1.0 - b) * Z)
        accept_mask = kappa * W + (p - 1.0) * torch.log(1.0 - x0 * W) - c >= torch.log(U)

        # Filter accepted samples
        accepted_samples = W[accept_mask]
        accepted_iter = accepted_samples.shape[0]

        # Store accepted samples
        samples[n_accepted:n_accepted + accepted_iter] = accepted_samples.unsqueeze(1)
        n_accepted += accepted_iter

    return samples


# @validate_call(validate_return=True, config={"arbitrary_types_allowed": True})
def torch_null_space(A: torch.Tensor, rcond: Optional[float] = None) -> torch.Tensor:
    r"""
        Computes the null space of a matrix A.
        Mimics scipy.linalg.null_space for PyTorch.

        :param A: torch.Tensor - Shape: (1, p) - Matrix A.
        :param rcond: float - Cutoff for small singular values.
        :return: torch.Tensor - Shape: (p, p-1) - Null space of A.
    """
    assert A.dim() == 2 and A.shape[0] == 1, f"The shape of A is not correct: {A.shape}."
    u, s, vh = torch.linalg.svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = torch.finfo(s.dtype).eps * max(M, N)
    tol = torch.amax(s) * rcond
    num = torch.sum(s > tol, dtype=torch.int64)
    Q = vh[num:, :].conj().T
    return Q

# @validate_call(validate_return=True, config={"arbitrary_types_allowed": True})
def torch_null_space_batch(A: torch.Tensor, rcond: Optional[float] = None) -> torch.Tensor:
    r"""
        Computes the null space of N matrices simultaneously.
        Mimics scipy.linalg.null_space for PyTorch.

        :param A: torch.Tensor - Shape: (N, 1, p) - Batch of matrices A.
        :param rcond: float - Cutoff for small singular values.
        :return: torch.Tensor - Shape: (N, p, p-1) - Null spaces of matrices.
    """
    assert A.dim() == 3 and A.shape[1] == 1, "The shape of A is not correct."
    u, s, vh = torch.linalg.svd(A, full_matrices=True)
    M, N = u.shape[1], vh.shape[2]
    if rcond is None:
        rcond = torch.finfo(s.dtype).eps * max(M, N)
    tol = torch.amax(s, dim=-1, keepdim=True) * rcond
    num = torch.sum(s > tol, dim=-1, keepdim=True)
    indices = torch.arange(vh.shape[2], device=vh.device)
    mask = indices.unsqueeze(0) >= num
    Q = vh[mask].reshape(A.shape[0], vh.shape[2] - 1, vh.shape[2]).transpose(1, 2)
    return Q

# @validate_call(validate_return=True, config={"arbitrary_types_allowed": True})
def rotate_sample_to_mu(samples: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    r"""
        Rotates the samples to the desired mean direction mu.

        :param samples: torch.Tensor - Shape (1, p) - Samples to be rotated.
        :param mu: torch.Tensor - Shape (p, 1) - Desired mean direction.
        :return: torch.Tensor - Shape (1, p) - Rotated samples.
    """
    # Consistency checks
    assert samples.dim() == 2, "samples must have shape (N, p)"
    assert mu.dim() == 2, "mu must have shape (p, 1)"
    assert mu.shape[1] == 1, "mu must have shape (p, 1)"
    assert samples.shape[1] == mu.shape[0], "The feature dimension p must match between samples and mu"

    # Rotation of samples to desired mu
    O = torch_null_space(mu.T)  # Shape: (p, p-1)
    R = torch.cat((mu, O), dim=1)  # Shape: (p, p)
    samples = torch.matmul(R, samples.T).T  # Shape: (1, p)

    return samples


# @validate_call(validate_return=True, config={"arbitrary_types_allowed": True})
def rotate_sample_to_mu_batch(samples: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    r"""
        Rotates the samples to the desired mean direction mu for a batch of samples.

        :param samples: torch.Tensor - Shape (N, p) - Samples to be rotated.
        :param mu: torch.Tensor - Shape (N, p, 1) - Desired mean directions for each sample.
        :return: torch.Tensor - Shape (N, p) - Rotated samples.
    """
    # Consistency checks
    assert samples.dim() == 2, "samples must have shape (N, p)"
    assert mu.dim() == 3, "mu must have shape (N, p, 1)"
    assert samples.shape[0] == mu.shape[0], "The batch size N must match between samples and mu"
    assert samples.shape[1] == mu.shape[1], "The feature dimension p must match between samples and mu"

    # Compute the null space for each mu.T in the batch
    O = torch_null_space_batch(mu.transpose(1, 2))  # Shape: (N, p, p-1)

    # Concatenate mu with the null space to form the rotation matrix R
    R = torch.cat((mu, O), dim=2)  # Shape: (N, p, p)

    # Rotate the samples by matrix multiplication
    samples_rotated = torch.bmm(R, samples.unsqueeze(2)).squeeze(2)  # Shape: (N, p)

    return samples_rotated


# @validate_call(validate_return=True, config={"arbitrary_types_allowed": True})
def rand_von_mises_fisher(mu: torch.Tensor, kappa: float, N: int = 1) -> torch.Tensor:
    r"""
        PyTorch Implementation of rand_von_mises_fisher(mu,kappa,N=1).

        Samples the von Mises-Fisher distribution with mean direction mu and concentration kappa.

        :param mu: torch.Tensor - Shape (p,) - Mean direction. This should be a unit vector.
        :param kappa: float - Concentration.
        :param N: int - Number of samples.

        :return: torch.Tensor - Shape (N, p) -
            Samples of the von Mises-Fisher distribution with mean direction mu and concentration kappa.
    """
    # Check that mu is one dimensional and a unit vector (norm - 1 < eps)
    eps = 1e-5  # Precision
    if mu.dim() > 1 or (torch.norm(mu) - 1.0).abs() > eps:
        log.warning(f"mu must be a unit vector. Norm: {torch.norm(mu):.2e} / Diff: {(torch.norm(mu) - 1.0).abs():.2e}")

    # Check kappa >= 0 is numeric
    if (kappa < 0) or not isinstance(kappa, (float, int)):
        raise ValueError("kappa must be a non-negative number.")

    # Check N>0 and is an int
    if (N <= 0) or not isinstance(N, int):
        raise ValueError("N must be a non-zero positive integer.")

    # Dimension p
    # p = len(mu)
    p = mu.shape[0]

    # Make sure that mu has a shape of px1
    # mu = np.reshape(mu, (p, 1))
    mu = mu.reshape(p, 1)

    # Array to store samples
    # samples = np.zeros((N, p))
    samples = torch.zeros((N, p), device=mu.device, dtype=mu.dtype)

    #  Component in the direction of mu (Nx1)
    t: torch.Tensor = rand_t_marginal(kappa, p, N, device=mu.device)
    # assert t.shape == (N, 1), "The shape of t is not correct."

    # Component orthogonal to mu (Nx(p-1))
    xi: torch.Tensor = rand_uniform_hypersphere(N, p - 1, device=mu.device)
    # assert xi.shape == (N, p - 1), "The shape of xi is not correct."

    # von-Mises-Fisher samples Nxp

    # Component in the direction of mu (Nx1).
    # Note that here we are choosing an
    # intermediate mu = [1, 0, 0, 0, ..., 0] later
    # we rotate to the desired mu below
    samples[:, [0]] = t

    # Component orthogonal to mu (Nx(p-1))
    tmp_torch = torch.sqrt(1 - t ** 2).reshape(-1, 1).repeat(1, p - 1) * xi
    # assert tmp_torch.shape == (N, p - 1), "The shape of tmp_torch is not correct."
    samples[:, 1:] = tmp_torch

    # compare the numpy and torch results
    # tmp_npy = np.tile(np.sqrt(1 - t ** 2), (1, p - 1)) * xi
    # assert np.allclose(tmp_npy, tmp_torch.numpy()), "The numpy and torch results are not the same."

    # Rotation of samples to desired mu
    samples = rotate_sample_to_mu(samples, mu)

    return samples


class RandVonMisesFisher(nn.Module):
    r"""
        PyTorch class version of rand_von_mises_fisher(mu,kappa,N=1).
        Allows re-use of the same object for multiple calls and implements Vectorization.

        Samples the von Mises-Fisher distribution with mean direction mu and concentration kappa.

        :param kappa: float - Concentration.
        :param p: int - The dimension of the generated samples on the (p-1)-dimensional
    """

    @validate_call
    def __init__(self, kappa: float, p: int):
        super().__init__()

        # Check kappa >= 0 is numeric
        if (kappa < 0) or not isinstance(kappa, (float, int)):
            raise ValueError("kappa must be a non-negative number.")

        # Check p>0 and is an int
        if (p <= 0) or not isinstance(p, int):
            raise ValueError("p must be a positive integer.")

        # Save the parameters
        self.kappa = torch.tensor(kappa)
        self.p = p

    # @validate_call(config={"arbitrary_types_allowed": True})
    def sample(self, mu: torch.Tensor) -> torch.Tensor:
        """
            Samples the von Mises-Fisher distribution with mean direction mu and concentration kappa.

            :param mu: torch.Tensor - Shape (N, p) - Mean directions.
                       Each mean direction should be a unit vector.
            :return: torch.Tensor - Shape: (N, p) -
                One sample from the von Mises-Fisher distribution per mean direction.
        """
        # Check that each mu is a unit vector
        eps = 1e-5  # Precision
        if mu.dim() != 2 or (torch.norm(mu, dim=1) - 1.0).abs().max() > eps:
            raise ValueError(f"mu must be a unit vector. Dim: {mu.dim()} / "
                             f"iff: {(torch.norm(mu, dim=1) - 1.0).abs().max():.2e}")

        # Dimension p
        if mu.shape[1] != self.p:
            raise ValueError(f"mu must have the same dimension as p. mu.shape: {mu.shape} / p: {self.p}")

        # Array to store samples
        samples = torch.zeros_like(mu)
        N = mu.shape[0]

        #  Components in the direction of mu (Nx1)
        t: torch.Tensor = rand_t_marginal(kappa=self.kappa, p=self.p, N=mu.shape[0], device=mu.device)

        # Component orthogonal to mu (Nx(p-1))
        xi: torch.Tensor = rand_uniform_hypersphere(N=mu.shape[0], p=self.p - 1, device=mu.device)

        # Fill the samples
        samples[:, [0]] = t
        samples[:, 1:] = torch.sqrt(1 - t ** 2).reshape(-1, 1).repeat(1, self.p - 1) * xi

        # Rotation of samples to desired mu
        mu = mu.unsqueeze(2)
        assert mu.shape == (N, self.p, 1)
        samples = rotate_sample_to_mu_batch(samples, mu)

        return samples


    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        return self.sample(mu)