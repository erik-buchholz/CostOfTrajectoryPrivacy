#!/usr/bin/env python3
"""Test Diffusion Capabilities."""
import logging
from unittest import TestCase

import torch

from privtrajgen.models.diffTraj import q_xt_x0, remove_noise

log = logging.getLogger()
# Set level to debug
# logging.basicConfig(level=logging.DEBUG)

class TestDiffusion(TestCase):

    def test_reversible_noise_addition(self):
        # Test that the noise addition and removal is reversible

        n_diff_steps = 500
        beta = torch.linspace(0.0001,0.05, n_diff_steps)
        alpha = 1. - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        x0 = torch.randn(5, 2, 3)
        log.debug(f"Shape of x0:\t{x0.shape}")
        # x0 has shape: (batch_size, 2, traj_length)

        # Sample time steps
        t = torch.randint(low=0, high=n_diff_steps,
                          size=(len(x0) // 2 + 1,))
        log.debug(f"Shape of t:\t{t.shape}")
        t = torch.cat([t, n_diff_steps - t - 1], dim=0)[:len(x0)]
        log.debug(f"Shape of t:\t{t.shape}")

        # Get the noised images (xt) and the noise (our target)
        xt, noise = q_xt_x0(x0, t, alpha_bar=alpha_bar)
        log.debug(f"Shape of xt:\t{xt.shape}")

        # Remove the noise again
        x0_reconstructed = remove_noise(xt, noise, t, alpha_bar)
        log.debug(f"Shape of x0_reconstructed:\t{x0_reconstructed.shape}")

        # Check that the reconstructed x0 is the same as the original x0
        if not torch.allclose(x0, x0_reconstructed, atol=1e-4):
            diff = (x0 - x0_reconstructed).abs()
            max_diff = diff.max().item()
            self.fail(f'Tensors are not close. Max difference: {max_diff}\nDifference tensor: {diff}')

