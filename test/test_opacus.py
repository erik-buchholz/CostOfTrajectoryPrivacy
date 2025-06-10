#!/usr/bin/env python3
"""Testing of certain DP-SGD and Opacus functionalities."""
import unittest

import math
import opacus
import torch
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch.utils.data import DataLoader


class TestBaseModel(unittest.TestCase):
    def setUp(self):
        # Create a mock dataset
        self.epochs = 1
        self.batch_size = 10
        self.dataset = torch.randn(100, 10)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, drop_last=True)
        self.model = torch.nn.Linear(10, 10)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        # Create a mock DP optimizer without calling expensive dp_init
        self.model, self.dp_optimizer, self.dataloader = opacus.PrivacyEngine().make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.dataloader,
                noise_multiplier=1.0,
                max_grad_norm=1.0,
                poisson_sampling=False
            )

    def test_BatchMemoryManager(self):
        # Test whether the number of steps is the physical number of steps
        physical_batch_size = self.batch_size // 3


        # Number of real batches (rounded down b/c of drop_last=True)
        expected_opt_steps = len(self.dataset) // self.batch_size
        # Number of physical batches (rounded up b/c all batches are entirely used)
        expected_steps = math.ceil(self.batch_size / physical_batch_size) * expected_opt_steps

        steps = 0
        seen_samples = 0
        complete_batches = 0
        # Create a BatchMemoryManager
        with BatchMemoryManager(data_loader=self.dataloader,
                                max_physical_batch_size=physical_batch_size,
                                optimizer=self.dp_optimizer
                                ) as bmm:
            for i, batch in enumerate(bmm):
                steps += 1
                seen_samples += len(batch)
                # Copy parameters of the model before the step
                old_params = [p.clone() for p in self.model.parameters()]

                # Imitate a training step
                self.dp_optimizer.zero_grad()
                loss = self.model(batch).sum()
                loss.backward()
                self.dp_optimizer.step()
                # Check that the parameters have changed only after a real batch size
                if seen_samples >= self.batch_size:
                    for old_param, new_param in zip(old_params, self.model.parameters()):
                        self.assertFalse(torch.allclose(old_param, new_param))
                    seen_samples -= self.batch_size
                    complete_batches += 1
                else:
                    for old_param, new_param in zip(old_params, self.model.parameters()):
                        self.assertTrue(torch.allclose(old_param, new_param))


        self.assertEqual(expected_steps, steps)
        self.assertNotEqual(expected_opt_steps, steps)
