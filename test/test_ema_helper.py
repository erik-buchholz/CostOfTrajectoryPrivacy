#!/usr/bin/env python3
"""Test the EMA helper functions."""
from unittest import TestCase

import torch
from opacus import GradSampleModule
from torch import nn
from torch.utils.data import DataLoader

from privtrajgen.models import EMAHelper
from privtrajgen.models.dp_wrapper import DP_Wrapper


class DummyModel(nn.Module):
    def __init__(self, input_dim: int = 100):
        super().__init__()
        # Use 3 Linear layers going from `input_dim` to 1
        self.sequential = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.sequential(x)


class TestEMA(TestCase):

    def setUp(self):
        self.mock_data = torch.rand(100, 100)
        self.dl = DataLoader(self.mock_data, batch_size=10)

    def test_ema_helper_no_update(self):
        """Test for mu = 1.0 (no update)"""
        # Create a dummy model
        model = DummyModel()
        # Create a dummy optimizer
        optimizer = torch.optim.Adam(model.parameters())
        # Create a dummy EMA helper
        ema_helper = EMAHelper(mu=1.0)
        # Register the model
        ema_helper.register(model)
        # Save current model parameters for comparison
        original_params = {name: param.clone() for name, param in model.named_parameters()}

        # Do a few updates
        for data in self.dl:
            optimizer.zero_grad()
            output = model(data)
            loss = output.mean()
            loss.backward()
            optimizer.step()
            ema_helper.update(model)

        # Verify that the EMA's shadow parameters have not changed (mu = 1.0)
        for name in original_params:
            self.assertTrue(torch.equal(ema_helper.shadow[name], original_params[name]))
        # Verify that the EMA parameters differ from the model's parameters
        model_params = {name: param for name, param in model.named_parameters()}
        for name in ema_helper.shadow:
            self.assertFalse(torch.equal(ema_helper.shadow[name], model_params[name]))

    def test_ema_helper_update(self):
        """Test for mu = 0.0 (always identical to model)"""
        # Create a dummy model
        model = DummyModel()
        # Create a dummy optimizer
        optimizer = torch.optim.Adam(model.parameters())
        # Create a dummy EMA helper
        ema_helper = EMAHelper(mu=0.0)
        # Register the model
        ema_helper.register(model)

        # Do a few updates
        for data in self.dl:
            optimizer.zero_grad()
            output = model(data)
            loss = output.mean()
            loss.backward()
            optimizer.step()
            ema_helper.update(model)

        # Verify that the EMA's shadow parameters are identical to the model's parameters
        model_params = {name: param for name, param in model.named_parameters()}
        for name in model_params:
            self.assertTrue(torch.equal(ema_helper.shadow[name], model_params[name]))

    def test_ema_helper_update_dp_wrapper(self):
        # Create a dummy model
        model = DummyModel()
        # Create a dummy optimizer
        optimizer = torch.optim.Adam(model.parameters())
        # Create a DP Wrapper
        wrapper = DP_Wrapper(name="dp_wrapper", model=model, optimizer=optimizer, dp=True)
        # Initialize DP
        dataloader = wrapper.init_dp(dataloader=self.dl, epochs=1, max_grad_norm=1.0, noise_multiplier=1.0)
        # Create a dummy EMA helper
        ema_helper = EMAHelper(mu=0.0)
        # Register the model
        ema_helper.register(wrapper)

        # Do a few updates
        for data in dataloader:
            wrapper.opt.zero_grad()
            output = wrapper(data)
            loss = output.mean()
            loss.backward()
            wrapper.opt.step()
            ema_helper.update(wrapper)

        # Verify that the EMA's shadow parameters are identical to the model's parameters
        model_params = {name: param for name, param in wrapper.model._module.named_parameters()}
        for name in model_params:
            self.assertTrue(torch.equal(ema_helper.shadow[name], model_params[name]))

    def test_ema_init_dp_wrapper(self):
        """
        There are three ways we can initialize the EMAHelper with a DP_Wrapper:
        1. Initialize the EMAHelper with the DP_Wrapper itself.
        2. Initialize the EMAHelper with the DP_Wrapper's model.
        3. Initialize the EMAHelper with the DP_Wrapper's model's module.

        With the current implementation, all three methods should be equivalent.
        """

        # Create a dummy model
        model = DummyModel()
        # Create a dummy optimizer
        optimizer = torch.optim.Adam(model.parameters())
        # Create a DP Wrapper
        wrapper = DP_Wrapper(name="dp_wrapper", model=model, optimizer=optimizer, dp=True)
        # Initialize DP
        dataloader = wrapper.init_dp(dataloader=self.dl, epochs=1, max_grad_norm=1.0, noise_multiplier=1.0)

        self.assertTrue(isinstance(wrapper.model, GradSampleModule), "DP Initialization failed")

        # Create a dummy EMA helpers
        ema_wrapper = EMAHelper(mu=0.0, module=wrapper)
        ema_model = EMAHelper(mu=0.0, module=wrapper.model)
        ema_module = EMAHelper(mu=0.0, module=wrapper.model._module)

        # Assert that all three ema_helpers are equivalent
        self.assertEqual(ema_wrapper, ema_model)
        self.assertEqual(ema_wrapper, ema_module)
        self.assertEqual(ema_model, ema_module)

        # Do a few updates
        for data in dataloader:
            wrapper.opt.zero_grad()
            output = wrapper(data)
            loss = output.mean()
            loss.backward()
            wrapper.opt.step()
            ema_wrapper.update(wrapper)
            ema_model.update(wrapper.model)
            ema_module.update(wrapper.model._module)

        # Verify that the EMA's shadow parameters are identical to the model's parameters
        self.assertEqual(ema_wrapper, ema_model)
        self.assertEqual(ema_wrapper, ema_module)
        self.assertEqual(ema_model, ema_module)

    def test_register_optimizer(self):
        """Register an optimizer such that each update of the optimizer updates the EMA."""
        # Create a dummy model
        model = DummyModel()
        # Create a dummy optimizer
        optimizer = torch.optim.Adam(model.parameters())
        # Create a dummy EMA helper
        ema_helper = EMAHelper(mu=0.0, module=model)
        # Register the optimizer
        ema_helper.register_optimizer(optimizer)

        # Save current model parameters for comparison
        original_params = {name: param.clone() for name, param in model.named_parameters()}

        # Do a few updates
        for data in self.dl:
            optimizer.zero_grad()
            output = model(data)
            loss = output.mean()
            loss.backward()
            optimizer.step()
            # Note that the optimizer is updated, not the ema_helper

        # Verify that the EMA's shadow parameters have changed
        model_params = {name: param for name, param in model.named_parameters()}
        for name in model_params:
            self.assertFalse(torch.equal(ema_helper.shadow[name], original_params[name]))
            self.assertTrue(torch.equal(ema_helper.shadow[name], model_params[name]))

        # Try to manually update the EMA -> Should raise an error
        with self.assertRaises(ValueError):
            ema_helper.update(model)

    def test_register_optimizer_wrapper(self):
        """Register an optimizer such that each update of the optimizer updates the EMA."""
        # Create a dummy model
        model = DummyModel()
        # Create a dummy optimizer
        optimizer = torch.optim.Adam(model.parameters())
        # Create a DP Wrapper
        wrapper = DP_Wrapper(name="dp_wrapper", model=model, optimizer=optimizer, dp=True)
        # Initialize DP
        dataloader = wrapper.init_dp(dataloader=self.dl, epochs=1, max_grad_norm=1.0, noise_multiplier=1.0)
        # Create a dummy EMA helper
        ema_helper = EMAHelper(mu=0.0, module=wrapper)
        # Register the optimizer
        ema_helper.register_optimizer(wrapper.opt)

        # Save current model parameters for comparison
        original_params = {name: param.clone() for name, param in model.named_parameters()}

        # Do a few updates
        for data in dataloader:
            wrapper.opt.zero_grad()
            output = wrapper(data)
            loss = output.mean()
            loss.backward()
            wrapper.opt.step()
            # Note that the optimizer is updated, not the ema_helper

        # Verify that the EMA's shadow parameters have changed
        model_params = {name: param for name, param in model.named_parameters()}
        for name in model_params:
            self.assertFalse(torch.equal(ema_helper.shadow[name], original_params[name]))
            self.assertTrue(torch.equal(ema_helper.shadow[name], model_params[name]))

        # Try to manually update the EMA -> Should raise an error
        with self.assertRaises(ValueError):
            ema_helper.update(wrapper)


