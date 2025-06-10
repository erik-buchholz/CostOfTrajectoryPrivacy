#!/usr/bin/env python3
"""Test the BaseModel class."""
import logging
import os
import tempfile
import unittest

import opacus.validators.errors
import torch
import torch.nn as nn
from parameterized import parameterized
from torch.utils.data import DataLoader

from privtrajgen.models.dp_wrapper import DP_Wrapper
from privtrajgen.models.ema import EMAHelper
from privtrajgen.models.save_and_load import load_model
from privtrajgen.utils import configure_root_loger

configure_root_loger(logging.INFO,error_file="error_unittest.log")

class TestBaseModel(unittest.TestCase):

    def compare_models_equal(self, model1, model2):
        # Iterate over all named parameters and compare them
        for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
            # print(name1, name2)
            self.assertTrue(torch.allclose(param1, param2))

    def compare_models_unequal(self, model1, model2):
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            self.assertFalse(torch.allclose(p1, p2))

    def setUp(self):
        self.model_name = "test_model"
        # Get a temporary directory path
        self.temp_dir = tempfile.TemporaryDirectory()
        self.param_dir = self.temp_dir.name
        self.model = nn.Linear(10, 10)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.1)
        self.wrapper = DP_Wrapper(name=self.model_name, model=self.model, param_dir=self.param_dir,
                                  optimizer=None, dp=False)
        self.epochs = 10
        self.target_epsilon = 1.0
        self.max_grad_norm = 1.0
        self.dataset = torch.randn(100, 10)
        self.dataloader = DataLoader(self.dataset, batch_size=10)


    def tearDown(self):
        self.temp_dir.cleanup()

    def test_init(self):
        wrapper = self.wrapper
        self.assertEqual(wrapper.name, self.model_name)
        self.assertEqual(wrapper.model, self.model)
        self.assertFalse(wrapper.dp)
        self.assertFalse(wrapper.dp_initialised)
        self.assertIn(self.param_dir, wrapper.param_file)
        self.assertIsNone(wrapper.opt)

    def test_forward(self):
        x = torch.randn(5, 10)
        self.assertTrue(
            torch.allclose(self.wrapper(x), self.model(x))
        )

    def test_save_and_load_parameters_no_dp(self):
        epoch = 1
        wrapper = self.wrapper
        # Save parameters
        wrapper.save_parameters(epoch=epoch)
        # Create a new model
        new_model = nn.Linear(10, 10)
        # Randomly initialize the new model
        torch.nn.init.normal_(new_model.weight)
        new_wrapper = DP_Wrapper(name=self.model_name, model=new_model, param_dir=self.param_dir,
                                 optimizer=self.optimizer, dp=False)
        # Verify that the parameters are different
        self.assertFalse(torch.allclose(wrapper.model.weight, new_wrapper.model.weight))
        # Load parameters
        new_wrapper.load_parameters(epoch=epoch)
        # Verify that the parameters are the same
        self.assertTrue(torch.allclose(wrapper.model.weight, new_wrapper.model.weight))



    def test_save_and_load_parameters_dp_uninitialised(self):
        epoch = 1
        wrapper = DP_Wrapper(name=self.model_name, model=self.model, param_dir=self.param_dir, dp=True, optimizer=self.optimizer)
        # Save parameters
        with self.assertRaises(RuntimeError):
            wrapper.save_parameters(epoch=epoch)

    def test_save_and_load_non_dp(self):
        epoch = 1
        wrapper = DP_Wrapper(name=self.model_name, model=self.model, param_dir=self.param_dir, dp=True, optimizer=self.optimizer)
        wrapper.init_dp(dataloader=self.dataloader, epochs=self.epochs,
                        target_epsilon=self.target_epsilon, max_grad_norm=self.max_grad_norm)
        # Save parameters
        wrapper.save_parameters(epoch=epoch)
        # Create a new model
        new_model = nn.Linear(10, 10)
        # Randomly initialize the new model
        torch.nn.init.normal_(new_model.weight)
        # Verify that the parameters are different
        self.assertFalse(torch.allclose(wrapper.model.state_dict()["_module.weight"], new_model.state_dict()["weight"]))
        # Load parameters
        new_wrapper = DP_Wrapper(name=self.model_name, model=new_model, param_dir=self.param_dir, dp=False, optimizer=None)
        new_wrapper.load_non_dp_parameters(epoch=epoch)
        # Verify that the parameters are the same
        self.assertTrue(torch.allclose(wrapper.model.state_dict()["_module.weight"], new_wrapper.model.state_dict()["weight"]))
        # Test manual loading
        param_path = wrapper.param_file
        new_model2 = nn.Linear(10, 10)
        torch.nn.init.normal_(new_model2.weight)
        self.assertFalse(torch.allclose(new_model.weight, new_model2.weight))
        new_model2.load_state_dict(torch.load(param_path.format(EPOCH=epoch), weights_only=True))
        self.assertTrue(torch.allclose(new_model.weight, new_model2.weight))

    def test_save_and_load_parameters_underlying_model_not_dp(self):
        """This test verifies that the parameters saved by a non-dp wrapper can also be loaded into a model
        without wrapping it first.
        """
        model = nn.Linear(10, 10)
        opt = torch.optim.AdamW(model.parameters(), lr=0.1)
        wrapper = DP_Wrapper(name=self.model_name, model=model, param_dir=self.param_dir, dp=False, optimizer=opt)
        param_file = os.path.join(self.param_dir, f"model.pth")
        # Save wrapper parameters
        wrapper.save_parameters(param_file=param_file)
        # Create a new model
        new_model = nn.Linear(10, 10)
        # Create a new wrapper
        new_wrapper = DP_Wrapper(name=self.model_name, model=new_model, param_dir=self.param_dir, dp=False, optimizer=self.optimizer)

        # Verify that the parameters of model and new_model are different
        self.compare_models_unequal(model, new_model)
        # Verify that the wrapper parameters are different too
        self.compare_models_unequal(wrapper, new_wrapper)

        # Load parameters into new model
        load_model(model=new_model, param_path=param_file)
        # Verify that the parameters of model and new_model are the same
        self.compare_models_equal(model, new_model)
        # Un-initialize the new wrapper
        new_wrapper.model = nn.Linear(10, 10)
        # Verify that the wrapper parameters are still different
        self.compare_models_unequal(wrapper, new_wrapper)
        # Load parameters into new wrapper
        new_wrapper.load_parameters(param_file=param_file)
        # Verify that the parameters of wrapper and new_wrapper are the same
        self.compare_models_equal(wrapper, new_wrapper)

    def test_save_and_load_wrapper_to_wrapper_no_dp(self):
        """Wrapper -> Wrapper without DP"""
        model = nn.Linear(10, 10)
        opt = torch.optim.AdamW(model.parameters(), lr=0.1)
        loaded_model = nn.Linear(10, 10)
        opt2 = torch.optim.AdamW(loaded_model.parameters(), lr=0.1)
        wrapper = DP_Wrapper(name=self.model_name, model=model, dp=False, param_dir=self.param_dir,
                             optimizer=opt)
        loaded_wrapper = DP_Wrapper(name=self.model_name, model=loaded_model, dp=False, param_dir=self.param_dir,
                                    optimizer=opt2)
        # Assert that the models are different
        self.compare_models_unequal(model, loaded_model)
        # Assert that model and the wrapper and the model are the same
        self.compare_models_equal(model, wrapper)
        # Assert that the wrappers are the different
        self.compare_models_unequal(wrapper, loaded_wrapper)

        # Do one training step
        x = torch.randn(5, 10)
        loss = wrapper(x).sum()
        loss.backward()
        wrapper.opt.step()
        wrapper.opt.zero_grad()

        # Save the wrapper
        wrapper.save_parameters(epoch=7)
        # Load saved parameters into the new wrapper
        loaded_wrapper.load_parameters(epoch=7)
        # Assert that the models are the same
        self.compare_models_equal(model, loaded_model)
        # Assert that the wrappers are the same
        self.compare_models_equal(wrapper, loaded_wrapper)

    def test_save_and_load_wrapper_to_wrapper_dp(self):
        """Wrapper -> Wrapper with DP"""
        model = nn.Linear(10, 10)
        loaded_model = nn.Linear(10, 10)
        # Create optimizers
        opt = torch.optim.AdamW(model.parameters(), lr=0.1)
        new_opt = torch.optim.AdamW(loaded_model.parameters(), lr=0.1)
        wrapper = DP_Wrapper(name=self.model_name, model=model, dp=True, param_dir=self.param_dir,
                             optimizer=opt)
        loaded_wrapper = DP_Wrapper(name=self.model_name, model=loaded_model, dp=True, param_dir=self.param_dir,
                                    optimizer=new_opt)
        # Assert that the models are different
        self.compare_models_unequal(model, loaded_model)
        # Assert that model and the wrapper and the model are the same
        self.compare_models_equal(model, wrapper)
        # Assert that the wrappers are the different
        self.compare_models_unequal(wrapper, loaded_wrapper)
        # Initialise DP
        wrapper.init_dp(dataloader=self.dataloader, epochs=self.epochs,
                        target_epsilon=self.target_epsilon, max_grad_norm=self.max_grad_norm)

        # Do one training step
        x = torch.randn(5, 10)
        loss = wrapper(x).sum()
        loss.backward()
        wrapper.opt.step()
        wrapper.opt.zero_grad()

        # Save the wrapper
        wrapper.save_parameters(epoch=7)
        # Load saved parameters into the new wrapper
        loaded_wrapper.load_parameters(epoch=7, dataloader=self.dataloader)
        # Assert that the models are the same
        self.compare_models_equal(model, loaded_model)
        # Assert that the wrappers are the same
        self.compare_models_equal(wrapper, loaded_wrapper)

    def test_save_and_load_wrapper_to_model_no_dp(self):
        """Wrapper -> Model without DP"""
        model = nn.Linear(10, 10)
        opt = torch.optim.AdamW(model.parameters(), lr=0.1)
        loaded_model = nn.Linear(10, 10)
        wrapper = DP_Wrapper(name=self.model_name, model=model, dp=False, param_dir=self.param_dir,
                             optimizer=opt)
        # Assert that the models are different
        self.compare_models_unequal(model, loaded_model)
        # Assert that model and the wrapper and the model are the same
        self.compare_models_equal(model, wrapper)

        # Do one training step
        x = torch.randn(5, 10)
        loss = wrapper(x).sum()
        loss.backward()
        wrapper.opt.step()
        wrapper.opt.zero_grad()

        # Save the wrapper
        wrapper.save_parameters(epoch=7)
        param_file = wrapper.param_file
        # Load saved parameters directly into the new model w/o wrapper
        load_model(model=loaded_model, param_path=param_file, epoch=7)
        # Assert that the models are the same
        self.compare_models_equal(model, loaded_model)

    def test_save_and_load_wrapper_to_model_dp(self):
        """Wrapper -> Model with DP"""
        model = nn.Linear(10, 10)
        loaded_model = nn.Linear(10, 10)

        # Create Wrapper
        opt = torch.optim.AdamW(model.parameters(), lr=0.1)
        wrapper = DP_Wrapper(name=self.model_name, model=model, dp=True, param_dir=self.param_dir,
                                optimizer=opt)
        # Initialise DP
        wrapper.init_dp(dataloader=self.dataloader, epochs=self.epochs,
                        target_epsilon=self.target_epsilon, max_grad_norm=self.max_grad_norm)

        # Assert models are different
        self.compare_models_unequal(model, loaded_model)
        # Assert model and wrapper are the same
        self.compare_models_equal(model, wrapper)

        # Do one training step
        x = torch.randn(5, 10)
        loss = wrapper(x).sum()
        loss.backward()
        wrapper.opt.step()
        wrapper.opt.zero_grad()

        # Save the wrapper
        wrapper.save_parameters(epoch=7)

        # Load saved parameters directly into the new model w/o wrapper
        load_model(model=loaded_model, param_path=wrapper.param_file, epoch=7)

        # Assert models are the same
        self.compare_models_equal(model, loaded_model)


    def test_save_and_load_ema_to_model_no_dp(self):
        """Model -> Wrapper without DP"""
        model = nn.Linear(10, 10)
        opt = torch.optim.AdamW(model.parameters(), lr=0.1)
        initial_model = nn.Linear(10, 10)
        # Copy all parameters from model to initial_model
        initial_model.load_state_dict(model.state_dict())
        loaded_model = nn.Linear(10, 10)
        wrapper = DP_Wrapper(name=self.model_name, model=model, dp=False, param_dir=self.param_dir,
                             optimizer=opt)


        # Create EMAHelper (set mu so low that the EMA is the same as the model)
        my = 0.0
        ema_helper = EMAHelper(mu=my)
        ema_helper.register(wrapper)

        # Do one training step
        x = torch.randn(5, 10)
        loss = wrapper(x).sum()
        loss.backward()
        wrapper.opt.step()
        wrapper.opt.zero_grad()
        ema_helper.update(model)

        # EMA should equal model
        for name, param in model.named_parameters():
            self.assertTrue(torch.allclose(param, ema_helper.shadow[name]))
        # EMA should differ from loaded model
        for name, param in loaded_model.named_parameters():
            self.assertFalse(torch.allclose(param, ema_helper.shadow[name]))
        # Model should differ from loaded model
        self.compare_models_unequal(model, loaded_model)

        # Save the wrapper
        wrapper.save_parameters(epoch=7, ema_helper=ema_helper)
        # Load saved parameters directly into the new model w/o wrapper
        load_model(model=loaded_model, param_path=wrapper.param_file, epoch=7, is_ema=True)
        # Assert that the models are the same
        self.compare_models_equal(model, loaded_model)
        # Assert ema equals model
        for name, param in loaded_model.named_parameters():
            self.assertTrue(torch.allclose(param, ema_helper.shadow[name]))
        # Assert initial model differs from loaded model
        self.compare_models_unequal(initial_model, loaded_model)

    def test_save_and_load_ema_to_model_dp(self):
        """Model -> Wrapper without DP"""
        model = nn.Linear(10, 10)
        opt = torch.optim.AdamW(model.parameters(), lr=0.1)
        initial_model = nn.Linear(10, 10)
        # Copy all parameters from model to initial_model
        initial_model.load_state_dict(model.state_dict())
        loaded_model = nn.Linear(10, 10)
        wrapper = DP_Wrapper(name=self.model_name, model=model, dp=True, param_dir=self.param_dir,
                             optimizer=opt)
        # Initialise DP
        wrapper.init_dp(dataloader=self.dataloader, epochs=self.epochs,
                        target_epsilon=self.target_epsilon, max_grad_norm=self.max_grad_norm)


        # Create EMAHelper (set mu so low that the EMA is the same as the model)
        my = 0.0
        ema_helper = EMAHelper(mu=my)
        ema_helper.register(wrapper)
        print(ema_helper.shadow)

        # Do one training step
        x = torch.randn(5, 10)
        loss = wrapper(x).sum()
        loss.backward()
        wrapper.opt.step()
        wrapper.opt.zero_grad()
        ema_helper.update(wrapper)
        print(ema_helper.shadow)

        # Save the wrapper
        wrapper.save_parameters(epoch=7, ema_helper=ema_helper)

        # EMA should equal model
        for name, param in model.named_parameters():
            self.assertTrue(torch.allclose(param, ema_helper.shadow[name]))
        # EMA Should equal wrapper.model._module
        for name, param in wrapper.model._module.named_parameters():
            self.assertTrue(torch.allclose(param, ema_helper.shadow[name]))
        # EMA should differ from loaded model
        for name, param in loaded_model.named_parameters():
            self.assertFalse(torch.allclose(param, ema_helper.shadow[name]))
        # Model should differ from loaded model
        self.compare_models_unequal(model, loaded_model)

        # Load saved parameters directly into the new model w/o wrapper
        load_model(model=loaded_model, param_path=wrapper.param_file, epoch=7, is_ema=True)
        # Assert that the models are the same
        self.compare_models_equal(model, loaded_model)
        # Assert ema equals model
        for name, param in loaded_model.named_parameters():
            self.assertTrue(torch.allclose(param, ema_helper.shadow[name]))
        # Assert initial model differs from loaded model
        self.compare_models_unequal(initial_model, loaded_model)


    def test_validate_dp_non_compatible(self):
        # BatchNorm1d is not compatible with DP-SGD
        model = nn.Sequential(nn.Linear(10, 10), nn.BatchNorm1d(10))
        wrapper = DP_Wrapper(name=self.model_name, model=model, dp=True, param_dir=self.param_dir, optimizer=self.optimizer)
        with self.assertRaises(opacus.validators.errors.UnsupportedModuleError):
            wrapper.validate_dp()

    def test_validate_dp_compatible(self):
        model = nn.Linear(10, 10)
        wrapper = DP_Wrapper(name=self.model_name, model=model, dp=True, param_dir=self.param_dir, optimizer=self.optimizer)
        wrapper.validate_dp()

    def test_fix_model(self):
        model = nn.Sequential(nn.Linear(10, 10), nn.BatchNorm1d(10))
        wrapper = DP_Wrapper(name=self.model_name, model=model, dp=True, param_dir=self.param_dir, optimizer=self.optimizer)
        wrapper.fix_model()
        wrapper.validate_dp()

    def test_init_dp_non_compatible(self):
        model = nn.Sequential(nn.Linear(10, 10), nn.BatchNorm1d(10))
        wrapper = DP_Wrapper(name=self.model_name, model=model, dp=True, param_dir=self.param_dir, optimizer=self.optimizer)
        with self.assertRaises(opacus.validators.errors.UnsupportedModuleError):
            wrapper.init_dp(dataloader=self.dataloader, epochs=self.epochs,
                            target_epsilon=self.target_epsilon, max_grad_norm=self.max_grad_norm)

    def test_init_dp_no_optimizer(self):
        wrapper = DP_Wrapper(name=self.model_name, model=self.model, dp=True, param_dir=self.param_dir, optimizer=None)
        with self.assertRaises(ValueError):
            wrapper.init_dp(dataloader=self.dataloader, epochs=self.epochs,
                            target_epsilon=self.target_epsilon, max_grad_norm=self.max_grad_norm)

    def test_init_dp_no_epsilon(self):
        wrapper = DP_Wrapper(name=self.model_name, model=self.model, dp=True, param_dir=self.param_dir, optimizer=self.optimizer)
        with self.assertRaises(ValueError):
            wrapper.init_dp(dataloader=self.dataloader, epochs=self.epochs,
                            max_grad_norm=self.max_grad_norm, optimizer=self.optimizer)

    def test_init_dp_double_call(self):
        wrapper = DP_Wrapper(name=self.model_name, model=self.model, dp=True, param_dir=self.param_dir, optimizer=self.optimizer)
        wrapper.init_dp(dataloader=self.dataloader, epochs=self.epochs,
                        target_epsilon=self.target_epsilon, max_grad_norm=self.max_grad_norm, optimizer=self.optimizer)
        with self.assertRaises(RuntimeError):
            wrapper.init_dp(dataloader=self.dataloader, epochs=self.epochs,
                            target_epsilon=self.target_epsilon, max_grad_norm=self.max_grad_norm,
                            optimizer=self.optimizer)
    def test_optimizer_after_init(self):
        wrapper = DP_Wrapper(name=self.model_name, model=self.model, dp=True, param_dir=self.param_dir, optimizer=self.optimizer)
        wrapper.init_dp(dataloader=self.dataloader, epochs=self.epochs,
                        target_epsilon=self.target_epsilon, max_grad_norm=self.max_grad_norm, optimizer=self.optimizer)
        # Store state of current parameters
        dp_weights = wrapper.model.state_dict()["_module.weight"].clone()
        # Forward path
        x = torch.randn(5, 10)
        loss = wrapper(x).sum()
        # Backward path
        loss.backward()
        wrapper.opt.step()
        wrapper.opt.zero_grad()
        # Check if parameters or DP Model have changed
        self.assertFalse(torch.allclose(dp_weights, wrapper.model.state_dict()["_module.weight"]))

    def test_save_and_load_dp_uninit(self):
        wrapper = DP_Wrapper(name=self.model_name, model=self.model, dp=True, param_dir=self.param_dir, optimizer=self.optimizer)
        with self.assertRaises(RuntimeError):
            wrapper.save_parameters(epoch=1)

    def test_save_and_load_dp(self):
        wrapper = DP_Wrapper(name=self.model_name, model=self.model, dp=True, param_dir=self.param_dir,
                            optimizer=self.optimizer)
        wrapper.init_dp(dataloader=self.dataloader, epochs=self.epochs,
                        target_epsilon=self.target_epsilon, max_grad_norm=self.max_grad_norm)
        epoch = 1
        wrapper.save_parameters(epoch=epoch)
        # Create a new model
        new_model = nn.Linear(10, 10)
        # Randomly initialize the new model
        torch.nn.init.normal_(new_model.weight)
        new_opt = torch.optim.AdamW(new_model.parameters(), lr=0.1)
        new_wrapper = DP_Wrapper(name=self.model_name, model=new_model, param_dir=self.param_dir, dp=True,
                                 optimizer=new_opt)
        # Verify that the parameters are different
        self.assertFalse(torch.allclose(wrapper.model.state_dict()["_module.weight"], new_wrapper.model.state_dict()["weight"]))
        x0 = torch.randn(5, 10)
        new_wrapper.dp_initialised = True  # Otherwise we cannot call forward
        self.assertFalse(torch.allclose(wrapper(x0), new_wrapper(x0)))
        new_wrapper.dp_initialised = False  # Otherwise model won't be initialized
        # Load parameters
        new_wrapper.load_parameters(epoch=epoch, dataloader=self.dataloader)
        # Verify that the parameters are the same
        self.assertTrue(torch.allclose(wrapper.model.state_dict()["_module.weight"], new_wrapper.model.state_dict()["_module.weight"]))
        self.assertTrue(torch.allclose(wrapper(x0), new_wrapper(x0)))


    # Iterate over all possible values for dp
    @parameterized.expand([(False,), (True,)])
    def test_device(self, dp):
        wrapper = DP_Wrapper(name=self.model_name, model=self.model, dp=dp, optimizer=self.optimizer)
        self.assertEqual(wrapper.device, torch.device("cpu"))
        if torch.cuda.is_available():
            wrapper.to("cuda:0")
            self.assertEqual(wrapper.device, torch.device("cuda:0"))
            if torch.cuda.device_count() > 1:
                wrapper.to("cuda:1")
                self.assertEqual(wrapper.device, torch.device("cuda:1"))

if __name__ == "__main__":
    unittest.main()


