#!/usr/bin/env python3
"""Test the BaseModel class."""
import tempfile
import unittest

import torch
import torch.nn as nn

from privtrajgen.models.base_model import BaseModel


class DerivedModel(BaseModel):
    def __init__(self, name, param_dir):
        super().__init__(name, param_dir)
        self.model = nn.Linear(10, 10)

    def forward(self, *args, **kwargs):
        pass

    def training_loop(self, *args, **kwargs):
        pass


class TestBaseModel(unittest.TestCase):
    def setUp(self):
        self.model_name = "test_model"
        # Get a temporary directory path
        self.temp_dir = tempfile.TemporaryDirectory()
        self.param_dir = self.temp_dir.name
        self.model = DerivedModel(name=self.model_name, param_dir=self.param_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_init(self):
        self.assertEqual(self.model.name, "test_model")
        self.assertIn(self.param_dir, self.model.param_path)

    def test_save_and_load_parameters(self):
        epoch = 0
        # Save parameters
        self.model.save_parameters(epoch=epoch)
        # Create a new model
        new_model = DerivedModel(name=self.model_name, param_dir=self.param_dir)
        # Randomly initialize the new model
        torch.nn.init.normal_(new_model.model.weight)
        # Verify that the parameters are different
        self.assertFalse(torch.allclose(self.model.model.weight, new_model.model.weight))
        # Load parameters
        new_model.load_parameters(epoch=epoch)
        # Verify that the parameters are the same
        self.assertTrue(torch.allclose(self.model.model.weight, new_model.model.weight))

    def test_device(self):
        self.assertEqual(self.model.device, torch.device("cpu"))
        if torch.cuda.is_available():
            self.model.to("cuda:0")
            self.assertEqual(self.model.device, torch.device("cuda:0"))
            if torch.cuda.device_count() > 1:
                self.model.to("cuda:1")
                self.assertEqual(self.model.device, torch.device("cuda:1"))

if __name__ == "__main__":
    unittest.main()


