#!/usr/bin/env python3
import json
import pickle
import unittest
from dataclasses import asdict, fields, is_dataclass

from privtrajgen.evaluation.config.baseconfig import *
from privtrajgen.evaluation.config.difftraj_config import DiffTrajConfig
from privtrajgen.evaluation.config.gan_config import *
from privtrajgen.models import get_difftraj_model
from privtrajgen.utils import dict_to_namespace


class TestEvalConfigs(unittest.TestCase):

    def test_dp_config(self):
        # Initialize DPConfig
        dp_config = DPConfig(dp=True)
        self.assertTrue(dp_config.dp)
        # with self.assertRaises(ValueError):
        #     # No value for dp
        #     # noinspection PyArgumentList,
        #     dp_config = DPConfig()
        # Ensure that adaptive_grad_norm is set to False b/c of implementation error.
        self.assertFalse(dp_config.adaptive_grad_norm)
        # Make sure that this cannot be changed
        with self.assertRaises(Exception):
            dp_config.adaptive_grad_norm = True

    def test_mandatory_fields(self):
        # Test mandatory fields for BaseConfig
        with self.assertRaises(ValueError):
            BaseConfig(conditional=False)

    def test_pickle(self):
        # Initialize BaseConfig
        base_config = BaseConfig(model_type=Models.UNetVAE, conditional=False, dataset=Datasets.PORTO)
        # Pickle BaseConfig
        base_config_pickle = pickle.dumps(base_config)
        # Unpickle BaseConfig
        base_config_unpickle = pickle.loads(base_config_pickle)
        self.assertEqual(base_config, base_config_unpickle)

        # Repeat for a BaseConfig with DPConfig
        base_config = BaseConfig(model_type=Models.UNetVAE, conditional=False, dataset=Datasets.PORTO, dp=DPConfig(dp=True))
        base_config_pickle = pickle.dumps(base_config)
        base_config_unpickle = pickle.loads(base_config_pickle)
        self.assertEqual(base_config, base_config_unpickle)

class TestGANConfig(unittest.TestCase):
    def test_gan_config(self):
        # Initialize GANConfig
        gan_config = GANConfig(model_type=Models.UNetGAN, conditional=False, dataset=Datasets.PORTO)
        self.assertEqual(gan_config.model_type, Models.UNetGAN)
        self.assertFalse(gan_config.conditional)
        self.assertEqual(gan_config.dataset, Datasets.PORTO)

    def test_loss(self):
        # Initialize LossConfig
        gan_config = GANConfig(model_type=Models.UNetGAN, conditional=False, dataset=Datasets.PORTO,
                               loss_option=GANLossOptions.WGAN)
        self.assertEqual(gan_config.loss, gan_loss_configs[GANLossOptions.WGAN])
        # Test name
        self.assertEqual(gan_config.loss.name, "WGAN")
        # Test boolean values
        self.assertTrue(gan_config.loss.wgan)
        self.assertFalse(gan_config.loss.gradient_penalty)
        self.assertFalse(gan_config.loss.lipschitz_penalty)
        # Change to WGAN_LP
        gan_config.loss_option = GANLossOptions.WGAN_LP
        # Test that the underlying loss has changed
        self.assertEqual(gan_config.loss, gan_loss_configs[GANLossOptions.WGAN_LP])
        # Test name
        self.assertEqual(gan_config.loss.name, "WGAN-LP")
        # Test boolean values
        self.assertTrue(gan_config.loss.wgan)
        self.assertTrue(gan_config.loss.gradient_penalty)
        self.assertTrue(gan_config.loss.lipschitz_penalty)

class TestDiffTrajConfig(unittest.TestCase):

    def test_bwd_compatibility(self):
        from privtrajgen.models.diffTraj.configs.config_cond import args
        from types import SimpleNamespace

        # List of keys to ignore
        ignore = ['ema', 'ema_rate', 'sampling']

        # Create DiffTrajConfig from legacy dictionary
        cfg = DiffTrajConfig.from_legacy_dict(args)

        # Convert the dictionary 'args' into a SimpleNamespace
        args_namespace: SimpleNamespace = dict_to_namespace(args)  # type: ignore

        # Recursively compare SimpleNamespace and DiffTrajConfig
        def compare_namespace_to_cfg(ns: SimpleNamespace, cfg: BaseConfig, ignore: Sequence[str]=()) -> None:
            for key, value in ns.__dict__.items():
                if str(key) in ignore:
                    continue
                self.assertTrue(hasattr(cfg, key), f"{key} not found in {type(cfg)}")
                cfg_value = getattr(cfg, key)
                if isinstance(value, SimpleNamespace):
                    # Recursively compare nested SimpleNamespace objects
                    compare_namespace_to_cfg(value, cfg_value, ignore)
                else:
                    # Assert values are equal for non-namespace attributes
                    self.assertEqual(value, cfg_value, f"Mismatch for key {key}")

        compare_namespace_to_cfg(args_namespace, cfg, ignore)

    def test_model_init_unconditional(self):
        # Initialize DiffTrajConfig
        difftraj_config = DiffTrajConfig(model_type=Models.DiffTraj, conditional=False, dataset=Datasets.PORTO)
        # Create model based on the config
        _ = get_difftraj_model(difftraj_config, conditional=False)

    def test_model_init_conditional(self):
        # Initialize DiffTrajConfig
        difftraj_config = DiffTrajConfig(model_type=Models.DiffTraj, conditional=True, dataset=Datasets.PORTO)
        # Create model based on the config
        _ = get_difftraj_model(difftraj_config, conditional=True)

    # noinspection PyPropertyAccess, PyDataclass
    def test_properties_setting(self):
        # Initialize DiffTrajConfig
        difftraj_config = DiffTrajConfig(model_type=Models.DiffTraj, conditional=False, dataset=Datasets.PORTO)
        # .data property - should not be modifiable
        with self.assertRaises(Exception):
            difftraj_config.data = None
        data = difftraj_config.data
        # Check is dataset is frozen
        self.assertEqual(data.dataset, Datasets.PORTO)
        with self.assertRaises(Exception):
            data.dataset = Datasets.FSNYC
        self.assertNotEqual(data.dataset, Datasets.FSNYC)
        # Dataset should not be changed
        self.assertEqual(difftraj_config.dataset, Datasets.PORTO)

        with self.assertRaises(Exception):
            difftraj_config.data.traj_length = 200
        # .training property - should not be modifiable

    # noinspection PyDataclass
    def test_frozen(self):
        cfg = DataConfig(traj_length=200, dataset=Datasets.PORTO)
        with self.assertRaises(Exception):
            cfg.dataset = Datasets.FSNYC

    def test_training_proxy(self):
        # Initialize DiffTrajConfig
        cfg = DiffTrajConfig(model_type=Models.DiffTraj, conditional=False, dataset=Datasets.PORTO)
        # Check that the training property is a TrainingProxy
        self.assertIsInstance(cfg.training, TrainingProxy)
        # Check that the training property has the n_epochs property
        self.assertEqual(cfg.training.n_epochs, cfg.epochs)
        # Check that the training property has the n_iters property
        self.assertEqual(cfg.training.n_iters, cfg.steps)
        # Check that the training property has the snapshot_freq property
        self.assertEqual(cfg.training.snapshot_freq, cfg.save_freq)
        # Test Setting
        cfg.training.n_epochs = 55
        self.assertEqual(cfg.epochs, 55)
        cfg.training.n_iters = 1000
        self.assertEqual(cfg.steps, 1000)

    def test_to_dict(self):
        # Initialize DiffTrajConfig
        difftraj_config = DiffTrajConfig(model_type=Models.DiffTraj, conditional=False, dataset=Datasets.PORTO)
        # Convert to dictionary
        difftraj_dict = asdict(difftraj_config)
        # pprint(difftraj_dict)
        # Check that properties are not part of the dictionary
        self.assertNotIn("data", difftraj_dict)
        # Check fields are part of the dictionary
        for field in fields(difftraj_config):
            # print(field.name)
            self.assertIn(field.name, difftraj_dict)
            # Deal with nested dataclasses
            field_value = getattr(difftraj_config, field.name)
            if is_dataclass(field_value):
                # print(field.name)
                # Check that the nested dataclass is converted to a dictionary
                self.assertIsInstance(difftraj_dict[field.name], dict)
                for nested_field in fields(field_value):
                    # print(nested_field.name)
                    self.assertIn(nested_field.name, difftraj_dict[field.name])

    def test_to_json(self):
        # Initialize DiffTrajConfig
        difftraj_config = DiffTrajConfig(model_type=Models.DiffTraj, conditional=False, dataset=Datasets.PORTO)
        # Convert to dictionary
        difftraj_dict = asdict(difftraj_config)
        # Convert to json (print nicely, with indentation of 4)
        difftraj_json = json.dumps(difftraj_dict, indent=4)
        # print(difftraj_json)
        # Check that properties are not part of the json
        self.assertNotIn('"data"', difftraj_json)
        # Check fields are part of the json
        for field in fields(difftraj_config):
            # print(field.name)
            self.assertIn(field.name, difftraj_json)
            # Deal with nested dataclasses
            field_value = getattr(difftraj_config, field.name)
            if is_dataclass(field_value):
                # print(field.name)
                for nested_field in fields(field_value):
                    # print(nested_field.name)
                    self.assertIn(nested_field.name, difftraj_json)


if __name__ == '__main__':
    unittest.main()
