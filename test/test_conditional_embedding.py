#!/usr/bin/env python3
""" """
from unittest import TestCase

import numpy as np
import torch

from privtrajgen.models.clipping import clip_to_norm, clip_to_max


class TestClipToNorm(TestCase):

    def test_clip_to_norm_no_clipping_needed(self):
        # Case where no clipping is needed (norm <= max_norm)
        data = torch.tensor([[1.0, 2.0], [2.0, 1.0], [1.0, 1.0]])
        # Norm of each sample is sqrt(1^2 + 2^2) = sqrt(5) ~ 2.236
        max_norm = 3.0  # Larger than the norms of data
        norm_type = 2.0
        result = clip_to_norm(data, max_norm, norm_type)

        # Check if result is unchanged
        self.assertTrue(torch.allclose(result, data), "Tensor should not be modified if norm <= max_norm")

    def test_clip_to_norm_with_clipping(self):
        # Case where clipping is needed (norm > max_norm)
        data = torch.tensor([[1.0, 2.0], [2.0, 1.0], [2.0, 2.0]])
        # Norm of each sample is sqrt(1^2 + 2^2) = sqrt(5) ~ 2.236
        max_norm = 2.0
        norm_type = 2.0
        result = clip_to_norm(data, max_norm, norm_type)

        self.assertEqual(result.shape, data.shape)

        # Check if the norm of each row is now <= max_norm
        norms = result.norm(p=norm_type, dim=1)
        # print(norms)
        self.assertTrue(torch.allclose(norms, torch.tensor([max_norm] * norms.size(0))),
                        "Each sample's norm should be clipped to max_norm")

    def test_clip_to_norm_random(self):
        """Check with random numbers that the clipping always results in a norm smaller than max_norm"""
        data = torch.randn(50, 100)  # Use gaussian b/c we don't need bounded sensitivity
        max_norm = 0.1
        norm_type = 2.0
        result = clip_to_norm(data, max_norm, norm_type)

        # Check that norm <= max_norm
        norms = result.norm(p=norm_type, dim=1)
        self.assertTrue(norms.shape == (50,))
        self.assertTrue(torch.all(norms <= max_norm + 1e-7), "Each sample's norm should be clipped to max_norm")

        # Verify that all samples that were clipped are now equal to max_norm
        clipped = data.norm(p=norm_type, dim=1) > max_norm
        self.assertTrue(
            torch.allclose(result[clipped].norm(p=norm_type, dim=1), torch.tensor([max_norm] * clipped.sum())),
            "Clipped samples should have norm equal to max_norm")

    def test_scale_up(self):
        data = torch.tensor([[1.0, 1.0], [2.0, 2.0], [0.5, 0.5]])
        max_norm = 2.0
        norm_type = 2.0
        result = clip_to_norm(data, max_norm, norm_type, scale_up=False)
        norms = result.norm(p=norm_type, dim=1)
        print(norms)
        # Some of the norms should be smaller than max_norm
        self.assertTrue(torch.any(norms < max_norm), "Some norms should be smaller than max_norm")
        result = clip_to_norm(data, max_norm, norm_type, scale_up=True)
        norms = result.norm(p=norm_type, dim=1)
        print(norms)
        # All norms should be equal to max_norm
        torch.testing.assert_allclose(norms, torch.tensor([max_norm] * norms.size(0)), rtol=1e-6, atol=1e-6,
                                      msg="All norms should be equal to max_norm")


class TestClipToMax(TestCase):

    def test_clip_to_max_uniform_sensitivity(self):
        # Test with uniform feature sensitivity (all features have sensitivity 1)
        data = torch.tensor([[3.0, 4.0], [5.0, 6.0]])
        max_norm = 5.0
        norm_type = 2.0
        feature_sensitivity = 6.0
        result = clip_to_max(data, max_norm, norm_type, apply_sigmoid=False, feature_sensitivity=feature_sensitivity)

        # Calculate the expected value.
        scaling_factor = max_norm / (np.sqrt(2 * 36) + 1e-6)
        expected_result = data * scaling_factor

        # Check if the norm of each row is now <= max_norm
        norms = result.norm(p=norm_type, dim=1)
        self.assertTrue(torch.all(norms <= max_norm), "Each sample's norm should be clipped to max_norm")

        # Check if the result is as expected
        self.assertTrue(torch.allclose(result, expected_result), "Result should be scaled by the same factor")

    def test_clip_to_max(self):
        # Test with uniform feature sensitivity (all features have sensitivity 1)
        data = torch.tensor([[3.0, 4.0], [5.0, 6.0], [1.0, 1.0]])
        max_norm = 10.0
        norm_type = 1.0
        feature_sensitivity = 10.0
        result = clip_to_max(data, max_norm, norm_type, apply_sigmoid=False, feature_sensitivity=feature_sensitivity)

        sensitivity = feature_sensitivity * data.shape[1]

        # Calculate the expected value.
        expected = data * (max_norm / (sensitivity + 1e-6))

        # Check if result is unchanged
        self.assertTrue(torch.allclose(result, expected), "Result should be scaled by the same factor")

        # Check that norm <= max_norm
        norms = result.norm(p=norm_type, dim=1)
        self.assertTrue(torch.all(norms <= max_norm), "Each sample's norm should be clipped to max_norm")

    def test_clip_max_random_no_sigmoid(self):
        """Check with random numbers that the clipping always results in a norm smaller than max_norm"""
        data = torch.rand(100, 10)  # Output range bound to [0, 1]
        feature_sensitivity = 1.0
        max_norm = 0.1
        norm_type = 2.0
        result = clip_to_max(data, max_norm, norm_type, False, feature_sensitivity)

        # Check that norm <= max_norm
        norms = result.norm(p=norm_type, dim=1)
        self.assertTrue(torch.all(norms <= max_norm), "Each sample's norm should be clipped to max_norm")

    def test_clip_max_random_sigmoid(self):
        """Check with random numbers that the clipping always results in a norm smaller than max_norm"""
        data = torch.rand(100, 10)
        feature_sensitivity = 1.0
        max_norm = 0.1
        norm_type = 2.0
        result = clip_to_max(data, max_norm, norm_type, True)

        # Check that norm <= max_norm
        norms = result.norm(p=norm_type, dim=1)
        self.assertTrue(torch.all(norms <= max_norm), "Each sample's norm should be clipped to max_norm")

    def test_wrong_feature_sensitive(self):
        data = torch.tensor([[3.0, 4.0], [5.0, 6.0], [1.0, 1.0]])
        max_norm = 10.0
        norm_type = 1.0
        feature_sensitivity = 1.0  # Some values are larger than 1
        with self.assertRaises(ValueError):
            clip_to_max(data, max_norm, norm_type, apply_sigmoid=False, feature_sensitivity=feature_sensitivity)
        # No error if apply_sigmoid is True
        res = clip_to_max(data, max_norm, norm_type, apply_sigmoid=True, feature_sensitivity=feature_sensitivity)

        # Check that norm <= max_norm
        norms = res.norm(p=norm_type, dim=1)
        self.assertTrue(torch.all(norms <= max_norm), "Each sample's norm should be clipped to max_norm")
