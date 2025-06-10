#!/usr/bin/env python3
""" """
from unittest import TestCase
from unittest.mock import patch

from privtrajgen.privacy.vmf import *


class TestVMF(TestCase):
    def setUp(self):
        self.p = 8
        self.N = 10
        self.kappa = 0.5
        self.mu = np.random.randn(self.p)
        self.mu = self.mu / np.linalg.norm(self.mu)
        self.mu = np.reshape(self.mu, (-1, 1))
        self.normal_v = np.random.normal(0, 1, (self.N, self.p))
        self.t = np.array([[0.84462341],
                      [0.59870297],
                      [0.14892983],
                      [0.41675367],
                      [0.85811989],
                      [0.61850634],
                      [0.66449528],
                      [0.7289653],
                      [0.54810156],
                      [0.51268302]])
        self.xi = np.array(
            [[0.2618276, -0.24487466, 0.11093308, -0.85466379, -0.35216052, 0.03210312, 0.0606525],
             [0.25127797, 0.49900751, -0.02100929, 0.40800569, 0.05517929, 0.23255197, -0.68104014],
             [-0.04133007, 0.1505179, -0.05373118, -0.2667067, -0.73969449, -0.41185893, -0.42993136],
             [0.0662264, -0.03008504, 0.65389815, 0.32210991, -0.32336375, 0.38221118, -0.46121781],
             [0.62019749, -0.06476663, -0.28951977, 0.00719854, -0.59403996, 0.1398041, -0.39352024],
             [-0.68009085, -0.43802753, 0.38666562, 0.06556067, 0.41957946, 0.11423681, -0.05198873],
             [0.6626429, -0.38721318, 0.19728798, -0.35348799, -0.22980606, 0.27351185, -0.3456508],
             [0.38461752, -0.50752193, 0.30395431, 0.378664, -0.15319445, 0.55650274, -0.15985108],
             [0.2952418, 0.0059521, 0.14439805, 0.63694069, -0.07545093, 0.26691227, -0.63977935],
             [0.09527902, -0.36887487, -0.56053031, -0.57947161, 0.35697342, 0.17979842, -0.21240096]]
        )

    def test_rotate_sample_to_mu(self):
        """Assert that the torch version is the same as the numpy version"""
        # Create realistic Samples
        samples = np.zeros((self.N, self.p))
        t = rand_t_marginal_np(self.kappa, self.p, self.N)
        xi = rand_uniform_hypersphere_np(self.N, self.p - 1)
        samples[:, [0]] = t
        samples[:, 1:] = np.tile(np.sqrt(1 - t ** 2), (1, self.p - 1)) * xi

        # Run the numpy version (Ground Truth)
        rotated_samples_np = rotate_sample_to_mu_np(samples, self.mu)

        # convert data to torch
        samples = torch.from_numpy(samples).float()
        mu = torch.from_numpy(self.mu).float()

        # Run the torch version
        rotated_samples = rotate_sample_to_mu(samples, mu)

        # Assert that the results are the same
        self.assertTrue(np.allclose(rotated_samples_np, rotated_samples.numpy(), atol=1e-6))

    def test_torch_null_space(self):
        """Assert that the torch version is the same as the scipy version"""
        A = np.random.randn(1, 10)

        ground_truth = scipy.linalg.null_space(A)

        torch_nullspace = torch_null_space(torch.from_numpy(A).float())

        self.assertTrue(np.allclose(ground_truth, torch_nullspace.numpy(), atol=1e-6))

    @patch("torch.distributions.Beta.sample")
    @patch("torch.rand")
    def test_class_func_equivalence_RandTMarginal(self, mock_rand, mock_beta_sample):
        """Assert that the class and functional implementation are equivalent"""
        # Mock the random numbers
        # Beta.shape = [1,], rand.shape = []
        mock_beta_sample.return_value = torch.tensor(0.2550)
        mock_rand.return_value = torch.tensor(0.2527)

        # Functional Interface
        data_func = rand_t_marginal(self.kappa, self.p, self.N)

        # Class Interface
        data_cls = RandTMarginal(self.kappa, self.p).sample(self.N)

        # Assert that the results are the same
        self.assertTrue(torch.allclose(data_func, data_cls, atol=1e-6))

    @patch("numpy.random.normal")
    @patch("torch.normal")
    def test_rand_uniform_hypersphere(self, mock_torch_normal, mock_np_normal):
        """Assert that the torch version is the same as the numpy version"""
        mock_np_normal.return_value = self.normal_v
        mock_torch_normal.return_value = torch.from_numpy(self.normal_v).float()

        # Numpy version
        data_np = rand_uniform_hypersphere_np(self.N, self.p)

        # Torch version
        data = rand_uniform_hypersphere(self.N, self.p)

        # Assert that the results are the same
        self.assertTrue(np.allclose(data_np, data.numpy(), atol=1e-6))

    @patch("numpy.random.beta")
    @patch("torch.distributions.Beta.sample")
    @patch("numpy.random.uniform")
    @patch("torch.rand")
    def test_rand_t_marginal(self, mock_torch_rand, mock_np_uniform, mock_beta_sample, mock_np_beta):
        """Assert that the torch version is the same as the numpy version"""
        # Mock the random numbers
        # Beta.shape = [1,], rand.shape = []
        mock_beta_sample.return_value = torch.tensor(0.2550)
        mock_torch_rand.return_value = torch.tensor(0.2527)
        mock_np_uniform.return_value = 0.2527
        mock_np_beta.return_value = 0.2550

        # Numpy version
        data_np = rand_t_marginal_np(self.kappa, self.p, self.N)

        # Torch version
        data = rand_t_marginal(self.kappa, self.p, self.N)

        # Assert that the results are the same
        self.assertTrue(np.allclose(data_np, data.numpy(), atol=1e-6))

    @patch("privtrajgen.privacy.vmf.rand_t_marginal_np")
    @patch("privtrajgen.privacy.vmf.rand_t_marginal")
    @patch("privtrajgen.privacy.vmf.rand_uniform_hypersphere_np")
    @patch("privtrajgen.privacy.vmf.rand_uniform_hypersphere")
    def test_rand_von_mises_fisher(self, mock_torch_rand_uniform_hypersphere, mock_np_rand_uniform_hypersphere,
                                   mock_torch_rand_t_marginal, mock_np_rand_t_marginal):
        """Assert that the torch version is the same as the numpy version"""
        # Use a fixed case
        mu = np.random.randn(self.p)
        mu = mu / np.linalg.norm(mu)
        mu = np.reshape(mu, (-1, 1))

        # Assign the mock values
        mock_np_rand_t_marginal.return_value = self.t
        mock_torch_rand_t_marginal.return_value = torch.from_numpy(self.t).float()
        mock_np_rand_uniform_hypersphere.return_value = self.xi
        mock_torch_rand_uniform_hypersphere.return_value = torch.from_numpy(self.xi).float()

        # Numpy version
        data_np = rand_von_mises_fisher_np(mu, self.kappa, self.N)

        # Torch version
        data = rand_von_mises_fisher(torch.from_numpy(mu).float(), self.kappa, self.N)

        # Assert that the results are the same
        self.assertTrue(np.allclose(data_np, data.numpy(), atol=1e-6))

    @patch("privtrajgen.privacy.vmf.rand_t_marginal")
    @patch("privtrajgen.privacy.vmf.rand_uniform_hypersphere")
    def test_class_func_equivalence_VMF(self, mock_rand_uniform_hypersphere, mock_rand_t_marginal):
        """Assert that the class and functional implementation are equivalent"""
        # The new interface only produces one sample per mu
        N = 1
        # Use a fixed case
        mu = np.random.randn(self.p)
        mu = mu / np.linalg.norm(mu)
        mu = np.reshape(mu, (-1,))
        mu = torch.from_numpy(mu).float()

        # Assign the mock values
        mock_rand_t_marginal.return_value = torch.from_numpy(self.t[:N]).float()
        mock_rand_uniform_hypersphere.return_value = torch.from_numpy(self.xi[:N]).float()

        # Functional Interface
        data_func = rand_von_mises_fisher(mu.flatten(), self.kappa, N)

        # Class Interface
        mu = mu.reshape(1, -1)
        data_cls = RandVonMisesFisher(kappa=self.kappa, p=self.p).sample(mu)

        # print(data_func)
        # print(data_cls)

        # Assert that the results are the same
        self.assertTrue(torch.allclose(data_func, data_cls, atol=1e-6))


class TestVectorization(TestCase):

    def setUp(self):
        self.p = 8
        self.N = 10
        self.kappa = np.random.rand() * 10
        self.mus = torch.randn((self.N, self.p))
        # Normalise each row
        self.mus = self.mus / torch.norm(self.mus, dim=1, keepdim=True)
        # Shape: (N, p)

        self.t = np.array([[0.84462341],
                      [0.59870297],
                      [0.14892983],
                      [0.41675367],
                      [0.85811989],
                      [0.61850634],
                      [0.66449528],
                      [0.7289653],
                      [0.54810156],
                      [0.51268302]])
        self.xi = np.array(
            [[0.2618276, -0.24487466, 0.11093308, -0.85466379, -0.35216052, 0.03210312, 0.0606525],
             [0.25127797, 0.49900751, -0.02100929, 0.40800569, 0.05517929, 0.23255197, -0.68104014],
             [-0.04133007, 0.1505179, -0.05373118, -0.2667067, -0.73969449, -0.41185893, -0.42993136],
             [0.0662264, -0.03008504, 0.65389815, 0.32210991, -0.32336375, 0.38221118, -0.46121781],
             [0.62019749, -0.06476663, -0.28951977, 0.00719854, -0.59403996, 0.1398041, -0.39352024],
             [-0.68009085, -0.43802753, 0.38666562, 0.06556067, 0.41957946, 0.11423681, -0.05198873],
             [0.6626429, -0.38721318, 0.19728798, -0.35348799, -0.22980606, 0.27351185, -0.3456508],
             [0.38461752, -0.50752193, 0.30395431, 0.378664, -0.15319445, 0.55650274, -0.15985108],
             [0.2952418, 0.0059521, 0.14439805, 0.63694069, -0.07545093, 0.26691227, -0.63977935],
             [0.09527902, -0.36887487, -0.56053031, -0.57947161, 0.35697342, 0.17979842, -0.21240096]]
        )

    def test_vectorized_null_space(self):
        assert self.mus.shape == (self.N, self.p)
        baseline = []
        for mu in self.mus:
            mu = mu.reshape(-1, 1)
            # Mu has shape (p, 1)
            baseline.append(torch_null_space(mu.T))
        # bring to shape (N, p, p - 1)
        baseline = torch.stack(baseline)
        assert baseline.shape == (self.N, self.p, self.p - 1)

        # Bring to shape (N, 1, p) by adding a dimension and then transpose
        self.mus = self.mus.unsqueeze(-1).transpose(1, 2)
        assert self.mus.shape == (self.N, 1, self.p)
        torch_nullspace = torch_null_space_batch(self.mus)
        assert torch_nullspace.shape == (self.N, self.p, self.p - 1)

        self.assertTrue(np.allclose(baseline, torch_nullspace, atol=1e-6))

    def test_vectorized_rotate_single(self):
        """For a single mu, the results should be the same"""
        mu = self.mus[0].reshape(-1, 1)

        # Generate samples with shape (1, p)
        t = rand_t_marginal(kappa=self.kappa, p=self.p, N=1)
        xi = rand_uniform_hypersphere(N=1, p=self.p - 1)
        sample = torch.zeros((1, self.p))
        sample[:, [0]] = t
        sample[:, 1:] = torch.sqrt(1 - t ** 2).reshape(-1, 1).repeat(1, self.p - 1) * xi

        # Run the numpy version (Ground Truth)
        rotated_sample_np = rotate_sample_to_mu_np(sample.numpy(), mu.numpy())

        # Run the torch version
        rotated_samples = rotate_sample_to_mu_batch(sample, mu.unsqueeze(0))

        # Assert that the results are the same
        self.assertTrue(np.allclose(rotated_sample_np, rotated_samples.numpy(), atol=1e-6))

    def test_vectorized_rotate(self):
        assert self.mus.shape == (self.N, self.p)
        self.mus = self.mus.unsqueeze(-1)

        # Generate samples with shape (N, p)
        t = rand_t_marginal(kappa=self.kappa, p=self.p, N=self.N)
        xi = rand_uniform_hypersphere(N=self.N, p=self.p - 1)
        samples = torch.zeros((self.N, self.p))
        samples[:, [0]] = t
        samples[:, 1:] = torch.sqrt(1 - t ** 2).reshape(-1, 1).repeat(1, self.p - 1) * xi

        baseline = []
        for i, mu in enumerate(self.mus):
            mu = mu.reshape(-1, 1)
            # Mu has shape (p, 1)
            sample = samples[i].reshape(1, -1)
            baseline.append(rotate_sample_to_mu(sample, mu))  # Each has shape (1, p)
        # bring to shape (N, p)
        baseline = torch.cat(baseline)
        assert baseline.shape == (self.N, self.p), f"Shape: {baseline.shape}"

        # Vectorized
        rotated_samples = rotate_sample_to_mu_batch(samples, self.mus)

        # print(baseline)
        # print(rotated_samples)

        self.assertTrue(torch.allclose(baseline, rotated_samples, atol=1e-6))

    @patch("privtrajgen.privacy.vmf.rand_t_marginal")
    @patch("privtrajgen.privacy.vmf.rand_uniform_hypersphere")
    def test_vectroized_rand_von_mises_fisher(self, mock_rand_uniform_hypersphere, mock_rand_t_marginal):
        # Generate with for loop and old function
        baseline = []
        for i, mu in enumerate(self.mus):
            # Mock for individual
            mock_rand_t_marginal.return_value = torch.from_numpy(self.t[i]).float()
            mock_rand_uniform_hypersphere.return_value = torch.from_numpy(self.xi[i]).float()
            baseline.append(rand_von_mises_fisher(mu, self.kappa, 1))
        baseline = torch.cat(baseline)

        # Update Mocks
        mock_rand_t_marginal.return_value = torch.from_numpy(self.t).float()
        mock_rand_uniform_hypersphere.return_value = torch.from_numpy(self.xi).float()

        # Vectorized interface
        cls = RandVonMisesFisher(self.kappa, self.p)
        vectorized = cls.sample(self.mus)

        # print(baseline)
        # print(vectorized)

        # Assert
        self.assertTrue(torch.allclose(baseline, vectorized, atol=1e-6))



