import unittest
from unittest import TestCase

import numpy as np
import torch
from torch.utils.data import TensorDataset

from privtrajgen.evaluation.split_dataset import *


# Test class
class TestDatasetSplit(TestCase):

    def setUp(self):
        self.num_samples = 1000

    def test_select_indices(self):
        # Create indices to select
        indices = [0, 1, 2, 3, 4]

        # Create datasets of different types
        np_dataset = np.arange(self.num_samples)
        torch_tensor_dataset = torch.arange(self.num_samples)
        tensor_dataset = TensorDataset(torch.arange(self.num_samples),
                                       torch.arange(self.num_samples, self.num_samples * 2))

        # Test with np.ndarray
        selected_np = select_indices(indices, np_dataset)
        np.testing.assert_array_equal(selected_np, np_dataset[indices])

        # Test with torch.Tensor
        selected_tensor = select_indices(indices, torch_tensor_dataset)
        self.assertTrue(torch.equal(selected_tensor, torch_tensor_dataset[indices]))

        # Test with TensorDataset
        selected_tensor_dataset = select_indices(indices, tensor_dataset)
        # Check that selected_tensor_dataset is a TensorDataset
        self.assertIsInstance(selected_tensor_dataset, TensorDataset)
        # Check that tensors in selected_tensor_dataset are correctly indexed
        for i, (batch, label) in enumerate(selected_tensor_dataset):
            self.assertTrue(torch.equal(batch, tensor_dataset[i][0]))
            self.assertTrue(torch.equal(label, tensor_dataset[i][1]))

    def test_disjoint_and_complete(self):
        # Iterate over all datasets in Datasets enum
        for dataset in Datasets:
            if dataset == Datasets.MNISTSequential:
                continue
            # Load train and test sets
            try:
                for k in (3, 5):
                    for s in range(k):
                        train_set = get_train_set_idx(dataset, s, k)
                        test_set = get_test_set_idx(dataset, s, k)
                        # Check that train and test sets are disjoint
                        self.assertTrue(np.intersect1d(train_set, test_set).size == 0)
                        # Check that train and test sets are complete
                        self.assertTrue(np.union1d(train_set, test_set).size == number_of_trajectories[dataset])
            except FileNotFoundError as e:
                # In case the data/ directory is not available, we skip the test
                print(f"Skipping test for {dataset.name}: {e}")

if __name__ == '__main__':
    unittest.main()
