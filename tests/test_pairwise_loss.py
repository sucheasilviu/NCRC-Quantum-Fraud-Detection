"""Smoke check for the opt-in pairwise term in the VQC trainer."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import train_vqc  # noqa: E402


class PairwiseLossTest(unittest.TestCase):
    def test_pairwise_weight_trains_on_a_mixed_minibatch(self):
        x = np.linspace(-1.0, 1.0, 16)
        features = np.column_stack((x, np.ones_like(x)))
        labels = (x > 0).astype(float)

        def model(batch, weights):
            return torch.sigmoid(batch @ weights)

        def init(seed):
            return torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))

        weights, history, validation_passes = train_vqc(
            model, init, features, labels, features, labels, seed=3,
            epochs=1, pairwise_weight=0.5,
        )

        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(np.isfinite(history.loc[0, "train loss"]))
        self.assertEqual(validation_passes, 1)


if __name__ == "__main__":
    unittest.main()
