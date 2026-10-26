"""Exact-gradient and Adam-step checks for the packed NCRC gradient compiler."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import PackedNCRCGradient, active_executions_per_sample, make_noisy_model, make_model  # noqa: E402
from models import MODELS, ncrc_circuit  # noqa: E402


class PackedNcrcGradientTest(unittest.TestCase):
    def test_pruned_default_preserves_scores_and_reduces_standard_shifts(self):
        spec = MODELS["ncrc_4q_L4"]
        self.assertTrue(spec["prune_local_z"])
        self.assertEqual(spec["default_gradient_backend"], "combined")
        unpruned, _ = ncrc_circuit(4, 4, 4, prune=False)
        weights = spec["init"](0)
        X = torch.tensor(np.random.default_rng(11).random((20, 4)), dtype=torch.float64)
        with torch.no_grad():
            old_scores = make_model(unpruned, 4)(X, weights)
            new_scores = make_model(spec["circuit"], 4)(X, weights)
        self.assertTrue(torch.allclose(old_scores, new_scores, atol=1e-14, rtol=0))
        x = X[0].numpy()
        self.assertEqual(active_executions_per_sample(unpruned, weights, x, 4), 97)
        self.assertEqual(active_executions_per_sample(spec["circuit"], weights, x, 4), 81)

    def test_gradient_pullback_matches_exact_backprop(self):
        model_spec = MODELS["ncrc_4q_L4"]
        x = torch.tensor(np.random.default_rng(13).uniform(0, np.pi, 4), dtype=torch.float64)
        labels = torch.tensor([1.0], dtype=torch.float64)
        for noise in (0.0, 0.05):
            reference = model_spec["init"](7)
            model = make_noisy_model(model_spec["circuit"], noise, 4)
            loss = F.binary_cross_entropy(model(x[None], reference).clamp(1e-7, 1 - 1e-7), labels)
            loss.backward()
            for mode, expected_executions in (("euler", 57), ("parallel", 21), ("combined", 15)):
                reference_step = torch.nn.Parameter(reference.detach().clone())
                reference_step.grad = reference.grad.detach().clone()
                candidate = torch.nn.Parameter(reference.detach().clone())
                backend = PackedNCRCGradient(4, 4, False, mode)
                backend.backward(model, x[None], labels, candidate, F.binary_cross_entropy)
                self.assertTrue(torch.allclose(candidate.grad, reference.grad, atol=1e-9, rtol=1e-9), mode)
                self.assertEqual(1 + 2 * backend.n_slots, expected_executions)
                optimizer_reference = torch.optim.Adam([reference_step], lr=0.05)
                optimizer_candidate = torch.optim.Adam([candidate], lr=0.05)
                optimizer_reference.step()
                optimizer_candidate.step()
                self.assertTrue(torch.allclose(candidate, reference_step, atol=1e-9, rtol=1e-9), mode)


if __name__ == "__main__":
    unittest.main()
