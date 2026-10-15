"""Focused checks for M1's actual gradient, future-access and capacity changes."""
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from g1_embodiedskills.learning import use_upstream

use_upstream(os.environ.get("EMBODIEDSKILLS_ROOT", str(Path(__file__).resolve().parents[2] / "EmbodiedSkills")))
from g1_embodiedskills.m1_learning import (
    SACriticMember, build_m1_member, current_statistics, gradient_route_audit,
    load_m1_archive, loss_components, tensor_batch, forward_member,
)
from embodiedskills.training import fit_statistics, seed_everything
from embodiedskills.losses import _loss


def sample():
    rng = np.random.default_rng(17)
    visual = rng.normal(size=(4, 1024)).astype(np.float32)
    visual /= np.linalg.norm(visual, axis=-1, keepdims=True)
    return {
        "visual": visual, "future_visual": np.repeat(visual[:, None], 5, axis=1),
        "language": rng.normal(size=(4, 384)).astype(np.float32),
        "state": rng.normal(size=(4, 6)).astype(np.float32),
        "progress": np.full((4, 1), .5, np.float32),
        "actions": rng.normal(size=(4, 5, 3, 2)).astype(np.float32),
        "mask": np.ones((4, 5, 3), np.float32),
        "success": rng.integers(0, 2, (4, 5)).astype(np.float32),
        "return": rng.normal(size=(4, 5)).astype(np.float32),
        "expert_utility": rng.normal(size=(4, 5)).astype(np.float32),
        "task": np.full(4, "test"), "seed": np.arange(4), "anchor": np.zeros(4, np.int64),
    }


class M1Contract(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.data = sample()
        self.config = dict(hidden=192, success_priority=4., outcome_change_standardized=True, state_dim=6, action_dim=2, execution_horizon=3)

    def test_critic_contract_loads_and_trains_with_future_field_deleted(self):
        data = {key: value for key, value in self.data.items() if key != "future_visual"}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "without-future.npz"
            np.savez(path, **data)
            loaded = load_m1_archive(path, critic=True)
            self.assertNotIn("future_visual", loaded)
            with self.assertRaisesRegex(ValueError, "future_visual"):
                load_m1_archive(path, critic=False)
        statistics = current_statistics(loaded)
        self.assertFalse(any("delta" in key or "future" in key for key in statistics))
        model = SACriticMember(**self.config)
        batch = tensor_batch(loaded, np.arange(4), "cpu")
        out = forward_member(model, statistics, batch)
        loss, parts = loss_components(model, out, batch, data["task"], statistics, 1., .5, critic=True)
        loss.backward()
        self.assertNotIn("future_cosine", parts)
        self.assertGreater(float(model.action_gru.weight_ih_l0.grad.abs().sum()), 0.)

    def test_full_loss_matches_upstream_and_no_future_diff_is_exact(self):
        statistics = fit_statistics(self.data, "continuous")
        seed_everything(0)
        model = build_m1_member("full_g", self.config, statistics)
        batch = tensor_batch(self.data, np.arange(4), "cpu")
        out = forward_member(model, statistics, batch)
        ours, parts = loss_components(model, out, batch, self.data["task"], statistics, 1., .5)
        return_stats = batch["return"].new_tensor([statistics["returns"][str(task)] for task in self.data["task"]])
        expert = batch["expert_utility"] - batch["expert_utility"].mean(1, keepdim=True)
        expert /= expert.std(1, keepdim=True, unbiased=False).clamp_min(1e-5)
        expected, _ = _loss(out, {"future": batch["future_visual"], "success": batch["success"], "delta_standard": (batch["future_visual"] - batch["visual"][:, None] - model.delta_mean) / model.delta_std, "return_standard": (batch["return"] - return_stats[:, :1]) / return_stats[:, 1:], "expert_standard": expert}, success_priority=4., positive_weight=1., expert_weight=.5)
        torch.testing.assert_close(ours, expected)
        without, _ = loss_components(model, out, batch, self.data["task"], statistics, 1., .5, future_weight=0.)
        torch.testing.assert_close(ours - without, parts["delta"] + parts["relative_delta"] + parts["future_cosine"], atol=1e-5, rtol=1e-5)

    def test_paired_initialization_capacity_and_score_gradient_routes(self):
        statistics = fit_statistics(self.data, "continuous")
        seed_everything(0)
        full = build_m1_member("full_g", self.config, statistics)
        seed_everything(0)
        no_future = build_m1_member("no_future_g", self.config, statistics)
        for key, value in full.state_dict().items():
            torch.testing.assert_close(value, no_future.state_dict()[key], rtol=0, atol=0)
        critic = SACriticMember(**self.config)
        self.assertEqual(sum(p.numel() for p in full.parameters()), sum(p.numel() for p in critic.parameters()))
        for variant in ("full_d", "full_g", "no_future_g", "sa_critic"):
            data = {key: value for key, value in self.data.items() if variant != "sa_critic" or key != "future_visual"}
            audit = gradient_route_audit(variant, self.config, current_statistics(data) if variant == "sa_critic" else statistics, data, "cpu")
            norm = audit["score_only"]["groups"]["action_gru"]["gradient_squared_norm"]
            self.assertEqual(norm == 0., variant == "full_d")


if __name__ == "__main__":
    unittest.main()
