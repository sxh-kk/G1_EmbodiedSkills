"""Synthetic interface fixtures only; these are not G1 experiment results."""

import copy
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from g1_embodiedskills.learning import (
    DEFAULT_CONFIG, attach_language_cache, build_language_cache, cached_language_vectors,
    encode_arrays, episode_split, read_archive, split_archive, training_config,
    training_language_cache, use_upstream, validate_archive, validate_language_cache,
)


def features(episodes=6):
    n = episodes * 2
    return {
        "task": np.full(n, "pick_place"), "seed": np.repeat(np.arange(episodes), 2),
        "anchor": np.tile([0, 1], episodes), "state": np.zeros((n, 6), np.float32),
        "progress": np.full((n, 1), 0.25, np.float32),
        "actions": np.zeros((n, 5, 4, 7), np.float32),
        "mask": np.ones((n, 5, 4), np.float32),
        "visual": np.zeros((n, 1024), np.float32),
        "future_visual": np.zeros((n, 5, 1024), np.float32),
        "language": np.zeros((n, 384), np.float32),
        "success": np.tile([0, 1, 0, 1, 0], (n, 1)).astype(np.float32),
        "return": np.tile([-1, 2, -2, 3, -3], (n, 1)).astype(np.float32),
        "expert_utility": np.tile([0, -1, -2, -3, -4], (n, 1)).astype(np.float32),
    }


def subset(data, rows):
    return {key: value[rows] for key, value in data.items()}


class SingleItemLanguageEncoder:
    feature_dim = 384

    def __init__(self):
        self.calls = []

    def encode(self, texts, *, batch_size):
        self.calls.append((list(texts), batch_size))
        if len(texts) != 1 or batch_size != 1:
            raise AssertionError("Instruction extraction must use exactly one item")
        vector = np.zeros((1, 384), dtype=np.float32)
        vector[0, 0 if texts[0] == "first" else 1] = 1
        return vector


class LearningContractTests(unittest.TestCase):
    def test_unique_instructions_are_single_encoded_then_broadcast_exactly(self):
        encoder = SingleItemLanguageEncoder()
        cache = build_language_cache(["first", "second", "first", "first"], encoder)
        self.assertEqual(encoder.calls, [(["first"], 1), (["second"], 1)])
        vectors = cached_language_vectors(["first", "first", "second"], cache)
        self.assertTrue(np.array_equal(vectors[0], vectors[1]))
        self.assertEqual(vectors.dtype, np.float32)
        with self.assertRaisesRegex(ValueError, "absent.*generalization"):
            cached_language_vectors(["unseen instruction"], cache)
        changed = copy.deepcopy(cache)
        changed["embeddings"]["first"][1] = .01
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            validate_language_cache(changed)

    def test_train_cache_requires_matching_metadata_and_exact_language_rows(self):
        cache = build_language_cache(["first"], SingleItemLanguageEncoder())
        vector = cached_language_vectors(["first"], cache)
        train, validation = {"language": vector.copy()}, {"language": vector.copy()}
        metadata = [{"g1_language_cache": cache}, {"g1_language_cache": copy.deepcopy(cache)}]
        self.assertEqual(training_language_cache(train, validation, metadata), cache)
        with self.assertRaisesRegex(ValueError, "re-extract"):
            training_language_cache(train, validation, [{}, {}])
        changed = copy.deepcopy(metadata)
        changed[1]["g1_language_cache"] = build_language_cache(["second"], SingleItemLanguageEncoder())
        with self.assertRaisesRegex(ValueError, "caches differ"):
            training_language_cache(train, validation, changed)
        validation["language"][0, 2] = 1e-8
        with self.assertRaisesRegex(ValueError, "exact frozen cache"):
            training_language_cache(train, validation, metadata)

    def test_split_inherits_exact_language_cache_metadata(self):
        cache = build_language_cache(["first"], SingleItemLanguageEncoder())
        data = features()
        data["language"][:] = cached_language_vectors(["first"], cache)[0]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "features.npz"
            np.savez_compressed(source, **data)
            import json
            source.with_suffix(".meta.json").write_text(json.dumps({"g1_language_cache": cache}))
            split_archive(source, directory / "splits")
            for name in ("train", "validation", "test"):
                metadata = json.loads((directory / "splits" / f"{name}.meta.json").read_text())
                self.assertEqual(metadata["g1_language_cache"], cache)

    def test_missing_labels_fail_instead_of_creating_placeholders(self):
        data = features()
        for key in ("return", "expert_utility"):
            malformed = {name: value for name, value in data.items() if name != key}
            with self.assertRaisesRegex(ValueError, key):
                validate_archive(malformed)

    def test_whole_episodes_stay_disjoint_and_split_is_reproducible(self):
        data = features()
        parts = episode_split(data, seed=21)
        repeated = episode_split(data, seed=21)
        identities = {}
        for name, rows in parts.items():
            np.testing.assert_array_equal(rows, repeated[name])
            identities[name] = set(zip(data["task"][rows], data["seed"][rows]))
            self.assertEqual(len(rows), 2 * len(identities[name]))
        self.assertFalse(identities["train"] & identities["validation"])
        self.assertFalse(identities["test"] & identities["validation"])
        self.assertFalse(identities["test"] & identities["train"])
        self.assertEqual(sum(map(len, parts.values())), len(data["task"]))

    def test_config_infers_g1_dimensions_and_rejects_leakage_or_mismatch(self):
        data = features()
        parts = episode_split(data)
        train, validation = subset(data, parts["train"]), subset(data, parts["validation"])
        config = training_config(train, validation)
        self.assertEqual({key: config["model"][key] for key in ("state_dim", "action_dim", "execution_horizon")}, {"state_dim": 6, "action_dim": 7, "execution_horizon": 4})
        config["model"]["state_dim"] = 4
        with self.assertRaisesRegex(ValueError, "Configured state_dim"):
            training_config(train, validation, config)
        with self.assertRaisesRegex(ValueError, "episode overlap"):
            training_config(train, subset(train, np.array([0])))

    def test_candidate_masks_cannot_hide_empty_or_nonprefix_references(self):
        data = features()
        data["mask"][0, 1] = [1, 0, 1, 0]
        with self.assertRaisesRegex(ValueError, "valid prefix"):
            validate_archive(data)
        data["mask"][0, 1] = 0
        with self.assertRaisesRegex(ValueError, "nonempty"):
            validate_archive(data)

    def test_feature_conversion_preserves_true_labels_and_uses_pooling_l2(self):
        class Visual:
            feature_dim = 1024
            def encode_rgb(self, images, *, batch_size):
                return np.full((len(images), 1, 1024), 2, np.float32)
        class Language:
            feature_dim = 384
            def encode(self, texts, *, batch_size):
                return np.ones((len(texts), 384), np.float32) / np.sqrt(384)
        data = features(episodes=1)
        for key in ("visual", "future_visual", "language"):
            data.pop(key)
        n = len(data["task"])
        data.update(rgb=np.zeros((n, 256, 256, 3), np.uint8), future_rgb=np.zeros((n, 5, 256, 256, 3), np.uint8), instruction=np.full(n, "Place the block in the tray."))
        output = encode_arrays(data, Visual(), Language(), batch_size=2)
        for key in ("return", "expert_utility", "success", "actions", "mask"):
            np.testing.assert_array_equal(output[key], data[key])
        np.testing.assert_allclose(np.linalg.norm(output["visual"], axis=-1), 1)
        self.assertEqual(output["future_visual"].shape, (n, 5, 1024))
        self.assertNotIn("instruction", output)

    def test_split_archives_load_with_upstream_and_standard_bundle_roundtrips(self):
        if not os.environ.get("EMBODIEDSKILLS_ROOT") and importlib.util.find_spec("embodiedskills") is None:
            self.skipTest("Set EMBODIEDSKILLS_ROOT to run upstream compatibility checks")
        use_upstream()
        import torch
        from embodiedskills.checkpoint import build_member, cpu_state, save_bundle
        from embodiedskills.continuous_consensus import build_future_consequence_critic
        from embodiedskills.data import ensure_disjoint, load_archive
        from embodiedskills.runtime import Selector
        from embodiedskills.training import fit_statistics

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "source.npz"
            np.savez_compressed(source, **features())
            split_archive(source, directory / "splits")
            train = load_archive(directory / "splits/train.npz", "continuous", training=True)
            validation = load_archive(directory / "splits/validation.npz", "continuous", training=True)
            ensure_disjoint(train, validation)
            config = training_config(train, validation)
            statistics = fit_statistics(train, "continuous")
            members = []
            for _ in range(3):
                model = build_member("continuous", config["model"], statistics)
                members.append({"config": config["model"], "statistics": statistics, "state_dict": cpu_state(model)})
            head = build_future_consequence_critic(effect_rms=np.ones(1024), effect_mean=np.zeros(1024), pca_basis=np.eye(1024, 32))
            bundle = directory / "selector.pt"
            save_bundle(bundle, "continuous", members, head, encoder_config=config["encoders"])
            cache = build_language_cache(["first"], SingleItemLanguageEncoder())
            attach_language_cache(bundle, cache)
            selector = Selector(bundle)
            self.assertEqual(selector.payload["g1_language_cache"], cache)
            result = selector.predict(**{key: validation[key][:1] for key in ("visual", "state", "language", "progress", "actions", "mask")})
            self.assertEqual(len(selector.members), 3)
            self.assertEqual(result["scores"].shape, (1, 5))
            self.assertTrue(np.isfinite(result["scores"]).all())
            self.assertIn(int(result["selected"][0]), range(5))


if __name__ == "__main__":
    unittest.main()
