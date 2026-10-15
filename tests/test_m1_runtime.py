"""Restricted-input M1 policy checks; no simulator trials or efficacy claims."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from g1_embodiedskills.io import sha256
from g1_embodiedskills.m1_runtime import (PHASE_SCHEMA, fixed_candidate, load_phase_policy,
                                        phase_selection, validate_phase_policy)


class M1RuntimeContractTests(unittest.TestCase):
    def policy(self):
        return {"schema": PHASE_SCHEMA, "bounds_seconds": [0, 6, 10, 16, 20, 24],
                "indices": [4, 3, 2, 1, 0]}

    def test_phase_boundaries_use_time_only_with_closed_final_interval(self):
        policy = self.policy()
        for elapsed, expected in ((0, (4, 0)), (5.8, (4, 0)), (6, (3, 1)), (9.8, (3, 1)),
                                  (10, (2, 2)), (16, (1, 3)), (20, (0, 4)), (24, (0, 4))):
            self.assertEqual(phase_selection(policy, elapsed), expected)
        for bad in (-.2, 24.2, float("nan"), float("inf"), True, {"elapsed_seconds": 10}):
            with self.assertRaises(ValueError):
                phase_selection(policy, bad)

    def test_fixed_indices_cannot_coerce_invalid_or_noninteger_values(self):
        for value in range(5):
            self.assertEqual(fixed_candidate(np.int64(value)), value)
        for bad in (None, -1, 5, True, 2., "2"):
            with self.assertRaises(ValueError):
                fixed_candidate(bad)

    def test_policy_file_identity_binds_exact_frozen_table(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "phase.json"
            path.write_text(json.dumps(self.policy()), encoding="utf-8")
            policy, identity = load_phase_policy(path)
            self.assertEqual(identity["phase_policy_sha256"], sha256(path))
            self.assertEqual(identity["selection_input_names"], ["elapsed_seconds"])
            self.assertEqual(identity["phase_indices"], policy["indices"])
            self.assertIsNone(identity["selector"])

    def test_cannot_change_predeclared_schedule_or_use_invalid_indices(self):
        for key, value in (("schema", "other"), ("bounds_seconds", [0, 5, 10, 16, 20, 24]),
                           ("indices", [0, 1, 2, 3, 5]), ("indices", [0, 1, 2, 3])):
            policy = self.policy()
            policy[key] = value
            with self.assertRaises(ValueError):
                validate_phase_policy(policy)


if __name__ == "__main__":
    unittest.main()
