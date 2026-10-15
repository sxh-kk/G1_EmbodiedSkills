"""Deterministic M1 policies with explicit, restricted selection inputs.

This module deliberately imports neither simulator state nor neural models.
The phase policy's only run-time input is elapsed seconds. Perception and
candidate generation remain unchanged in the common closed-loop executor.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .io import sha256


PHASE_SCHEMA = "g1_embodiedskills.phase_only_policy.v1"
PHASE_BOUNDS = (0., 6., 10., 16., 20., 24.)
SELECTOR_KINDS = ("legacy", "full_d", "full_g", "no_future_g", "sa_critic")


def fixed_candidate(value):
    """Validate a candidate index without silently accepting booleans/floats."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 <= value < 5:
        raise ValueError("fixed_candidate_index must be an integer from 0 through 4")
    return int(value)


def validate_phase_policy(policy):
    if not isinstance(policy, dict) or policy.get("schema") != PHASE_SCHEMA:
        raise ValueError(f"Phase policy must declare schema {PHASE_SCHEMA}")
    bounds = policy.get("bounds_seconds")
    if (not isinstance(bounds, list) or len(bounds) != 6
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bounds)
            or not np.array_equal(np.asarray(bounds, dtype=float), PHASE_BOUNDS)):
        raise ValueError("M1 phase boundaries must be exactly [0,6,10,16,20,24] seconds")
    indices = policy.get("indices")
    if not isinstance(indices, list) or len(indices) != 5:
        raise ValueError("Phase policy must contain exactly five fixed candidate indices")
    for value in indices:
        fixed_candidate(value)
    return policy


def load_phase_policy(path):
    path = Path(path)
    policy = validate_phase_policy(json.loads(path.read_text(encoding="utf-8")))
    identity = {"selector": None, "selector_kind": "phase_only",
        "phase_policy": str(path), "phase_policy_sha256": sha256(path),
        "phase_bounds_seconds": list(PHASE_BOUNDS), "phase_indices": list(policy["indices"]),
        "selection_input_names": ["elapsed_seconds"],
        "selection_rule": "Predeclared elapsed-time lookup only; no learned predictor or state input"}
    return policy, identity


def phase_selection(policy, elapsed_seconds):
    """Return (candidate, phase) using elapsed time as the sole observation."""
    validate_phase_policy(policy)
    if (isinstance(elapsed_seconds, bool) or not isinstance(elapsed_seconds, (int, float, np.number))
            or not np.isfinite(elapsed_seconds) or not 0 <= elapsed_seconds <= 24):
        raise ValueError("Phase selection requires finite elapsed_seconds within [0,24]")
    phase = min(int(np.searchsorted(PHASE_BOUNDS, float(elapsed_seconds), side="right")-1), 4)
    return int(policy["indices"][phase]), phase
