"""Vector-first, paired-noise diagnostics for the predeclared M2 factor grids.

This module contains no simulator/model calls. Its unit of resampling is an
original episode, retaining all five anchors and all nine interventions.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np

CELLS = tuple((u, b) for u in range(3) for b in range(3))
CELL_INDEX = {cell: index for index, cell in enumerate(CELLS)}
INTERACTIONS = ((1, 1), (1, 2), (2, 1), (2, 2))
HAND_PAIRS = ((0, 1), (0, 2), (1, 2))
RELATIVE_PAIRS = tuple((left, right, body) for left, right in HAND_PAIRS for body in (1, 2))
KINEMATIC_NULL_VARIANTS = ("unclipped", "joint_clipped")
OFFSETS = (10, 25, 50, 150)
CONDITIONS = ("low", "high", "far", "right")
FAMILIES = {"low": "height", "high": "height", "far": "position", "right": "position"}
ANCHOR_STEPS = (300, 480, 680, 830, 980)
NOISE_SEEDS = {"height": (920000, 920010, 920001, 920011, 920002),
               "position": (920020, 920030, 920021, 920031, 920022)}


def formal_roster():
    return {(condition, 920000+10*condition_index+episode, anchor)
            for condition_index, condition in enumerate(CONDITIONS)
            for episode in range(10) for anchor in ANCHOR_STEPS}


def validate_formal_groups(groups):
    identities = [(group["condition"], int(group["seed"]), int(group["anchor_step"])) for group in groups]
    if len(identities) != len(set(identities)) or set(identities) != formal_roster():
        raise ValueError("Formal report requires the complete frozen 40-episode/200-anchor roster, including rejected grids")
    selected = []
    for group in groups:
        if group["family"] != FAMILIES[group["condition"]]:
            raise ValueError("Scene family does not match the frozen condition")
        expected = int(group["seed"]) == NOISE_SEEDS[group["family"]][ANCHOR_STEPS.index(int(group["anchor_step"]))]
        if bool(group.get("noise_anchor", False)) != expected:
            raise ValueError("Noise anchors differ from the preselected family/phase roster")
        if expected:
            selected.append(group)
        included = [(int(run["repeat"]), int(run["candidate"])) for run in group.get("runs", [])
                    if run.get("analysis_included", True)]
        required = {(repeat, cell) for repeat in range(3 if expected else 1) for cell in range(9)}
        if len(included) != len(set(included)) or set(included) != required:
            raise ValueError("Each formal grid needs all nine first-pass records and every preselected repeat, including physical failures")
    if len(selected) != 10:
        raise ValueError("Exactly ten preselected noise anchors must be retained")


def vector_interactions(grid, valid=None):
    """Compute four contrasts before any norm or threshold operation.

    grid is [9, ...], and optional valid is [9]. Missing one of the four
    necessary cells censors that contrast rather than inventing a zero.
    """
    grid = np.asarray(grid, dtype=float)
    if grid.shape[0] != 9:
        raise ValueError("A factor grid must contain all nine declared cells")
    observed = np.isfinite(grid).reshape(9, -1).all(1)
    if valid is not None:
        supplied = np.asarray(valid, dtype=bool)
        if supplied.shape != (9,):
            raise ValueError("valid must have shape [9]")
        observed &= supplied
    values, masks = [], []
    for u, b in INTERACTIONS:
        indices = [CELL_INDEX[u, b], CELL_INDEX[u, 0], CELL_INDEX[0, b], 0]
        ok = bool(observed[indices].all())
        values.append(grid[indices[0]]-grid[indices[1]]-grid[indices[2]]+grid[0]
                      if ok else np.full(grid.shape[1:], np.nan))
        masks.append(ok)
    return np.asarray(values), np.asarray(masks, dtype=bool)


def centered_repeat_noise(repeated):
    """R=3 sample scatter of a paired scalar/vector contrast, not raw RMS."""
    repeated = np.asarray(repeated, dtype=float)
    if repeated.shape[0] != 3 or not np.isfinite(repeated).all():
        return None
    residual = repeated-repeated.mean(0)
    return float(np.sqrt(np.sum(residual**2)/(len(repeated)-1)))


def relative_pair_interactions(grid, valid=None):
    """Six direct hand-pair/body contrasts, before norms or rank thresholds.

    T(u,v,b)=[Y(u,b)-Y(v,b)]-[Y(u,0)-Y(v,0)]. A contrast requires
    exactly those four cells. In particular (+,-) never depends on (0,0)
    or (0,b), as subtracting two separately masked Gamma values would do.
    The default-hand body-execution gate is a separate measurement and may
    remain unavailable even while T itself is observed.
    """
    grid = np.asarray(grid, dtype=float)
    if grid.shape[0] != 9:
        raise ValueError("A relative-pair factor grid must contain nine cells")
    observed = np.isfinite(grid).reshape(9, -1).all(1)
    if valid is not None:
        supplied = np.asarray(valid, dtype=bool)
        if supplied.shape != (9,):
            raise ValueError("valid must have shape [9]")
        observed &= supplied
    values, masks = [], []
    for left, right, body in RELATIVE_PAIRS:
        indices = [CELL_INDEX[left, body], CELL_INDEX[right, body],
                   CELL_INDEX[left, 0], CELL_INDEX[right, 0]]
        ok = bool(observed[indices].all())
        values.append((grid[indices[0]]-grid[indices[1]])-
                      (grid[indices[2]]-grid[indices[3]]) if ok
                      else np.full(grid.shape[1:], np.nan))
        masks.append(ok)
    return np.asarray(values), np.asarray(masks, dtype=bool)


def rank_contrasts(grid):
    """Pairwise hand differences at each body reference; higher is better."""
    grid = np.asarray(grid, dtype=float)
    if grid.shape != (9,):
        raise ValueError("Expected nine scalar utilities")
    pairs = ((0, 1), (0, 2), (1, 2))
    return {(left, right, body): float(grid[CELL_INDEX[left, body]]-grid[CELL_INDEX[right, body]])
            for left, right in pairs for body in range(3)}


def robust_rank_flips(utilities, repeated_reference, floor=1e-8):
    """Require opposite resolved differences and a resolved difference-change.

    Every noise estimate uses a within-repeat paired contrast from the same
    grid. A missing/noisy reference is not treated as evidence of no noise.
    """
    point = rank_contrasts(utilities)
    reference = np.asarray(repeated_reference, dtype=float)
    if reference.shape != (3, 9):
        raise ValueError("A rank-noise reference must contain three nine-cell repeats")
    repeated = [rank_contrasts(row) for row in reference]
    rows = []
    for left, right in ((0, 1), (0, 2), (1, 2)):
        for body_a, body_b in ((0, 1), (0, 2), (1, 2)):
            key_a, key_b = (left, right, body_a), (left, right, body_b)
            da, db = point[key_a], point[key_b]
            na = centered_repeat_noise([row[key_a] for row in repeated])
            nb = centered_repeat_noise([row[key_b] for row in repeated])
            nc = centered_repeat_noise([row[key_a]-row[key_b] for row in repeated])
            finite = bool(np.isfinite([da, db]).all())
            reference_ok = na is not None and nb is not None and nc is not None
            raw = bool(finite and da*db < 0)
            resolved = bool(raw and reference_ok
                and abs(da) > max(3*na, floor) and abs(db) > max(3*nb, floor)
                and abs(da-db) > max(3*nc, floor))
            rows.append({"hand_pair": [left, right], "body_pair": [body_a, body_b],
                "differences": [da if finite else None, db if finite else None],
                "noise": [na, nb], "difference_change_noise": nc,
                "raw_flip": raw, "observed": finite, "noise_available": reference_ok,
                "scope": "primary_body_comparison" if body_a == 0 else "exploratory_plus_vs_minus",
                "flip_margin": min(abs(da), abs(db)) if raw else 0. if finite else None,
                "resolved_flip": resolved})
    return rows


def stratified_episode_bootstrap(records, value_key, *, repetitions=10000, seed=20261015):
    """Episode-cluster bootstrap, stratified by the four scene conditions.

    A row is an original episode summary, not an anchor/branch. Missing values
    stay missing; each draw retains the same condition-specific episode count.
    """
    groups = defaultdict(list)
    seen = set()
    for row in records:
        key = (row["condition"], int(row["seed"]))
        if key in seen:
            raise ValueError("Bootstrap input must have one row per original episode")
        seen.add(key)
        groups[row["condition"]].append(row.get(value_key))
    if not groups:
        return {"mean": None, "ci95": None, "episodes": 0, "observed_episodes": 0}
    arrays = [np.asarray([np.nan if value is None else value for value in groups[key]], float)
              for key in sorted(groups)]
    flat = np.concatenate(arrays)
    finite = np.isfinite(flat)
    result = {"mean": float(flat[finite].mean()) if finite.any() else None,
              "episodes": len(flat), "observed_episodes": int(finite.sum()),
              "repetitions": repetitions, "seed": seed,
              "unit": "original episode, stratified by scene condition"}
    if not finite.any():
        return {**result, "ci95": None}
    rng = np.random.default_rng(seed)
    draws = np.concatenate([array[rng.integers(len(array), size=(repetitions, len(array)))]
                            for array in arrays], axis=1)
    count = np.isfinite(draws).sum(1)
    means = np.divide(np.nansum(draws, axis=1), count,
                      out=np.full(repetitions, np.nan), where=count > 0)
    valid = means[np.isfinite(means)]
    return {**result, "ci95": np.quantile(valid, [.025, .975]).tolist() if len(valid) else None}


def _finite_list(array):
    array = np.asarray(array, float)
    return array.tolist() if np.isfinite(array).all() else None


def _distribution(values):
    array = np.asarray([value for value in values if value is not None], dtype=float)
    array = array[np.isfinite(array)]
    return {"n": len(array), "mean": float(array.mean()) if len(array) else None,
            "quantiles_0_25_50_75_100": np.quantile(array, [0, .25, .5, .75, 1]).tolist()
            if len(array) else None}


def _resolve(base, value):
    path = Path(value)
    return path if path.is_absolute() else base/path


def _read_npz(path, expected=None, audited=None):
    """Hash the exact bytes used in the calculation, closing the audit/read gap."""
    path = Path(path).resolve()
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f"Analysis input differs from collection-time SHA256: {path}")
    if audited is not None and (expected is None or audited.get(str(path)) != digest):
        raise ValueError(f"Analysis input is not covered by the passed independent audit: {path}")
    with np.load(BytesIO(data), allow_pickle=False) as raw:
        return {key: np.asarray(raw[key]) for key in raw.files}


def load_groups(manifests):
    """Load every declared physical group and preserve rejected coverage rows."""
    groups = []
    for manifest_path in manifests:
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != "g1_embodiedskills.m2_collection.v1":
            raise ValueError(f"Unexpected collection schema: {manifest_path}")
        if manifest.get("completed") is not True:
            raise ValueError(f"Cannot analyze an interrupted/incomplete collection as completed: {manifest_path}")
        for group in manifest["groups"]:
            group = dict(group)
            group["_base"] = manifest_path.parent
            group["_simulator_joint_names"] = manifest.get("simulator_joint_names")
            group["_mjcf_joint_names"] = manifest.get("mjcf_joint_names")
            groups.append(group)
    return groups


def _run_arrays(group):
    samples, runs = {}, {}
    for run in group.get("runs", []):
        if not run.get("analysis_included", True):
            continue
        candidate, repeat = int(run["candidate"]), int(run["repeat"])
        key = (repeat, candidate)
        if key in runs:
            raise ValueError(f"Duplicate repeat/cell in {group['id']}")
        if not 0 <= candidate < 9 or repeat not in (0, 1, 2):
            raise ValueError("Invalid declared repeat/cell")
        run = {**run, "valid": bool(group.get("accepted") and run.get("valid"))}
        runs[key] = run
        if group.get("accepted") and run.get("valid"):
            sample = _read_npz(_resolve(group["_base"], run["samples_archive"]),
                               run.get("samples_sha256"), group.get("_audited_evidence"))
            if tuple(sample["offset_steps"]) != OFFSETS:
                raise ValueError("Sample times differ from the frozen M2 protocol")
            anchor_waist = run.get("anchor_waist_rad")
            sample["waist_change"] = sample["waist"]-float(anchor_waist) if anchor_waist is not None else np.full(4, np.nan)
            samples[key] = sample
    return runs, samples


def _grid(samples, repeat, key, index):
    shape = {"actual_tcp": (3,), "nominal_tcp": (3,), "actual_R": (3, 3), "nominal_R": (3, 3), "waist": (), "waist_change": (), "q": (31,),
             "cube_root": (13,), "root": (13,), "soles": (2, 3)}[key]
    values = np.full((9, *shape), np.nan)
    valid = np.zeros(9, dtype=bool)
    for cell in range(9):
        sample = samples.get((repeat, cell))
        if sample is not None and bool(sample["valid"][index]):
            values[cell] = sample[key][index]
            valid[cell] = bool(np.isfinite(values[cell]).all())
    return values, valid


def _kinematic_null_results(group, samples, model):
    """One official-FK oracle per measured repeat at the fixed +.5 s clock.

    These masks are independent of the original TCP Gamma/T diagnostics.
    Canonical reference bytes and observed q/root sample bytes retain their
    collection-time and passed-audit SHA bindings.
    """
    if model is None or not group.get("accepted"):
        return {}
    from .m2_kinematic_null import (kinematic_additive_error_null,
                                    root_rotations_from_wxyz, simulator_to_mjcf)

    if group.get("_mjcf_joint_names") != list(model.joint_names):
        raise ValueError("Oracle model/body joint names differ from the physical manifest")
    simulator_names = group.get("_simulator_joint_names")
    if not isinstance(simulator_names, list):
        raise ValueError("Oracle null requires the manifest simulator joint inventory")
    canonical = _read_npz(_resolve(group["_base"], group["canonical_archive"]),
                         group.get("canonical_sha256"), group.get("_audited_evidence"))
    for key, shape in (("qtraj_mjcf", (9, 105, 29)), ("roots", (9, 105, 3)),
                       ("root_quaternion_wxyz", (9, 105, 4)), ("nominal_tcp", (9, 105, 3))):
        if key not in canonical or canonical[key].shape != shape or not np.isfinite(canonical[key]).all():
            raise ValueError(f"Oracle canonical {key} must be finite {shape}")
    nominal_q = canonical["qtraj_mjcf"][:, 25]
    nominal_roots = canonical["roots"][:, 25]
    nominal_rotations = root_rotations_from_wxyz(canonical["root_quaternion_wxyz"][:, 25])
    results = {}
    for repeat in sorted({repeat for repeat, _ in samples}):
        q31, qv = _grid(samples, repeat, "q", 1)
        roots, rv = _grid(samples, repeat, "root", 1)
        actual, av = _grid(samples, repeat, "actual_tcp", 1)
        result = kinematic_additive_error_null(model,
            actual_q=simulator_to_mjcf(q31, simulator_names, model.joint_names),
            actual_roots=roots[:, :3], actual_root_rotations=root_rotations_from_wxyz(roots[:, 3:7]),
            nominal_q=nominal_q, nominal_roots=nominal_roots,
            nominal_root_rotations=nominal_rotations, actual_tcp=actual,
            valid=qv & rv & av)
        observed = result["cell_valid"]
        if observed.any():
            nominal_error = np.abs(result["nominal_tcp"][observed]-canonical["nominal_tcp"][observed, 25]).max()
            physical_error = np.linalg.norm(result["actual_tcp_minus_fk_m"][observed], axis=-1).max()
            if nominal_error > 1e-7 or physical_error > 1e-4:
                raise ValueError("Oracle FK disagrees with the independently audited canonical/physical geometry")
        results[repeat] = result
    return results


def _diagnostic_grids(group, samples, repeat, time_index):
    """World-frame vectors and kinematic relative motion, before taking norms."""
    from scipy.spatial.transform import Rotation

    actual, av = _grid(samples, repeat, "actual_tcp", time_index)
    nominal, nv = _grid(samples, repeat, "nominal_tcp", time_index)
    ar, arv = _grid(samples, repeat, "actual_R", time_index)
    nr, nrv = _grid(samples, repeat, "nominal_R", time_index)
    cube, cv = _grid(samples, repeat, "cube_root", time_index)
    roots, rv = _grid(samples, repeat, "root", time_index)
    soles, sv = _grid(samples, repeat, "soles", time_index)
    waist, wv = _grid(samples, repeat, "waist", time_index)
    orientation = np.full((9, 3), np.nan)
    rotation_valid = arv & nrv
    if rotation_valid.any():
        matrices = ar[rotation_valid] @ np.swapaxes(nr[rotation_valid], -1, -2)
        orientation[rotation_valid] = Rotation.from_matrix(matrices).as_rotvec()
    result = {"tcp_actual_m": (actual, av), "tcp_nominal_m": (nominal, nv),
        "tcp_tracking_m": (actual-nominal, av & nv),
        "orientation_tracking_world_rad": (orientation, rotation_valid),
        "cube_position_m": (cube[:, :3], cv), "cube_velocity_mps": (cube[:, 7:10], cv),
        "root_velocity_mps": (roots[:, 7:10], rv), "root_angular_velocity_radps": (roots[:, 10:13], rv),
        "sole_position_m": (soles.reshape(9, 6), sv), "waist_rad": (waist[:, None], wv)}
    anchor_path = group.get("anchor_sample_archive")
    if anchor_path:
        anchor = _read_npz(_resolve(group["_base"], anchor_path),
                           group.get("anchor_sample_sha256"), group.get("_audited_evidence"))
        preheld = bool(float(anchor["prefix_max_cube_z"])-float(anchor["prefix_cube_initial_z"]) >= .03
            and np.max(anchor["gripper"]) <= .01
            and np.linalg.norm(anchor["cube_root"][:3]-anchor["actual_tcp"]) <= .06)
        if preheld:
            base_relative = anchor["actual_R"].T @ (anchor["cube_root"][:3]-anchor["actual_tcp"])
            relative = np.einsum("nji,nj->ni", ar, cube[:, :3]-actual)-base_relative
            result["preheld_cube_relative_tcp_change_m"] = (relative, cv & av & arv)
        result["sole_displacement_m"] = ((soles-anchor["soles"]).reshape(9, 6), sv)
        # Root roll/pitch magnitude is a supplemental scalar diagnostic; its
        # nonlinearity is never used as evidence of mechanical interaction.
        tilt = 2*np.arccos(np.clip(np.sqrt(roots[:, 3]**2+roots[:, 6]**2), 0, 1))
        result["root_tilt_scalar_rad"] = (tilt[:, None], rv)
    return result


def _utility(runs, repeat, metric):
    result = np.full(9, np.nan)
    for cell in range(9):
        run = runs.get((repeat, cell))
        if not run or not run.get("valid"):
            continue
        if metric == "distance":
            if run.get("d24_valid"):
                result[cell] = -float(run["outcome"]["final_target_distance_m"])
        elif metric in ("return", "dense_return"):
            result[cell] = float(run["outcome"]["return_value"])
            if metric == "dense_return":
                outcome = run["outcome"]
                result[cell] += -4*bool(outcome["success"])-bool(outcome["lifted"])+2*(not bool(outcome["standing"]))
        else:
            raise ValueError(metric)
    return result


def summarize_m2(groups, *, formal=False, model=None):
    """Summarize first passes with family/phase-matched repeat-grid noise."""
    if formal:
        validate_formal_groups(groups)
        if model is None:
            raise ValueError("Formal complete M2 analysis requires the official CPU FK oracle model")
    by_identity = {}
    for group in groups:
        identity = (group["condition"], int(group["seed"]), int(group["anchor_step"]))
        if identity in by_identity:
            raise ValueError("Merge collection repeats into one group before analysis")
        by_identity[identity] = (group, *_run_arrays(group))
    null_results = {identity: _kinematic_null_results(group, samples, model)
                    for identity, (group, _, samples) in by_identity.items()}
    noise, rank_noise, supplemental_noise, body_noise = {}, {}, {}, {}
    relative_pair_noise = {}
    null_noise = {}
    for identity, (group, runs, samples) in by_identity.items():
        if not group.get("noise_anchor", False):
            continue
        family, anchor = group["family"], int(group["anchor_step"])
        if not group.get("accepted"):
            runs = {}
        for t in range(len(OFFSETS)):
            contrast_repeats, waist_repeats = [], []
            diagnostic_repeats = defaultdict(list)
            for repeat in range(3):
                actual, av = _grid(samples, repeat, "actual_tcp", t)
                nominal, nv = _grid(samples, repeat, "nominal_tcp", t)
                contrast, _ = vector_interactions(actual-nominal, av & nv)
                contrast_repeats.append(contrast)
                waist, wv = _grid(samples, repeat, "waist_change", t)
                waist_repeats.append([waist[CELL_INDEX[u, b]]-waist[CELL_INDEX[u, 0]]
                                      for u, b in INTERACTIONS])
                for metric, (grid, observed) in _diagnostic_grids(group, samples, repeat, t).items():
                    diagnostic_repeats[metric].append(vector_interactions(grid, observed)[0])
            for combo in range(4):
                key = (family, anchor, t, combo)
                if key in noise:
                    raise ValueError("Exactly one preselected noise anchor per family/phase is required")
                noise[key] = {"source_seed": group["seed"], "source_condition": group["condition"],
                    "source_effective_amplitude_m": group.get("candidate_metadata", {}).get("effective_amplitude_m"),
                    "source_effective_body_pitch_rad": group.get("candidate_metadata", {}).get("effective_body_pitch_rad"),
                    "gamma_m": centered_repeat_noise(np.asarray(contrast_repeats)[:, combo]),
                    "waist_rad": centered_repeat_noise(np.asarray(waist_repeats)[:, combo])}
                for metric, repeated in diagnostic_repeats.items():
                    supplemental_noise[family, anchor, t, metric, combo] = centered_repeat_noise(np.asarray(repeated)[:, combo])
            for body in (1, 2):
                repeated = []
                for repeat in range(3):
                    waist, _ = _grid(samples, repeat, "waist_change", t)
                    repeated.append(waist[CELL_INDEX[0, body]]-waist[0])
                body_noise[family, anchor, t, body] = centered_repeat_noise(repeated)
        for metric in ("distance", "return", "dense_return"):
            rank_noise[family, anchor, metric] = np.stack([_utility(runs, repeat, metric) for repeat in range(3)])
        # Task-association contrasts supplement the unchanged four-Gamma
        # primary diagnostic, at the predeclared +.5 s clock only.
        pair_repeats = []
        for repeat in range(3):
            actual, av = _grid(samples, repeat, "actual_tcp", 1)
            nominal, nv = _grid(samples, repeat, "nominal_tcp", 1)
            pair_repeats.append(relative_pair_interactions(actual-nominal, av & nv)[0])
        for pair in range(len(RELATIVE_PAIRS)):
            key = (family, anchor, pair)
            if key in relative_pair_noise:
                raise ValueError("Exactly one relative-pair noise anchor per family/phase is required")
            relative_pair_noise[key] = {
                "source_seed": group["seed"], "source_condition": group["condition"],
                "source_effective_amplitude_m": group.get("candidate_metadata", {}).get("effective_amplitude_m"),
                "source_effective_body_pitch_rad": group.get("candidate_metadata", {}).get("effective_body_pitch_rad"),
                "contrast_m": centered_repeat_noise(np.asarray(pair_repeats)[:, pair])}
        for variant in KINEMATIC_NULL_VARIANTS:
            repeated = [null_results[identity][repeat][variant]["gamma_excess"]
                        if repeat in null_results[identity] else np.full((4, 3), np.nan)
                        for repeat in range(3)]
            for combo in range(4):
                null_noise[family, anchor, variant, combo] = {
                    "source_seed": group["seed"], "source_condition": group["condition"],
                    "source_effective_amplitude_m": group.get("candidate_metadata", {}).get("effective_amplitude_m"),
                    "source_effective_body_pitch_rad": group.get("candidate_metadata", {}).get("effective_body_pitch_rad"),
                    "excess_m": centered_repeat_noise(np.asarray(repeated)[:, combo])}

    anchor_rows = []
    for identity, (group, runs, samples) in sorted(by_identity.items()):
        family, anchor = group["family"], int(group["anchor_step"])
        row = {key: group[key] for key in ("id", "condition", "family", "seed", "anchor_step")}
        metadata = group.get("candidate_metadata", {})
        row["factor_amplitudes"] = {key: metadata.get(key) for key in ("requested_amplitude_m", "requested_body_pitch_rad",
            "effective_amplitude_m", "effective_body_pitch_rad", "common_scale")}
        row.update(accepted=bool(group.get("accepted")), samples=[], ranks={}, diagnostics=[], body_response=[])
        for t, offset in enumerate(OFFSETS):
            actual, av = _grid(samples, 0, "actual_tcp", t)
            nominal, nv = _grid(samples, 0, "nominal_tcp", t)
            actual_gamma, valid = vector_interactions(actual, av & nv)
            nominal_gamma, _ = vector_interactions(nominal, av & nv)
            residual_gamma = actual_gamma-nominal_gamma
            waist, wv = _grid(samples, 0, "waist_change", t)
            bodies = []
            for body, sign in ((1, 1), (2, -1)):
                delta = float(waist[CELL_INDEX[0, body]]-waist[0]) if wv[CELL_INDEX[0, body]] and wv[0] else None
                nbody = body_noise.get((family, anchor, t, body))
                bodies.append({"b_index": body, "observed": delta is not None,
                    "waist_change_contrast_rad": delta, "noise_rad": nbody,
                    "gain_at_peak": sign*delta/metadata["effective_body_pitch_rad"]
                        if offset == 25 and delta is not None and metadata.get("effective_body_pitch_rad", 0) > 0 else None,
                    "direction_correct_above_noise": bool(delta is not None and nbody is not None
                        and sign*delta > max(3*nbody, 1e-4))})
            row["body_response"].append({"offset_steps": offset, "u_index": 0, "bodies": bodies})
            rows = []
            for combo, (u, b) in enumerate(INTERACTIONS):
                reference = noise.get((family, anchor, t, combo), {})
                n, nw = reference.get("gamma_m"), reference.get("waist_rad")
                magnitude = float(np.linalg.norm(residual_gamma[combo])) if valid[combo] else None
                body_observed = bool(wv[CELL_INDEX[u, b]] and wv[CELL_INDEX[u, 0]])
                body_change = float(waist[CELL_INDEX[u, b]]-waist[CELL_INDEX[u, 0]]) if body_observed else None
                threshold = max(3*n, .002) if n is not None else None
                same_scale = bool(metadata.get("effective_amplitude_m") is not None
                    and metadata.get("effective_body_pitch_rad") is not None
                    and metadata["effective_amplitude_m"] == reference.get("source_effective_amplitude_m")
                    and metadata["effective_body_pitch_rad"] == reference.get("source_effective_body_pitch_rad"))
                body_gate = bodies[b-1]["direction_correct_above_noise"]
                rows.append({"u_index": u, "b_index": b, "observed": bool(valid[combo]),
                    "actual_gamma_m": _finite_list(actual_gamma[combo]),
                    "nominal_gamma_m": _finite_list(nominal_gamma[combo]),
                    "residual_gamma_m": _finite_list(residual_gamma[combo]),
                    "residual_norm_m": magnitude, "noise_m": n, "engineering_threshold_m": threshold,
                    "above_threshold": bool(magnitude is not None and threshold is not None and magnitude > threshold),
                    "body_change_rad": body_change, "body_noise_rad": nw,
                    "body_executed_above_noise": body_gate,
                    "default_hand_body_response": bodies[b-1], "noise_actual_amplitudes_match": same_scale,
                    "noise_reference": {key: reference[key] for key in ("source_seed", "source_condition", "source_effective_amplitude_m",
                                                                       "source_effective_body_pitch_rad") if key in reference}})
            row["samples"].append({"offset_steps": offset, "seconds": offset*.02, "interactions": rows})
            diagnostics = {}
            for metric, (grid, observed) in _diagnostic_grids(group, samples, 0, t).items():
                contrasts, mask = vector_interactions(grid, observed)
                diagnostics[metric] = [{"u_index": u, "b_index": b, "observed": bool(mask[combo]),
                    "gamma": _finite_list(contrasts[combo]),
                    "gamma_norm": float(np.linalg.norm(contrasts[combo])) if mask[combo] else None,
                    "centered_repeat_noise": supplemental_noise.get((family, anchor, t, metric, combo))}
                    for combo, (u, b) in enumerate(INTERACTIONS)]
            row["diagnostics"].append({"offset_steps": offset, "metrics": diagnostics})
        actual, av = _grid(samples, 0, "actual_tcp", 1)
        nominal, nv = _grid(samples, 0, "nominal_tcp", 1)
        pair_contrasts, pair_valid = relative_pair_interactions(actual-nominal, av & nv)
        pair_rows = []
        for pair, (left, right, body) in enumerate(RELATIVE_PAIRS):
            reference = relative_pair_noise.get((family, anchor, pair), {})
            n = reference.get("contrast_m")
            magnitude = float(np.linalg.norm(pair_contrasts[pair])) if pair_valid[pair] else None
            threshold = max(3*n, .002) if n is not None else None
            same_scale = bool(metadata.get("effective_amplitude_m") is not None
                and metadata.get("effective_body_pitch_rad") is not None
                and metadata["effective_amplitude_m"] == reference.get("source_effective_amplitude_m")
                and metadata["effective_body_pitch_rad"] == reference.get("source_effective_body_pitch_rad"))
            body_response = row["body_response"][1]["bodies"][body-1]
            pair_rows.append({"hand_pair": [left, right], "body_pair": [0, body],
                "offset_steps": 25, "seconds": .5,
                "observed": bool(pair_valid[pair]),
                "residual_contrast_m": _finite_list(pair_contrasts[pair]),
                "residual_norm_m": magnitude, "noise_m": n,
                "engineering_threshold_m": threshold,
                "above_threshold": bool(magnitude is not None and threshold is not None and magnitude > threshold),
                "body_executed_above_noise": body_response["direction_correct_above_noise"],
                "default_hand_body_response": body_response,
                "noise_actual_amplitudes_match": same_scale,
                "noise_reference": {key: reference[key] for key in ("source_seed", "source_condition",
                    "source_effective_amplitude_m", "source_effective_body_pitch_rad") if key in reference},
                "scope": "supplemental_task_association; original four-Gamma primary unchanged"})
        row["relative_pair_interactions"] = pair_rows
        row["relative_pair_observed_combinations"] = sum(entry["observed"] for entry in pair_rows)
        row["relative_pair_gate"] = any(entry["above_threshold"] and entry["body_executed_above_noise"] for entry in pair_rows)
        for metric in ("distance", "return", "dense_return"):
            reference = rank_noise.get((family, anchor, metric), np.full((3, 9), np.nan))
            row["ranks"][metric] = robust_rank_flips(_utility(runs, 0, metric), reference,
                                                     floor=1e-6 if metric == "distance" else 1e-8)
        primary = row["samples"][1]["interactions"]
        row["primary_observed_combinations"] = sum(value["observed"] for value in primary)
        row["primary_gate"] = any(value["above_threshold"] and value["body_executed_above_noise"] for value in primary)
        for metric in ("distance", "return", "dense_return"):
            row[f"resolved_{metric}_rank_flip"] = any(value["resolved_flip"] and value["scope"] == "primary_body_comparison"
                for value in row["ranks"][metric])
            matched = []
            for entry in primary:
                matching = next(value for value in row["ranks"][metric]
                    if value["hand_pair"] == [0, entry["u_index"]] and value["body_pair"] == [0, entry["b_index"]])
                matched.append({"u_index": entry["u_index"], "b_index": entry["b_index"],
                    "primary_observed": entry["observed"], "rank_observed": matching["observed"],
                    "noise_actual_amplitudes_match": entry["noise_actual_amplitudes_match"],
                    "resolved_joint_evidence": bool(entry["above_threshold"] and entry["body_executed_above_noise"]
                                                     and matching["resolved_flip"])})
            row[f"matched_{metric}_interaction_rank"] = matched
            row[f"matched_{metric}_joint_gate"] = any(value["resolved_joint_evidence"] for value in matched)
            row[f"matched_{metric}_joint_gate_same_amplitudes"] = any(value["resolved_joint_evidence"]
                and value["noise_actual_amplitudes_match"] for value in matched)
            relative_matched = []
            for entry in pair_rows:
                matching = next(value for value in row["ranks"][metric]
                    if value["hand_pair"] == entry["hand_pair"] and value["body_pair"] == entry["body_pair"])
                relative_matched.append({"hand_pair": entry["hand_pair"], "body_pair": entry["body_pair"],
                    "contrast_observed": entry["observed"], "rank_observed": matching["observed"],
                    "contrast_noise_available": entry["noise_m"] is not None,
                    "rank_noise_available": matching["noise_available"],
                    "body_observed": entry["default_hand_body_response"]["observed"],
                    "noise_actual_amplitudes_match": entry["noise_actual_amplitudes_match"],
                    "resolved_joint_evidence": bool(entry["above_threshold"] and entry["body_executed_above_noise"]
                                                     and matching["resolved_flip"])})
            row[f"relative_pair_{metric}_interaction_rank"] = relative_matched
            row[f"relative_pair_{metric}_joint_gate"] = any(value["resolved_joint_evidence"] for value in relative_matched)
            row[f"relative_pair_{metric}_joint_gate_same_amplitudes"] = any(value["resolved_joint_evidence"]
                and value["noise_actual_amplitudes_match"] for value in relative_matched)
        # Explicit convenience alias; the three metric-specific gates above
        # remain available, with no selection of a best-looking endpoint.
        row["relative_pair_joint_gate"] = row["relative_pair_distance_joint_gate"]
        row["relative_pair_joint_gate_same_amplitudes"] = row["relative_pair_distance_joint_gate_same_amplitudes"]
        result = null_results[identity].get(0)
        row["kinematic_oracle_null"] = {}
        fk_error = result["actual_tcp_minus_fk_m"] if result is not None else np.full((9, 3), np.nan)
        fk_observed = np.isfinite(fk_error).all(1)
        row["kinematic_oracle_null_actual_fk_max_error_m"] = (
            float(np.linalg.norm(fk_error[fk_observed], axis=-1).max()) if fk_observed.any() else None)
        for variant in KINEMATIC_NULL_VARIANTS:
            null_rows = []
            for combo, (u, b) in enumerate(INTERACTIONS):
                observed = bool(result is not None and result[variant]["observed"][combo])
                excess = result[variant]["gamma_excess"][combo] if observed else np.full(3, np.nan)
                reference = null_noise.get((family, anchor, variant, combo), {})
                n = reference.get("excess_m")
                magnitude = float(np.linalg.norm(excess)) if observed else None
                threshold = max(3*n, .002) if n is not None else None
                same_scale = bool(metadata.get("effective_amplitude_m") is not None
                    and metadata.get("effective_body_pitch_rad") is not None
                    and metadata["effective_amplitude_m"] == reference.get("source_effective_amplitude_m")
                    and metadata["effective_body_pitch_rad"] == reference.get("source_effective_body_pitch_rad"))
                body_response = row["body_response"][1]["bodies"][b-1]
                null_rows.append({"u_index": u, "b_index": b, "observed": observed,
                    "gamma_null_m": _finite_list(result[variant]["gamma_null"][combo]) if observed else None,
                    "gamma_tracking_null_m": _finite_list(result[variant]["gamma_tracking_null"][combo]) if observed else None,
                    "gamma_excess_m": _finite_list(excess), "excess_norm_m": magnitude,
                    "noise_m": n, "engineering_threshold_m": threshold,
                    "above_threshold": bool(magnitude is not None and threshold is not None and magnitude > threshold),
                    "body_executed_above_noise": body_response["direction_correct_above_noise"],
                    "noise_actual_amplitudes_match": same_scale,
                    "noise_reference": {key: reference[key] for key in ("source_seed", "source_condition",
                        "source_effective_amplitude_m", "source_effective_body_pitch_rad") if key in reference},
                    "unclipped_joint_limits_exceeded": bool(result["null_joint_limit_exceeded"][combo]) if observed else None,
                    "unclipped_max_joint_limit_violation_rad": float(result["null_max_joint_limit_violation_rad"][combo]) if observed else None})
            row["kinematic_oracle_null"][variant] = {"interactions": null_rows,
                "observed_combinations": sum(entry["observed"] for entry in null_rows),
                "gate": any(entry["above_threshold"] and entry["body_executed_above_noise"] for entry in null_rows),
                "gate_same_amplitudes": any(entry["above_threshold"] and entry["body_executed_above_noise"]
                    and entry["noise_actual_amplitudes_match"] for entry in null_rows)}
        first_runs = [runs.get((0, index)) for index in range(9)]
        row["successful_candidate_count"] = sum(bool(run and run.get("valid") and run["outcome"]["success"]) for run in first_runs)
        row["default_body_success_coverage"] = any(first_runs[index] and first_runs[index].get("valid")
            and first_runs[index]["outcome"]["success"] for index in (0, 3, 6))
        row["all_body_success_coverage"] = row["successful_candidate_count"] > 0
        row["first_pass_observed_branches"] = sum(bool(run and run.get("valid")) for run in first_runs)
        row["first_pass_terminations"] = sum(bool(run and run.get("first_termination_step") is not None) for run in first_runs)
        anchor_rows.append(row)

    episodes = defaultdict(list)
    for row in anchor_rows:
        episodes[row["condition"], row["seed"]].append(row)
    episode_rows = []
    for (condition, seed), anchors in sorted(episodes.items()):
        magnitudes = [entry["residual_norm_m"] for row in anchors
                      for entry in row["samples"][1]["interactions"] if entry["observed"]]
        pair_magnitudes = [entry["residual_norm_m"] for row in anchors
                           for entry in row["relative_pair_interactions"] if entry["observed"]]
        episode = {"condition": condition, "family": FAMILIES[condition], "seed": seed,
            "planned_anchors": 5, "recorded_anchors": len(anchors),
            "primary_mean_residual_norm_m": float(np.mean(magnitudes)) if magnitudes else None,
            "primary_observed_fraction": sum(row["primary_observed_combinations"] for row in anchors)/20.,
            "primary_gate_anchor_fraction": sum(row["primary_gate"] for row in anchors)/5.,
            "distance_flip_anchor_fraction": sum(row["resolved_distance_rank_flip"] for row in anchors)/5.,
            "return_flip_anchor_fraction": sum(row["resolved_return_rank_flip"] for row in anchors)/5.,
            "matched_distance_joint_anchor_fraction": sum(row["matched_distance_joint_gate"] for row in anchors)/5.,
            "successful_grid_anchor_fraction": sum(row["all_body_success_coverage"] for row in anchors)/5.,
            "relative_pair_mean_residual_norm_m": float(np.mean(pair_magnitudes)) if pair_magnitudes else None,
            "relative_pair_observed_fraction": sum(row["relative_pair_observed_combinations"] for row in anchors)/30.,
            "relative_pair_gate_anchor_fraction": sum(row["relative_pair_gate"] for row in anchors)/5.}
        for metric in ("distance", "return", "dense_return"):
            episode[f"relative_pair_{metric}_joint_anchor_fraction"] = sum(row[f"relative_pair_{metric}_joint_gate"] for row in anchors)/5.
            episode[f"relative_pair_{metric}_joint_same_amplitudes_anchor_fraction"] = sum(row[f"relative_pair_{metric}_joint_gate_same_amplitudes"] for row in anchors)/5.
        for variant in KINEMATIC_NULL_VARIANTS:
            excesses = [entry["excess_norm_m"] for row in anchors
                        for entry in row["kinematic_oracle_null"][variant]["interactions"] if entry["observed"]]
            episode[f"kinematic_null_{variant}_mean_excess_norm_m"] = float(np.mean(excesses)) if excesses else None
            episode[f"kinematic_null_{variant}_observed_fraction"] = sum(row["kinematic_oracle_null"][variant]["observed_combinations"] for row in anchors)/20.
            episode[f"kinematic_null_{variant}_gate_anchor_fraction"] = sum(row["kinematic_oracle_null"][variant]["gate"] for row in anchors)/5.
            episode[f"kinematic_null_{variant}_gate_same_amplitudes_anchor_fraction"] = sum(row["kinematic_oracle_null"][variant]["gate_same_amplitudes"] for row in anchors)/5.
        episode_rows.append(episode)
    summaries = {}
    for condition in (*CONDITIONS, "height", "position", "all"):
        subset = [row for row in anchor_rows if condition == "all" or row["condition"] == condition or row["family"] == condition]
        selected_episodes = [row for row in episode_rows if condition == "all" or row["condition"] == condition or row["family"] == condition]
        primary = [entry for row in subset for entry in row["samples"][1]["interactions"]]
        relative = [entry for row in subset for entry in row["relative_pair_interactions"]]
        planned_anchors = {**dict.fromkeys(CONDITIONS, 50), "height": 100, "position": 100, "all": 200}[condition]
        summaries[condition] = {"planned_anchors": planned_anchors, "recorded_anchors": len(subset),
            "accepted_grids": sum(row["accepted"] for row in subset),
            "primary_planned_contrasts": planned_anchors*4,
            "primary_observed_contrasts": sum(value["observed"] for value in primary),
            "primary_above_noise_and_floor_contrasts": sum(value["above_threshold"] for value in primary),
            "primary_gate_episodes": len({row["seed"] for row in subset if row["primary_gate"]}),
            "distance_flip_episodes": len({row["seed"] for row in subset if row["resolved_distance_rank_flip"]}),
            "return_flip_episodes": len({row["seed"] for row in subset if row["resolved_return_rank_flip"]}),
            "cooccurring_primary_and_distance_flip_episodes": len({row["seed"] for row in subset if row["primary_gate"] and row["resolved_distance_rank_flip"]}),
            "matched_interaction_and_distance_flip_episodes": len({row["seed"] for row in subset if row["matched_distance_joint_gate"]}),
            "matched_interaction_and_distance_flip_same_amplitudes_episodes": len({row["seed"] for row in subset if row["matched_distance_joint_gate_same_amplitudes"]}),
            "successful_grids": sum(row["all_body_success_coverage"] for row in subset),
            "relative_pair_planned_contrasts": planned_anchors*6,
            "relative_pair_observed_contrasts": sum(entry["observed"] for entry in relative),
            "relative_pair_noise_available_contrasts": sum(entry["noise_m"] is not None for entry in relative),
            "relative_pair_above_noise_and_floor_contrasts": sum(entry["above_threshold"] for entry in relative),
            "relative_pair_gate_episodes": len({row["seed"] for row in subset if row["relative_pair_gate"]}),
            "relative_pair_residual_norm_m": _distribution([entry["residual_norm_m"] for entry in relative]),
            "primary_residual_norm_m": _distribution([value["residual_norm_m"] for value in primary]),
            "episode_cluster_intervals": {key: stratified_episode_bootstrap(selected_episodes, key)
                for key in ("primary_mean_residual_norm_m", "primary_observed_fraction", "primary_gate_anchor_fraction",
                            "distance_flip_anchor_fraction", "return_flip_anchor_fraction", "matched_distance_joint_anchor_fraction", "successful_grid_anchor_fraction",
                            "relative_pair_mean_residual_norm_m", "relative_pair_observed_fraction", "relative_pair_gate_anchor_fraction",
                            *(f"relative_pair_{metric}_joint{suffix}_anchor_fraction"
                              for metric in ("distance", "return", "dense_return") for suffix in ("", "_same_amplitudes")),
                            *(f"kinematic_null_{variant}_{metric}"
                              for variant in KINEMATIC_NULL_VARIANTS
                              for metric in ("mean_excess_norm_m", "observed_fraction", "gate_anchor_fraction", "gate_same_amplitudes_anchor_fraction")))}}
        for metric in ("distance", "return", "dense_return"):
            summaries[condition][f"relative_pair_{metric}_joint_episodes"] = len({row["seed"] for row in subset if row[f"relative_pair_{metric}_joint_gate"]})
            summaries[condition][f"relative_pair_{metric}_joint_same_amplitudes_episodes"] = len({row["seed"] for row in subset if row[f"relative_pair_{metric}_joint_gate_same_amplitudes"]})
        summaries[condition]["kinematic_oracle_null"] = {}
        for variant in KINEMATIC_NULL_VARIANTS:
            null_entries = [entry for row in subset for entry in row["kinematic_oracle_null"][variant]["interactions"]]
            summaries[condition]["kinematic_oracle_null"][variant] = {
                "planned_contrasts": planned_anchors*4,
                "observed_contrasts": sum(entry["observed"] for entry in null_entries),
                "noise_available_contrasts": sum(entry["noise_m"] is not None for entry in null_entries),
                "above_noise_and_floor_contrasts": sum(entry["above_threshold"] for entry in null_entries),
                "gate_episodes": len({row["seed"] for row in subset if row["kinematic_oracle_null"][variant]["gate"]}),
                "gate_same_amplitudes_episodes": len({row["seed"] for row in subset if row["kinematic_oracle_null"][variant]["gate_same_amplitudes"]}),
                "unclipped_joint_limit_exceeded_contrasts": sum(bool(entry["unclipped_joint_limits_exceeded"]) for entry in null_entries),
                "excess_norm_m": _distribution([entry["excess_norm_m"] for entry in null_entries])}
    return {"schema": "g1_embodiedskills.m2_analysis.v1", "primary_seconds": .5,
        "planned_episodes": 40, "planned_anchors": 200, "planned_first_pass_branches": 1800,
        "planned_additional_noise_analysis_branches": 180,
        "preselected_noise_reference_groups": sum(bool(group.get("noise_anchor")) for group in groups),
        "complete_primary_gamma_noise_references": sum(value["gamma_m"] is not None for key, value in noise.items() if key[2] == 1),
        "complete_relative_pair_noise_references": sum(value["contrast_m"] is not None for value in relative_pair_noise.values()),
        "relative_pair_seconds": .5,
        "relative_pair_order": [{"hand_pair": [left, right], "body_pair": [0, body]} for left, right, body in RELATIVE_PAIRS],
        "relative_pair_rule": "T=[Y(u,b)-Y(v,b)]-[Y(u,0)-Y(v,0)], Y=actual-minus-nominal TCP vector; direct four-cell R=3 scatter; norm>max(3*nT,.002m), default-hand signed body execution and the same hand/body-pair rank flip",
        "relative_pair_scope": "Supplemental task association at+.5s; original four-Gamma primary and matched fields unchanged; distance is the relative_pair_joint_gate alias; pairs are never independent bootstrap units",
        "kinematic_oracle_null": {"enabled": model is not None, "seconds": .5,
            "variants": list(KINEMATIC_NULL_VARIANTS),
            "model_class": f"{type(model).__module__}.{type(model).__name__}" if model is not None else None,
            "scope": "Supplementary oracle using same-time future measured axis-cell joint/root tracking errors; not deployable and not proof of novel dynamics by itself",
            "rule": "Gamma_excess=Gamma_actual-Gamma_null; direct R=3 paired excess scatter; norm>max(3*nExcess,.002m) and default-hand signed body execution",
            "variant_rule": "Unclipped and joint-limit-clipped variants always reported; no selection by observed excess",
            "missingness": "Null requires its own four q/root/TCP cells; unavailable null inputs do not remove original Gamma/T observations",
            "noise_reference_counts": {variant: sum(value["excess_m"] is not None for key, value in null_noise.items() if key[2] == variant)
                                       for variant in KINEMATIC_NULL_VARIANTS}},
        "noise_rule": "Centered R=3 scatter of the same four-cell vector contrast; family/phase/combo matched",
        "engineering_gate": "residual norm > max(3*nGamma, .002 m); default-hand signed body contrast > max(3*nWaist, .0001 rad)",
        "rank_rule": "Opposite hand utility differences, both >3x paired noise; difference-change also >3x its paired noise",
        "missingness": "Unobserved contrasts remain missing; reported gate/flip fractions over all planned anchors are known lower bounds, not no-effect classifications",
        "scope": "Mechanism development diagnostic, not neural algorithm superiority; terminal d24 is censored",
        "summaries": summaries, "episodes": episode_rows, "anchors": anchor_rows}
