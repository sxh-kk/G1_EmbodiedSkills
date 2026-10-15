"""Oracle additive joint/root-error null for the M2 mechanism diagnostic.

This uses future measured tracking errors from the three axis cells of the
same physical grid. It is an explanatory oracle, never a deployable predictor.
The original measured interaction is retained; neither variant selects data
or replaces missing measurements.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


CELLS = tuple((u, b) for u in range(3) for b in range(3))
CELL_INDEX = {cell: index for index, cell in enumerate(CELLS)}
INTERACTIONS = ((1, 1), (1, 2), (2, 1), (2, 2))


def simulator_to_mjcf(q, simulator_joint_names, mjcf_joint_names):
    """Map recorded q[...,31] to named 29-body-DOF coordinates, preserving NaN."""
    values = np.asarray(q)
    simulator_joint_names, mjcf_joint_names = list(simulator_joint_names), list(mjcf_joint_names)
    if (len(simulator_joint_names) != 31 or len(set(simulator_joint_names)) != 31
            or len(mjcf_joint_names) != 29 or len(set(mjcf_joint_names)) != 29
            or not set(mjcf_joint_names) <= set(simulator_joint_names)):
        raise ValueError("Expected unique simulator[31] and contained MJCF body[29] joint names")
    if values.shape[-1:] != (31,) or values.dtype.kind not in "fiu":
        raise ValueError("Recorded simulator q must be a numeric [...,31] array")
    indices = [simulator_joint_names.index(name) for name in mjcf_joint_names]
    return values[..., indices].astype(float, copy=True)


def root_rotations_from_wxyz(quaternions):
    """Convert recorded root quaternions while retaining absent rows as NaN."""
    values = np.asarray(quaternions, dtype=float)
    if values.shape[-1:] != (4,):
        raise ValueError("Expected root quaternions [...,4] in WXYZ order")
    flat = values.reshape(-1, 4)
    finite = np.isfinite(flat).all(1)
    norms = np.linalg.norm(flat[finite], axis=-1)
    if np.any(np.abs(norms-1.) > 1e-4):
        raise ValueError("Observed root quaternions must already be normalized")
    rotations = np.full((len(flat), 3, 3), np.nan)
    if finite.any():
        rotations[finite] = Rotation.from_quat(flat[finite][:, [1, 2, 3, 0]]).as_matrix()
    return rotations.reshape(values.shape[:-1]+(3, 3))


def _array(value, shape, label):
    result = np.asarray(value)
    if result.shape != shape or result.dtype.kind not in "fiu":
        raise ValueError(f"{label} must be a numeric {shape} array")
    return result.astype(float, copy=True)


def _proper_rotations(values, observed, label):
    if not observed.any():
        return
    rotations = values[observed]
    gram = rotations @ np.swapaxes(rotations, -1, -2)
    if (np.max(np.abs(gram-np.eye(3))) > 1e-6
            or np.max(np.abs(np.linalg.det(rotations)-1.)) > 1e-6):
        raise ValueError(f"{label} contains a non-SO(3) observed matrix")


def _fk_tcp(model, q, roots, rotations, tcp_offset):
    if not len(q):
        return np.empty((0, 3))
    quaternions = Rotation.from_matrix(rotations).as_quat()[:, [3, 0, 1, 2]]
    positions, matrices = model.transforms(q, roots, quaternions)
    positions, matrices = np.asarray(positions), np.asarray(matrices)
    wrist_positions = positions[..., model.right_wrist_index, :]
    wrist_matrices = matrices[..., model.right_wrist_index, :, :]
    result = wrist_positions+np.einsum("...ij,j->...i", wrist_matrices, tcp_offset)
    if result.shape != (len(q), 3) or not np.isfinite(result).all():
        raise ValueError("Official FK returned missing/nonfinite TCP values")
    return result


def _contrasts(grid, observed):
    values = np.full((4, 3), np.nan)
    for index, (u, b) in enumerate(INTERACTIONS):
        indices = (CELL_INDEX[u, b], CELL_INDEX[u, 0], CELL_INDEX[0, b], 0)
        if observed[list(indices)].all():
            values[index] = grid[indices[0]]-grid[indices[1]]-grid[indices[2]]+grid[0]
    return values


def kinematic_additive_error_null(model, *, actual_q, actual_roots, actual_root_rotations,
        nominal_q, nominal_roots, nominal_root_rotations, actual_tcp,
        valid=None, tcp_offset=(.085, 0., 0.)):
    """Propagate additive axis-cell tracking errors through nonlinear G1 FK.

    All arguments use environment-local world axes. q follows MJCF body order.
    Each is one same-anchor/sample-time K9 grid in hand-major (0,+,-) order.
    A cross-cell prediction uses only its nominal reference and the measured
    errors of (u,0),(0,b),(0,0). Its actual state is used for availability and
    measured comparison, never to fit the oracle prediction.

    Both the unconstrained extrapolation and fixed joint-limit-clipped version
    are always returned. Axis TCPs remain the original physical measurements,
    so each excess equals actual_tcp[u,b] - tcp_null[u,b] rather than adding
    numerical FK/sampling discrepancies from the other three cells.
    """
    aq = _array(actual_q, (9, 29), "actual_q")
    ap = _array(actual_roots, (9, 3), "actual_roots")
    ar = _array(actual_root_rotations, (9, 3, 3), "actual_root_rotations")
    nq = _array(nominal_q, (9, 29), "nominal_q")
    np_ = _array(nominal_roots, (9, 3), "nominal_roots")
    nr = _array(nominal_root_rotations, (9, 3, 3), "nominal_root_rotations")
    measured_tcp = _array(actual_tcp, (9, 3), "actual_tcp")
    offset = _array(tcp_offset, (3,), "tcp_offset")
    if not np.isfinite(offset).all():
        raise ValueError("TCP offset must be finite")
    observed = np.ones(9, bool)
    if valid is not None:
        supplied = np.asarray(valid)
        if supplied.shape != (9,) or supplied.dtype != np.bool_:
            raise ValueError("valid must be an explicit bool[9] mask")
        observed &= supplied
    for value in (aq, ap, ar, nq, np_, nr, measured_tcp):
        observed &= np.isfinite(value).reshape(9, -1).all(1)
    _proper_rotations(ar, observed, "actual_root_rotations")
    _proper_rotations(nr, observed, "nominal_root_rotations")
    limits = np.asarray(model.joint_limits, float)
    if (limits.shape != (29, 2) or not np.isfinite(limits).all()
            or np.any(limits[:, 0] > limits[:, 1])):
        raise ValueError("Official model must expose finite ordered joint_limits[29,2]")

    nominal_tcp = np.full((9, 3), np.nan)
    actual_fk_tcp = np.full((9, 3), np.nan)
    nominal_tcp[observed] = _fk_tcp(model, nq[observed], np_[observed], nr[observed], offset)
    actual_fk_tcp[observed] = _fk_tcp(model, aq[observed], ap[observed], ar[observed], offset)
    q_error, position_error = aq-nq, ap-np_
    rotation_error = np.full((9, 3), np.nan)
    if observed.any():
        rotation_error[observed] = Rotation.from_matrix(
            ar[observed] @ np.swapaxes(nr[observed], -1, -2)).as_rotvec()

    combo_observed = np.asarray([observed[[CELL_INDEX[u, b], CELL_INDEX[u, 0],
        CELL_INDEX[0, b], 0]].all() for u, b in INTERACTIONS], bool)
    q_null = np.full((4, 29), np.nan)
    root_null = np.full((4, 3), np.nan)
    root_rotation_null = np.full((4, 3, 3), np.nan)
    additive_q_error = np.full((4, 29), np.nan)
    additive_root_error = np.full((4, 3), np.nan)
    additive_rotation_error = np.full((4, 3), np.nan)
    for combo, (u, b) in enumerate(INTERACTIONS):
        if not combo_observed[combo]:
            continue
        joint, hand, body = CELL_INDEX[u, b], CELL_INDEX[u, 0], CELL_INDEX[0, b]
        additive_q_error[combo] = q_error[hand]+q_error[body]-q_error[0]
        additive_root_error[combo] = position_error[hand]+position_error[body]-position_error[0]
        additive_rotation_error[combo] = rotation_error[hand]+rotation_error[body]-rotation_error[0]
        q_null[combo] = nq[joint]+additive_q_error[combo]
        root_null[combo] = np_[joint]+additive_root_error[combo]
        root_rotation_null[combo] = Rotation.from_rotvec(additive_rotation_error[combo]).as_matrix() @ nr[joint]

    violation = np.full((4, 29), np.nan)
    violation[combo_observed] = np.maximum(limits[:, 0]-q_null[combo_observed], 0.)+np.maximum(q_null[combo_observed]-limits[:, 1], 0.)
    maximum_violation = np.full(4, np.nan)
    maximum_violation[combo_observed] = violation[combo_observed].max(1)
    exceeds_limits = np.zeros(4, bool)
    exceeds_limits[combo_observed] = maximum_violation[combo_observed] > 0.
    gamma_actual = _contrasts(measured_tcp, observed)
    gamma_nominal = _contrasts(nominal_tcp, observed)
    variants = {}
    for name, use_clip in (("unclipped", False), ("joint_clipped", True)):
        delivered_q = q_null.copy()
        if use_clip:
            delivered_q[combo_observed] = np.clip(delivered_q[combo_observed], limits[:, 0], limits[:, 1])
        # Only the five axis positions are copied. Unobserved cross cells stay
        # NaN; they are never filled by predictions or zeros for statistics.
        null_tcp = np.full((9, 3), np.nan)
        axes = (0, 1, 2, 3, 6)
        for axis in axes:
            if observed[axis]:
                null_tcp[axis] = measured_tcp[axis]
        predicted = _fk_tcp(model, delivered_q[combo_observed], root_null[combo_observed],
            root_rotation_null[combo_observed], offset)
        selected_combos = np.flatnonzero(combo_observed)
        excess = np.full((4, 3), np.nan)
        for row, combo in enumerate(selected_combos):
            joint = CELL_INDEX[INTERACTIONS[combo]]
            null_tcp[joint] = predicted[row]
            excess[combo] = measured_tcp[joint]-predicted[row]
        gamma_null = _contrasts(null_tcp, observed)
        identity_difference = np.abs((gamma_actual-gamma_null)-excess)
        variants[name] = {
            "observed": combo_observed.copy(), "tcp_null": null_tcp,
            "q_null": delivered_q, "root_position_null": root_null.copy(),
            "root_rotation_null": root_rotation_null.copy(), "gamma_null": gamma_null,
            "gamma_tracking_null": gamma_null-gamma_nominal, "gamma_excess": excess,
            "contrast_identity_max_abs_error": float(identity_difference[combo_observed].max()) if combo_observed.any() else None,
            "joint_clipping_applied": use_clip,
        }
    return {
        "schema": "g1_embodiedskills.m2_kinematic_additive_error_null.v1",
        "cell_valid": observed, "observed": combo_observed, "interactions": INTERACTIONS,
        "nominal_tcp": nominal_tcp, "actual_fk_tcp": actual_fk_tcp,
        "actual_tcp_minus_fk_m": measured_tcp-actual_fk_tcp,
        "gamma_actual": gamma_actual, "gamma_nominal": gamma_nominal,
        "gamma_tracking_actual": gamma_actual-gamma_nominal,
        "additive_axis_joint_error": additive_q_error,
        "additive_axis_root_position_error": additive_root_error,
        "additive_axis_world_rotation_error": additive_rotation_error,
        "q_null_unclipped": q_null, "null_joint_limit_violation_rad": violation,
        "null_max_joint_limit_violation_rad": maximum_violation,
        "null_joint_limit_exceeded": exceeds_limits,
        "unclipped": variants["unclipped"], "joint_clipped": variants["joint_clipped"],
        "root_error_convention": "world-left eta=Log(R_actual R_nominal.T); R_null=Exp(eta_u0+eta_0b-eta_00) R_nominal_ub",
        "scope": "Oracle mechanism control using same-time future measured axis-cell errors; not a deployable predictor or evidence of new dynamics by itself",
        "variant_rule": "Unclipped and fixed joint-limit-clipped FK are both reported; no selection based on observed excess",
    }
