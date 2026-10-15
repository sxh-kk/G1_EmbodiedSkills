"""Auditable 3 hand x 3 waist reference factors for the M2 physical diagnostic.

This is a separate K=9, H>=105 interface. It is not compatible with the
pre-existing K=5/H=75 neural selector and never invokes that selector.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .candidates import CandidateBatch, NativeCommand, encode_native
from .m2_kinematics import solve_compensated_trajectory, tcp_transforms


FACTOR_LEVELS = (0, 1, -1)
FACTOR_GRID = tuple((u, b) for u in FACTOR_LEVELS for b in FACTOR_LEVELS)


@dataclass(frozen=True)
class FactorCandidateBatch(CandidateBatch):
    metadata: dict
    targets_tcp: np.ndarray
    targets_quaternion_wxyz: np.ndarray
    nominal_tcp: np.ndarray
    nominal_quaternion_wxyz: np.ndarray


def _readonly(value):
    copied = np.array(value, copy=True)
    copied.setflags(write=False)
    return copied


def pulse_envelope(frames, *, fps=50.):
    """One second pulse: .4 s rise, .2 s hold, .4 s return, then zero."""
    if not isinstance(frames, (int, np.integer)) or frames < 51:
        raise ValueError("Need at least 51 frames to include the return endpoint")
    if fps != 50.:
        raise ValueError("M2 pulse is frozen to the audited 50-Hz clock")
    times = np.arange(frames, dtype=float)/fps
    values = np.zeros(frames, dtype=float)
    rising = (times > 0.) & (times < .4)
    holding = (times >= .4) & (times <= .6)
    falling = (times > .6) & (times < 1.)
    def smooth(x):
        return x*x*x*(10.-15.*x+6.*x*x)
    values[rising] = smooth(times[rising]/.4)
    values[holding] = 1.
    values[falling] = smooth((1.-times[falling])/.4)
    return values


def _validate_inputs(q, roots, quaternions, fingers):
    frames = len(q)
    for value, shape in ((q, (frames, 29)), (roots, (frames, 3)),
                         (quaternions, (frames, 4)), (fingers, (frames, 2))):
        if (value.shape != shape or not np.issubdtype(value.dtype, np.floating)
                or not np.isfinite(value).all()):
            raise ValueError(f"Expected finite floating native input {shape}")
    if frames < 105:
        raise ValueError("50 execution + 45 preview + 1 + 9 context needs H>=105")
    if not np.allclose(np.linalg.norm(quaternions, axis=-1), 1., rtol=0., atol=1e-4):
        raise ValueError("Root quaternion must already be normalized WXYZ")
    return frames


def generate_factor_candidates(
        model, qtraj_mjcf, roots, root_quaternion_wxyz, finger_joint_targets,
        *, amplitude_m=.010, body_pitch_rad=.03, max_halvings=3,
        preview_fps=50., execution_steps=50, preview_frames=45,
        max_position_error_m=.002, max_orientation_error_rad=.02,
        max_joint_speed_rad_s=6., max_offset_speed_rad_s=2.,
        regularization_weight_m=.001, max_ik_evaluations=80,
        tcp_offset=(.085, 0., 0.)):
    """Generate one immutable full factor grid with a common applied scale.

    Cell order is hand-major: (0,0),(0,+),(0,-),(+,0),(+,+),(+,-),
    (-,0),(-,+),(-,-). Hand +/- denotes world X, body +/- denotes
    waist_pitch_joint. Every body level of the same hand condition targets
    the same world TCP position and orientation, including the preview tail.

    All nine cells are generated at each common shrink attempt. The first
    entirely feasible grid is returned. On exhaustion the last full grid is
    returned with explicit infeasibility flags; the collector must not execute
    it as a valid factor group. No per-cell shrink or output clipping occurs.
    """
    started = time.perf_counter()
    qbase, rootbase, quatbase, fingerbase = tuple(np.asarray(value) for value in
        (qtraj_mjcf, roots, root_quaternion_wxyz, finger_joint_targets))
    frames = _validate_inputs(qbase, rootbase, quatbase, fingerbase)
    if preview_fps != 50. or execution_steps != 50 or preview_frames != 45:
        raise ValueError("M2 protocol fixes 50-Hz, 50 execution and 45 preview frames")
    if isinstance(max_halvings, (bool, np.bool_)) or not isinstance(max_halvings, (int, np.integer)) or not 0 <= max_halvings <= 8:
        raise ValueError("max_halvings must be an integer from zero to eight")
    for number in (amplitude_m, body_pitch_rad, max_position_error_m,
                   max_orientation_error_rad, regularization_weight_m):
        if not np.isfinite(number) or number <= 0:
            raise ValueError("M2 amplitudes, pose tolerances and regularizer must be positive")
    for speed in (max_joint_speed_rad_s, max_offset_speed_rad_s):
        if speed is not None and (not np.isfinite(speed) or speed <= 0):
            raise ValueError("Optional speed bounds must be finite and positive")
    if isinstance(max_ik_evaluations, (bool, np.bool_)) or not isinstance(max_ik_evaluations, (int, np.integer)) or max_ik_evaluations < 1:
        raise ValueError("max_ik_evaluations must be a positive integer")
    waist = model.joint_names.index("waist_pitch_joint")
    ids = np.asarray(model.right_arm_indices, dtype=int)
    limits = np.asarray(model.joint_limits, dtype=float)
    if waist in ids or limits.shape != (29, 2):
        raise ValueError("Invalid G1 waist/arm/limit interface")
    envelope = pulse_envelope(frames, fps=preview_fps)
    active = envelope > 0.
    base_tcp, base_rotation = tcp_transforms(model, qbase, rootbase, quatbase, tcp_offset)
    target_rotation = np.broadcast_to(base_rotation, (9, frames, 3, 3)).copy()
    attempts = []
    result = None
    for level in range(max_halvings+1):
        scale = .5**level
        delta, theta = amplitude_m*scale, body_pitch_rad*scale
        target_tcp = np.broadcast_to(base_tcp, (9, frames, 3)).copy()
        commands, solver_rows = [], []
        for cell, (u, b) in enumerate(FACTOR_GRID):
            target_tcp[cell, :, 0] += u*delta*envelope
            if cell == 0:
                q = qbase.copy()
                solver_rows.append({"fk_calls": 0, "total_evaluations": 0,
                                    "seed_projection_count": 0,
                                    "baseline_reused_bitwise": True})
            else:
                solved = solve_compensated_trajectory(
                    model, qbase, rootbase, quatbase, target_tcp[cell],
                    base_rotation, qbase[:, waist]+b*theta*envelope, active,
                    waist_index=waist, tcp_offset=tcp_offset,
                    position_tolerance_m=min(.0004, max_position_error_m*.2),
                    orientation_tolerance_rad=min(.004, max_orientation_error_rad*.2),
                    regularization_weight_m=regularization_weight_m,
                    max_evaluations=max_ik_evaluations)
                q = solved.q.astype(qbase.dtype)
                solver_rows.append({"fk_calls": solved.fk_calls,
                    "total_evaluations": int(solved.solver_evaluations.sum()),
                    "maximum_frame_evaluations": int(solved.solver_evaluations.max()),
                    "inner_tolerance_failed_frames": int((~solved.frame_passed).sum()),
                    "seed_projection_count": solved.seed_projection_count,
                    "baseline_reused_bitwise": False})
            commands.append(NativeCommand(*(_readonly(value) for value in
                (q, rootbase, quatbase, fingerbase))))
        native = tuple(commands)
        actions = np.stack([encode_native(command) for command in native])
        all_q = np.stack([command.qtraj_mjcf for command in native])
        nominal_tcp, nominal_rotation = tcp_transforms(
            model, all_q, np.broadcast_to(rootbase, (9, frames, 3)),
            np.broadcast_to(quatbase, (9, frames, 4)), tcp_offset)
        position_error = np.linalg.norm(nominal_tcp-target_tcp, axis=-1)
        orientation_error = Rotation.from_matrix(
            (np.swapaxes(target_rotation, -1, -2)@nominal_rotation).reshape(-1, 3, 3)
        ).magnitude().reshape(9, frames)
        minimum_margin = np.minimum(all_q-limits[None, None, :, 0],
                                    limits[None, None, :, 1]-all_q).min(axis=(1, 2))
        joint_speed = np.max(np.abs(np.diff(all_q, axis=1))*preview_fps, axis=(1, 2))
        offset_speed = np.max(np.abs(np.diff(all_q-all_q[0:1], axis=1))*preview_fps,
                              axis=(1, 2))
        offset_acceleration = np.max(np.abs(np.diff(all_q-all_q[0:1], n=2, axis=1))
                                     *preview_fps**2, axis=(1, 2))
        cross_position, cross_orientation = [], []
        for start in (0, 3, 6):
            for a, b in ((start, start+1), (start, start+2), (start+1, start+2)):
                cross_position.append(float(np.linalg.norm(nominal_tcp[a]-nominal_tcp[b], axis=-1).max()))
                cross_orientation.append(float(Rotation.from_matrix(
                    np.swapaxes(nominal_rotation[a], -1, -2)@nominal_rotation[b]
                ).magnitude().max()))
        global_cross_pass = (max(cross_position) <= max_position_error_m and
                             max(cross_orientation) <= max_orientation_error_rad)
        diagnostics, feasible = [], []
        for cell, (u, b) in enumerate(FACTOR_GRID):
            command = native[cell]
            reasons = []
            if fingerbase.min() < -1e-6 or fingerbase.max() > .035+1e-6:
                reasons.append("finger_target_outside_gripper_limits")
            if minimum_margin[cell] < -1e-7:
                reasons.append("joint_limit_violation")
            if position_error[cell].max() > max_position_error_m:
                reasons.append("world_tcp_position_mismatch")
            if orientation_error[cell].max() > max_orientation_error_rad:
                reasons.append("world_tcp_orientation_mismatch")
            if not global_cross_pass:
                reasons.append("same_hand_cross_body_pose_mismatch")
            if max_joint_speed_rad_s is not None and joint_speed[cell] > max_joint_speed_rad_s+1e-6:
                reasons.append("reference_joint_speed_exceeds_bound")
            if max_offset_speed_rad_s is not None and offset_speed[cell] > max_offset_speed_rad_s+1e-6:
                reasons.append("reference_offset_speed_exceeds_bound")
            identity = {
                "first_q_bitwise_equal": bool(np.array_equal(command.qtraj_mjcf[0], qbase[0])),
                "tail_q_bitwise_equal": bool(np.array_equal(command.qtraj_mjcf[50:], qbase[50:])),
                "root_bitwise_equal": bool(np.array_equal(command.roots, rootbase)),
                "root_quaternion_bitwise_equal": bool(np.array_equal(command.root_quaternion_wxyz, quatbase)),
                "fingers_bitwise_equal": bool(np.array_equal(command.finger_joint_targets, fingerbase)),
                "codec_bitwise_equal": bool(np.array_equal(actions[cell], encode_native(command))),
            }
            if not all(identity.values()):
                reasons.append("native_invariance_failure")
            unchanged = np.setdiff1d(np.arange(29), np.r_[ids, waist])
            if not np.array_equal(command.qtraj_mjcf[:, unchanged], qbase[:, unchanged]):
                reasons.append("unrequested_joint_changed")
            requested_waist = qbase[:, waist]+b*theta*envelope
            waist_error = float(np.abs(command.qtraj_mjcf[:, waist]-requested_waist).max())
            if waist_error > 1e-6:
                reasons.append("waist_reference_mismatch")
            if cell and any(np.array_equal(actions[cell], actions[other]) for other in range(cell)):
                reasons.append("duplicate_encoded_candidate")
            feasible.append(not reasons)
            diagnostics.append({"candidate_index": cell, "hand_factor": u, "body_factor": b,
                "reasons": reasons, "status": "feasible" if not reasons else "infeasible",
                "requested_world_x_m": u*amplitude_m, "requested_waist_pitch_rad": b*body_pitch_rad,
                "effective_world_x_m": u*delta, "effective_waist_pitch_rad": b*theta,
                "common_scale": scale, "maximum_position_error_m": float(position_error[cell].max()),
                "maximum_orientation_error_rad": float(orientation_error[cell].max()),
                "maximum_joint_speed_rad_s": float(joint_speed[cell]),
                "maximum_offset_speed_rad_s": float(offset_speed[cell]),
                "maximum_offset_acceleration_rad_s2": float(offset_acceleration[cell]),
                "minimum_joint_limit_margin_rad": float(minimum_margin[cell]),
                "maximum_waist_reference_error_rad": waist_error, "identity": identity,
                "solver": solver_rows[cell], "command_clipped": False})
        attempts.append({"level": level, "common_scale": scale,
            "effective_amplitude_m": delta, "effective_body_pitch_rad": theta,
            "feasible": feasible,
            "maximum_position_error_m": float(position_error.max()),
            "maximum_orientation_error_rad": float(orientation_error.max()),
            "maximum_same_hand_cross_body_position_error_m": max(cross_position),
            "maximum_same_hand_cross_body_orientation_error_rad": max(cross_orientation),
            "reasons_by_candidate": [row["reasons"] for row in diagnostics]})
        quaternion_targets = Rotation.from_matrix(target_rotation.reshape(-1, 3, 3)).as_quat()[:, [3, 0, 1, 2]].reshape(9, frames, 4)
        quaternion_nominal = Rotation.from_matrix(nominal_rotation.reshape(-1, 3, 3)).as_quat()[:, [3, 0, 1, 2]].reshape(9, frames, 4)
        metadata = {
            "schema": "g1_embodiedskills.m2_factor_grid.v1", "factor_grid": [list(row) for row in FACTOR_GRID],
            "candidate_count": 9, "native_frames": frames, "reference_fps": 50,
            "execution_steps": 50, "preview_frames": 45, "filter_context_frames": 9,
            "pulse_seconds": 1., "pulse_rise_hold_return_seconds": [.4, .2, .4],
            "pulse_envelope": envelope.tolist(), "waist_joint_name": "waist_pitch_joint",
            "waist_joint_index_mjcf": int(waist), "body_sign_semantics": "local positive Y pitch; verify direction with FK",
            "requested_amplitude_m": amplitude_m, "requested_body_pitch_rad": body_pitch_rad,
            "common_scale": scale, "effective_amplitude_m": delta, "effective_body_pitch_rad": theta,
            "attempts": attempts.copy(), "ik_regularization_weight_m": regularization_weight_m,
            "ik_position_tolerance_m": min(.0004, max_position_error_m*.2),
            "ik_orientation_tolerance_rad": min(.004, max_orientation_error_rad*.2),
            "ik_max_evaluations_per_frame": int(max_ik_evaluations),
            "maximum_joint_speed_bound_rad_s": max_joint_speed_rad_s,
            "maximum_offset_speed_bound_rad_s": max_offset_speed_rad_s,
            "finger_joint_target_bounds_m": [0., .035],
            "same_hand_world_position_tolerance_m": max_position_error_m,
            "same_hand_world_orientation_tolerance_rad": max_orientation_error_rad,
            "maximum_same_hand_cross_body_position_error_m": max(cross_position),
            "maximum_same_hand_cross_body_orientation_error_rad": max(cross_orientation),
            "world_axes": "motion-frame/world-aligned; add environment origin only for physical comparison",
            "baseline_cell": 0, "baseline_native_bitwise_equal": all(
                np.array_equal(value, original) for value, original in
                zip(native[0], (qbase, rootbase, quatbase, fingerbase))),
            "wall_seconds": time.perf_counter()-started,
        }
        result = FactorCandidateBatch(native, _readonly(actions), _readonly(np.asarray(feasible, dtype=bool)),
            tuple(diagnostics), metadata, _readonly(target_tcp), _readonly(quaternion_targets),
            _readonly(nominal_tcp), _readonly(quaternion_nominal))
        if result.all_feasible:
            return result
    return result
