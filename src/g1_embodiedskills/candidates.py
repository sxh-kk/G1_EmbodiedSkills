"""Five immutable native previews and their exact float32 action encoding.

No simulator rollout, state oracle, random sampling, clipping, or changes to
the frozen reference are used. Feasibility is returned to the caller: invalid
candidates remain visible and must trigger re-proposal before execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy.spatial.transform import Rotation


class NativeCommand(NamedTuple):
    qtraj_mjcf: np.ndarray
    roots: np.ndarray
    root_quaternion_wxyz: np.ndarray
    finger_joint_targets: np.ndarray


@dataclass(frozen=True)
class CandidateBatch:
    native: tuple[NativeCommand, ...]
    actions: np.ndarray
    feasible: np.ndarray
    diagnostics: tuple[dict, ...]

    @property
    def all_feasible(self):
        return bool(self.feasible.all())


def encode_native(command):
    """Losslessly specified codec: concatenate then cast once to float32.

    Native arrays retain their input dtypes. The neural representation uses
    float32, including when the input reference was computed in float64.
    """
    return np.concatenate(tuple(command), axis=-1).astype(np.float32)


def _readonly_copy(value):
    value = np.array(value, copy=True)
    value.setflags(write=False)
    return value


def _tcp_transforms(model, q, roots, quaternions, tcp_offset):
    positions, rotations = model.transforms(q, roots, quaternions)
    index = model.right_wrist_index
    wrist_rotations = rotations[..., index, :, :]
    tcp = positions[..., index, :] + np.einsum("...ij,j->...i", wrist_rotations, tcp_offset)
    return tcp, wrist_rotations


def generate_candidates(model, qtraj_mjcf, roots, root_quaternion_wxyz, finger_joint_targets,
                        amplitude_m=.012, preview_fps=50., ramp_seconds=.2,
                        jacobian_frame=0, tcp_offset=(.085, 0., 0.),
                        finite_difference_rad=.002, damping=.02,
                        orientation_weight_m=.12, max_position_error_m=.010,
                        max_orientation_change_rad=.05, jacobian_mode="per_frame",
                        max_delta_speed_rad_s=2., max_joint_speed_rad_s=6.):
    """Return candidate 0 and four world +X/-X/+Y/-Y native previews.

    A batched 6D TCP Jacobian at every preview frame preserves orientation to
    first order, avoiding singular current-pose inverses reused at later poses.
    Four damped least-squares solves share each factorization. Joint offsets
    are temporally smoothed and ramp in over .2s; frame zero is unchanged.
    When needed a single uniform scale per candidate bounds offset slew, with
    the effective Cartesian target explicitly reported. No joint is clipped.

    Joint bounds and actual offset/rotation are checked with one batched FK
    over all previews. No command is clipped. ``feasible`` is a proposal mask,
    not an execution permission: the caller must re-propose if any entry is
    false, since the original selector expects exactly five candidates.
    """
    inputs = tuple(np.asarray(value) for value in (qtraj_mjcf, roots, root_quaternion_wxyz, finger_joint_targets))
    shapes = ((75, 29), (75, 3), (75, 4), (75, 2))
    for value, shape in zip(inputs, shapes):
        if value.shape != shape or not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
            raise ValueError(f"Expected a finite floating-point preview with shape {shape}")
    q_original, roots_original, quaternions_original, fingers_original = inputs
    if not np.allclose(np.linalg.norm(quaternions_original, axis=-1), 1., atol=1e-4, rtol=0):
        raise ValueError("Root quaternions must be normalized WXYZ; input is never silently normalized")
    if not isinstance(jacobian_frame, (int, np.integer)) or not 0 <= jacobian_frame < 75:
        raise ValueError("jacobian_frame must index the 75-frame preview")
    if not np.isfinite(amplitude_m) or amplitude_m <= 0:
        raise ValueError("amplitude_m must be positive and finite")
    if min(preview_fps, ramp_seconds, finite_difference_rad, damping, orientation_weight_m) <= 0:
        raise ValueError("Timing and Jacobian parameters must be positive")
    if not np.isfinite([preview_fps, ramp_seconds, finite_difference_rad, damping, orientation_weight_m,
                        max_position_error_m, max_orientation_change_rad]).all():
        raise ValueError("All numerical configuration values must be finite")
    if max_position_error_m <= 0 or max_orientation_change_rad <= 0:
        raise ValueError("Diagnostic tolerances must be positive")
    if jacobian_mode not in ("per_frame", "current"):
        raise ValueError("jacobian_mode must be per_frame or current")
    for speed in (max_delta_speed_rad_s, max_joint_speed_rad_s):
        if speed is not None and (not np.isfinite(speed) or speed <= 0):
            raise ValueError("Optional speed bounds must be positive and finite")
    ids = np.asarray(model.right_arm_indices, dtype=int)
    limits = np.asarray(model.joint_limits, dtype=float)
    offset = np.asarray(tcp_offset, dtype=float)
    if ids.shape != (7,) or len(np.unique(ids)) != 7 or np.any(ids < 0) or np.any(ids >= 29):
        raise ValueError("Model must expose seven unique right-arm indices")
    if limits.shape != (29, 2) or offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("Expected joint limits [29,2] and a finite local TCP offset [3]")

    # One batched FK covers all central differences. G1Kinematics uses float64
    # here; a 2mrad probe also remains stable for float32 FK implementations.
    frames = np.arange(75) if jacobian_mode == "per_frame" else np.asarray([jacobian_frame])
    reference_q = np.asarray(q_original[frames], dtype=float)
    perturbed = np.repeat(reference_q[:, None], 14, axis=1)
    perturbed[:, np.arange(7), ids] += finite_difference_rad
    perturbed[:, np.arange(7) + 7, ids] -= finite_difference_rad
    jacobian_tcp, jacobian_rotation = _tcp_transforms(
        model, perturbed, np.repeat(roots_original[frames, None], 14, axis=1),
        np.repeat(quaternions_original[frames, None], 14, axis=1), offset)
    translation_jacobian = np.swapaxes((jacobian_tcp[:, :7] - jacobian_tcp[:, 7:]) / (2 * finite_difference_rad), -1, -2)
    rotation_jacobian = (Rotation.from_matrix(
        (jacobian_rotation[:, :7] @ np.swapaxes(jacobian_rotation[:, 7:], -1, -2)).reshape(-1, 3, 3)).as_rotvec()
        .reshape(len(frames), 7, 3) / (2 * finite_difference_rad))
    rotation_jacobian = np.swapaxes(rotation_jacobian, -1, -2)
    jacobian = np.concatenate([translation_jacobian, orientation_weight_m * rotation_jacobian], axis=1)
    directions = amplitude_m * np.asarray([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.]])
    right_hand_side = np.concatenate([directions.T, np.zeros((3, 4))], axis=0)
    transposed = np.swapaxes(jacobian, -1, -2)
    solved = np.linalg.solve(jacobian @ transposed + damping ** 2 * np.eye(6),
                             np.broadcast_to(right_hand_side, (len(frames), 6, 4)))
    joint_deltas = np.transpose(transposed @ solved, (2, 0, 1))
    if jacobian_mode == "current":
        joint_deltas = np.repeat(joint_deltas, 75, axis=1)
    else:
        # Symmetric three-tap filtering changes no baseline value and avoids
        # high-frequency derivative noise near small singular values.
        padded = np.pad(joint_deltas, ((0, 0), (1, 1), (0, 0)), mode="edge")
        joint_deltas = .25*padded[:, :-2] + .5*padded[:, 1:-1] + .25*padded[:, 2:]
    time = np.arange(75) / preview_fps
    ramp = np.clip(time / ramp_seconds, 0., 1.)
    ramp = ramp ** 3 * (10 - 15 * ramp + 6 * ramp ** 2)
    joint_deltas *= ramp[None, :, None]
    scales = np.ones(4)
    if max_delta_speed_rad_s is not None:
        peak_slew = np.max(np.abs(np.diff(joint_deltas, axis=1))*preview_fps, axis=(1, 2))
        scales = np.minimum(1., max_delta_speed_rad_s / np.maximum(peak_slew, 1e-12))
        joint_deltas *= scales[:, None, None]

    commands = []
    for candidate in range(5):
        q_native = np.array(q_original, copy=True)
        if candidate:
            q_native[:, ids] += joint_deltas[candidate - 1].astype(q_native.dtype)
        commands.append(NativeCommand(_readonly_copy(q_native), _readonly_copy(roots_original),
                                      _readonly_copy(quaternions_original), _readonly_copy(fingers_original)))
    native = tuple(commands)
    actions = np.stack([encode_native(command) for command in native])
    all_q = np.stack([command.qtraj_mjcf for command in native])
    tcp, rotations = _tcp_transforms(model, all_q, np.stack([command.roots for command in native]),
                                     np.stack([command.root_quaternion_wxyz for command in native]), offset)
    actual_delta = tcp - tcp[0:1]
    orientation_delta = Rotation.from_matrix(
        (rotations @ np.swapaxes(rotations[0:1], -1, -2)).reshape(-1, 3, 3)).magnitude().reshape(5, 75)
    desired_delta = np.zeros((5, 75, 3))
    desired_delta[1:] = directions[:, None] * ramp[None, :, None] * scales[:, None, None]
    pose_error = np.linalg.norm(actual_delta - desired_delta, axis=-1)
    requested_delta = np.zeros_like(desired_delta)
    requested_delta[1:] = directions[:, None] * ramp[None, :, None]
    requested_error = np.linalg.norm(actual_delta-requested_delta, axis=-1)
    minimum_margin = np.minimum(all_q - limits[None, None, :, 0], limits[None, None, :, 1] - all_q).min(axis=(1, 2))
    reference_speed = np.max(np.abs(np.diff(all_q, axis=1))*preview_fps, axis=(1, 2))
    offset_speed = np.max(np.abs(np.diff(all_q-all_q[0:1], axis=1))*preview_fps, axis=(1, 2))
    offset_acceleration = np.max(np.abs(np.diff(all_q-all_q[0:1], n=2, axis=1))*preview_fps**2, axis=(1, 2))
    feasible, diagnostics = [], []
    for candidate in range(5):
        reasons = []
        if minimum_margin[candidate] < -1e-6:
            reasons.append("joint_limit_violation")
        if max_joint_speed_rad_s is not None and reference_speed[candidate] > max_joint_speed_rad_s + 1e-5:
            reasons.append("reference_joint_speed_exceeds_bound")
        duplicate = candidate > 0 and any(np.array_equal(actions[candidate], actions[other]) for other in range(candidate))
        if duplicate:
            reasons.append("duplicate_encoded_candidate")
        if candidate and pose_error[candidate].max() > max_position_error_m:
            reasons.append("tcp_offset_approximation_error")
        if candidate and orientation_delta[candidate].max() > max_orientation_change_rad:
            reasons.append("orientation_change_exceeds_tolerance")
        direction = np.zeros(3) if candidate == 0 else directions[candidate - 1]
        # A reversed or vanishing displacement is explicitly infeasible even
        # if a loose absolute approximation threshold would otherwise pass it.
        minimum_projection = minimum_gain = None
        if candidate:
            active = ramp >= .5
            signed_projection = actual_delta[candidate, active] @ (direction / amplitude_m)
            minimum_projection = float(signed_projection.min())
            minimum_gain = float(np.min(signed_projection / (amplitude_m*scales[candidate-1]*ramp[active])))
            if np.any(signed_projection <= 0):
                reasons.append("offset_direction_not_realized")
        feasible.append(not reasons)
        diagnostics.append({
            "candidate_index": candidate, "status": "feasible" if not reasons else "infeasible",
            "reasons": reasons, "requested_world_offset_m": direction.tolist(),
            "jacobian_frame": int(jacobian_frame), "minimum_joint_limit_margin_rad": float(minimum_margin[candidate]),
            "jacobian_mode": jacobian_mode, "jacobian_pose_count": len(frames), "damping": float(damping),
            "finite_difference_rad": float(finite_difference_rad),
            "uniform_slew_scale": 1. if candidate == 0 else float(scales[candidate-1]),
            "effective_world_offset_m": np.zeros(3).tolist() if candidate == 0 else (direction*scales[candidate-1]).tolist(),
            "maximum_joint_speed_rad_s": float(reference_speed[candidate]),
            "maximum_offset_slew_rad_s": float(offset_speed[candidate]),
            "maximum_offset_acceleration_rad_s2": float(offset_acceleration[candidate]),
            "minimum_direction_projection_m": minimum_projection,
            "minimum_direction_gain": minimum_gain,
            "maximum_tcp_offset_error_m": float(pose_error[candidate].max()),
            "maximum_requested_tcp_offset_error_m": float(requested_error[candidate].max()),
            "maximum_orientation_change_rad": float(orientation_delta[candidate].max()),
            "final_actual_tcp_offset_m": actual_delta[candidate, -1].tolist(),
            "final_requested_tcp_offset_m": requested_delta[candidate, -1].tolist(),
            "final_effective_tcp_offset_m": desired_delta[candidate, -1].tolist(),
            "joint_delta_rad": np.zeros(7).tolist() if candidate == 0 else joint_deltas[candidate - 1, -1].tolist(),
            "maximum_joint_delta_rad": 0. if candidate == 0 else float(np.abs(joint_deltas[candidate-1]).max()),
            "clipped": False,
        })
    actions.setflags(write=False)
    feasible = np.asarray(feasible, dtype=bool)
    feasible.setflags(write=False)
    return CandidateBatch(native, actions, feasible, tuple(diagnostics))
