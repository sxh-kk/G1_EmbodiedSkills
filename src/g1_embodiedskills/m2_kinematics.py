"""CPU-only, temporally warm-started pose IK for the M2 factor diagnostic.

The supplied model is the existing G1Kinematics interface. Non-arm joints,
including the requested waist angle, stay fixed during each bounded solve.
No Isaac import, controller mutation, or random restart is used here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class CompensatedTrajectory:
    q: np.ndarray
    position_errors_m: np.ndarray
    orientation_errors_rad: np.ndarray
    frame_passed: np.ndarray
    solver_evaluations: np.ndarray
    fk_calls: int
    seed_projection_count: int
    velocity_feasible: np.ndarray


def tcp_transforms(model, q, roots, quaternions, tcp_offset=(.085, 0., 0.)):
    positions, rotations = model.transforms(q, roots, quaternions)
    wrist = rotations[..., model.right_wrist_index, :, :]
    tcp = positions[..., model.right_wrist_index, :] + np.einsum(
        "...ij,j->...i", wrist, np.asarray(tcp_offset, dtype=float))
    return tcp, wrist


def solve_compensated_trajectory(
        model, qbase, roots, quaternions, target_tcp, target_rotations,
        waist_values, active, *, waist_index, tcp_offset=(.085, 0., 0.),
        position_tolerance_m=.0004, orientation_tolerance_rad=.004,
        orientation_weight_m=.12, regularization_weight_m=.001,
        max_evaluations=80, fps=50., max_joint_speed_rad_s=6.,
        max_offset_speed_rad_s=2.):
    """Solve a fixed-target path, returning explicit per-frame feasibility.

    A small regularizer towards each baseline pose suppresses redundant-arm
    drift. The previous arm solution plus the nominal inter-frame increment
    initializes the next solve. Zero-pulse frames copy the baseline exactly;
    The existing speed limits constrain optimization coordinates at each frame,
    including enough remaining offset travel to reach the next zero-pulse
    baseline frame. An infeasible intersection is returned explicitly and must
    invalidate the candidate; no final output is projected or clipped.
    Any projection below applies only to an optimization starting guess, not
    to a final executable command. The delivered trajectory is never clipped.
    """
    qbase = np.asarray(qbase, dtype=float)
    roots = np.asarray(roots, dtype=float)
    quaternions = np.asarray(quaternions, dtype=float)
    target_tcp = np.asarray(target_tcp, dtype=float)
    target_rotations = np.asarray(target_rotations, dtype=float)
    waist_values = np.asarray(waist_values, dtype=float)
    active = np.asarray(active, dtype=bool)
    frames = len(qbase)
    expected = ((qbase, (frames, 29)), (roots, (frames, 3)),
                (quaternions, (frames, 4)), (target_tcp, (frames, 3)),
                (target_rotations, (frames, 3, 3)), (waist_values, (frames,)),
                (active, (frames,)))
    if any(value.shape != shape for value, shape in expected):
        raise ValueError("M2 IK trajectory inputs have inconsistent shapes")
    if any(not np.isfinite(value).all() for value, _ in expected):
        raise ValueError("M2 IK inputs must be finite")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("IK fps must be finite and positive")
    for speed in (max_joint_speed_rad_s, max_offset_speed_rad_s):
        if speed is not None and (not np.isfinite(speed) or speed <= 0):
            raise ValueError("Optional IK speed limits must be finite and positive")
    ids = np.asarray(model.right_arm_indices, dtype=int)
    if ids.shape != (7,) or waist_index in ids or not 0 <= waist_index < 29:
        raise ValueError("Expected a separate waist joint and seven arm joints")
    bounds = np.asarray(model.joint_limits, dtype=float)[ids]
    lower, upper = bounds[:, 0] + 1e-9, bounds[:, 1] - 1e-9
    q = qbase.copy()
    q[:, waist_index] = waist_values
    evaluations = np.zeros(frames, dtype=np.int64)
    velocity_feasible = np.ones(frames, dtype=bool)
    fk_calls, seed_projection_count = 0, 0
    previous = None
    for frame in range(frames):
        if not active[frame]:
            # The factor envelope must be zero on these frames.
            if waist_values[frame] != qbase[frame, waist_index]:
                raise ValueError("Inactive pulse frame changed the waist")
            q[frame] = qbase[frame]
            if previous is not None:
                total_change = np.abs(q[frame, ids]-previous)*fps
                offset_change = np.abs(previous-qbase[frame-1, ids])*fps
                velocity_feasible[frame] = (
                    (max_joint_speed_rad_s is None or total_change.max() <= max_joint_speed_rad_s+1e-8)
                    and (max_offset_speed_rad_s is None or offset_change.max() <= max_offset_speed_rad_s+1e-8))
            previous = q[frame, ids].copy()
            continue
        base_arm = qbase[frame, ids]
        frame_lower, frame_upper = lower.copy(), upper.copy()
        if previous is not None:
            if max_joint_speed_rad_s is not None:
                step = max_joint_speed_rad_s/fps
                frame_lower = np.maximum(frame_lower, previous-step)
                frame_upper = np.minimum(frame_upper, previous+step)
            if max_offset_speed_rad_s is not None:
                prior_offset = previous-qbase[frame-1, ids]
                step = max_offset_speed_rad_s/fps
                frame_lower = np.maximum(frame_lower, base_arm+prior_offset-step)
                frame_upper = np.minimum(frame_upper, base_arm+prior_offset+step)
        future_inactive = np.flatnonzero(~active[frame+1:])
        if len(future_inactive):
            endpoint = frame+1+int(future_inactive[0])
            remaining = endpoint-frame
            if max_offset_speed_rad_s is not None:
                travel = remaining*max_offset_speed_rad_s/fps
                frame_lower = np.maximum(frame_lower, base_arm-travel)
                frame_upper = np.minimum(frame_upper, base_arm+travel)
            if max_joint_speed_rad_s is not None:
                travel = remaining*max_joint_speed_rad_s/fps
                frame_lower = np.maximum(frame_lower, qbase[endpoint, ids]-travel)
                frame_upper = np.minimum(frame_upper, qbase[endpoint, ids]+travel)
        if np.any(frame_lower > frame_upper):
            # A diagnostic fallback is retained, never an executable solution.
            # The caller must reject velocity_feasible=False even if this
            # baseline fallback happens to meet the requested pose tolerance.
            velocity_feasible[frame] = False
            q[frame, ids] = base_arm
            previous = base_arm.copy()
            continue
        guess = (base_arm.copy() if previous is None else
                 previous + qbase[frame, ids] - qbase[max(frame-1, 0), ids])
        numerical_width = 32*np.finfo(float).eps*np.maximum(
            1., np.maximum(np.abs(frame_lower),np.abs(frame_upper)))
        free = frame_upper-frame_lower > numerical_width
        # Fixed coordinates (including roundoff-width intersections) are not
        # passed as numerically indistinguishable SciPy bounds.
        # For free coordinates, a tiny interior seed margin avoids TRF's
        # zero-distance trust-region scaling at an active speed boundary.
        # This changes only the optimization initial guess, never solved q.
        inset = np.minimum((frame_upper-frame_lower)*.25,
                           1e-8*np.maximum(1., np.maximum(np.abs(frame_lower),np.abs(frame_upper))))
        projected = np.minimum(np.maximum(guess, frame_lower+inset), frame_upper-inset)
        projected[~free] = (frame_lower[~free]+frame_upper[~free])/2.
        seed_projection_count += int(np.count_nonzero(projected != guess))
        target_rotation = Rotation.from_matrix(target_rotations[frame])
        fixed = q[frame].copy()

        def residual(values):
            nonlocal fk_calls
            trial = fixed.copy()
            arm = projected.copy()
            arm[free] = values
            trial[ids] = arm
            position, rotation = tcp_transforms(
                model, trial, roots[frame], quaternions[frame], tcp_offset)
            fk_calls += 1
            angle = (target_rotation.inv() * Rotation.from_matrix(rotation)).as_rotvec()
            return np.r_[position - target_tcp[frame],
                         orientation_weight_m * angle,
                         regularization_weight_m * (arm-base_arm)]

        arm = projected.copy()
        if free.any():
            solved = least_squares(
                residual, projected[free], bounds=(frame_lower[free], frame_upper[free]), diff_step=1e-3,
                max_nfev=max_evaluations, gtol=1e-10, ftol=1e-11, xtol=1e-11)
            arm[free] = solved.x
            evaluations[frame] = solved.nfev
        q[frame, ids] = arm
        previous = arm.copy()
    tcp, rotations = tcp_transforms(model, q, roots, quaternions, tcp_offset)
    fk_calls += 1
    position_errors = np.linalg.norm(tcp-target_tcp, axis=-1)
    orientation_errors = Rotation.from_matrix(
        np.swapaxes(target_rotations, -1, -2) @ rotations).magnitude()
    passed = ((position_errors <= position_tolerance_m) &
              (orientation_errors <= orientation_tolerance_rad) & velocity_feasible)
    return CompensatedTrajectory(q, position_errors, orientation_errors,
                                 passed, evaluations, fk_calls,
                                 seed_projection_count, velocity_feasible)
