"""Frozen task-reference proposer used by the first G1 migration pilot.

This is a geometric task policy, not a learned vision-language policy. Its
reference is generated offline and tracked by the unchanged SONIC controller.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
import xml.etree.ElementTree as ET

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp


@dataclass(frozen=True)
class PickPlaceSpec:
    cube_position: tuple = (.28, -.23, .7275)
    cube_size_m: float = .035
    tray_position_xy: tuple = (.28, -.03)
    tray_top_z: float = .722
    table_top_z: float = .71
    table_center_xy: tuple = (.5, -.12)
    table_size_xy: tuple = (.55, .65)
    tray_size_xy: tuple = (.16, .16)
    grasp_tcp_z: float = .747
    release_tcp_z: float = .759
    transit_tcp_z: float = .83
    duration: float = 24.
    tail_seconds: float = 2.
    open_position_m: float = .035
    closed_position_m: float = 0.
    yaw_rad: float = np.pi / 2

    def as_dict(self):
        return asdict(self)


class NominalPickPlacePolicy:
    """Task geometry and phase schedule, with no simulator oracle calls."""

    def __init__(self, initial_tcp, initial_quaternion_wxyz, spec=None):
        self.spec = spec or PickPlaceSpec()
        spec = self.spec
        self.initial_tcp = np.asarray(initial_tcp, dtype=float)
        self.initial_quaternion_wxyz = np.asarray(initial_quaternion_wxyz, dtype=float)
        down = (Rotation.from_euler("z", spec.yaw_rad) * Rotation.from_euler("y", np.pi / 2)).as_quat()[[3, 0, 1, 2]]
        cx, cy, _ = spec.cube_position
        tx, ty = spec.tray_position_xy
        self.keyframes = [
            (0., "settle", self.initial_tcp, self.initial_quaternion_wxyz),
            (1., "raise_clear", self.initial_tcp, self.initial_quaternion_wxyz),
            (4., "orient_and_approach", [.15, cy, .85], self.initial_quaternion_wxyz),
            (7., "descend", [cx, cy, spec.transit_tcp_z], down),
            (10., "close", [cx, cy, spec.grasp_tcp_z], down),
            (11., "lift", [cx, cy, spec.grasp_tcp_z], down),
            (14., "carry", [cx, cy, spec.transit_tcp_z], down),
            (17., "lower", [tx, ty, spec.transit_tcp_z], down),
            (20., "release", [tx, ty, spec.release_tcp_z], down),
            (21., "retreat", [tx, ty, spec.release_tcp_z], down),
            (24., "finished", [tx, ty, .84], down),
        ]
        self.gripper_schedule = [
            {"time": 0., "position": spec.open_position_m, "joint_target_m": spec.open_position_m, "event": "open"},
            {"time": 10., "position": spec.closed_position_m, "joint_target_m": spec.closed_position_m, "event": "close"},
            {"time": 20., "position": spec.open_position_m, "joint_target_m": spec.open_position_m, "event": "release"},
        ]

    def target_pose(self, time):
        if time >= self.keyframes[-1][0]:
            return np.asarray(self.keyframes[-1][2]), np.asarray(self.keyframes[-1][3])
        index = max(i for i, row in enumerate(self.keyframes[:-1]) if row[0] <= max(time, 0.))
        left, right = self.keyframes[index:index + 2]
        fraction = np.clip((time - left[0]) / (right[0] - left[0]), 0., 1.)
        fraction = fraction ** 3 * (10 - 15 * fraction + 6 * fraction ** 2)
        position = (1 - fraction) * np.asarray(left[2]) + fraction * np.asarray(right[2])
        rotations = Rotation.from_quat(np.asarray([left[3], right[3]])[:, [1, 2, 3, 0]])
        quaternion = Slerp([0, 1], rotations)(fraction).as_quat()[[3, 0, 1, 2]]
        return position, quaternion

    def gripper_target(self, time):
        return next(event["position"] for event in reversed(self.gripper_schedule) if time >= event["time"])

    def phase_at(self, time):
        return next(row[1] for row in reversed(self.keyframes) if time >= row[0])

    def manifest(self):
        return {
            "schema_version": 1, "policy": "frozen_geometric_reference",
            "status": "offline_kinematic_pilot_physics_pending", "geometry": self.spec.as_dict(),
            "duration": self.spec.duration, "tail_seconds": self.spec.tail_seconds,
            "phases": [{"name": left[1], "start_time": left[0], "end_time": right[0],
                        "target_tcp_position": list(right[2]), "target_wrist_quaternion_wxyz": list(right[3])}
                       for left, right in zip(self.keyframes[:-1], self.keyframes[1:])],
            "gripper_schedule": self.gripper_schedule,
            "gripper_semantics": "two prismatic finger joint targets in meters; zero closes",
            "tcp_offset_wrist_local_m": [.085, 0., 0.],
            "grasp_geometry_note": "tips extend .035m beyond TCP; grasp TCP is .0195m above cube center",
        }


def solve_task_trajectory(model, policy, q_initial, root_position, seed=0, knot_fps=10):
    """Solve smooth Cartesian target knots, then verify the full 50 Hz path."""
    rng = np.random.default_rng(seed)
    total = policy.spec.duration + policy.spec.tail_seconds
    knot_time = np.unique(np.r_[np.arange(0, policy.spec.duration + .5 / knot_fps, 1 / knot_fps),
                                [row[0] for row in policy.keyframes], total])
    q_previous = np.asarray(q_initial).copy()
    q_knots = []
    ik_diagnostics = []
    previous_target = None
    for time in knot_time:
        position, quaternion = policy.target_pose(time)
        target = np.r_[position, quaternion]
        if previous_target is None or not np.allclose(target, previous_target, atol=1e-10, rtol=0):
            result = model.solve_right_wrist(position, quaternion, q_previous, root_position,
                                            max_evaluations=180)
            # A different elbow seed is permitted only when the local solve
            # fails the explicit physical pose tolerances.
            for _ in range(8):
                if result.success:
                    break
                candidate_seed = q_previous.copy()
                candidate_seed[model.right_arm_indices] += rng.normal(0, .35, 7)
                candidate = model.solve_right_wrist(position, quaternion, candidate_seed, root_position,
                                                   max_evaluations=200)
                if candidate.position_error_m + .12 * candidate.orientation_error_rad < result.position_error_m + .12 * result.orientation_error_rad:
                    result = candidate
            if not result.success:
                raise RuntimeError(f"IK failed at t={time:.3f}, phase={policy.phase_at(time)}, "
                                   f"position_error={result.position_error_m:.6f}m, rotation_error={result.orientation_error_rad:.6f}rad")
            q_previous = result.q
            ik_diagnostics.append({"time": float(time), "position_error_m": result.position_error_m,
                                   "orientation_error_rad": result.orientation_error_rad,
                                   "evaluations": result.evaluations})
        q_knots.append(q_previous.copy())
        previous_target = target
    time = np.arange(int(round(total * 50)) + 1) / 50
    q = CubicSpline(knot_time, np.asarray(q_knots), bc_type="clamped")(time)
    return time, q, ik_diagnostics


def _box_corners(center, dimensions):
    return np.asarray(center) + np.asarray(list(itertools.product([-1, 1], repeat=3))) * np.asarray(dimensions) / 2


def audit_task_trajectory(model, policy, time, q, root_position):
    """Check joint bounds, speed, tracking geometry and all gripper boxes.

    Surface tests conservatively use each oriented box's world AABB overlap
    with the support rectangle. These are offline checks, not contact evidence.
    """
    positions, matrices = model.transforms(q, root_position)
    wrist_positions, wrist_matrices = positions[:, model.right_wrist_index], matrices[:, model.right_wrist_index]
    tcp = wrist_positions + np.einsum("nij,j->ni", wrist_matrices, [.085, 0, 0])
    desired = [policy.target_pose(t) for t in time]
    targets = np.asarray([row[0] for row in desired])
    target_rotation = Rotation.from_quat(np.asarray([row[1] for row in desired])[:, [1, 2, 3, 0]])
    position_error = np.linalg.norm(tcp - targets, axis=-1)
    rotation_error = (target_rotation.inv() * Rotation.from_matrix(wrist_matrices)).magnitude()
    lower_margin = q - model.joint_limits[:, 0]
    upper_margin = model.joint_limits[:, 1] - q
    speed = np.abs(np.gradient(q, time, axis=0))
    urdf = ET.parse(model.sonic_root / "gear_sonic/data/assets/robot_description/urdf/g1/main.urdf").getroot()
    joint_nodes = {joint.get("name"): joint for joint in urdf.findall("joint")}
    velocity_limits = np.asarray([float(joint_nodes[name].find("limit").get("velocity")) for name in model.joint_names])
    surface_clearance = {"table": [], "tray": []}
    spec = policy.spec
    surfaces = [("table", spec.table_center_xy, spec.table_size_xy, spec.table_top_z),
                ("tray", spec.tray_position_xy, spec.tray_size_xy, spec.tray_top_z)]
    for i, t in enumerate(time):
        opening = policy.gripper_target(t)
        boxes = [([.02, 0, 0], [.04, .06, .035])]
        boxes += [([.08, sign * (.008 + opening), 0], [.08, .012, .024]) for sign in (-1, 1)]
        for center, dimensions in boxes:
            world = wrist_positions[i] + _box_corners(center, dimensions) @ wrist_matrices[i].T
            low, high = world.min(axis=0), world.max(axis=0)
            for name, surface_center, surface_size, z in surfaces:
                center_xy, half = np.asarray(surface_center), np.asarray(surface_size) / 2
                if np.all(high[:2] >= center_xy - half) and np.all(low[:2] <= center_xy + half):
                    surface_clearance[name].append(float(low[2] - z))
    report = {
        "frame_count": len(time), "maximum_ik_interpolation_error_m": float(position_error.max()),
        "maximum_orientation_error_rad": float(rotation_error.max()),
        "minimum_joint_limit_margin_rad": float(min(lower_margin.min(), upper_margin.min())),
        "maximum_joint_speed_rad_s": float(speed.max()),
        "maximum_joint_speed_by_name": dict(zip(model.joint_names, speed.max(axis=0).tolist())),
        "maximum_urdf_velocity_fraction": float((speed / velocity_limits).max()),
        "speed_review_note": "exceeds conservative 2rad/s pilot hint" if speed.max() > 2 else "below conservative 2rad/s pilot hint",
        "minimum_gripper_clearance_m": {name: min(values) if values else None for name, values in surface_clearance.items()},
        "collision_check_scope": "three gripper boxes against table/tray top rectangles; full-body collision needs physics validation",
    }
    if report["minimum_joint_limit_margin_rad"] < -1e-5:
        raise RuntimeError(f"Joint limit violation: {report}")
    if report["maximum_urdf_velocity_fraction"] > 1.:
        raise RuntimeError(f"Offline pilot exceeds URDF joint velocity limits: {report}")
    if position_error.max() > .003 or rotation_error.max() > .035:
        raise RuntimeError(f"Interpolated pose tolerance failed: {report}")
    if any(clearance is not None and clearance < -.0001 for clearance in report["minimum_gripper_clearance_m"].values()):
        raise RuntimeError(f"Gripper intersects support geometry: {report}")
    return report, tcp
