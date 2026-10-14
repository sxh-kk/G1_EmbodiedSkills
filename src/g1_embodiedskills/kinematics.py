"""Offline G1 reference kinematics using the installed SONIC model.

Public quaternions are WXYZ; SONIC ``global_rotation`` is XYZW. Joint values
always follow MJCF traversal order and are mapped by name when used in Isaac.
This module imports neither Isaac Sim nor Isaac Lab.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import os
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import torch


@dataclass
class IKResult:
    q: np.ndarray
    success: bool
    position_error_m: float
    orientation_error_rad: float
    evaluations: int
    message: str


def _rotation_wxyz(quaternion):
    quaternion = np.asarray(quaternion, dtype=float)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("Quaternion must be a finite WXYZ vector of length 4")
    return Rotation.from_quat(quaternion[[1, 2, 3, 0]])


class G1Kinematics:
    """Name-checked 29-DOF G1 FK and bounded right-arm pose IK."""

    def __init__(self, sonic_root=None):
        supplied = sonic_root or os.environ.get("SONIC_ROOT")
        if not supplied:
            raise ValueError("Pass sonic_root or set SONIC_ROOT")
        self.sonic_root = Path(supplied).expanduser().resolve()
        if not (self.sonic_root / "gear_sonic").is_dir():
            raise FileNotFoundError(f"SONIC source package missing: {self.sonic_root}")
        sys.path.insert(0, str(self.sonic_root))
        from omegaconf import OmegaConf
        from gear_sonic.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch

        cfg = OmegaConf.create({
            "asset": {
                "assetRoot": str(self.sonic_root / "gear_sonic/data/assets/robot_description/mjcf"),
                "assetFileName": "g1_29dof_rev_1_0.xml",
            },
            "extend_config": [], "target_fps": 50,
        })
        self.fk = Humanoid_Batch(cfg)
        self.body_names = list(self.fk.body_names)
        world = ET.parse(self.fk.mjcf_file).getroot().find("worldbody")
        pairs = [(body, joint) for body in world.iter("body")
                 for joint in body.findall("joint") if joint.get("type") != "free"]
        self.joint_names = [joint.get("name") for _, joint in pairs]
        self.joint_body_indices = np.asarray([self.body_names.index(body.get("name"))
                                              for body, _ in pairs])
        if len(self.joint_names) != 29 or self.fk.num_dof != 29:
            raise ValueError("Expected the SONIC 29-DOF G1 body model")
        # Some SONIC versions include the free root in actuated_joints_idx.
        # Validate the actual non-root body sequence, not that cached helper.
        if not np.array_equal(self.joint_body_indices, np.arange(1, 30)):
            raise ValueError("G1 FK expects one hinge joint per non-root body")
        self.axes = self.fk.dof_axis.cpu().numpy().astype(float)
        if not np.allclose(self.axes, [np.fromstring(joint.get("axis"), sep=" ") for _, joint in pairs]):
            raise ValueError("MJCF axis ordering differs from SONIC FK ordering")
        self.joint_limits = self.fk.joints_range.cpu().numpy().astype(float)
        self.right_arm_indices = np.asarray([self.joint_names.index("right_" + name + "_joint")
            for name in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                         "wrist_roll", "wrist_pitch", "wrist_yaw")])
        self.right_wrist_index = self.body_names.index("right_wrist_yaw_link")
        self.sole_offsets = self._sole_offsets()

    def _default_joint_positions(self):
        # Parse only the declaration, avoiding Isaac imports before AppLauncher.
        tree = ast.parse((self.sonic_root / "gear_sonic/envs/manager_env/robots/g1.py").read_text())
        declaration = next(node.value for node in tree.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "G1_CYLINDER_MODEL_12_DEX_CFG"
                    for target in node.targets))
        init = next(keyword.value for keyword in declaration.keywords if keyword.arg == "init_state")
        mapping = ast.literal_eval(next(keyword.value for keyword in init.keywords if keyword.arg == "joint_pos"))
        q = np.zeros(29)
        for pattern, value in mapping.items():
            for i, name in enumerate(self.joint_names):
                if re.fullmatch(pattern, name):
                    q[i] = value
        return q

    def _sole_offsets(self):
        """Foot envelope for the runtime's cylinder-to-capsule import setting.

        The marker uses the lowest surface of the collision capsules, rather
        than the visually rendered foot mesh. Caller must retain that importer
        setting, or recompute the marker for a different collision geometry.
        """
        xml = ET.parse(self.sonic_root / "gear_sonic/data/assets/robot_description/urdf/g1/main.urdf").getroot()
        offsets = {}
        for side in ("left", "right"):
            name = side + "_ankle_roll_link"
            link = next(node for node in xml.findall("link") if node.get("name") == name)
            lows, highs = [], []
            for collision in link.findall("collision"):
                cylinder = collision.find("geometry/cylinder")
                if cylinder is None:
                    raise ValueError(f"Unsupported foot collision geometry in {name}")
                origin = collision.find("origin")
                center = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
                rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
                axis = Rotation.from_euler("xyz", rpy).as_matrix()[:, 2] * float(cylinder.get("length")) / 2
                radius = float(cylinder.get("radius"))
                lows.append(np.minimum(center - axis, center + axis) - radius)
                highs.append(np.maximum(center - axis, center + axis) + radius)
            low, high = np.min(lows, axis=0), np.max(highs, axis=0)
            offsets[name] = np.array([(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, low[2]])
        return offsets

    def transforms(self, q, root_position, root_quaternion_wxyz=(1., 0., 0., 0.)):
        """Return official SONIC body world positions and rotation matrices."""
        q = np.asarray(q, dtype=float)
        root = np.asarray(root_position, dtype=float)
        quat = np.asarray(root_quaternion_wxyz, dtype=float)
        if q.shape[-1:] != (29,) or root.shape[-1:] != (3,) or quat.shape[-1:] != (4,):
            raise ValueError("Expected q[...,29], root[...,3], quaternion[...,4]")
        shape = np.broadcast_shapes(q.shape[:-1], root.shape[:-1], quat.shape[:-1])
        qpos = np.concatenate([np.broadcast_to(root, shape + (3,)),
                              np.broadcast_to(quat, shape + (4,)),
                              np.broadcast_to(q, shape + (29,))], axis=-1)
        if not np.isfinite(qpos).all():
            raise ValueError("Non-finite FK input")
        with torch.no_grad():
            positions, matrices = self.fk.qpos_to_global_transforms(
                torch.as_tensor(qpos, dtype=torch.float64), use_parallel_fk=True)
        return positions.cpu().numpy(), matrices.cpu().numpy()

    def pose_axis_angle(self, q, root_quaternion_wxyz=(1., 0., 0., 0.)):
        q = np.asarray(q, dtype=float)
        pose = np.zeros(q.shape[:-1] + (len(self.body_names), 3))
        pose[..., self.joint_body_indices, :] = q[..., :, None] * self.axes
        pose[..., 0, :] = _rotation_wxyz(root_quaternion_wxyz).as_rotvec()
        return pose

    def default_stand(self):
        """Level both ankle frames and set their capsule soles on z=0."""
        q = self._default_joint_positions()
        root = np.array([0., 0., .76])
        ids = np.asarray([self.joint_names.index(side + "_" + name + "_joint")
                          for side in ("left", "right") for name in ("ankle_pitch", "ankle_roll")])
        def residual(values):
            trial = q.copy()
            trial[ids] = values
            _, matrices = self.transforms(trial, root)
            return np.concatenate([Rotation.from_matrix(matrices[self.body_names.index(side + "_ankle_roll_link")]).as_rotvec()
                                   for side in ("left", "right")])
        result = least_squares(residual, q[ids], bounds=(self.joint_limits[ids, 0], self.joint_limits[ids, 1]),
                               diff_step=1e-3, max_nfev=100, gtol=1e-10, ftol=1e-10, xtol=1e-10)
        q[ids] = result.x
        positions, matrices = self.transforms(q, root)
        soles = [positions[self.body_names.index(name)] + matrices[self.body_names.index(name)] @ offset
                 for name, offset in self.sole_offsets.items()]
        spread = float(np.ptp(np.asarray(soles)[:, 2]))
        if spread > .002 or np.linalg.norm(residual(result.x)) > .005:
            raise RuntimeError(f"Default stance feet cannot be leveled: height spread {spread}")
        root[2] -= min(sole[2] for sole in soles)
        return q, root, {
            "ankle_ik_residual_rad": residual(result.x).tolist(),
            "sole_height_spread_m": spread,
            "sole_geometry": "URDF cylinders converted to collision capsules",
            "physics_validation": "pending",
        }

    @staticmethod
    def downward_quaternion_wxyz(yaw=0.):
        """Wrist local +X points down; yaw rotates its world approach frame."""
        xyzw = (Rotation.from_euler("z", yaw) * Rotation.from_euler("y", np.pi / 2)).as_quat()
        return xyzw[[3, 0, 1, 2]]

    def solve_right_wrist(self, target_position, target_quaternion_wxyz, q_seed,
                          root_position, tcp_offset=(.085, 0., 0.),
                          root_quaternion_wxyz=(1., 0., 0., 0.),
                          position_tolerance=.003, orientation_tolerance=.035,
                          max_evaluations=120):
        """Bounded 7-DOF IK for the TCP pose, leaving all other joints fixed.

        ``target_position`` is the TCP world position, not the wrist origin.
        ``target_quaternion_wxyz`` specifies the wrist/TCP frame orientation.
        A failed result is returned explicitly; it is never silently accepted.
        """
        seed = np.asarray(q_seed, dtype=float).copy()
        target = np.asarray(target_position, dtype=float)
        offset = np.asarray(tcp_offset, dtype=float)
        if seed.shape != (29,) or target.shape != (3,) or offset.shape != (3,):
            raise ValueError("Expected q_seed[29], target_position[3], tcp_offset[3]")
        target_rotation = _rotation_wxyz(target_quaternion_wxyz)
        ids = self.right_arm_indices
        lower, upper = self.joint_limits[ids, 0], self.joint_limits[ids, 1]
        seed[ids] = np.clip(seed[ids], lower + 1e-8, upper - 1e-8)
        def errors(values):
            trial = seed.copy()
            trial[ids] = values
            positions, matrices = self.transforms(trial, root_position, root_quaternion_wxyz)
            matrix = matrices[self.right_wrist_index]
            position = positions[self.right_wrist_index] + matrix @ offset
            position_error = position - target
            rotation_error = (target_rotation.inv() * Rotation.from_matrix(matrix)).as_rotvec()
            return position_error, rotation_error
        def residual(values):
            position_error, rotation_error = errors(values)
            return np.concatenate([position_error, .12 * rotation_error])
        result = least_squares(residual, seed[ids], bounds=(lower, upper),
                               diff_step=1e-3, max_nfev=max_evaluations,
                               gtol=1e-9, ftol=1e-10, xtol=1e-10)
        position_error, rotation_error = errors(result.x)
        q = seed.copy()
        q[ids] = result.x
        pos_norm, rot_norm = float(np.linalg.norm(position_error)), float(np.linalg.norm(rotation_error))
        return IKResult(q, bool(pos_norm <= position_tolerance and rot_norm <= orientation_tolerance),
                        pos_norm, rot_norm, int(result.nfev), str(result.message))

    def fk_arrays(self, pose_aa, roots, fps=50):
        """Full official SONIC trajectory FK, including finite-difference velocities."""
        if len(pose_aa) < 3:
            raise ValueError("SONIC full FK requires at least three time frames")
        with torch.no_grad():
            result = self.fk.fk_batch(torch.as_tensor(pose_aa, dtype=torch.float32)[None],
                torch.as_tensor(roots, dtype=torch.float32)[None], return_full=True,
                fps=fps, target_fps=fps)
        return {key: value[0].cpu().numpy() for key, value in result.items()
                if isinstance(value, torch.Tensor)}
