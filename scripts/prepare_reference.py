#!/usr/bin/env python3
"""Generate fresh SONIC standing and small-wrist-motion references offline."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from g1_embodiedskills.io import sha256, write_json
from g1_embodiedskills.kinematics import G1Kinematics


def write_reference(model, output, kind, q_trajectory, root, duration, tail, diagnostics):
    output.mkdir(parents=True, exist_ok=False)
    time = np.arange(len(q_trajectory)) / 50
    pose = model.pose_axis_angle(q_trajectory).astype(np.float32)
    roots = np.repeat(root[None], len(time), axis=0).astype(np.float32)
    joblib.dump({kind: {"pose_aa": pose, "root_trans_offset": roots, "fps": 50}}, output / "motion.pkl")
    full = model.fk_arrays(pose, roots)
    np.savez_compressed(output / "planned.npz", time=time, q=q_trajectory.astype(np.float32),
                        pose_aa=pose, root_trans_offset=roots, **full)
    markers = {}
    for name in ("right_wrist_yaw_link", "left_wrist_yaw_link", "left_ankle_roll_link", "right_ankle_roll_link"):
        index = model.body_names.index(name)
        offset = model.sole_offsets.get(name, np.zeros(3))
        matrix = full["global_rotation_mat"][0, index]
        markers[name] = {"offset": offset, "position": full["global_translation"][0, index] + matrix @ offset,
                         "quaternion_wxyz": full["global_rotation"][0, index][[3, 0, 1, 2]]}
    write_json(output / "reference.json", {
        "schema_version": 1, "kind": kind, "fps": 50, "duration": duration,
        "tail_seconds": tail, "frame_count": len(time),
        "source": {"kind": "fresh_asset_default_with_leveled_ankles", "model": "g1_29dof_rev_1_0.xml"},
        "joint_names_mjcf": model.joint_names, "body_names_mjcf": model.body_names,
        "q": q_trajectory[0], "root_position": root, "root_quaternion_wxyz": [1, 0, 0, 0],
        "markers": markers, "tcp_offset_wrist_local_m": [.085, 0, 0],
        "quaternion_convention": "public WXYZ; planned.npz SONIC global_rotation XYZW",
        "nominal_status": "offline_kinematics_passed_physics_pending",
        "timing_status": "50Hz_with_2s_tail_pending_runtime_preview_audit",
        "diagnostics": diagnostics, "motion_sha256": sha256(output / "motion.pkl"),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sonic", type=Path, default=os.environ.get("SONIC_ROOT"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=20.)
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration < 2:
        parser.error("--duration must be at least 2 seconds")
    model = G1Kinematics(args.sonic)
    q, root, diagnostics = model.default_stand()
    tail = 2.
    time = np.arange(int(np.ceil((args.duration + tail) * 50)) + 1) / 50
    q_stand = np.repeat(q[None], len(time), axis=0)
    write_reference(model, args.output / "standing", "standing", q_stand, root,
                    args.duration, tail, diagnostics)
    positions, matrices = model.transforms(q, root)
    wrist = model.right_wrist_index
    tcp = positions[wrist] + matrices[wrist] @ np.array([.085, 0., 0.])
    wrist_quaternion = Rotation.from_matrix(matrices[wrist]).as_quat()[[3, 0, 1, 2]]
    ik = model.solve_right_wrist(tcp + [.025, 0., .020], wrist_quaternion, q, root)
    if not ik.success:
        raise RuntimeError(f"Small wrist motion IK failed: {asdict(ik)}")
    # Smooth excursion and return; after duration, maintain the standing tail.
    blend = np.where(time < args.duration, np.sin(np.pi * np.minimum(time / args.duration, 1)) ** 2, 0.)
    q_move = q[None] + blend[:, None] * (ik.q - q)[None]
    motion_diagnostics = dict(diagnostics, wrist_goal_ik=asdict(ik),
                              goal_tcp_position=(tcp + [.025, 0., .020]).tolist())
    write_reference(model, args.output / "wrist_motion", "wrist_motion", q_move, root,
                    args.duration, tail, motion_diagnostics)
    print(json.dumps({"standing": str(args.output / "standing/reference.json"),
                      "wrist_motion": str(args.output / "wrist_motion/reference.json"),
                      "root_position": root.tolist(), "ik_position_error_m": ik.position_error_m,
                      "ik_orientation_error_rad": ik.orientation_error_rad}, ensure_ascii=False))


if __name__ == "__main__":
    main()
