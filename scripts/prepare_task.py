#!/usr/bin/env python3
"""Generate the fresh frozen-proposer pick/place reference, without Isaac."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from g1_embodiedskills.io import write_json
from g1_embodiedskills.kinematics import G1Kinematics
from g1_embodiedskills.policies import NominalPickPlacePolicy, spec_from_geometry, audit_task_trajectory, solve_task_trajectory
from prepare_reference import write_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sonic", type=Path, default=os.environ.get("SONIC_ROOT"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--knot-fps", type=int, default=10)
    parser.add_argument("--geometry", type=Path, help="JSON geometry mapping; old fields and all defaults remain supported")
    args = parser.parse_args()
    if args.knot_fps < 5:
        parser.error("Use at least five IK knots per second")
    torch.set_num_threads(1)
    model = G1Kinematics(args.sonic)
    q_initial, root, stance = model.default_stand()
    positions, matrices = model.transforms(q_initial, root)
    wrist = model.right_wrist_index
    initial_tcp = positions[wrist] + matrices[wrist] @ [.085, 0., 0.]
    initial_quat = Rotation.from_matrix(matrices[wrist]).as_quat()[[3, 0, 1, 2]]
    spec = spec_from_geometry(json.loads(args.geometry.read_text(encoding="utf-8")) if args.geometry else None)
    policy = NominalPickPlacePolicy(initial_tcp, initial_quat, spec)
    time, q, ik = solve_task_trajectory(model, policy, q_initial, root, seed=args.seed, knot_fps=args.knot_fps)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "ik_cache.npz", time=time, q=q, root_position=root)
    write_json(args.output / "ik_knots.json", ik)
    try:
        audit, tcp = audit_task_trajectory(model, policy, time, q, root)
    except RuntimeError as error:
        write_json(args.output / "offline_audit_failure.json", {"status": "failed", "reason": str(error)})
        raise
    native = args.output / "native"
    write_reference(model, native, "pick_place", q, root, policy.spec.duration,
                    policy.spec.tail_seconds, {"stance": stance, "trajectory_audit": audit,
                                              "task_geometry": spec.as_dict()})
    manifest = policy.manifest()
    manifest.update({"reference_subdirectory": "native", "seed": args.seed,
                     "ik_knot_fps": args.knot_fps, "trajectory_audit": audit, "ik_knots": ik})
    write_json(args.output / "task.json", manifest)
    write_json(native / "task.json", manifest)
    np.savez_compressed(args.output / "task_plan.npz", time=time, q=q, tcp=tcp,
                        gripper=np.asarray([policy.gripper_target(t) for t in time]))
    print(json.dumps({"task": str(args.output / "task.json"), "reference": str(native),
                      "audit": audit}, ensure_ascii=False))


if __name__ == "__main__":
    main()
