"""Read-only official SONIC FK smoke; writes only an optional /tmp report."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

LOCAL_PACKAGE = Path(__file__).resolve().parents[1]/"src/g1_embodiedskills"
PROJECT_ROOT = (LOCAL_PACKAGE.parents[1] if (LOCAL_PACKAGE/"__init__.py").is_file()
                else Path(os.environ["G1_M2_PROJECT_ROOT"]))
sys.path.insert(0, str(PROJECT_ROOT/"src"))
import g1_embodiedskills
g1_embodiedskills.__path__.insert(0, str(LOCAL_PACKAGE))
from g1_embodiedskills.kinematics import G1Kinematics
from g1_embodiedskills.m2_candidates import generate_factor_candidates
from g1_embodiedskills.collection import _fingers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cursors", type=int, nargs="+", default=[480])
    parser.add_argument("--report", type=Path)
    parser.add_argument("--sonic", type=Path, default=os.environ.get("SONIC_ROOT"))
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = G1Kinematics(args.sonic)
    reference = args.reference
    planned = dict(np.load(reference/"planned.npz", allow_pickle=False))
    manifest = json.loads((reference/"reference.json").read_text())
    task = json.loads((reference/"task.json").read_text())
    reports = []
    for cursor in args.cursors:
        q = planned["q"][cursor:cursor+105]
        roots = np.tile(np.asarray(manifest["root_position"], dtype=np.float32), (105, 1))
        quats = np.tile(np.asarray(manifest["root_quaternion_wxyz"], dtype=np.float32), (105, 1))
        fingers = _fingers(task["gripper_schedule"], cursor, frames=105)
        batch = generate_factor_candidates(model, q, roots, quats, fingers)
        row = {"cursor": cursor, "all_feasible": batch.all_feasible,
               "metadata": batch.metadata, "diagnostics": batch.diagnostics,
               "kind": "CPU kinematic smoke only; no physical outcome"}
        reports.append(row)
        print(json.dumps({"cursor": cursor, "all_feasible": batch.all_feasible,
                          "scale": batch.metadata["common_scale"],
                          "seconds": batch.metadata["wall_seconds"],
                          "max_cross_position_m": batch.metadata["maximum_same_hand_cross_body_position_error_m"],
                          "max_cross_orientation_rad": batch.metadata["maximum_same_hand_cross_body_orientation_error_rad"],
                          "reasons": [item["reasons"] for item in batch.diagnostics]}, ensure_ascii=False), flush=True)
    if args.report:
        if not str(args.report.resolve()).startswith("/tmp/"):
            raise ValueError("This staging smoke writes reports only under /tmp")
        args.report.write_text(json.dumps(reports, indent=2, ensure_ascii=False)+"\n")
    return 0 if all(item["all_feasible"] for item in reports) else 2


if __name__ == "__main__":
    raise SystemExit(main())
