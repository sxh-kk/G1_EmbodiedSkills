#!/usr/bin/env python3
"""Publish a complete M2 diagnostic only after every physical shard is audited."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/"src"))
from g1_embodiedskills.io import sha256, write_json
from g1_embodiedskills.m2_analysis import load_groups, summarize_m2


def verify_evidence(expected):
    for path, digest in expected.items():
        if not Path(path).is_file() or sha256(path) != digest:
            raise ValueError(f"Passed-audit source/evidence changed before or during analysis: {path}")


def official_fk_input_hashes(sonic):
    """Pin the exact local G1 FK implementation and mechanical model inputs."""
    sonic = Path(sonic).expanduser().resolve()
    paths = [PROJECT/"src/g1_embodiedskills"/name for name in
             ("kinematics.py", "m2_kinematic_null.py", "m2_analysis.py")]
    paths.extend(sonic/name for name in (
        "gear_sonic/utils/motion_lib/torch_humanoid_batch.py",
        "gear_sonic/isaac_utils/rotations.py",
        "gear_sonic/trl/utils/torch_transform.py",
        "gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml",
        "gear_sonic/data/assets/robot_description/urdf/g1/main.urdf",
        "gear_sonic/envs/manager_env/robots/g1.py"))
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("Official G1 FK sources/model inputs are incomplete")
    return {str(path.resolve()): sha256(path) for path in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--audits", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sonic", type=Path, required=True,
                        help="Official SONIC G1 CPU FK for the supplementary oracle null")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Analysis output must be a fresh file")
    started = time.monotonic()
    audits, evidence = {}, {}
    for path in args.audits:
        report = json.loads(path.read_text())
        if report.get("completed") is not True or report.get("passed") is not True:
            raise ValueError(f"Physical/record audit did not pass: {path}")
        if report.get("official_cpu_fk_verified") is not True:
            raise ValueError(f"Formal M2 analysis requires independent official FK verification: {path}")
        digest = report["manifest_sha256"]
        if digest in audits:
            raise ValueError("Duplicate audit for the same physical shard")
        audits[digest] = {"path": str(path.resolve()), "sha256": sha256(path)}
        locked = report["evidence_hashes_before_after"]
        if locked.get("unchanged") is not True or locked["before"] != locked["after"]:
            raise ValueError("Passed audit has no immutable evidence inventory")
        for item, item_digest in locked["after"].items():
            if item in evidence and evidence[item] != item_digest:
                raise ValueError("Shards were audited with different shared sources/inputs")
            evidence[item] = item_digest
    records = []
    for path in args.manifests:
        digest = sha256(path)
        if digest not in audits:
            raise ValueError(f"No passed audit for the current manifest bytes: {path}")
        records.append({"path": str(path.resolve()), "sha256": digest, "audit": audits[digest]})
    if len(records) != len(audits) or len({row["sha256"] for row in records}) != len(records):
        raise ValueError("Manifest/audit pairing is not one-to-one")
    verify_evidence(evidence)
    model_inputs = official_fk_input_hashes(args.sonic)
    for path, digest in model_inputs.items():
        if path in evidence and evidence[path] != digest:
            raise ValueError(f"Oracle model input differs from passed-audit sources: {path}")
    import torch
    import scipy
    torch.set_num_threads(1)
    from g1_embodiedskills.kinematics import G1Kinematics
    model = G1Kinematics(args.sonic)
    verify_evidence(model_inputs)
    groups = load_groups(args.manifests)
    for group in groups:
        group["_audited_evidence"] = evidence
    report = summarize_m2(groups, formal=True, model=model)
    verify_evidence(evidence)
    verify_evidence(model_inputs)
    report["kinematic_oracle_null"].update(official_cpu_fk_verified=True,
        official_model_inputs_sha256=model_inputs, model_inputs_unchanged=True,
        device="cpu", torch_version=torch.__version__, scipy_version=scipy.__version__,
        sonic_root=str(args.sonic.expanduser().resolve()))
    report.update(completed=True, created_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        inputs=records, analysis_source_sha256=sha256(PROJECT/"src/g1_embodiedskills/m2_analysis.py"),
        evidence_unchanged_during_analysis=True, independently_audited_artifacts=len(evidence),
        entrypoint_sha256=sha256(__file__), numpy_version=np.__version__,
        bootstrap_uncertainty_scope="Conditional on the ten fixed R=3 noise references; their sampling uncertainty is not separately estimated",
        noise_scaling="Actual amplitudes recorded per grid; no linear rescaling or post-hoc noise-reference substitution",
        wall_seconds=time.monotonic()-started)
    write_json(args.output, report)
    print(json.dumps({"completed": True, "episodes": len(report["episodes"]), "anchors": len(report["anchors"]),
        "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
