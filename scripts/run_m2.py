#!/usr/bin/env python3
"""Run explicit M2 scene/phase shards with frozen inputs and real resource logs."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import evaluate_online as helpers

PROJECT = helpers.PROJECT
sys.path.insert(0, str(PROJECT/"src"))
from g1_embodiedskills.io import sha256, write_json
from g1_embodiedskills.m2_analysis import (ANCHOR_STEPS, CONDITIONS, FAMILIES, NOISE_SEEDS,
                                         formal_roster)


def timestamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def registry(protocol, phase, conditions, anchors):
    """Declare complete companions in advance, without looking at outcomes."""
    if phase == "formal" and (tuple(conditions) != CONDITIONS or tuple(anchors) != ANCHOR_STEPS):
        raise ValueError("Formal M2 must retain all four conditions and five anchors")
    envs = protocol["num_envs"]
    shards = []
    for condition in conditions:
        index = CONDITIONS.index(condition)
        family = FAMILIES[condition]
        seeds = list(range(920000+10*index, 920010+10*index)) if phase == "formal" else list(range(929000+2*index, 929002+2*index))
        for anchor in anchors:
            batches = []
            for batch_index, start in enumerate(range(0, len(seeds), envs)):
                active = seeds[start:start+envs]
                selected_seed = NOISE_SEEDS[family][ANCHOR_STEPS.index(anchor)] if phase == "formal" else 929000
                items = [{"seed": seed, "anchor_step": anchor, "slot": slot,
                          "noise_repeat": bool(seed == selected_seed and (phase == "formal" or anchor == 480))}
                         for slot, seed in enumerate(active)]
                batches.append({"capsule_id": f"{condition}-anchor{anchor}-batch{batch_index}",
                    "condition": condition, "family": family, "anchors": items,
                    "companion_seeds": [active[-1]]*(envs-len(active))})
            shards.append({"id": f"{condition}-anchor{anchor}", "condition": condition, "family": family,
                "config": {"schema": "g1_embodiedskills.m2_collection_config.v1",
                    "registry_scope": "shard", "dataset_phase": phase, "formal": False,
                    "amplitude_m": protocol["amplitude_m"], "body_pitch_rad": protocol["body_pitch_rad"],
                    "max_amplitude_halvings": protocol["max_amplitude_halvings"],
                    "prefix_tolerance": protocol["prefix_tolerance"], "batches": batches}})
    if phase == "formal":
        actual = {(row["condition"], anchor["seed"], anchor["anchor_step"])
                  for row in shards for batch in row["config"]["batches"] for anchor in batch["anchors"]}
        selected = [(anchor["seed"], anchor["anchor_step"]) for row in shards
                    for batch in row["config"]["batches"] for anchor in batch["anchors"] if anchor["noise_repeat"]]
        if actual != formal_roster() or len(selected) != 10:
            raise ValueError("Formal registry lost an episode/anchor or a preselected repeat")
    return shards


def validate_protocol(protocol):
    required = {"schema": "g1_embodiedskills.m2_protocol.v1", "seconds": 24,
                "execution_steps": 50, "native_horizon": 105, "position_jitter_m": .02,
                "sample_offsets_steps": [10, 25, 50, 150]}
    for key, value in required.items():
        if protocol.get(key) != value:
            raise ValueError(f"M2 protocol requires {key}={value}")
    if type(protocol.get("num_envs")) is not int or protocol["num_envs"] not in (8, 10):
        raise ValueError("Start development at 8 environments; 10 may be frozen after measured engineering acceptance")
    if protocol.get("amplitude_m") not in (.005, .010, .015) or protocol.get("body_pitch_rad") not in (.03, .06):
        raise ValueError("Development choices must remain in the declared hand/body ladder")
    if protocol.get("max_amplitude_halvings") != 3:
        raise ValueError("This implementation predeclares three common grid halvings")
    return protocol


def physical_files(args):
    files = {"isaac_python_wrapper": args.isaac_python, "robot_asset": args.asset,
        "sonic_checkpoint": args.sonic/"sonic_release/last.pt",
        "sonic_checkpoint_configuration": args.sonic/"sonic_release/config.yaml",
        "sonic_fk_mjcf": args.sonic/"gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml",
        "sonic_sole_urdf": args.sonic/"gear_sonic/data/assets/robot_description/urdf/g1/main.urdf"}
    for condition in args.conditions:
        for name in ("motion.pkl", "reference.json", "planned.npz", "task.json"):
            files[f"reference/{condition}/{name}"] = args.reference_root/condition/"native"/name
    return files


def frozen_files(args, shards):
    files = {**helpers.source_files(args, args.protocol), **physical_files(args)}
    if args.acceptance:
        files["development_acceptance"] = args.acceptance
    for shard in shards:
        files[f"shard/{shard['id']}"] = args.output/"configurations"/(shard["id"]+".json")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("isaac-python", "sonic", "embodiedskills-root", "reference-root", "asset", "protocol", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--phase", choices=("development", "formal"), required=True)
    parser.add_argument("--conditions", choices=CONDITIONS, nargs="+", default=list(CONDITIONS))
    parser.add_argument("--anchor-steps", type=int, nargs="+", default=list(ANCHOR_STEPS))
    parser.add_argument("--acceptance", type=Path)
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    if len(args.conditions) != len(set(args.conditions)) or len(args.anchor_steps) != len(set(args.anchor_steps)):
        raise ValueError("Duplicate scene or phase in the registry")
    if any(step not in ANCHOR_STEPS for step in args.anchor_steps):
        raise ValueError("Only predeclared M2 anchor clocks are supported")
    protocol = validate_protocol(json.loads(args.protocol.read_text()))
    if args.phase == "formal":
        if args.acceptance is None:
            raise ValueError("Formal collection needs the recorded development engineering acceptance")
        acceptance = json.loads(args.acceptance.read_text())
        if acceptance.get("engineering_passed") is not True or acceptance.get("protocol_sha256") != sha256(args.protocol):
            raise ValueError("Development acceptance does not cover this exact frozen protocol")
        expected_sources = acceptance.get("source_sha256")
        actual_sources = {name: sha256(path) for name, path in helpers.source_files(args, args.protocol).items()}
        if expected_sources != actual_sources:
            raise ValueError("Development acceptance does not cover the current executable sources")
        physical_hashes = {name: sha256(path) for name, path in physical_files(args).items()}
        if acceptance.get("physical_input_sha256") != physical_hashes:
            raise ValueError("Development acceptance does not cover the current robot, controller, wrappers and scene references")
    shards = registry(protocol, args.phase, args.conditions, args.anchor_steps)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"configurations").mkdir()
    for shard in shards:
        write_json(args.output/"configurations"/(shard["id"]+".json"), shard["config"])
    files = frozen_files(args, shards)
    before = helpers.fingerprint(files)
    pipeline = {"schema": "g1_embodiedskills.m2_pipeline.v1", "phase": args.phase,
        "started_at": timestamp(), "completed": False, "status": "running", "runs": [],
        "protocol": protocol, "inputs_and_sources": before,
        "scope": "Physical factor diagnostic with complete original-batch noise repeats; no neural selector"}
    write_json(args.output/"pipeline.json", pipeline)
    started = time.monotonic()
    (args.output/"audits").mkdir()

    def interrupt(signum, _frame):
        raise helpers.PipelineInterrupted(f"M2 parent received signal {signum}")
    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with (args.output/"resources.jsonl").open("x") as resources:
            for shard in shards:
                destination = args.output/shard["id"]
                row = {"name": shard["id"], "status": "pending", "output": str(destination)}
                pipeline["runs"].append(row)
                command = ["bash", str(args.isaac_python), str(PROJECT/"scripts/run_sim.py"),
                    "--sonic", str(args.sonic), "--reference", str(args.reference_root/shard["condition"]/"native"),
                    "--embodiedskills-root", str(args.embodiedskills_root),
                    "--asset", str(args.asset), "--output", str(destination), "--mode", "m2_collect",
                    "--m2-collection-config", str(args.output/"configurations"/(shard["id"]+".json")),
                    "--num-envs", str(protocol["num_envs"]), "--seconds", "24", "--cameras",
                    "--position-jitter", str(protocol["position_jitter_m"])]
                helpers.run_process(command, args.output/(shard["id"]+".log"), row, pipeline, args.output, resources)
                changes = helpers.changed_inputs(before, frozen_files(args, shards))
                if changes:
                    raise ValueError(f"M2 sources/inputs changed during collection: {changes}")
                manifest = destination/"m2-collection/manifest.json"
                data = json.loads(manifest.read_text())
                if data.get("completed") is not True:
                    raise ValueError("Physical process returned without a complete collection")
                audit_path = args.output/"audits"/(shard["id"]+".json")
                audit_command = [sys.executable, str(PROJECT/"scripts/audit_m2.py"),
                                 "--manifests", str(manifest), "--output", str(audit_path), "--sonic", str(args.sonic)]
                with (args.output/(shard["id"]+"-audit.log")).open("x") as stream:
                    audited = subprocess.run(audit_command, cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT)
                if audited.returncode != 0:
                    raise ValueError(f"Independent M2 physical audit failed: {audit_path}")
                audit = json.loads(audit_path.read_text())
                if audit.get("passed") is not True or audit.get("manifest_sha256") != sha256(manifest):
                    raise ValueError("Independent M2 audit does not verify the current evidence")
                row.update(status="complete", physical_audit_passed=True, audit=str(audit_path),
                    audit_sha256=sha256(audit_path), manifest=str(manifest), manifest_sha256=sha256(manifest),
                    attempted_anchors=data["attempted_anchors"], analysis_branches=data["analysis_branches"],
                    physical_total_slot_rollouts=data["physical_total_slot_rollouts"])
                write_json(args.output/"pipeline.json", pipeline)
        changes = helpers.changed_inputs(before, frozen_files(args, shards))
        pipeline["source_changes"] = changes
        if changes:
            raise ValueError("M2 sources/inputs changed during the last collection/audit window")
        pipeline.update(status="complete", completed=True)
    except BaseException as error:
        pipeline.update(status="failed", completed=False, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        pipeline.update(finished_at=timestamp(), wall_seconds=time.monotonic()-started)
        write_json(args.output/"pipeline.json", pipeline)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    print(json.dumps({"completed": True, "shards": len(shards),
        "scope": "Physical collection complete; independent audit/analysis still required"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
