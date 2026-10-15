#!/usr/bin/env python3
"""Run one new fixed-candidate-2 control and re-audit the frozen paired records.

The direct/selector episodes are reused read-only. The fixed policy is evaluated
once in a fresh Isaac process on the original seeds, without a selector bundle,
retries, seed substitution or replacement trials. A failed integrity check stops
the pipeline and preserves the actual logs and records.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np

import evaluate_online as helpers
from summarize_evaluation import INITIAL_TOLERANCE, audit_method, paired, require

PROJECT = helpers.PROJECT
BASELINE_REVISION = "527c603d7516a511f10a693ac7f98744a99f2833"
ALLOWED_CHANGES = frozenset({"project/src/g1_embodiedskills/online.py",
    "project/scripts/run_sim.py", "project/scripts/summarize_evaluation.py"})
CORE = {"seed_base": 900000, "repeats": 30, "num_envs": 8, "seconds": 24,
        "position_jitter_m": .02, "amplitude_m": .025, "execution_steps": 10}
METHODS = ("direct", "selector", "fixed2")


def timestamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


# The shared process/resource helpers also record Beijing time explicitly.
helpers.timestamp = timestamp
read_json, write_json, sha256 = helpers.read_json, helpers.write_json, helpers.sha256


def current_files(args):
    return {**helpers.source_files(args, PROJECT / "configs/evaluation.json"),
            **helpers.input_files(args),
            "configuration/evaluation-fixed2.json": PROJECT / "configs/evaluation-fixed2.json"}


def baseline_files(directory):
    """Fingerprint every existing raw file, including decision archives/videos."""
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            require(path.resolve().is_relative_to(directory), f"Baseline symlink escapes the run: {path}")
            files[str(path.relative_to(directory))] = path
    require(bool(files), "Baseline directory is empty")
    return files


def validate_fingerprint(record):
    require(isinstance(record.get("files"), dict) and bool(record["files"]), "Missing baseline fingerprint")
    encoded = json.dumps({key: value["sha256"] for key, value in record["files"].items()},
                         sort_keys=True).encode()
    require(hashlib.sha256(encoded).hexdigest() == record.get("combined_sha256"),
            "Baseline fingerprint has an inconsistent combined hash")


def validate_protocol(args):
    base = helpers.validate_config(read_json(PROJECT / "configs/evaluation.json"))
    fixed = read_json(PROJECT / "configs/evaluation-fixed2.json")
    for name, expected in CORE.items():
        require(not isinstance(base.get(name), bool) and base.get(name) == expected,
                f"Original evaluation protocol requires {name}={expected}")
        require(not isinstance(fixed.get(name), bool) and fixed.get(name) == expected,
                f"Fixed2 protocol changed {name}; expected {expected}")
    require(fixed.get("method") == "fixed2" and fixed.get("fixed_candidate_index") == 2
            and fixed.get("baseline_revision") == BASELINE_REVISION,
            "Fixed2 configuration must pin candidate 2 and the original baseline revision")
    baseline = read_json(args.baseline / "pipeline.json")
    require(baseline.get("completed") is True and baseline.get("status") == "complete"
            and baseline.get("configuration") == base and baseline.get("final_input_changes") == [],
            "Original baseline pipeline must be complete, unchanged and use the original configuration")
    require(baseline.get("summary", {}).get("status") == "complete", "Original paired audit is incomplete")
    methods = baseline.get("methods", [])
    require([row.get("name") for row in methods] == list(helpers.METHODS)
            and all(row.get("status") == "complete" and row.get("returncode") == 0
                    and row.get("input_changes") == [] and row.get("result", {}).get("completed") is True
                    for row in methods), "Original baseline has incomplete or changed method records")
    pair_path = args.baseline / "paired-report.json"
    pair_hash = sha256(pair_path)
    require(pair_hash == baseline["summary"].get("report_sha256"), "Original paired report hash changed")
    original = baseline.get("inputs_and_sources", {})
    validate_fingerprint(original)
    now = helpers.fingerprint(current_files(args))
    differences = helpers.changed_inputs(original, current_files(args))
    authorized, added = [], []
    for change in differences:
        name = change["name"]
        if change.get("before") is None and name not in original["files"] and not change.get("error"):
            added.append(change)
        else:
            require(name in ALLOWED_CHANGES and not change.get("error") and change.get("after"),
                    f"Frozen baseline source/model/input changed without authorization: {change}")
            authorized.append(change)
    # Bind the allowed old source bytes to the named historical Git revision.
    historical = {}
    for name in sorted(ALLOWED_CHANGES):
        command = ["git", "show", f"{BASELINE_REVISION}:{name.removeprefix('project/')}"]
        process = subprocess.run(command, cwd=PROJECT, capture_output=True, check=False)
        require(process.returncode == 0, f"Cannot read baseline source revision for {name}")
        digest = hashlib.sha256(process.stdout).hexdigest()
        require(digest == original["files"][name]["sha256"], f"Named baseline revision differs for {name}")
        historical[name] = digest
    selector_hash = original["files"]["selector_checkpoint"]["sha256"]
    selector_result = read_json(args.baseline / "selector/result.json")
    require(selector_result.get("model_identity", {}).get("checkpoint_sha256") == selector_hash,
            "Original selector result/checkpoint identity disagree")
    original_report = read_json(pair_path)
    require(original_report.get("methods", {}).get("direct", {}).get("counts", {}).get("success") == 27
            and original_report.get("methods", {}).get("selector", {}).get("counts", {}).get("success") == 29,
            "Expected the original 27/30 direct and 29/30 selector records")
    return base, fixed, baseline, now, {"baseline_revision": BASELINE_REVISION,
        "baseline_pipeline": str(args.baseline / "pipeline.json"),
        "baseline_pipeline_sha256": sha256(args.baseline / "pipeline.json"),
        "baseline_paired_report": str(pair_path), "baseline_paired_report_sha256": pair_hash,
        "baseline_input_fingerprint_sha256": original["combined_sha256"],
        "selector_checkpoint_sha256": selector_hash,
        "authorized_old_source_changes": authorized, "new_files": added,
        "historical_source_hashes_verified": historical,
        "unchanged_original_inputs_required": True}


def compare_runs(args):
    """Independent full physical/decision audits, followed by paired comparison."""
    pipeline = read_json(args.output / "pipeline.json")
    require(not (args.output / "comparison-report.json").exists(), "Comparison report already exists")
    runs = {method: audit_method(args.output / method if method == "fixed2" else args.baseline / method, method)
            for method in METHODS}
    base = runs["direct"]
    for method, run in runs.items():
        require(run["pairing"] == base["pairing"], f"{method}: paired settings/reference/asset differ")
        require(set(run["episodes"]) == set(range(900000, 900030)), f"{method}: wrong paired seed set")
    require(runs["selector"]["summary"]["model_identity"].get("checkpoint_sha256") ==
            pipeline["baseline_provenance"]["selector_checkpoint_sha256"],
                "Re-audited selector checkpoint differs from the pinned original bundle")
    fixed = runs["fixed2"]["summary"]
    require(fixed["selected_histogram"] == [0, 0, fixed["decisions"], 0, 0]
            and fixed["executed_control_steps"] <= fixed["maximum_possible_control_steps"] == 36000,
            "Fixed2 contains a non-2 selection or inconsistent actual execution coverage")
    require(fixed["latency"]["encoder"]["count"] == fixed["latency"]["selector"]["count"] == 0,
            "Fixed2 must not perform encoder or selector inference")
    errors, per_seed = {}, []
    for seed in sorted(base["episodes"]):
        per_seed.append({"seed": seed, "methods": {method: run["episodes"][seed] for method, run in runs.items()}})
        errors[str(seed)] = {}
        for method in ("selector", "fixed2"):
            reference, actual = base["initial"][seed], runs[method]["initial"][seed]
            require(runs[method]["episodes"][seed]["slot"] == base["episodes"][seed]["slot"]
                    and runs[method]["episodes"][seed]["batch"] == base["episodes"][seed]["batch"],
                    f"{method}/seed={seed}: original batch/slot assignment changed")
            require(np.array_equal(reference["env_origin"], actual["env_origin"]),
                    f"{method}/seed={seed}: environment origins differ")
            fields = {"env_origin": {"max_abs": 0., "rms": 0.}}
            for key in ("root", "q", "qdot", "cube_root"):
                delta = np.abs(actual[key] - reference[key])
                fields[key] = {"max_abs": float(delta.max()), "rms": float(np.sqrt(np.mean(delta**2)))}
                if key in ("root", "cube_root"):
                    fields[key].update({name: float(delta[part].max()) for name, part in
                        (("position_m", slice(0, 3)), ("quaternion_components", slice(3, 7)),
                         ("linear_velocity_m_s", slice(7, 10)), ("angular_velocity_rad_s", slice(10, 13)))})
                require(fields[key]["max_abs"] <= INITIAL_TOLERANCE,
                        f"{method}/seed={seed}: initial {key} differs by {fields[key]['max_abs']}")
            errors[str(seed)][method] = fields
    original = read_json(args.baseline / "paired-report.json")
    for method in ("direct", "selector"):
        for key in ("counts", "selected_histogram", "decisions", "success_rate"):
            require(runs[method]["summary"][key] == original["methods"][method][key],
                    f"Re-audited original {method}/{key} differs from the saved paired report")
    report = {"schema": "g1_embodiedskills.fixed_candidate2_control.v1", "audited_at": timestamp(),
        "scope": "One new post-hoc fixed2 run; original direct and selector records reused and independently re-audited",
        "baseline_provenance": pipeline["baseline_provenance"],
        "frozen_sources_and_inputs_sha256": pipeline["inputs_and_sources"]["combined_sha256"],
        "baseline_records_before": pipeline["baseline_records_before"],
        "pairing": {**base["pairing"], "episodes": 30, "seeds": sorted(base["episodes"]),
            "same_seed_slot_batch": True, "initial_state_tolerance": INITIAL_TOLERANCE,
            "initial_state_fields": ["root13", "q31", "qdot31", "cube_root13", "env_origin3"],
            "initial_state_errors_vs_direct": errors},
        "methods": {method: {"directory": run["directory"],
            "evidence_origin": "new measured run" if method == "fixed2" else "original records re-audited; no rerun",
            **run["summary"]} for method, run in runs.items()},
        "paired_observed_outcomes": {
            "fixed2_vs_direct": paired(base["episodes"], runs["fixed2"]["episodes"]),
            "selector_vs_direct": paired(base["episodes"], runs["selector"]["episodes"]),
            "selector_vs_fixed2": paired(runs["fixed2"]["episodes"], runs["selector"]["episodes"])},
        "per_seed": per_seed,
        "audit": {"physical_outcomes_recomputed": True, "full_object_pose_and_velocity_required": True,
            "initial_states_and_origins_compared": True, "actual_control_step_coverage_verified": True,
            "all_selected_native_codecs_and_hashes_verified": True,
            "selector_actual_score_argmax_verified": True, "fixed2_all_choices_equal_2": True,
            "fixed2_complete_control_budget": fixed["executed_control_steps"] == 36000,
            "fixed2_scores_and_head_argmax_null": True, "fixed2_encoder_selector_inference_count": 0,
            "fixed2_loaded_selector_bundle": False, "missing_episodes_are_errors": True,
            "padding_slots_counted": False, "early_ended_or_failed_episodes_can_succeed": False,
            "original_direct_and_selector_summaries_match_saved_report": True,
            "unselected_native_hashes": "Only selected native commands are saved; other native hashes cannot be independently reconstructed"},
        "limitations": [
            "Candidate 2 was chosen post hoc after seeing selector preferences on these 30 seeds; this is a diagnostic control, not a new held-out confirmatory test",
            "Only fixed2 is newly run; direct/selector retain their original process times and are re-audited without retraining or rerunning",
            "Three old source files add the fixed2 branch/audit; their before/after hashes are recorded, while other original source/input bytes must match",
            "Fixed2 uses the same current-state perception, proposal feasibility checks, amplitude shrinking and feedback loop, and always selects current feasible index 2",
            "Same initial states/settings do not imply identical later trajectories after different choices; rescues/harms are separate measured closed-loop episodes",
            "Thirty seeds do not establish statistical superiority, broad generalization or a contribution specific to the three WAM predictors",
            "GPU telemetry is sampled whole-GPU use; runtime PyTorch peaks exclude renderer allocations",
            "Selected native commands are independently verified; unselected native hashes cannot be reconstructed from the saved selected-command archive"]}
    write_json(args.output / "comparison-report.json", report)
    print(json.dumps({"comparison_report": str(args.output / "comparison-report.json"),
        "success_counts": {method: run["summary"]["counts"]["success"] for method, run in runs.items()},
        "paired": report["paired_observed_outcomes"]}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("isaac-python", "sonic", "reference", "asset", "selector", "embodiedskills-root", "baseline", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    require(not args.output.is_relative_to(args.baseline) and not args.baseline.is_relative_to(args.output),
            "New output and original baseline must be separate directory trees")
    if args.audit_only:
        compare_runs(args)
        return
    require(shutil.which("nvidia-smi") is not None, "nvidia-smi is required for actual GPU resource records")
    base, fixed, _baseline, before, provenance = validate_protocol(args)
    baseline_before = helpers.fingerprint(baseline_files(args.baseline))
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "baseline-records-before.json", baseline_before)
    row = {"name": "fixed2", "status": "pending", "output": str(args.output / "fixed2")}
    audit_row = {"name": "comparison_audit", "status": "pending"}
    pipeline = {"schema": "g1_embodiedskills.fixed_candidate2_pipeline.v1", "started_at": timestamp(),
        "completed": False, "status": "running", "configuration": fixed,
        "original_configuration": base, "configuration_path": str(PROJECT / "configs/evaluation-fixed2.json"),
        "configuration_sha256": sha256(PROJECT / "configs/evaluation-fixed2.json"),
        "inputs_and_sources": before, "baseline_provenance": provenance,
        "baseline_records_before": {"manifest": "baseline-records-before.json", "files": len(baseline_before["files"]),
            "combined_sha256": baseline_before["combined_sha256"]},
        "python": {"executable": sys.executable, "version": sys.version},
        "arguments": {key: str(value) for key, value in vars(args).items() if key != "audit_only"},
        "methods": [row], "summary": audit_row, "resource_log": "resources.jsonl", "resource_interval_seconds": 10,
        "resource_scope": "Sampled whole-GPU memory/utilization including renderer and unrelated processes",
        "failure_policy": "Preserve all task failures; stop on process/integrity failure; no retries or replacement seeds",
        "fixed2_model_loading": "Selector checkpoint is fingerprinted only; no selector argument is passed to Isaac"}
    write_json(args.output / "pipeline.json", pipeline)
    started = time.monotonic()

    def interrupt(signum, _frame):
        raise helpers.PipelineInterrupted(f"Parent received signal {signum}")

    previous = {signum: signal.signal(signum, interrupt) for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        with (args.output / "resources.jsonl").open("x", encoding="utf-8") as resources:
            command = helpers.command_for(args, base, "fixed2", args.output / "fixed2")
            require("--selector" not in command, "Fixed2 subprocess must never receive the selector checkpoint")
            try:
                helpers.run_process(command, args.output / "fixed2.log", row, pipeline, args.output, resources)
            finally:
                try:
                    row["result"] = helpers.inspect_result(args.output / "fixed2", "fixed2", base,
                                                          provenance["selector_checkpoint_sha256"])
                except (OSError, ValueError, TypeError, AttributeError) as error:
                    row["result"] = {"completed": False, "error": f"{type(error).__name__}: {error}"}
                write_json(args.output / "pipeline.json", pipeline)
            row["input_changes"] = helpers.changed_inputs(before, current_files(args))
            require(not row["input_changes"], "Frozen inputs/sources changed during the fixed2 run")
            require(row["result"]["completed"], "Fixed2 did not produce the complete original 30-seed protocol")
            row["status"] = "complete"
            audit_command = [sys.executable, str(Path(__file__).resolve())]
            for name, value in vars(args).items():
                if name != "audit_only":
                    audit_command.extend(("--" + name.replace("_", "-"), str(value)))
            audit_command.append("--audit-only")
            helpers.run_process(audit_command, args.output / "comparison-audit.log", audit_row,
                                pipeline, args.output, resources)
            report_path = args.output / "comparison-report.json"
            require(report_path.is_file(), "Comparison audit did not create its report")
            audit_row.update(status="complete", report=str(report_path), report_sha256=sha256(report_path))
            pipeline["final_input_changes"] = helpers.changed_inputs(before, current_files(args))
            require(not pipeline["final_input_changes"], "Frozen inputs changed during comparison audit")
            baseline_after = helpers.fingerprint(baseline_files(args.baseline))
            write_json(args.output / "baseline-records-after.json", baseline_after)
            pipeline["baseline_record_changes"] = helpers.changed_inputs(baseline_before, baseline_files(args.baseline))
            pipeline["baseline_records_after"] = {"manifest": "baseline-records-after.json",
                "files": len(baseline_after["files"]), "combined_sha256": baseline_after["combined_sha256"]}
            require(not pipeline["baseline_record_changes"], "Original read-only baseline records changed")
            report = read_json(report_path)
            report["baseline_records_after"] = pipeline["baseline_records_after"]
            report["audit"]["all_original_baseline_file_hashes_unchanged"] = True
            report["audit"]["baseline_records_compared"] = len(baseline_before["files"])
            write_json(report_path, report)
            audit_row["report_sha256"] = sha256(report_path)
        pipeline.update(completed=True, status="complete")
    except BaseException as error:
        pipeline.update(completed=False, status="interrupted" if isinstance(error,
            (KeyboardInterrupt, helpers.PipelineInterrupted)) else "failed",
            error={"type": type(error).__name__, "message": str(error)})
        for stage in (row, audit_row):
            if stage["status"] == "pending":
                stage["status"] = "not_started_due_to_pipeline_stop"
        raise
    finally:
        # Preserve the read-only baseline evidence even after an interrupted or
        # failed simulator/audit process; never repair or rewrite its records.
        if "baseline_records_after" not in pipeline:
            try:
                baseline_after = helpers.fingerprint(baseline_files(args.baseline))
                write_json(args.output / "baseline-records-after.json", baseline_after)
                pipeline["baseline_records_after"] = {"manifest": "baseline-records-after.json",
                    "files": len(baseline_after["files"]), "combined_sha256": baseline_after["combined_sha256"]}
                pipeline["baseline_record_changes"] = helpers.changed_inputs(baseline_before,
                                                                             baseline_files(args.baseline))
            except (OSError, ValueError, TypeError, KeyError) as error:
                pipeline["baseline_finalization_error"] = f"{type(error).__name__}: {error}"
        if "final_input_changes" not in pipeline:
            try:
                pipeline["final_input_changes"] = helpers.changed_inputs(before, current_files(args))
            except (OSError, ValueError, TypeError, KeyError) as error:
                pipeline["input_finalization_error"] = f"{type(error).__name__}: {error}"
        pipeline.update(finished_at=timestamp(), wall_seconds=time.monotonic() - started)
        write_json(args.output / "pipeline.json", pipeline)
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    print(json.dumps({"completed": True, "pipeline": str(args.output / "pipeline.json"),
                     "comparison_report": str(args.output / "comparison-report.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, TypeError, KeyError) as error:
        raise SystemExit(f"Fixed-candidate-2 evaluation failed: {error}") from error
