#!/usr/bin/env python3
"""Run the predeclared M1 calibration and sealed paired test, preserving failures.

Calibration selects one fixed candidate only. Every declared learned method is
evaluated regardless of its calibration score. Sources and inputs are immutable
from the first calibration process through the final prediction/physical audit.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import signal
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np

import evaluate_online as helpers
from summarize_evaluation import INITIAL_TOLERANCE, audit_method, paired, require

PROJECT = helpers.PROJECT
sys.path.insert(0, str(PROJECT / "src"))
from g1_embodiedskills.io import sha256, write_json


def timestamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_config(config):
    expected = {"calibration_seed_base": 910000, "calibration_repeats": 40,
        "test_seed_base": 911000, "test_repeats": 100, "num_envs": 8,
        "seconds": 24, "position_jitter_m": .02, "amplitude_m": .025,
        "execution_steps": 10, "training_repetitions": 1}
    for key, value in expected.items():
        require(config.get(key) == value and not isinstance(config.get(key), bool),
                f"Predeclared protocol requires {key}={value}")
    require(config.get("models") == [
        {"id": "full-d", "kind": "full_d"}, {"id": "full-g", "kind": "full_g"},
        {"id": "no-future-loss-g", "kind": "no_future_g"}, {"id": "sa-critic", "kind": "sa_critic"}],
        "The four learned methods must not be selected using calibration outcomes")
    return config


def phase_policy(archive, config):
    with np.load(archive, allow_pickle=False) as raw:
        anchors = np.asarray(raw["anchor"])
        outcomes = np.asarray(raw["success"], dtype=np.float64)
        returns = np.asarray(raw["return"], dtype=np.float64)
        seeds = np.asarray(raw["seed"])
    require(len(anchors) == 140 and len(set(seeds.tolist())) == 28,
            "Phase lookup must use the original 140-row/28-episode training split only")
    return_mean, return_std = float(returns.mean()), max(float(returns.std()), 1e-6)
    utility = (returns-return_mean)/return_std + 4.*outcomes
    rows, indices = [], []
    for anchor in config["phase_anchor_steps"]:
        values = utility[anchors == anchor]
        require(values.shape == (28, 5), f"Phase anchor {anchor} must contain 28 training rows")
        means = values.mean(0)
        best = np.flatnonzero(means == means.max()).tolist()
        selected = 0 if 0 in best else min(best)
        rows.append({"anchor_step": anchor, "training_rows": len(values),
                     "mean_utility": means.tolist(), "selected": selected})
        indices.append(selected)
    return {"schema": "g1_embodiedskills.phase_only_policy.v1",
        "bounds_seconds": config["phase_bounds_seconds"], "indices": indices,
        "training_archive_sha256": sha256(archive), "training_seeds": sorted(set(seeds.tolist())),
        "return_mean": return_mean, "return_std": return_std, "training_rows": rows,
        "online_inputs": ["elapsed_seconds"], "future_images_used": False,
        "training_objective": "mean(training_return_standard + 4*training_success) per phase/index"}


def methods_for_calibration(config):
    return ([{"id": f"fixed-{index}", "method": "fixed", "fixed_index": index} for index in range(5)]
        + [{"id": "phase-only", "method": "phase"}]
        + [{"id": row["id"], "method": "selector", "kind": row["kind"]} for row in config["models"]])


def methods_for_test(config, fixed_index):
    result = [{"id": "direct", "method": "direct"}, {"id": "fixed-2", "method": "fixed2"}]
    if fixed_index not in (0, 2):
        result.append({"id": "fixed-best", "method": "fixed", "fixed_index": fixed_index})
    result += [{"id": "phase-only", "method": "phase"},
               {"id": "es-legacy", "method": "selector", "kind": "legacy"}]
    result += [{"id": row["id"], "method": "selector", "kind": row["kind"]} for row in config["models"]]
    return result


def checkpoint_for(args, method):
    return args.selector if method.get("kind") == "legacy" else args.training/method["id"]/"selector.pt"


def command_for(args, config, method, destination, seed_base, repeats):
    command = ["bash", str(args.isaac_python), str(PROJECT/"scripts/run_sim.py"),
        "--sonic", str(args.sonic), "--reference", str(args.reference), "--asset", str(args.asset),
        "--output", str(destination), "--mode", "closed_loop", "--method", method["method"],
        "--num-envs", str(config["num_envs"]), "--repeats", str(repeats),
        "--seconds", str(config["seconds"]), "--seed-base", str(seed_base),
        "--position-jitter", str(config["position_jitter_m"]), "--amplitude-m", str(config["amplitude_m"]),
        "--embodiedskills-root", str(args.embodiedskills_root), "--cameras", "--record-video"]
    if method["method"] == "selector":
        command += ["--selector", str(checkpoint_for(args, method)), "--selector-kind", method["kind"]]
    elif method["method"] == "fixed":
        command += ["--fixed-candidate-index", str(method["fixed_index"])]
    elif method["method"] == "phase":
        command += ["--phase-policy", str(args.output/"phase-policy.json")]
    return command


def frozen_files(args, config):
    files = {**helpers.source_files(args, args.configuration), **helpers.input_files(args),
             "training_archive": args.train_archive, "phase_policy": args.output/"phase-policy.json",
             "training_configuration": PROJECT/"configs/m1-training.json"}
    for row in config["models"]:
        files[f"checkpoint/{row['id']}"] = args.training/row["id"]/"selector.pt"
    return files


def legacy_files():
    files = {}
    for folder in ("stage4", "stage5", "stage6"):
        root = PROJECT/"outputs"/folder
        for path in sorted(root.rglob("*")):
            if path.is_file():
                files[str(path.relative_to(PROJECT))] = path
    return files


def input_changes(args, config, before, files):
    # Re-scan to detect newly introduced source files as well as edits/deletions
    # among the originally frozen dependency inventory.
    current = {**files, **frozen_files(args, config)}
    return helpers.changed_inputs(before, current)


def verify_paired(runs, reference_id):
    baseline = runs[reference_id]
    errors = {}
    for name, run in runs.items():
        require(run["pairing"] == baseline["pairing"], f"{name}: paired physical/proposal conditions differ")
        require(set(run["episodes"]) == set(baseline["episodes"]), f"{name}: paired seeds differ")
        maxima = {}
        for seed in sorted(baseline["episodes"]):
            expected, actual = baseline["initial"][seed], run["initial"][seed]
            require(set(expected) == set(actual), f"{name}/{seed}: initial history fields differ")
            for key in expected:
                left, right = np.asarray(expected[key]), np.asarray(actual[key])
                require(left.shape == right.shape, f"{name}/{seed}/{key}: initial shape mismatch")
                delta = float(np.max(np.abs(left.astype(float)-right.astype(float)), initial=0))
                maxima[key] = max(maxima.get(key, 0.), delta)
                tolerance = 0. if key == "env_origin" else INITIAL_TOLERANCE
                require(delta <= tolerance, f"{name}/{seed}/{key}: initial maximum difference {delta}")
        errors[name] = maxima
    return errors


def exact_paired(left, right, bootstrap_seed, repetitions):
    seeds = sorted(left)
    observation = paired(left, right)
    wins, losses = observation["rescues"], observation["harms"]
    discordant = wins+losses
    p_value = (min(1., 2.*sum(math.comb(discordant, i) for i in range(min(wins, losses)+1))/2**discordant)
               if discordant else 1.)
    differences = np.array([float(right[s]["success"])-float(left[s]["success"]) for s in seeds])
    rng = np.random.default_rng(bootstrap_seed)
    distribution = differences[rng.integers(0, len(seeds), size=(repetitions, len(seeds)))].mean(1)*100.
    return {**observation, "success_rate_change_pp": float(differences.mean()*100.),
        "paired_seed_bootstrap_95_percentile_pp": np.quantile(distribution, [.025, .975]).tolist(),
        "bootstrap_repetitions": repetitions, "bootstrap_seed": bootstrap_seed,
        "exact_two_sided_p": p_value, "unit": "one complete held-out initial seed, one training bundle"}


def outcome_summary(run):
    episodes = list(run["episodes"].values())
    distances = [row["final_target_distance_m"] for row in episodes]
    return {**run["summary"], "all_episode_final_distance_mean_m": float(np.mean(distances)),
        "all_episode_final_distance_median_m": float(np.median(distances))}


def audit_run(directory, method, repeats):
    run = audit_method(directory, method["method"], expected_episodes=repeats)
    result = read_json(directory/"result.json")
    identity = result["model_identity"]
    if method["method"] == "selector":
        require(identity.get("selector_kind") == method["kind"],
                f"{method['id']}: wrong actual selector kind")
    return run


def run_phase(args, config, phase, methods, pipeline, resources, before, files):
    seed_base, repeats = config[f"{phase}_seed_base"], config[f"{phase}_repeats"]
    runs = {}
    destination = args.output/phase
    destination.mkdir(exist_ok=False)
    for method in methods:
        row = {"name": f"{phase}/{method['id']}", "policy_id": method["id"], "runtime_method": method["method"],
               "status": "pending", "output": str(destination/method["id"])}
        pipeline["runs"].append(row)
        write_json(args.output/"pipeline.json", pipeline)
        helpers.run_process(command_for(args, config, method, destination/method["id"], seed_base, repeats),
            args.output/f"{phase}-{method['id']}.log", row, pipeline, args.output, resources)
        require(not input_changes(args, config, before, files), "Frozen input/source changed during M1")
        run = audit_run(destination/method["id"], method, repeats)
        if method["method"] == "selector":
            actual_hash = run["summary"]["model_identity"]["checkpoint_sha256"]
            require(actual_hash == sha256(checkpoint_for(args, method)), f"{method['id']}: checkpoint identity changed")
        row.update(status="complete", physical_audit_passed=True,
                   audited_episodes=len(run["episodes"]), decisions=run["summary"]["decisions"])
        runs[method["id"]] = run
        write_json(args.output/"pipeline.json", pipeline)
        # Deliberately print counts/identity only, not sealed outcomes.
        print(f"M1_AUDIT phase={phase} policy={method['id']} episodes={len(run['episodes'])} passed", flush=True)
    reference = "fixed-0" if phase == "calibration" else "direct"
    errors = verify_paired(runs, reference)
    write_json(destination/"paired-audit.json", {"completed": True, "reference": reference,
        "initial_maximum_errors": errors, "seeds": sorted(runs[reference]["episodes"]),
        "physical_outcomes_recomputed": True, "source_changes": input_changes(args, config, before, files)})
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("isaac-python", "sonic", "reference", "asset", "selector", "embodiedskills-root", "training", "train-archive", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--configuration", type=Path, default=PROJECT/"configs/m1-evaluation.json")
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    config = validate_config(read_json(args.configuration))
    require(sha256(args.train_archive) == config["training_archive_sha256"], "Original training split changed")
    require(sha256(args.selector) == config["legacy_checkpoint_sha256"], "Legacy checkpoint changed")
    require(shutil.which("nvidia-smi") is not None, "nvidia-smi is required for resource records")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/"phase-policy.json", phase_policy(args.train_archive, config))
    files = frozen_files(args, config)
    before = helpers.fingerprint(files)
    legacy = legacy_files()
    legacy_before = helpers.fingerprint(legacy)
    write_json(args.output/"legacy-records-before.json", legacy_before)
    pipeline = {"schema": "g1_embodiedskills.m1_evaluation_pipeline.v1", "started_at": timestamp(),
        "completed": False, "status": "running", "configuration": config, "inputs_and_sources": before,
        "runs": [], "resource_interval_seconds": 10, "legacy_records": len(legacy),
        "feedback_policy": config["test_feedback"], "scope": "One-bundle mechanism screen; all declared methods retained"}
    started = time.monotonic()
    write_json(args.output/"pipeline.json", pipeline)

    def interrupt(signum, _frame):
        raise helpers.PipelineInterrupted(f"Parent received signal {signum}")
    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with (args.output/"resources.jsonl").open("x", encoding="utf-8") as resources:
            calibration = run_phase(args, config, "calibration", methods_for_calibration(config),
                                    pipeline, resources, before, files)
            fixed_ranking = sorted(range(5), key=lambda index: (
                -sum(row["success"] for row in calibration[f"fixed-{index}"]["episodes"].values()),
                np.mean([row["final_target_distance_m"] for row in calibration[f"fixed-{index}"]["episodes"].values()]),
                index != 0, index))
            fixed_index = fixed_ranking[0]
            selection = {"schema": "g1_embodiedskills.fixed_validation_selection.v1", "selected_index": fixed_index,
                "ordered_indices": fixed_ranking, "tiebreak": config["fixed_tiebreak"],
                "calibration_seeds": sorted(calibration["fixed-0"]["episodes"]),
                "fixed_candidates": {str(i): outcome_summary(calibration[f"fixed-{i}"]) for i in range(5)},
                "decided_before_test": True}
            write_json(args.output/"fixed-selection.json", selection)
            # Selection is frozen before any sealed-test process is launched.
            files["fixed_validation_selection"] = args.output/"fixed-selection.json"
            before["files"]["fixed_validation_selection"] = helpers.fingerprint({"fixed_validation_selection": files["fixed_validation_selection"]})["files"]["fixed_validation_selection"]
            before["combined_sha256"] = hashlib.sha256(json.dumps(
                {key: value["sha256"] for key, value in before["files"].items()}, sort_keys=True).encode()).hexdigest()
            pipeline.update(test_frozen_at=timestamp(), fixed_candidate_index=fixed_index,
                inputs_and_sources=before, training_and_method_selection_unchanged=True)
            write_json(args.output/"pipeline.json", pipeline)
            tests = run_phase(args, config, "test", methods_for_test(config, fixed_index),
                              pipeline, resources, before, files)
            aliases = {"fixed-best": "direct" if fixed_index == 0 else "fixed-2"} if fixed_index in (0, 2) else {}
            if aliases:
                tests["fixed-best"] = tests[aliases["fixed-best"]]
            comparisons = {"primary_full_g_vs_no_future_loss_g": exact_paired(
                tests["no-future-loss-g"]["episodes"], tests["full-g"]["episodes"],
                config["bootstrap_seed"], config["bootstrap_repetitions"])}
            for left, right in (("full-d", "full-g"), ("sa-critic", "full-d"),
                                ("direct", "es-legacy"), ("fixed-best", "es-legacy"),
                                ("phase-only", "es-legacy"), ("direct", "full-d")):
                comparisons[f"exploratory_{right}_vs_{left}"] = exact_paired(
                    tests[left]["episodes"], tests[right]["episodes"],
                    config["bootstrap_seed"], config["bootstrap_repetitions"])
            report = {"schema": "g1_embodiedskills.m1_paired_mechanism_screen.v1", "completed": True,
                "finished_at": timestamp(), "configuration": config, "fixed_selected": fixed_index,
                "aliases": aliases, "calibration": {name: outcome_summary(run) for name, run in calibration.items()},
                "test": {name: outcome_summary(run) for name, run in tests.items()}, "comparisons": comparisons,
                "per_seed": [{"seed": seed, "methods": {name: run["episodes"][seed] for name, run in tests.items()}}
                    for seed in sorted(tests["direct"]["episodes"])],
                "physical_and_initial_audit": "All methods independently audited; paired-audit.json retained per phase",
                "prediction_recomputation": "Required separately before reporting final stage completion",
                "scope": "One training bundle per method, 100 held-out seeds; exploratory mechanism evidence",
                "limitations": ["NoFutureLoss-G retains training-set future residual scale statistics",
                    "Full-G/NoFutureLoss-G differ only in future losses; SA-Critic also changes representation",
                    "Original 28 training episodes contain few informative anchors",
                    "The same seed is the statistical unit; decisions/ensemble members are not independent trials",
                    "No test feedback used to change methods, sample size or success criteria"]}
            write_json(args.output/"comparison-report.json", report)
            require(not input_changes(args, config, before, files), "Frozen sources/inputs changed before M1 completion")
            legacy_after = helpers.fingerprint(legacy_files())
            write_json(args.output/"legacy-records-after.json", legacy_after)
            require(legacy_before == legacy_after, "A pre-existing experiment/model record changed")
            pipeline.update(completed=True, status="complete", original_records_unchanged=True,
                comparison_report_sha256=sha256(args.output/"comparison-report.json"),
                evaluated_calibration_episodes=sum(len(run["episodes"]) for run in calibration.values()),
                evaluated_test_episodes=sum(len(run["episodes"]) for name, run in tests.items() if name not in aliases))
    except BaseException as error:
        pipeline.update(completed=False, status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        pipeline.update(finished_at=timestamp(), wall_seconds=time.monotonic()-started)
        write_json(args.output/"pipeline.json", pipeline)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    print(json.dumps({"completed": True, "output": str(args.output), "physical_audit_passed": True,
                      "original_records_unchanged": True}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
