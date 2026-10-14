#!/usr/bin/env python3
"""Audit three measured online runs on exactly 30 paired held-out seeds.

No simulation, model inference, fabricated trials or statistical improvement
claims are performed. Missing episodes, invalid decisions or contradictory
physical records fail explicitly rather than disappearing from a denominator.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from g1_embodiedskills.io import sha256, write_json
from g1_embodiedskills.metrics import episode_outcome

EPISODES, STEPS, DT = 30, 1200, .02
INITIAL_TOLERANCE = 1e-6
OUTCOME_FLAGS = ("lifted", "placed", "released", "retreated", "standing", "terminated", "success")
NATIVE_FIELDS = ("qtraj_mjcf", "roots", "root_quaternion_wxyz", "finger_joint_targets")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing required evaluation record: {path}")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream, parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"Nonfinite JSON value {value} in {path}")))


def read_npz(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing required measured archive: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def integer(value, name, low=0, high=None):
    require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)),
            f"{name} must be an integer")
    require(value >= low and (high is None or value <= high), f"{name} is outside its allowed range")
    return int(value)


def number(value, name, low=None):
    require(isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_)) and np.isfinite(value), f"{name} must be finite")
    require(low is None or value >= low, f"{name} is below its allowed minimum")
    return float(value)


def numeric_array(value, shape, name):
    array = np.asarray(value)
    require(array.shape == shape and np.issubdtype(array.dtype, np.number)
            and np.isfinite(array).all(), f"{name}: expected finite numeric shape {shape}, got {array.shape}")
    return array


def digest(arrays):
    """The runtime's dtype/shape-aware native digest, without importing Torch."""
    result = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        result.update(str(array.dtype).encode())
        result.update(str(array.shape).encode())
        result.update(array.tobytes())
    return result.hexdigest()


def close(left, right, label, atol=1e-6):
    require(np.asarray(left).shape == np.asarray(right).shape
            and np.isfinite(np.asarray(left, dtype=float)).all()
            and np.isfinite(np.asarray(right, dtype=float)).all()
            and np.allclose(left, right, atol=atol, rtol=1e-6), f"{label}: recorded and measured values disagree")


def canonical_state(value):
    value = np.asarray(value).copy()
    if value.shape == (13,) and value[3] < 0:
        value[3:7] *= -1
    return value


def check_trace(trace, count, label):
    require("root" in trace and trace["root"].ndim == 3, f"{label}: missing batched physical trace")
    samples = len(trace["root"])
    require(1 <= samples <= STEPS+1, f"{label}: invalid physical sample count")
    widths = {"root": 13, "q": 31, "qdot": 31, "tcp": 3, "cube": 3, "cube_root": 13, "gripper": 2}
    for key, width in widths.items():
        require(key in trace, f"{label}: missing {key}; older traces cannot certify the full object state")
        numeric_array(trace[key], (samples, count, width), f"{label}/{key}")
    numeric_array(trace.get("env_origins"), (count, 3), f"{label}/env_origins")
    times = numeric_array(trace.get("time"), (samples, count), f"{label}/time")
    expected = np.broadcast_to(np.arange(samples)[:, None]*DT, times.shape)
    require(np.allclose(times, expected, rtol=0, atol=2e-5), f"{label}: physical time does not match 50-Hz samples")
    flags = np.asarray(trace.get("terminated"))
    require(flags.shape == (samples, count) and np.all((flags == 0) | (flags == 1)),
            f"{label}: missing or malformed actual termination flags")
    require(not flags[0].any(), f"{label}: nonzero termination flags at reset")
    require(np.array_equal(trace["cube"], trace["cube_root"][..., :3]),
            f"{label}: object position and full object-state records disagree")
    for key in ("root", "cube_root"):
        require(np.allclose(np.linalg.norm(trace[key][..., 3:7], axis=-1), 1., atol=1e-4, rtol=0),
                f"{label}: non-unit WXYZ quaternion in {key}")


def archive_path(directory, relative):
    require(isinstance(relative, str) and bool(relative), "Decision audit_archive must name a saved archive")
    path = (directory / relative).resolve()
    require(path.is_relative_to(directory), f"Audit path escapes its evaluation directory: {relative}")
    return path


def audit_decision(directory, row, trace, method, completed, seed, slot):
    label = f"{method}/seed={seed}/step={row.get('step')}"
    step = integer(row.get("step"), f"{label}/step", high=STEPS-10)
    require(step % 10 == 0 and step <= completed, f"{label}: invalid decision clock")
    require(row.get("seed") == seed and row.get("slot") == slot, f"{label}: decision identity mismatch")
    archive = archive_path(directory, row.get("audit_archive"))
    data = read_npz(archive)
    state = numeric_array(data.get("state"), (1041,), f"{label}/state")
    root = numeric_array(data.get("root_state"), (13,), f"{label}/root_state")
    actions = numeric_array(data.get("actions"), (5, 75, 38), f"{label}/actions")
    require(actions.dtype == np.float32, f"{label}: native neural encoding must be float32")
    require(digest((state,)) == row.get("state_sha256") and digest((root,)) == row.get("root_sha256"),
            f"{label}: state/root digest mismatch")
    physical_root = trace["root"][step, slot].copy()
    physical_root[:3] -= trace["env_origins"][slot]
    close(root, canonical_state(physical_root), f"{label}/physical_root")
    close(state[930:943], root, f"{label}/state_root")
    close(state[943:974], trace["q"][step, slot], f"{label}/state_q")
    close(state[974:1005], trace["qdot"][step, slot], f"{label}/state_qdot")
    close(state[-1], step*DT, f"{label}/state_elapsed")
    executed = integer(row.get("executed_control_steps"), f"{label}/executed_control_steps", high=10)
    require(step+executed <= completed, f"{label}: claimed execution exceeds actual episode endpoint")
    status = row.get("status")
    if status == "proposal_failed":
        require(row.get("selected") is None and row.get("scores") is None and executed == 0,
                f"{label}: failed proposal cannot claim selection or execution")
        feasible = np.asarray(data.get("feasible"))
        require(feasible.shape == (5,) and np.all((feasible == 0) | (feasible == 1)) and not feasible.all(),
                f"{label}: proposal failure lacks its measured feasibility mask")
        return {"step": step, "executed": 0, "selected": None, "complete": False, "archive": str(archive)}
    require(status == "selected" and row.get("method") == method, f"{label}: invalid selected decision record")
    selected = integer(row.get("selected"), f"{label}/selected", high=4)
    require(executed > 0, f"{label}: a selected command has no measured execution")
    saved_selected = np.asarray(data.get("selected"))
    require(saved_selected.shape == () and saved_selected.dtype.kind in "iu" and int(saved_selected) == selected,
            f"{label}: JSON and archive selections disagree")
    native = tuple(numeric_array(data.get(key), shape, f"{label}/{key}") for key, shape in
                   zip(NATIVE_FIELDS, ((75, 29), (75, 3), (75, 4), (75, 2)), strict=True))
    require(np.allclose(np.linalg.norm(native[2], axis=-1), 1., atol=1e-4, rtol=0),
            f"{label}: native root quaternions are not normalized")
    require(len({candidate.tobytes() for candidate in actions}) == 5,
            f"{label}: the five actual candidate encodings are not distinct")
    encoded = np.concatenate(native, axis=-1).astype(np.float32)
    require(np.array_equal(encoded, actions[selected]), f"{label}: selected native command differs from its encoding")
    native_hash = digest(native)
    canonical_hashes = row.get("canonical_native_sha256")
    require(isinstance(canonical_hashes, list) and len(canonical_hashes) == 5
            and native_hash == canonical_hashes[selected] == row.get("selected_native_sha256"),
            f"{label}: selected/canonical/native hashes disagree")
    require(digest((encoded,)) == row.get("encoded_action_sha256") and row.get("encoded_native_match") is True,
            f"{label}: selected encoding hash or runtime identity check failed")
    close(numeric_array(data.get("progress"), (1,), f"{label}/progress"),
          [step/STEPS], f"{label}/elapsed_budget")
    scores = row.get("scores")
    if scores is not None:
        scores = numeric_array(scores, (5,), f"{label}/scores")
        saved_scores = numeric_array(data.get("scores"), (5,), f"{label}/saved_scores")
        require(np.array_equal(scores, saved_scores), f"{label}: actual JSON/NPZ scores differ")
    else:
        require("scores" not in data, f"{label}: null JSON scores contradict actual saved scores")
    if method == "selector":
        head_argmax = integer(row.get("head_argmax"), f"{label}/head_argmax", high=4)
        require(scores is not None and selected == int(np.argmax(scores)) == head_argmax,
                f"{label}: selector choice is not the actual comparison-head argmax")
    if method == "direct":
        require(selected == 0, f"{label}: direct baseline selected a nonzero candidate")
    patch = row.get("patch", {})
    require(patch.get("cursor") == step and patch.get("patched_frames") == 75
            and patch.get("last_execution_preview_frame") == step+55, f"{label}: native patch clock/horizon mismatch")
    close(row.get("time"), step*DT, f"{label}/decision_time")
    return {"step": step, "executed": executed, "selected": selected, "complete": executed == 10,
            "archive": str(archive)}


def validate_latency(result, decisions, label):
    latency = result.get("latency")
    require(isinstance(latency, dict), f"{label}: missing measured latency summaries")
    groups = {(batch, row["step"]) for batch, row in decisions}
    selected = [(batch, row) for batch, row in decisions if row.get("selected") is not None]
    modeled = {(batch, row["step"]) for batch, row in selected if row.get("scores") is not None}
    expected = {"decision": len(groups), "render": len(groups), "proposal": len(decisions),
                "patch": len(selected), "encoder": len(modeled), "selector": len(modeled)}
    for key, count in expected.items():
        summary = latency.get(key, {})
        require(summary.get("count") == count, f"{label}: {key} latency count contradicts actual decision records")
        for field in ("mean_ms", "p50_ms", "p95_ms", "max_ms"):
            if count:
                number(summary.get(field), f"{label}/latency/{key}/{field}", low=0)
            else:
                require(summary.get(field) is None, f"{label}: empty latency series must use null statistics")
        if count:
            require(summary["p50_ms"] <= summary["p95_ms"] <= summary["max_ms"]+1e-8
                    and summary["mean_ms"] <= summary["max_ms"]+1e-8, f"{label}: inconsistent latency summary")
    return latency


def audit_method(directory, method):
    directory = Path(directory).resolve()
    request, result, interface = (read_json(directory / name) for name in ("request.json", "result.json", "interface.json"))
    process = read_json(directory / "process_result.json")
    require(process.get("returncode") == 0 and process.get("result_present") is True,
            f"{method}: evaluation process did not finish normally")
    require(result.get("completed") is True and result.get("mode") == request.get("mode") == "closed_loop",
            f"{method}: run is not a completed online evaluation")
    require(request.get("method") == result.get("method") == method, f"{method}: requested/reported method mismatch")
    require(request.get("repeats") == result.get("requested_episodes") == result.get("counted_episodes") == EPISODES,
            f"{method}: exactly 30 requested and counted episodes are required")
    require(number(request.get("seconds"), f"{method}/seconds") == 24., f"{method}: expected a 24-second budget")
    require(request.get("cameras") is True and result.get("hard_physics_reset") is True,
            f"{method}: real camera observations and full physical reset are required")
    count = integer(request.get("num_envs"), f"{method}/num_envs", low=1)
    seed_base = integer(request.get("seed_base"), f"{method}/seed_base", low=900000)
    jitter = number(request.get("position_jitter"), f"{method}/position_jitter", low=0)
    amplitude = number(request.get("amplitude_m"), f"{method}/amplitude_m", low=0)
    require(jitter <= .04 and 0 < amplitude <= .05, f"{method}: invalid task/proposal amplitude")
    require(result.get("seed_base") == seed_base and result.get("position_jitter_m") == jitter,
            f"{method}: request/result task parameters differ")
    require(result.get("execution_steps") == 10 and result.get("native_preview_frames") == 75
            and result.get("state_dim") == 1041 and result.get("action_dim") == 38,
            f"{method}: unexpected observation/action/horizon contract")
    close(interface.get("control_dt"), DT, f"{method}/control_dt", atol=1e-12)
    require(request.get("instruction") == result.get("instruction"), f"{method}: language instruction mismatch")
    identity = result.get("model_identity", {})
    if method == "selector":
        require(identity.get("family") == "continuous" and identity.get("members") == 3
                and bool(identity.get("comparison_head")) and isinstance(identity.get("checkpoint_sha256"), str)
                and len(identity["checkpoint_sha256"]) == 64,
                "selector: missing actual three-WAM comparison-bundle identity")
    episodes = read_json(directory / "episodes.json")
    require(isinstance(episodes, list) and len(episodes) == EPISODES and episodes == result.get("episodes"),
            f"{method}: episode records missing or result/episodes.json disagree")
    by_seed = {}
    for row in episodes:
        seed = integer(row.get("seed"), f"{method}/episode_seed", low=900000)
        require(seed not in by_seed, f"{method}: duplicate episode seed {seed}")
        by_seed[seed] = row
    require(set(by_seed) == set(range(seed_base, seed_base+EPISODES)),
            f"{method}: missing/extra seeds; expected all 30 consecutive requested seeds")
    expected_batches = {f"batch-{i:04d}" for i in range((EPISODES+count-1)//count)}
    require({p.name for p in directory.glob("batch-*") if p.is_dir()} == expected_batches,
            f"{method}: missing or unexpected episode batches")
    checked, initial, selections, all_decisions, audited_archives = {}, {}, [], [], set()
    for batch_id in range(len(expected_batches)):
        batch_dir = directory / f"batch-{batch_id:04d}"
        trace = read_npz(batch_dir / "trace.npz")
        check_trace(trace, count, f"{method}/{batch_dir.name}")
        rows = read_json(batch_dir / "decisions.json")
        require(isinstance(rows, list), f"{method}: decisions.json must be an array")
        active_count = min(count, EPISODES-batch_id*count)
        expected_seed_slot = {seed_base+batch_id*count+slot: slot for slot in range(active_count)}
        grouped = {seed: [] for seed in expected_seed_slot}
        for row in rows:
            require(isinstance(row, dict) and row.get("seed") in grouped
                    and row.get("slot") == expected_seed_slot[row["seed"]],
                    f"{method}: unknown seed, duplicate/padded slot or batch mismatch in decisions")
            grouped[row["seed"]].append(row)
            all_decisions.append((batch_id, row))
        for seed, slot in expected_seed_slot.items():
            row = by_seed[seed]
            episode_index = seed-seed_base
            require(row.get("episode") == episode_index and row.get("slot") == slot and row.get("batch") == batch_id
                    and row.get("method") == method, f"{method}/seed={seed}: episode batch/slot identity mismatch")
            require(read_json(batch_dir / f"episode-slot-{slot}.json") == row,
                    f"{method}/seed={seed}: per-episode record differs from aggregate")
            completed = integer(row.get("completed_steps"), f"{method}/seed={seed}/completed_steps", high=STEPS)
            require(row.get("requested_steps") == STEPS and completed < len(trace["time"]),
                    f"{method}/seed={seed}: missing actual physical endpoint")
            local_trace = {key: value[:completed+1] for key, value in trace.items() if key != "env_origins"}
            terminal_flags = np.asarray(local_trace["terminated"])[:, slot].astype(bool)
            terminated = bool(terminal_flags.any())
            if terminated:
                require(int(np.flatnonzero(terminal_flags)[0]) == completed,
                        f"{method}/seed={seed}: episode continued beyond its first physical termination")
            physical = episode_outcome(local_trace, slot=slot, origin=trace["env_origins"][slot],
                                       terminated=terminated, control_dt=DT)
            budget_complete = completed == STEPS
            failure = row.get("failure")
            require(failure is None or isinstance(failure, str) and bool(failure),
                    f"{method}/seed={seed}: malformed failure reason")
            if not budget_complete or terminated:
                require(failure is not None, f"{method}/seed={seed}: early/terminated episode has no recorded failure")
            physical["success"] = bool(physical["success"] and failure is None and budget_complete)
            for key in OUTCOME_FLAGS:
                require(isinstance(row.get(key), bool) and row[key] == physical[key],
                        f"{method}/seed={seed}: logged {key} contradicts its actual physical trace")
            for key in ("maximum_cube_height_m", "final_cube_position", "final_target_distance_m", "return_value"):
                close(row.get(key), physical[key], f"{method}/seed={seed}/{key}")
            close(row.get("duration"), completed*DT, f"{method}/seed={seed}/duration", atol=2e-5)
            decisions = grouped[seed]
            require(bool(decisions), f"{method}/seed={seed}: no decision or explicit proposal-failure record")
            require([v["step"] for v in decisions] == sorted(v["step"] for v in decisions),
                    f"{method}/seed={seed}: decision records are out of order")
            cursor, choices, full, partial, rejected = 0, [], 0, 0, 0
            for index, decision in enumerate(decisions):
                audit = audit_decision(directory, decision, trace, method, completed, seed, slot)
                require(audit["archive"] not in audited_archives, f"{method}: a decision archive was reused")
                audited_archives.add(audit["archive"])
                require(audit["step"] == cursor, f"{method}/seed={seed}: gap/overlap in delivered control coverage")
                cursor += audit["executed"]
                if audit["selected"] is None:
                    rejected += 1
                    require(index == len(decisions)-1 and failure is not None,
                            f"{method}/seed={seed}: proposal failure did not end this episode")
                else:
                    choices.append(audit["selected"])
                    full += int(audit["complete"])
                    partial += int(not audit["complete"])
                    if not audit["complete"]:
                        require(index == len(decisions)-1 and terminated,
                                f"{method}/seed={seed}: partial execution is not an actual terminal decision")
            require(cursor == completed, f"{method}/seed={seed}: claimed execution does not cover actual control steps")
            require(row.get("decisions") == len(choices), f"{method}/seed={seed}: selection count mismatch")
            switches = sum(value != 0 for value in choices)
            changes = sum(a != b for a, b in zip(choices, choices[1:]))
            require(row.get("switches_from_direct") == switches and row.get("index_changes") == changes,
                    f"{method}/seed={seed}: recorded switching metrics are inconsistent")
            selections.extend(choices)
            initial[seed] = {key: canonical_state(trace[key][0, slot]) if key in ("root", "cube_root")
                             else trace[key][0, slot].copy() for key in ("root", "q", "qdot", "cube_root")}
            initial[seed]["env_origin"] = trace["env_origins"][slot].copy()
            checked[seed] = {"seed": seed, "episode": episode_index, "batch": batch_id, "slot": slot,
                "completed_steps": completed, "requested_steps": STEPS, "budget_complete": budget_complete,
                "failure": failure, **physical, "decisions": len(choices), "complete_decisions": full,
                "partial_decisions": partial, "proposal_failures": rejected, "executed_control_steps": cursor,
                "switches_from_direct": switches, "index_changes": changes,
                "selected_histogram": np.bincount(choices, minlength=5).tolist()}
    histogram = np.bincount(selections, minlength=5).tolist()
    require(result.get("decisions") == len(selections) and result.get("selected_histogram") == histogram
            and result.get("switches_from_direct") == sum(v != 0 for v in selections),
            f"{method}: aggregate selection counts contradict audited decisions")
    success = sum(row["success"] for row in checked.values())
    close(result.get("success_rate"), success/EPISODES, f"{method}/success_rate")
    gpu = result.get("gpu_peak")
    if gpu is not None:
        allocated = integer(gpu.get("allocated_bytes"), f"{method}/gpu_allocated")
        reserved = integer(gpu.get("reserved_bytes"), f"{method}/gpu_reserved")
        require(reserved >= allocated and isinstance(gpu.get("scope"), str), f"{method}: malformed GPU peak record")
    summary = {"episodes": EPISODES, "counts": {key: sum(row[key] for row in checked.values()) for key in OUTCOME_FLAGS},
        "success_rate": success/EPISODES, "complete_budget_episodes": sum(row["budget_complete"] for row in checked.values()),
        "failed_episodes": sum(row["failure"] is not None for row in checked.values()),
        "executed_control_steps": sum(row["executed_control_steps"] for row in checked.values()),
        "maximum_possible_control_steps": EPISODES*STEPS,
        "decisions": len(selections), "complete_decisions": sum(row["complete_decisions"] for row in checked.values()),
        "partial_decisions": sum(row["partial_decisions"] for row in checked.values()),
        "proposal_failures": sum(row["proposal_failures"] for row in checked.values()),
        "switches_from_direct": sum(value != 0 for value in selections),
        "index_changes": sum(row["index_changes"] for row in checked.values()), "selected_histogram": histogram,
        "latency": validate_latency(result, all_decisions, method), "gpu_peak": gpu,
        "gpu_limitation": "Reported PyTorch allocator peaks exclude RTX and other non-PyTorch allocations",
        "wall_seconds": number(result.get("wall_seconds"), f"{method}/wall_seconds", low=0),
        "model_identity": identity, "audit_archives_checked": len(audited_archives),
        "provenance": {"request_sha256": sha256(directory/"request.json"),
            "result_sha256": sha256(directory/"result.json"), "interface_sha256": sha256(directory/"interface.json")}}
    pairing = {"num_envs": count, "seed_base": seed_base, "seconds": 24., "position_jitter_m": jitter,
               "amplitude_m": amplitude, "instruction": result["instruction"],
               "asset_sha256": interface.get("asset_sha256"), "reference_sha256": interface.get("reference_sha256")}
    require(all(isinstance(pairing[key], str) and len(pairing[key]) == 64
                for key in ("asset_sha256", "reference_sha256")), f"{method}: missing source asset/reference hashes")
    return {"directory": str(directory), "summary": summary, "episodes": checked, "initial": initial, "pairing": pairing}


def paired(left, right):
    """Descriptive observed differences, without independence/p-value claims."""
    seeds = sorted(left)
    rescues = [seed for seed in seeds if not left[seed]["success"] and right[seed]["success"]]
    harms = [seed for seed in seeds if left[seed]["success"] and not right[seed]["success"]]
    return {"rescues": len(rescues), "harms": len(harms), "net_success_count_change": len(rescues)-len(harms),
            "rescued_seeds": rescues, "harmed_seeds": harms,
            "both_succeed": sum(left[s]["success"] and right[s]["success"] for s in seeds),
            "both_fail": sum(not left[s]["success"] and not right[s]["success"] for s in seeds)}


def summarize(direct, random, selector):
    runs = {method: audit_method(path, method) for method, path in
            (("direct", direct), ("random", random), ("selector", selector))}
    baseline = runs["direct"]
    for method, run in runs.items():
        require(run["pairing"] == baseline["pairing"], f"{method}: paired run settings/asset/reference differ from direct")
        require(set(run["episodes"]) == set(baseline["episodes"]), f"{method}: missing paired seeds")
    initial_errors, per_seed = {}, []
    for seed in sorted(baseline["episodes"]):
        per_seed.append({"seed": seed, "methods": {m: run["episodes"][seed] for m, run in runs.items()}})
        initial_errors[str(seed)] = {}
        for method in ("random", "selector"):
            reference, actual = baseline["initial"][seed], runs[method]["initial"][seed]
            require(np.array_equal(reference["env_origin"], actual["env_origin"]),
                    f"{method}/seed={seed}: environment origins differ")
            errors = {}
            for key in ("root", "q", "qdot", "cube_root"):
                delta = np.abs(actual[key]-reference[key])
                errors[key] = {"max_abs": float(delta.max()), "rms": float(np.sqrt(np.mean(delta**2)))}
                if key in ("root", "cube_root"):
                    errors[key].update({name: float(delta[part].max()) for name, part in
                        (("position_m", slice(0, 3)), ("quaternion_components", slice(3, 7)),
                         ("linear_velocity_m_s", slice(7, 10)), ("angular_velocity_rad_s", slice(10, 13)))})
                require(errors[key]["max_abs"] <= INITIAL_TOLERANCE,
                        f"{method}/seed={seed}: paired initial {key} differs by {errors[key]['max_abs']}; errors={errors}")
            initial_errors[str(seed)][method] = errors
    return {"schema": "g1_embodiedskills.paired_online_engineering_acceptance.v1",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "30-sample measured online engineering acceptance; no statistical improvement claim",
        "pairing": {**baseline["pairing"], "episodes": EPISODES, "seeds": sorted(baseline["episodes"]),
            "same_seed_slot_batch": True, "initial_state_tolerance": INITIAL_TOLERANCE,
            "initial_state_fields": ["root13", "q31", "qdot31", "cube_root13"],
            "initial_state_errors_vs_direct": initial_errors},
        "methods": {m: {"directory": run["directory"], **run["summary"]} for m, run in runs.items()},
        "paired_observed_outcomes": {
            "random_vs_direct": paired(baseline["episodes"], runs["random"]["episodes"]),
            "selector_vs_direct": paired(baseline["episodes"], runs["selector"]["episodes"]),
            "selector_vs_random": paired(runs["random"]["episodes"], runs["selector"]["episodes"])},
        "per_seed": per_seed,
        "audit": {"physical_outcomes_recomputed": True, "full_object_pose_and_velocity_required": True,
            "initial_states_compared": True, "actual_control_step_coverage_verified": True,
            "all_selected_native_codecs_and_hashes_verified": True, "selector_actual_score_argmax_verified": True,
            "missing_episodes_are_errors": True, "padding_slots_counted": False,
            "early_ended_or_failed_episodes_can_succeed": False,
            "latency": "Actual runtime summaries, checked against decision counts; raw timing distributions are not all retained",
            "unselected_native_hashes": "Only the selected native command is saved; unselected hashes cannot be independently reconstructed"},
        "limitations": ["30 seeds establish engineering operation; no statistical superiority or significance is claimed",
            "Paired refers to verified initial states/settings, not identical states after different policy choices",
            "Rescues/harms are observed separate closed-loop episodes, not restored-state counterfactuals",
            "Quaternion signs are canonicalized before componentwise initial-state comparison",
            "Full physics traces and decision audits remain local; reported GPU allocator peaks exclude RTX"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--random", type=Path, required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="JSON report; never a raw evaluation record")
    args = parser.parse_args()
    require(args.output.suffix.lower() == ".json", "--output must name a JSON report")
    require(not any(args.output.resolve().is_relative_to(directory.resolve())
                    for directory in (args.direct, args.random, args.selector)),
            "Save the summary outside the three raw evaluation directories")
    report = summarize(args.direct, args.random, args.selector)
    write_json(args.output, report)
    print(json.dumps({"output": str(args.output.resolve()), "scope": report["scope"],
        "success_counts": {m: value["counts"]["success"] for m, value in report["methods"].items()},
        "paired": report["paired_observed_outcomes"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, TypeError, KeyError, OSError) as error:
        raise SystemExit(f"Evaluation audit failed: {error}") from error
