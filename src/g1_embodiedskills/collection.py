"""Measured five-branch archives from same-slot, whole-prefix replays.

Each physical slot owns one (seed, anchor) group for five complete rollouts.
There is no simulator-state snapshot restoration and no cross-slot equivalence
assumption. Replayed anchors are accepted only after componentwise prefix and
candidate checks. Failed groups remain in the audit files, never as fake rows.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .io import sha256, write_json


def _array(value):
    value = value.torch if hasattr(value, "torch") else value
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def _save_npz(path, **fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **fields)
    temporary.replace(path)


def _local_estimates(estimates, origin):
    result = copy.deepcopy(estimates)
    for value in result.values():
        for key in ("position", "surface_position"):
            if value.get(key) is not None:
                value[key] = np.asarray(value[key], dtype=float) - origin
    return result


def _servo_state(servo):
    return {"correction": servo.correction.copy(), "cube_offset": servo.cube_offset.copy(),
        "tray_offset": servo.tray_offset.copy(), "last_estimate_time": copy.deepcopy(servo.last_estimate_time)}


def _restore_servo(servo, state):
    for key in ("correction", "cube_offset", "tray_offset"):
        setattr(servo, key, np.asarray(state[key], dtype=float).copy())
    servo.last_estimate_time = copy.deepcopy(state["last_estimate_time"])


def _native_digest(native):
    digest = hashlib.sha256()
    for value in native:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _snapshot(measured, slot, origin, observations, servo, include_physical_object=False):
    root = _array(measured["root"])[slot]
    root[:3] -= origin
    if root[3] < 0:  # Equivalent root quaternions share a stable sign.
        root[3:7] *= -1
    snapshot = {
        "root": root, "q": _array(measured["q"])[slot],
        "qdot": _array(measured["qdot"])[slot],
        "actor_obs": _array(observations["actor_obs"])[slot].reshape(-1),
        "tokenizer": _array(observations["tokenizer"])[slot].reshape(-1),
        "servo_correction": servo.correction.copy(),
        "cube_offset": servo.cube_offset.copy(), "tray_offset": servo.tray_offset.copy(),
    }
    if include_physical_object:
        cube_root = _array(measured["cube_root"])[slot]
        cube_root[:3] -= origin
        if cube_root[3] < 0:
            cube_root[3:7] *= -1
        snapshot["cube_root"] = cube_root
    return snapshot


def _tolerance(settings, key):
    value = settings.get(key, settings.get("default", 1e-3)) if isinstance(settings, dict) else settings
    value = float(value)
    if value < 0 or not np.isfinite(value):
        raise ValueError(f"Invalid prefix tolerance for {key}")
    return value


def prefix_comparison(reference, actual, tolerance=1e-3):
    """Return an auditable componentwise gate; no hidden relative tolerance."""
    fields = {}
    for key, baseline in reference.items():
        limit = _tolerance(tolerance, key)
        value = np.asarray(actual[key])
        if value.shape != baseline.shape or not np.isfinite(value).all():
            fields[key] = {"passed": False, "shape_mismatch_or_nonfinite": True}
            continue
        delta = np.abs(value - baseline)
        fields[key] = {"max_abs": float(delta.max(initial=0)), "rms": float(np.sqrt(np.mean(delta**2))),
            "tolerance": limit, "exceeding_components": int(np.sum(delta > limit)), "passed": bool(np.all(delta <= limit))}
    return {"passed": all(row["passed"] for row in fields.values()), "fields": fields,
        "meaning": "Tolerance-checked replay agreement; not bit-exact PhysX restoration"}


def state_from_snapshot(snapshot, elapsed):
    """Deployment-visible proprioception, planner memory and elapsed time only."""
    names = ("actor_obs", "root", "q", "qdot", "servo_correction", "cube_offset", "tray_offset")
    state = np.concatenate([np.asarray(snapshot[key]).reshape(-1) for key in names] + [np.asarray([elapsed])]).astype(np.float32)
    if not np.isfinite(state).all():
        raise ValueError("Nonfinite deployment-visible state")
    return state


def reference_utility(action, direct_action):
    """Negative physical-unit-normalized RMS distance to the frozen direct plan.

    Scale vector: hinge 1 rad, root translation .1 m, quaternion components 1,
    finger travel .035 m. This is a measured planner-distance auxiliary target,
    not a claim that the direct planner is an optimal success expert.
    """
    action, direct = np.asarray(action), np.asarray(direct_action)
    if action.shape != direct.shape or action.ndim != 2 or action.shape[1] != 38:
        raise ValueError("Expected matching native action encodings [H,38]")
    scale = np.r_[np.ones(29), np.full(3, .1), np.ones(4), np.full(2, .035)]
    return -float(np.sqrt(np.mean(((action - direct) / scale)**2)))


def _fingers(schedule, cursor, frames=75):
    times = (cursor + np.arange(frames)) / 50.
    targets = np.full(frames, .035, dtype=float)
    for event in schedule:
        targets[times >= float(event["time"])] = float(event["joint_target_m"])
    return np.repeat(targets[:, None], 2, axis=1)


def _propose(model, q, roots, quaternions, fingers, amplitude, max_halvings):
    from .candidates import generate_candidates
    attempts = []
    for level in range(max_halvings + 1):
        actual_amplitude = amplitude * .5**level
        candidates = generate_candidates(model, q, roots, quaternions, fingers, amplitude_m=actual_amplitude)
        attempts.append({"amplitude_m": actual_amplitude, "feasible": candidates.feasible.tolist(), "diagnostics": candidates.diagnostics})
        if bool(candidates.feasible.all()):
            return candidates, attempts
    return candidates, attempts


def _raw_row(group, task_name, instruction, total_steps):
    return {
        "rgb": group["rgb"], "future_rgb": np.stack(group["future_rgb"]),
        "instruction": instruction, "state": group["state"],
        "progress": np.asarray([group["anchor_step"] / total_steps], dtype=np.float32),
        "actions": np.stack(group["actions"]), "mask": np.ones((5, 75), dtype=np.float32),
        "success": np.asarray([row["outcome"]["success"] for row in group["runs"]], dtype=np.float32),
        "return": np.asarray([row["outcome"]["return_value"] for row in group["runs"]], dtype=np.float32),
        "expert_utility": np.asarray(group["expert_utility"], dtype=np.float32),
        "task": task_name, "seed": int(group["seed"]), "anchor": int(group["anchor_step"]),
    }


def _group_report(group):
    return {key: group[key] for key in ("id", "seed", "anchor_step", "slot", "rejections", "runs")}


def collect_branches(wrapper, policy, request):
    """Collect raw NPZ archives using actual simulator rollouts and branch labels.

    Config requires ``anchors=[{'seed':int,'anchor_step':int}, ...]``. Anchors
    must align to the ten-step servo period. ``prefix_tolerance`` may be a scalar
    or per-field mapping with a ``default``; default is 1e-3. Collection is
    intentionally strict: reset discrepancies reject a whole group rather than
    weakening its gate or relabelling different states as a shared anchor.

    Integration prerequisites: cameras enabled; independent MotionLib slot per
    environment; reset_nominal respects env.episode_seeds; ReferenceServo accepts
    model=; render_observations returns a slot-keyed observation dictionary.
    """
    from .candidates import encode_native
    from .kinematics import G1Kinematics
    from .learning import validate_archive
    from .metrics import episode_outcome
    from .native_reference import NativeReferencePatcher, install_reference_chunk
    from .perception import render_observations
    from .servo import ReferenceServo

    env = wrapper.env
    if not request.cameras:
        raise ValueError("Branch collection requires actual RGB-D head cameras")
    cfg = json.loads(Path(request.collection_config).read_text(encoding="utf-8"))
    provenance = {
        "collection_config_sha256": sha256(request.collection_config),
        "robot_asset_sha256": sha256(request.asset),
        "reference_sha256": {name: sha256(Path(request.reference) / name)
            for name in ("motion.pkl", "planned.npz", "reference.json", "task.json")},
        "adapter_source_sha256": {name: sha256(Path(__file__).with_name(name))
            for name in ("collection.py", "candidates.py", "simulation.py", "servo.py", "native_reference.py", "metrics.py", "perception.py")},
    }
    requested = cfg.get("anchors", [])
    if not requested:
        raise ValueError("collection_config needs at least one (seed,anchor_step)")
    if int(cfg.get("execution_steps", 10)) != 10:
        raise ValueError("This collection protocol executes ten control steps per candidate")
    if not np.isclose(float(env.step_dt), .02):
        raise ValueError("Collection requires the audited 50-Hz control clock")
    total_steps = round(float(request.seconds) / env.step_dt)
    anchors = []
    identities = set()
    for item in requested:
        seed, step = item["seed"], item["anchor_step"]
        if isinstance(seed, bool) or isinstance(step, bool) or not isinstance(seed, int) or not isinstance(step, int) or seed < 0 or step < 0:
            raise ValueError("Anchor seed and step must be nonnegative integers")
        if step % 10 or step + 10 >= total_steps:
            raise ValueError("Anchors must be ten-step aligned with enough budget for future RGB and continuation")
        if (seed, step) in identities:
            raise ValueError("Duplicate seed/anchor identity in collection config")
        identities.add((seed, step))
        anchors.append({"seed": seed, "anchor_step": step})
    amplitude = float(cfg.get("amplitude_m", .012))
    if not 0 < amplitude <= .05:
        raise ValueError("amplitude_m must lie in (0,.05] meters")
    tolerance = cfg.get("prefix_tolerance", 1e-3)
    for key in ("root", "q", "qdot", "cube_root", "actor_obs", "tokenizer", "servo_correction", "cube_offset", "tray_offset", "actions"):
        _tolerance(tolerance, key)
    request.position_jitter = float(cfg.get("position_jitter", getattr(request, "position_jitter", 0.)))
    if not 0 <= request.position_jitter <= .04:
        raise ValueError("position_jitter must lie in [0,.04] meters")
    task_path = request.task or Path(request.reference) / "task.json"
    task = json.loads(Path(task_path).read_text(encoding="utf-8"))
    planned = dict(np.load(Path(request.reference) / "planned.npz", allow_pickle=False))
    schedule = task["gripper_schedule"]
    model = G1Kinematics(request.sonic)
    torch.set_num_threads(1)
    cmd = env.command_manager.get_term("motion")
    lib = cmd.motion_lib
    if int(lib._num_motions) < env.num_envs:
        raise ValueError("Need independent loaded motion timeline slots for every physical environment")
    original_fields = {name: value.clone() for name, value in NativeReferencePatcher(lib).fields.items()}
    origins = _array(env.scene.env_origins)
    output = Path(request.output) / "collection"
    output.mkdir(parents=True, exist_ok=True)
    task_name = str(cfg.get("task_name", "g1_pick_place"))
    instruction = str(cfg.get("instruction", "Place the red block into the green tray."))
    if not task_name.strip() or not instruction.strip():
        raise ValueError("task_name and instruction must be nonempty")
    max_halvings = cfg.get("max_amplitude_halvings", 5)
    if isinstance(max_halvings, bool) or not isinstance(max_halvings, int) or not 0 <= max_halvings <= 8:
        raise ValueError("max_amplitude_halvings must be an integer in [0,8]")
    all_rows, reports = [], []
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        if previous.get("provenance") != provenance:
            raise ValueError("Cannot resume a collection with changed source, assets, reference or configuration")
        reports.extend(previous.get("groups", []))
        archive = output / "raw.npz"
        if archive.is_file():
            with np.load(archive, allow_pickle=False) as source:
                all_rows.extend({key: source[key][index] for key in source.files} for index in range(len(source["task"])))
    completed_identities = {(int(row["seed"]), int(row["anchor"])) for row in all_rows}
    anchors = [item for item in anchors if (item["seed"], item["anchor_step"]) not in completed_identities]
    started = time.monotonic()
    first_batch = len(list(output.glob("batch-*")))

    for offset in range(0, len(anchors), env.num_envs):
        batch_id = first_batch + offset // env.num_envs
        batch_dir = output / f"batch-{batch_id:05d}"
        batch_dir.mkdir(parents=True, exist_ok=False)
        active = anchors[offset:offset + env.num_envs]
        if cfg.get("synchronize_batch_anchors", False) and len({item["anchor_step"] for item in active}) != 1:
            raise ValueError("Synchronized collection requires one common anchor time per physical batch")
        groups = [{**item, "id": f"seed-{item['seed']}-anchor-{item['anchor_step']}", "slot": slot,
            "rejections": [], "runs": [], "future_rgb": [None]*5, "actions": [None]*5,
            "expert_utility": [None]*5} for slot, item in enumerate(active)]
        baseline_actions, baseline_grippers, baseline_patches = [], [], {}
        baseline_initial = {}
        for candidate_id in range(5):
            pass_dir = batch_dir / f"candidate-{candidate_id}"
            pass_dir.mkdir()
            # Native reference arrays, policy state, simulator state and sensor/
            # observation histories all reset before replaying the common prefix.
            for name, value in original_fields.items():
                getattr(lib, name).copy_(value)
            env.episode_seeds = [item["seed"] for item in active] + [int(active[-1]["seed"])] * (env.num_envs-len(active))
            servos = [ReferenceServo(request.sonic, planned, env.reference, task["geometry"], model=model) for _ in range(env.num_envs)]
            estimates = [None] * env.num_envs
            branch_native = [None] * env.num_envs
            run_rows = [{"candidate": candidate_id, "seed": group["seed"], "anchor_step": group["anchor_step"], "slot": group["slot"], "native_executed": False} for group in groups]
            done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            trace_records, servo_log = [], []
            pass_error = None
            try:
                if cfg.get("hard_physics_reset", False):
                    # Stop invalidates views; reset alone can retain PhysX
                    # contact/solver state. Asset objects survive, while data
                    # and actuator views are recreated by their callbacks.
                    from .simulation import rebuild_physics
                    rebuild_physics(env)
                policy.init_rollout()
                observations = wrapper.process_raw_obs(env.reset_nominal(), True)
                if len(set(cmd.motion_ids.tolist())) != env.num_envs:
                    raise ValueError("Physical slots share one reference timeline after reset")
                trace_records.append(env.capture())
                _save_npz(pass_dir / "initial-observations.npz", actor_obs=_array(observations["actor_obs"]), tokenizer=_array(observations["tokenizer"]))
                for slot, group in enumerate(groups):
                    initial = _snapshot(trace_records[0], slot, origins[slot], observations, servos[slot], include_physical_object=True)
                    if candidate_id == 0:
                        baseline_initial[slot] = initial
                    elif slot in baseline_initial:
                        comparison = prefix_comparison(baseline_initial[slot], initial, tolerance)
                        run_rows[slot]["initial_comparison"] = comparison
                        if not comparison["passed"]:
                            group["rejections"].append(f"candidate {candidate_id}: initial physical/history tolerance exceeded")
                for step in range(total_steps):
                    elapsed = step * env.step_dt
                    sensing = step % 10 == 0 and 3.4 <= elapsed <= 6.6
                    # Later branches replay actual delivered controls before
                    # their anchor. Camera noise never regenerates that prefix.
                    replay_slots = {slot for slot in range(env.num_envs) if candidate_id > 0
                        and (slot >= len(groups) or step < groups[slot]["anchor_step"])}
                    sensing_slots = set(range(env.num_envs)) - replay_slots if sensing else set()
                    current_slots = [slot for slot, group in enumerate(groups) if step == group["anchor_step"]]
                    future_slots = [slot for slot, group in enumerate(groups) if step == group["anchor_step"]+10]
                    # Match online 5-Hz rendering in every branch. This also
                    # controls temporal renderer history at the future frame;
                    # replayed controls are never regenerated from these RGBs.
                    render_slots = set(range(env.num_envs)) if step % 10 == 0 else set()
                    render_slots.update(current_slots + future_slots)
                    images = render_observations(env, sorted(render_slots), refresh=True) if render_slots else {}
                    if sensing:
                        for slot in sensing_slots:
                            estimates[slot] = _local_estimates(images[slot]["estimates"], origins[slot])
                    for slot in future_slots:
                        rgb = np.asarray(images[slot]["rgb"])
                        if rgb.shape != (256, 256, 3) or rgb.dtype != np.uint8:
                            raise ValueError("Training RGB must be actual uint8[256,256,3]")
                        groups[slot]["future_rgb"][candidate_id] = rgb.copy()
                    if step % 10 == 0:
                        measured = env.capture()
                        for slot, servo in enumerate(servos):
                            candidates = None
                            if bool(done[slot]):
                                if slot < len(groups) and step == groups[slot]["anchor_step"]:
                                    groups[slot]["rejections"].append(f"candidate {candidate_id}: terminated before anchor")
                                continue
                            is_anchor = slot < len(groups) and step == groups[slot]["anchor_step"]
                            cursor = int(cmd.time_steps[slot] + cmd.motion_start_time_steps[slot])
                            if cursor != step:
                                raise ValueError(f"Control step/reference cursor mismatch in slot {slot}: {step} vs {cursor}")
                            if slot in replay_slots:
                                cached = baseline_patches.get((step, slot))
                                if cached is None:
                                    raise ValueError("Baseline delivered native prefix is unavailable")
                                q, roots, quaternions, fingers = cached["native"]
                                _restore_servo(servo, cached["servo_state"])
                                patch = install_reference_chunk(lib, int(cmd.motion_ids[slot]), cursor, q, roots, quaternions, execution_steps=10, preview_frames=45)
                                servo_log.append({"step": step, "slot": slot, "control_prefix_replay": True,
                                    "native_sha256": _native_digest(cached["native"]), "patch": patch})
                                continue
                            prefix_passed = True
                            if is_anchor:
                                group, row = groups[slot], run_rows[slot]
                                snapshot = _snapshot(measured, slot, origins[slot], observations, servo, include_physical_object=True)
                                _save_npz(pass_dir / f"prefix-slot-{slot}.npz", **snapshot)
                                if candidate_id == 0:
                                    group["prefix"] = snapshot
                                    group["state"] = state_from_snapshot(snapshot, elapsed)
                                    group["rgb"] = images[slot]["rgb"].copy()
                                elif "prefix" not in group:
                                    group["rejections"].append(f"candidate {candidate_id}: baseline anchor unavailable")
                                    prefix_passed = False
                                else:
                                    row["prefix_comparison"] = prefix_comparison(group["prefix"], snapshot, tolerance)
                                    row["rgb_mean_abs_delta"] = float(np.abs(images[slot]["rgb"].astype(float)-group["rgb"].astype(float)).mean())
                                    if not row["prefix_comparison"]["passed"]:
                                        group["rejections"].append(f"candidate {candidate_id}: prefix tolerance exceeded")
                                        prefix_passed = False
                            reuse_canonical = is_anchor and candidate_id > 0 and "canonical" in group and "anchor_servo_state" in group
                            if is_anchor and candidate_id > 0 and not reuse_canonical:
                                group["rejections"].append(f"candidate {candidate_id}: canonical baseline candidate set unavailable")
                                prefix_passed = False
                            if reuse_canonical:
                                _restore_servo(servo, group["anchor_servo_state"])
                                candidates = group["canonical"]
                                # A failed prefix is diagnostic only. Do not
                                # call an executed candidate a valid branch.
                                fallback = candidates.native[0]
                                q, roots, quaternions, fingers = fallback
                                servo_row = {"position_error": np.zeros(3)}
                                correction = servo.correction.copy()
                            else:
                                servo.observe_targets(estimates[slot], elapsed)
                                previous_correction, correction, servo_row = servo.update(cursor,
                                    _array(measured["tcp"])[slot] - origins[slot],
                                    _array(measured["wrist_quat_wxyz"])[slot])
                                q, roots, quaternions = servo.trajectory(cursor, previous_correction, correction, frames=75)
                                fingers = _fingers(schedule, cursor)
                            if is_anchor and candidate_id == 0:
                                candidates, attempts = _propose(model, q, roots, quaternions, fingers, amplitude, max_halvings)
                                row["proposal_attempts"] = attempts
                                group["canonical"] = candidates
                                group["anchor_servo_state"] = _servo_state(servo)
                                _save_npz(batch_dir / f"canonical-slot-{slot}.npz", actions=candidates.actions,
                                    feasible=candidates.feasible,
                                    qtraj_mjcf=np.stack([native.qtraj_mjcf for native in candidates.native]),
                                    roots=np.stack([native.roots for native in candidates.native]),
                                    root_quaternion_wxyz=np.stack([native.root_quaternion_wxyz for native in candidates.native]),
                                    finger_joint_targets=np.stack([native.finger_joint_targets for native in candidates.native]))
                            if is_anchor:
                                if candidates is None:
                                    group["rejections"].append(f"candidate {candidate_id}: canonical proposals unavailable")
                                elif not bool(candidates.feasible.all()):
                                    group["rejections"].append(f"candidate {candidate_id}: five feasible proposals unavailable")
                                elif prefix_passed:
                                    native = candidates.native[candidate_id]
                                    actual_encoding = encode_native(native)
                                    if not np.array_equal(actual_encoding, candidates.actions[candidate_id]):
                                        raise ValueError("Native executable/encoded candidate identity mismatch")
                                    group["actions"][candidate_id] = actual_encoding.copy()
                                    group["expert_utility"][candidate_id] = reference_utility(actual_encoding, candidates.actions[0])
                                    q, roots, quaternions, fingers = native
                                    branch_native[slot] = native
                                    row["native_executed"] = True
                                    row["expert_utility"] = group["expert_utility"][candidate_id]
                                    row["native_sha256"] = _native_digest(native)
                                    row["canonical_native_sha256"] = _native_digest(candidates.native[candidate_id])
                                    row["canonical_encoding_bitwise_equal"] = True
                                    _save_npz(pass_dir / f"native-slot-{slot}.npz", qtraj_mjcf=q, roots=roots,
                                        root_quaternion_wxyz=quaternions, finger_joint_targets=fingers, encoded_actions=actual_encoding)
                            patch = install_reference_chunk(lib, int(cmd.motion_ids[slot]), cursor, q, roots, quaternions, execution_steps=10, preview_frames=45)
                            if candidate_id == 0:
                                baseline_patches[(step, slot)] = {"native": tuple(np.array(value, copy=True) for value in (q, roots, quaternions, fingers)), "servo_state": _servo_state(servo)}
                            servo_log.append({"step": step, "slot": slot, "position_error": servo_row["position_error"], "correction": correction.copy(), "patch": patch})
                        raw = env.observation_manager.compute(update_history=False)
                        observations = wrapper.process_raw_obs(raw, True)
                    for slot in range(env.num_envs):
                        native = branch_native[slot]
                        anchor = groups[slot]["anchor_step"] if slot < len(groups) else -100
                        if slot in replay_slots:
                            if step >= len(baseline_grippers):
                                raise ValueError("Baseline gripper control prefix is unavailable")
                            target = baseline_grippers[step][slot]
                        elif native is not None and anchor <= step < anchor+10:
                            target = native.finger_joint_targets[step-anchor]
                        else:
                            target = _fingers(schedule, step, frames=1)[0]
                        env.gripper_target[slot] = env.gripper_target.new_tensor(target)
                    with torch.no_grad():
                        action = policy.act_inference(obs_dict=observations, cur_dones=done, skip_episode_attnmask=True)
                        # Advance inference histories, but the delivered control
                        # prefix is exactly the baseline action, not this newly
                        # inferred action from slightly different microphysics.
                        action = action.clone()
                        for slot in replay_slots:
                            if step >= len(baseline_actions):
                                raise ValueError("Baseline low-level control prefix is unavailable")
                            cached_action = baseline_actions[step][slot].to(action.device)
                            if step == 0 and slot < len(groups):
                                run_rows[slot]["inferred_first_action_max_abs_delta"] = float((action[slot]-cached_action).abs().max())
                            action[slot] = cached_action
                            if step == 0 and slot < len(groups):
                                run_rows[slot]["delivered_first_action_bitwise_equal"] = bool(torch.equal(action[slot], cached_action))
                        if candidate_id == 0:
                            baseline_actions.append(action.detach().clone())
                            baseline_grippers.append(_array(env.gripper_target))
                        observations, _, newly_done, _ = wrapper.step({"actions": action})
                    done |= newly_done.reshape(-1).bool()
                    trace_records.append(env.capture())
                    if step % 200 == 199:
                        print(f"G1_COLLECT batch={batch_id} candidate={candidate_id} step={step+1}/{total_steps} terminated={done.tolist()}", flush=True)
            except Exception as error:
                pass_error = {"type": type(error).__name__, "message": str(error)}
                for group in groups:
                    group["rejections"].append(f"candidate {candidate_id}: rollout error {type(error).__name__}: {error}")
            trace = {key: np.stack([_array(record[key]) for record in trace_records]) for key in trace_records[0]} if trace_records else {}
            if trace:
                _save_npz(pass_dir / "trace.npz", **trace, env_origins=origins)
            write_json(pass_dir / "servo.json", servo_log)
            if candidate_id == 0 and baseline_actions:
                _save_npz(batch_dir / "delivered-control-prefix.npz", actions=_array(torch.stack(baseline_actions)), gripper_targets=np.stack(baseline_grippers))
                patch_keys = sorted(baseline_patches)
                _save_npz(batch_dir / "delivered-native-prefix.npz", step_slot=np.asarray(patch_keys, dtype=np.int64),
                    qtraj_mjcf=np.stack([baseline_patches[key]["native"][0] for key in patch_keys]),
                    roots=np.stack([baseline_patches[key]["native"][1] for key in patch_keys]),
                    root_quaternion_wxyz=np.stack([baseline_patches[key]["native"][2] for key in patch_keys]),
                    finger_joint_targets=np.stack([baseline_patches[key]["native"][3] for key in patch_keys]))
                write_json(batch_dir / "delivered-servo-prefix.json", [{"step": step, "slot": slot, **baseline_patches[(step,slot)]["servo_state"]} for step,slot in patch_keys])
            for slot, group in enumerate(groups):
                row = run_rows[slot]
                row["completed_steps"] = max(0, len(trace_records)-1)
                if trace:
                    row["outcome"] = episode_outcome(trace, slot=slot, origin=origins[slot], terminated=bool(done[slot]), control_dt=env.step_dt)
                if pass_error:
                    row["rollout_error"] = pass_error
                if group["actions"][candidate_id] is None or group["future_rgb"][candidate_id] is None or row["completed_steps"] != total_steps:
                    group["rejections"].append(f"candidate {candidate_id}: incomplete measured branch")
                group["runs"].append(row)
            write_json(pass_dir / "branches.json", run_rows)
            write_json(batch_dir / "groups-in-progress.json", [_group_report(group) for group in groups])

        accepted = []
        for group in groups:
            if not group["rejections"]:
                accepted.append(_raw_row(group, task_name, instruction, total_steps))
            reports.append({**_group_report(group), "accepted": not bool(group["rejections"])})
        if accepted:
            data = {key: np.stack([row[key] for row in accepted]) for key in accepted[0]}
            validate_archive(data, raw=True)
            _save_npz(batch_dir / "raw.npz", **data)
            all_rows.extend(accepted)
            merged = {key: np.stack([row[key] for row in all_rows]) for key in all_rows[0]}
            validate_archive(merged, raw=True)
            _save_npz(output / "raw.npz", **merged)
        manifest = {
            "schema": "g1_embodiedskills.same_slot_replay.v1", "accepted_anchors": len(all_rows),
            "provenance": provenance,
            "attempted_groups": len(reports), "rejected_groups": sum(not row["accepted"] for row in reports),
            "groups": reports, "prefix_tolerance": tolerance, "position_jitter_m": request.position_jitter,
            "hard_physics_reset": bool(cfg.get("hard_physics_reset", False)),
            "physical_prefix_fields": ["root", "q", "qdot", "cube_root"],
            "synchronize_batch_anchors": bool(cfg.get("synchronize_batch_anchors", False)),
            "protocol": "Five full rollouts per same physical slot/seed; replay baseline delivered low-level actions/gripper/native references/planner memory before anchor; componentwise physical/history gate; one canonical five-candidate set; no snapshot or cross-slot matching",
            "execution_steps": 10, "native_preview_frames": 75, "future_frame_offset_steps": 10,
            "future_label_horizon_steps": total_steps,
            "rendering": "Same 5-Hz refresh cadence as online in all branches; replay controls never regenerated from images",
            "success_and_return": "Shared metrics.episode_outcome from actual whole-episode physical trace",
            "expert_utility": "Negative normalized RMS native-action distance to the same-run direct planner; scales rad=1, translation_m=.1, quat=1, finger_m=.035; not an optimal-success expert",
            "state_fields": ["actor_obs", "root_local_xyz_wxyz_world_velocities", "q31", "qdot31", "servo_correction29", "perceived_cube_offset3", "perceived_tray_offset3", "elapsed_seconds"],
            "progress": "elapsed control budget fraction; no privileged task completion",
            "wall_seconds": time.monotonic()-started,
        }
        write_json(manifest_path, manifest)
        print(f"G1_COLLECT batch={batch_id} accepted={len(accepted)}/{len(groups)} total_anchors={len(all_rows)}", flush=True)
    return {"completed": True, "mode": "collect", "accepted_anchors": len(all_rows),
        "rejected_groups": sum(not row["accepted"] for row in reports),
        "raw_archive": str(output / "raw.npz") if all_rows else None,
        "manifest": str(manifest_path), "all_requested_anchors_accepted": len(all_rows) == len(requested),
        "note": "Collection completion does not imply replay gates passed; rejected groups have no training row"}
