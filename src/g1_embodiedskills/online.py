"""Measured observe/propose/predict/select/execute evaluation on held-out seeds.

This module performs no simulator lookahead. It uses the same pre-proposal
state, geometric proposer, frozen encoders and native codec as collection.
Physical reset support, especially multi-environment resets, must be verified
by the caller on the installed Isaac version; this file makes no such claim.
"""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
import time

import numpy as np
import torch

from .collection import (_array, _fingers, _local_estimates, _native_digest,
                         _propose, _save_npz, _snapshot, state_from_snapshot)
from .io import sha256, write_json


@contextmanager
def _feature_precision():
    """Match offline FP32 matmul while preserving SONIC's TF32 setting."""
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _latency(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None, "max_ms": None}
    return {"count": len(values), "mean_ms": float(values.mean() * 1000),
            "p50_ms": float(np.median(values) * 1000), "p95_ms": float(np.quantile(values, .95) * 1000),
            "max_ms": float(values.max() * 1000)}


def _predictions(result, count):
    scores, selected = np.asarray(result["scores"]), np.asarray(result["selected"])
    if scores.shape != (count, 5) or not np.isfinite(scores).all():
        raise ValueError("Selector must return finite scores [batch,5]")
    if selected.shape != (count,) or not np.issubdtype(selected.dtype, np.integer):
        raise ValueError("Selector must return integer selected[batch]")
    if not np.array_equal(selected, scores.argmax(axis=1)):
        raise ValueError("Selector choice differs from its actual comparison-head argmax")
    return scores, selected


def _load_predictor(request, device, instruction):
    """Load the actual three-WAM bundle and its immutable offline encoders."""
    from .learning import (DEFAULT_LANGUAGE, DEFAULT_VISUAL, cached_language_vectors, encoder_revision,
                           resolve_offline_model, use_upstream)
    use_upstream(getattr(request, "upstream_root", None))
    from embodiedskills.runtime import Selector
    from embodiedskills.vision import FrozenPublicVisualEncoder
    checkpoint = Path(request.selector)
    selector = Selector(checkpoint, device=str(device))
    if selector.family != "continuous" or len(selector.members) != 3 or selector.head is None:
        raise ValueError("Online G1 evaluation requires the real continuous three-WAM comparison bundle")
    for member in selector.payload["members"]:
        config = member["config"]
        if tuple(config.get(key) for key in ("state_dim", "action_dim", "execution_horizon")) != (1041, 38, 75):
            raise ValueError("Selector bundle dimensions must be state=1041/action=38/horizon=75")
    config = selector.payload.get("encoders", {})
    if config.get("visual_backbone", "vjepa2") != "vjepa2" or int(config.get("spatial_grid", 1)) != 1:
        raise ValueError("Online G1 requires the same V-JEPA2 spatial_grid=1 contract as training")
    visual_id, language_id = config.get("visual_model", DEFAULT_VISUAL), config.get("language_model", DEFAULT_LANGUAGE)
    visual_revision = encoder_revision(visual_id, config.get("visual_revision"))
    language_revision = encoder_revision(language_id, config.get("language_revision"))
    visual_path = resolve_offline_model(visual_id, visual_revision)
    language_path = resolve_offline_model(language_id, language_revision)
    visual = FrozenPublicVisualEncoder(backbone="vjepa2", model=visual_path, device=str(device), spatial_grid=1)
    cache = selector.payload.get("g1_language_cache", {})
    language_vector = cached_language_vectors([instruction], cache)[0]
    if visual.feature_dim != 1024 or language_vector.shape != (384,) or not np.isfinite(language_vector).all():
        raise ValueError("Frozen encoder dimensions must be visual=1024/language=384")
    vector_hash = hashlib.sha256(language_vector.tobytes()).hexdigest()
    identity = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "family": selector.family, "members": 3, "comparison_head": type(selector.head).__name__,
        "dimensions": {"state": 1041, "action": 38, "horizon": 75, "candidates": 5},
        "visual_model": str(visual_id), "visual_revision": visual_revision,
        "language_model": str(language_id), "language_revision": language_revision,
        "visual_preprocessing": visual.preprocessing_contract, "visual_normalization": "L2 after pooled features",
        "language_preprocessing": "MiniLM masked mean pooling and L2; max_length=64", "network_access": False}
    identity["language_runtime"] = "Exact cached frozen single-instruction embedding from offline extraction"
    identity["language_vector_sha256"] = vector_hash
    identity["language_snapshot"] = str(language_path)
    identity["prediction_matmul"] = "FP32, TF32 disabled locally; SONIC setting restored before control"
    return selector, visual, language_vector, identity


def run_closed_loop(wrapper, policy, request):
    """Execute ten real control steps for each newly selected native preview.

    ``repeats`` is the total episode count. Direct/random can omit a bundle;
    their model scores are then explicitly null rather than invented. Providing
    a bundle for those baselines records real predictions without using them
    to choose actions. Every selector-method decision uses the genuine head.
    """
    from .candidates import encode_native
    from .kinematics import G1Kinematics
    from .metrics import episode_outcome
    from .native_reference import NativeReferencePatcher
    from .perception import render_observations
    from .servo import ReferenceServo
    from .simulation import rebuild_physics
    from .task_runtime import encode_video

    env = wrapper.env
    method = str(getattr(request, "method", "selector"))
    if method not in ("direct", "random", "selector"):
        raise ValueError("method must be direct, random or selector")
    if not request.cameras:
        raise ValueError("Online evaluation requires the actual head RGB-D camera")
    repeats, seed_base = int(request.repeats), int(request.seed_base)
    if repeats <= 0 or seed_base < 900000:
        raise ValueError("Use positive total repeats and held-out seed_base >= 900000")
    if not np.isclose(float(env.step_dt), .02):
        raise ValueError("Online execution requires audited 50-Hz control")
    jitter = float(getattr(request, "position_jitter", .02))
    if not 0 <= jitter <= .04:
        raise ValueError("position_jitter must be between zero and .04m")
    request.position_jitter = jitter
    total_steps = round(float(request.seconds) / env.step_dt)
    if total_steps < 10 or total_steps % 10:
        raise ValueError("Episode budget must contain an integer number of ten-step decisions")
    if method == "selector" and not getattr(request, "selector", None):
        raise ValueError("selector method requires a trained selector bundle")
    output = Path(request.output)
    output.mkdir(parents=True, exist_ok=True)
    task_path = request.task or Path(request.reference) / "task.json"
    task = json.loads(Path(task_path).read_text(encoding="utf-8"))
    with np.load(Path(request.reference) / "planned.npz", allow_pickle=False) as source:
        planned = {key: source[key].copy() for key in source.files}
    instruction = str(getattr(request, "instruction", "Place the red block into the green tray."))
    if not instruction.strip():
        raise ValueError("instruction must be nonempty")
    amplitude = float(getattr(request, "amplitude_m", .012))
    max_halvings = int(getattr(request, "max_amplitude_halvings", 5))
    if not 0 < amplitude <= .05 or not 0 <= max_halvings <= 8:
        raise ValueError("Invalid proposal amplitude/retry configuration")
    torch.set_num_threads(1)
    model = G1Kinematics(request.sonic)
    device = torch.device(env.device)
    selector = visual_encoder = language_vector = None
    identity = {"selector": None, "model_scores": "not applicable: baseline has no loaded bundle"}
    if getattr(request, "selector", None):
        selector, visual_encoder, language_vector, identity = _load_predictor(request, device, instruction)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    cmd = env.command_manager.get_term("motion")
    lib = cmd.motion_lib
    if int(lib._num_motions) < env.num_envs:
        raise ValueError("Every physical slot needs an independent reference timeline")
    original_fields = {name: value.clone() for name, value in NativeReferencePatcher(lib).fields.items()}
    episodes, global_selections = [], []
    timings = {key: [] for key in ("decision", "render", "proposal", "encoder", "selector", "patch")}
    started = time.monotonic()

    for offset in range(0, repeats, env.num_envs):
        batch_id = offset // env.num_envs
        batch_dir = output / f"batch-{batch_id:04d}"
        batch_dir.mkdir(exist_ok=False)
        decisions_dir = batch_dir / "decisions"
        decisions_dir.mkdir()
        count = min(env.num_envs, repeats-offset)
        seeds = [seed_base + offset + slot for slot in range(count)]
        # Stop invalidates data views. All robot/camera data and origins are
        # reacquired after this full physical reset; no old data cache is reused.
        rebuild_physics(env)
        cmd = env.command_manager.get_term("motion")
        lib = cmd.motion_lib
        for name, value in original_fields.items():
            getattr(lib, name).copy_(value)
        patcher = NativeReferencePatcher(lib)
        env.episode_seeds = seeds + [seeds[-1]] * (env.num_envs-count)
        servos = [ReferenceServo(request.sonic, planned, env.reference, task["geometry"], model=model)
                  for _ in range(env.num_envs)]
        policy.init_rollout()
        observations = wrapper.process_raw_obs(env.reset_nominal(), True)
        if len(set(cmd.motion_ids.tolist())) != env.num_envs:
            raise ValueError("Reset assigned the same native timeline to multiple physical slots")
        origins = _array(env.scene.env_origins)
        _save_npz(batch_dir / "reset-observations.npz", actor_obs=_array(observations["actor_obs"]),
                  tokenizer=_array(observations["tokenizer"]))
        physical_done = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        terminated_at_end = np.zeros(env.num_envs, dtype=bool)
        failed = np.zeros(env.num_envs, dtype=bool)
        end_steps = np.full(env.num_envs, -1, dtype=int)
        failures = [None] * env.num_envs
        selected_native = [None] * env.num_envs
        selected_rows = [None] * env.num_envs
        selections = [[] for _ in range(count)]
        estimates = [None] * env.num_envs
        random_generators = [np.random.default_rng(np.random.SeedSequence([seed, 0x474153])) for seed in seeds]
        trace_records, decisions, video_frames = [env.capture()], [], []
        batch_started = time.monotonic()

        for step in range(total_steps):
            elapsed = step * env.step_dt
            live_slots = [slot for slot in range(count) if end_steps[slot] < 0]
            if not live_slots:
                break
            if step % 10 == 0:
                decision_started = time.perf_counter()
                before_render = time.perf_counter()
                images = render_observations(env, live_slots, refresh=True)
                timings["render"].append(time.perf_counter()-before_render)
                if batch_id == 0 and getattr(request, "record_video", False) and 0 in images:
                    video_frames.append(images[0]["rgb"].copy())
                # Collection refreshes target estimates only in its clear-view
                # window. Current RGB is still encoded at every decision.
                if 3.4 <= elapsed <= 6.6:
                    for slot in live_slots:
                        estimates[slot] = _local_estimates(images[slot]["estimates"], origins[slot])
                measured = env.capture()
                proposals, states, snapshots, cursors, servo_logs, attempts_by_slot = {}, {}, {}, {}, {}, {}
                for slot in live_slots:
                    servo = servos[slot]
                    # Snapshot before observe_targets/update, exactly as at a
                    # collection anchor. Ground-truth cube pose is never read.
                    snapshot = _snapshot(measured, slot, origins[slot], observations, servo)
                    state = state_from_snapshot(snapshot, elapsed)
                    if state.shape != (1041,):
                        raise ValueError(f"Online state dimension mismatch: {state.shape}, expected (1041,)")
                    snapshots[slot], states[slot] = snapshot, state
                    cursor = int(cmd.time_steps[slot] + cmd.motion_start_time_steps[slot])
                    if cursor != step:
                        raise ValueError(f"Reference/control clock mismatch: slot={slot}, cursor={cursor}, step={step}")
                    cursors[slot] = cursor
                    servo.observe_targets(estimates[slot], elapsed)
                    previous, correction, servo_log = servo.update(cursor,
                        _array(measured["tcp"])[slot]-origins[slot], _array(measured["wrist_quat_wxyz"])[slot])
                    q, roots, quaternions = servo.trajectory(cursor, previous, correction, frames=75)
                    fingers = _fingers(task["gripper_schedule"], cursor)
                    proposal_started = time.perf_counter()
                    candidates, attempts = _propose(model, q, roots, quaternions, fingers, amplitude, max_halvings)
                    timings["proposal"].append(time.perf_counter()-proposal_started)
                    attempts_by_slot[slot], servo_logs[slot] = attempts, servo_log
                    if not bool(candidates.feasible.all()):
                        failed[slot], end_steps[slot] = True, step
                        failures[slot] = "five feasible native proposals unavailable after amplitude retries"
                        audit_path = decisions_dir / f"step-{step:05d}-slot-{slot}-failed.npz"
                        _save_npz(audit_path, state=state, root_state=snapshot["root"],
                                  actions=candidates.actions, feasible=candidates.feasible)
                        decisions.append({"step": step, "slot": slot, "seed": seeds[slot], "selected": None,
                            "scores": None, "status": "proposal_failed", "proposal_attempts": attempts,
                            "state_sha256": _native_digest((state,)), "root_sha256": _native_digest((snapshot["root"],)),
                            "canonical_native_sha256": [_native_digest(native) for native in candidates.native],
                            "audit_archive": str(audit_path.relative_to(output)), "executed_control_steps": 0})
                        continue
                    if candidates.actions.shape != (5, 75, 38):
                        raise ValueError("Online action dimensions must be [5,75,38]")
                    proposals[slot] = candidates
                valid_slots = sorted(proposals)
                scores = head_selected = None
                encoder_seconds = selector_seconds = 0.
                if valid_slots and selector is not None:
                    rgb = np.stack([images[slot]["rgb"] for slot in valid_slots])
                    if rgb.shape != (len(valid_slots), 256, 256, 3) or rgb.dtype != np.uint8:
                        raise ValueError("Online vision requires measured uint8 RGB [B,256,256,3]")
                    encoder_started = time.perf_counter()
                    with _feature_precision():
                        features = np.asarray(visual_encoder.encode_rgb(rgb, batch_size=len(valid_slots)), dtype=np.float32)
                    if features.shape != (len(valid_slots), 1, 1024):
                        raise ValueError("Frozen V-JEPA2 output must be [B,1,1024]")
                    features = features[:, 0]
                    features /= np.maximum(np.linalg.norm(features, axis=-1, keepdims=True), 1e-12)
                    encoder_seconds = time.perf_counter()-encoder_started
                    timings["encoder"].append(encoder_seconds)
                    selector_started = time.perf_counter()
                    with _feature_precision():
                        result = selector.predict(visual=features, state=np.stack([states[slot] for slot in valid_slots]),
                            language=np.repeat(language_vector[None], len(valid_slots), axis=0),
                            progress=np.full((len(valid_slots), 1), step/total_steps, dtype=np.float32),
                            actions=np.stack([proposals[slot].actions for slot in valid_slots]),
                            mask=np.ones((len(valid_slots), 5, 75), dtype=np.float32))
                    scores, head_selected = _predictions(result, len(valid_slots))
                    selector_seconds = time.perf_counter()-selector_started
                    timings["selector"].append(selector_seconds)
                for index, slot in enumerate(valid_slots):
                    candidates = proposals[slot]
                    selected = (0 if method == "direct" else int(random_generators[slot].integers(5))
                                if method == "random" else int(head_selected[index]))
                    native = candidates.native[selected]
                    encoded = encode_native(native)
                    if not np.array_equal(encoded, candidates.actions[selected]):
                        raise ValueError("Selected native executable and encoded action are not identical")
                    patch_started = time.perf_counter()
                    patch = patcher.patch(int(cmd.motion_ids[slot]), cursors[slot], *native[:3],
                                          execution_steps=10, preview_frames=45)
                    timings["patch"].append(time.perf_counter()-patch_started)
                    selected_native[slot] = native
                    selections[slot].append(selected)
                    global_selections.append(selected)
                    audit_path = decisions_dir / f"step-{step:05d}-slot-{slot}.npz"
                    arrays = {"state": states[slot], "actions": candidates.actions,
                        "selected": np.asarray(selected, dtype=np.int64), "qtraj_mjcf": native.qtraj_mjcf,
                        "roots": native.roots, "root_quaternion_wxyz": native.root_quaternion_wxyz,
                        "finger_joint_targets": native.finger_joint_targets, "root_state": snapshots[slot]["root"],
                        "progress": np.asarray([step/total_steps], dtype=np.float32)}
                    if scores is not None:
                        arrays["scores"] = scores[index]
                    if step % 250 == 0:
                        arrays["rgb"], arrays["depth"] = images[slot]["rgb"], images[slot]["depth"]
                    _save_npz(audit_path, **arrays)
                    row = {"step": step, "time": elapsed, "slot": slot, "seed": seeds[slot], "method": method,
                        "scores": None if scores is None else scores[index].tolist(), "selected": selected,
                        "head_argmax": None if head_selected is None else int(head_selected[index]),
                        "scores_status": "actual three-WAM comparison output" if scores is not None else "not applicable: baseline without bundle",
                        "canonical_native_sha256": [_native_digest(candidate) for candidate in candidates.native],
                        "selected_native_sha256": _native_digest(native), "encoded_action_sha256": _native_digest((encoded,)),
                        "root_sha256": _native_digest((snapshots[slot]["root"],)),
                        "state_sha256": _native_digest((states[slot],)), "encoded_native_match": True,
                        "audit_archive": str(audit_path.relative_to(output)), "proposal_attempts": attempts_by_slot[slot],
                        "servo": servo_logs[slot], "patch": patch, "encoder_batch_seconds": encoder_seconds,
                        "selector_batch_seconds": selector_seconds, "executed_control_steps": 0, "status": "selected"}
                    decisions.append(row)
                    selected_rows[slot] = row
                # Refresh cached observations only: no command compute/cursor
                # advancement and no history append is allowed here.
                cursor_before = _array(cmd.time_steps) + _array(cmd.motion_start_time_steps)
                raw = env.observation_manager.compute(update_history=False)
                observations = wrapper.process_raw_obs(raw, True)
                cursor_after = _array(cmd.time_steps) + _array(cmd.motion_start_time_steps)
                if not np.array_equal(cursor_before, cursor_after):
                    raise ValueError("Reference cursor advanced while refreshing patched observations")
                timings["decision"].append(time.perf_counter()-decision_started)
                if step % 100 == 0:
                    write_json(batch_dir / "decisions.json", decisions)
            if all(end_steps[slot] >= 0 for slot in range(count)):
                break
            for slot in range(count):
                if end_steps[slot] < 0:
                    native = selected_native[slot]
                    if native is None:
                        raise ValueError("Live slot has no selected native command")
                    env.gripper_target[slot] = env.gripper_target.new_tensor(native.finger_joint_targets[step % 10].copy())
            inference_done = physical_done | torch.as_tensor(failed, device=device)
            if count < env.num_envs:
                inference_done[count:] = True
            with torch.no_grad():
                action = policy.act_inference(obs_dict=observations, cur_dones=inference_done, skip_episode_attnmask=True)
                observations, _, newly_done, _ = wrapper.step({"actions": action})
            physical_done |= newly_done.reshape(-1).bool()
            trace_records.append(env.capture())
            for slot in range(count):
                if end_steps[slot] < 0:
                    selected_rows[slot]["executed_control_steps"] += 1
                    if bool(physical_done[slot]):
                        end_steps[slot] = step+1
                        terminated_at_end[slot] = True
                        failures[slot] = "physical termination"
            if step % 200 == 199:
                print(f"G1_ONLINE method={method} batch={batch_id} step={step+1}/{total_steps} choices={[row[-1] if row else None for row in selections]} terminated={physical_done.tolist()}", flush=True)

        trace = {key: np.stack([_array(record[key]) for record in trace_records]) for key in trace_records[0]}
        _save_npz(batch_dir / "trace.npz", **trace, env_origins=origins)
        write_json(batch_dir / "decisions.json", decisions)
        if video_frames:
            encode_video(output / "head-first-batch-slot-0.mp4", video_frames, fps=5)
        for slot in range(count):
            completed_steps = end_steps[slot] if end_steps[slot] >= 0 else len(trace_records)-1
            slot_trace = {key: value[:completed_steps+1] for key, value in trace.items()}
            outcome = episode_outcome(slot_trace, slot=slot, origin=origins[slot],
                terminated=bool(terminated_at_end[slot]), control_dt=env.step_dt)
            if failures[slot] is not None or completed_steps != total_steps:
                outcome["success"] = False
            row = {"episode": offset+slot, "seed": seeds[slot], "batch": batch_id, "slot": slot,
                "method": method, "completed_steps": int(completed_steps), "requested_steps": total_steps,
                "failure": failures[slot], "decisions": len(selections[slot]),
                "switches_from_direct": sum(value != 0 for value in selections[slot]),
                "index_changes": sum(a != b for a, b in zip(selections[slot], selections[slot][1:])),
                "duration": float(slot_trace["time"][-1, slot]), "wall_seconds": time.monotonic()-batch_started, **outcome}
            episodes.append(row)
            write_json(batch_dir / f"episode-slot-{slot}.json", row)
        write_json(output / "episodes.json", episodes)
        print(f"G1_ONLINE method={method} completed={len(episodes)}/{repeats} success={sum(row['success'] for row in episodes)}/{len(episodes)}", flush=True)

    gpu_peak = None
    if device.type == "cuda":
        gpu_peak = {"allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                    "scope": "PyTorch allocator for entire process; excludes non-PyTorch renderer allocations"}
    result = {"completed": len(episodes) == repeats, "mode": "closed_loop", "method": method,
        "episodes": episodes, "success_rate": sum(row["success"] for row in episodes)/len(episodes),
        "requested_episodes": repeats, "counted_episodes": len(episodes), "seed_base": seed_base,
        "position_jitter_m": jitter, "execution_steps": 10, "native_preview_frames": 75,
        "state_dim": 1041, "action_dim": 38, "instruction": instruction, "model_identity": identity,
        "decisions": len(global_selections), "switches_from_direct": sum(value != 0 for value in global_selections),
        "selected_histogram": np.bincount(global_selections, minlength=5).tolist(),
        "latency": {key: _latency(value) for key, value in timings.items()}, "gpu_peak": gpu_peak,
        "wall_seconds": time.monotonic()-started, "hard_physics_reset": True,
        "protocol": "Every decision uses present RGB/proprioception and five native previews; ten actual control steps, then feedback. No future labels, rollout lookahead, forced switching, or counted padding slots.",
        "video": "head-first-batch-slot-0.mp4" if getattr(request, "record_video", False) else None}
    write_json(output / "result.json", result)
    return result
