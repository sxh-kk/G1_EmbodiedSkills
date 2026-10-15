"""Independent K=9/H=105 same-slot physical factor-grid collection.

No learner is imported. Each batch owns an immutable capsule containing the
actual delivered controls, all companions, canonical grids and pre-intervention
history. Noise repeats rerun that complete capsule; only preselected groups are
included as additional observations. This module does not alter M1 paths.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from .collection import (_array, _fingers, _local_estimates, _native_digest,
                         _restore_servo, _save_npz, _servo_state, _snapshot,
                         prefix_comparison)
from .io import sha256, write_json

SCHEMA = "g1_embodiedskills.m2_collection.v1"
SAMPLE_OFFSETS = (10, 25, 50, 150)
CELL_FACTORS = ((0, 0), (0, 1), (0, -1), (1, 0), (1, 1), (1, -1),
                (-1, 0), (-1, 1), (-1, -1))
HORIZON, EXECUTION = 105, 50


def validate_batches(config, count, total_steps=1200):
    """Validate explicit slot layout before the first simulator step."""
    if config.get("schema") != "g1_embodiedskills.m2_collection_config.v1":
        raise ValueError("Explicit M2 collection_config schema is required")
    if total_steps != 1200 or count < 1:
        raise ValueError("M2 requires the predeclared 24-second 50-Hz budget")
    batches = config.get("batches")
    if not isinstance(batches, list) or not batches:
        raise ValueError("M2 needs explicit same-time physical batches")
    seen, capsules, selected = set(), set(), set()
    for batch in batches:
        capsule = batch.get("capsule_id")
        if not isinstance(capsule, str) or not capsule or "/" in capsule or capsule in capsules:
            raise ValueError("Capsule ids must be unique nonempty path components")
        capsules.add(capsule)
        anchors = batch.get("anchors")
        if not isinstance(anchors, list) or not 1 <= len(anchors) <= count:
            raise ValueError("A capsule needs 1..num_envs active anchors")
        anchor_steps = set()
        for slot, anchor in enumerate(anchors):
            seed, step = anchor.get("seed"), anchor.get("anchor_step")
            if (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
                    or isinstance(step, bool) or not isinstance(step, int)
                    or step < 0 or step % 10 or step + 150 > total_steps):
                raise ValueError("Seeds and anchor clocks must satisfy the M2 physical protocol")
            if anchor.get("slot", slot) != slot or (seed, step) in seen:
                raise ValueError("Slots must be contiguous and anchor identities unique")
            seen.add((seed, step))
            anchor_steps.add(step)
            if anchor.get("noise_repeat", False):
                selected.add((seed, step))
        if len(anchor_steps) != 1:
            raise ValueError("All physical companions must share one anchor time")
        companions = batch.get("companion_seeds")
        expected = len(anchors)
        if not isinstance(companions, list) or len(companions) != count - expected:
            raise ValueError("Inactive slot seeds must be explicit in companion_seeds")
        if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in companions):
            raise ValueError("Companion seeds must be nonnegative integers")
        if not isinstance(batch.get("condition"), str) or not isinstance(batch.get("family"), str):
            raise ValueError("Every capsule must declare condition and family")
    if config.get("formal", False):
        if len(seen) != 200 or len(selected) != 10 or len({s for s, _ in seen}) != 40:
            raise ValueError("Formal M2 requires 40 episodes / 200 anchors / 10 repeated anchors")
    return batches


def serialize_cached_estimates(value):
    """Canonical exact-value JSON for the continuation policy's RGB-D cache."""
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Unsupported cached observation value: {type(item).__name__}")
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False, default=convert)


def history_snapshot(policy, observations, servo, measured, slot, origin,
                     motion_lib=None, motion_id=None, cursor=None, *, cached_estimates=None):
    """Capture runtime actor buffers, Servo validity and visible reference history.

    This supplements physical/current observations: checking only actor_obs
    omits the temporal actor buffer and cannot establish a common policy state.
    """
    result = _snapshot(measured, slot, origin, observations, servo,
                       include_physical_object=True)
    result["cached_estimates_present"] = np.asarray(cached_estimates is not None, np.uint8)
    result["cached_estimates_json"] = np.frombuffer(
        serialize_cached_estimates(cached_estimates).encode("utf-8"), dtype=np.uint8).copy()
    result["servo_target_valid"] = np.asarray([
        servo.last_estimate_time[key] is not None for key in ("cube", "tray")], np.uint8)
    result["servo_target_time"] = np.asarray([
        0. if servo.last_estimate_time[key] is None else servo.last_estimate_time[key]
        for key in ("cube", "tray")], float)
    for name in ("obs_dict_buffer", "dones_buffer", "steps"):
        if not hasattr(policy, name):
            raise ValueError(f"Actor temporal-state contract missing {name}")
    result["policy_steps"] = np.asarray(policy.steps, np.int64)
    for name, value in sorted(policy.obs_dict_buffer.items()):
        array = _array(value)
        if array.shape[0] != len(_array(measured["root"])):
            raise ValueError("Actor history does not match the physical slot layout")
        result["policy_observation/" + name] = array[slot]
    result["policy_history_keys"] = np.frombuffer(
        "\n".join(sorted(policy.obs_dict_buffer.keys())).encode(), dtype=np.uint8).copy()
    result["policy_dones_present"] = np.asarray(policy.dones_buffer is not None, np.uint8)
    result["policy_dones"] = (np.zeros(0, np.uint8) if policy.dones_buffer is None
                              else _array(policy.dones_buffer)[slot].astype(np.uint8))
    if motion_lib is not None:
        from .native_reference import NativeReferencePatcher
        start = int(_array(motion_lib.length_starts)[motion_id]) + int(cursor)
        # The first frame has no earlier context; every later anchor preserves
        # nine historical frames plus the complete candidate query window.
        lower = max(int(_array(motion_lib.length_starts)[motion_id]), start - 9)
        for name, tensor in NativeReferencePatcher(motion_lib).fields.items():
            result["reference/" + name] = _array(tensor[lower:start + HORIZON])
    return result


def compare_history(reference, actual, tolerance):
    if set(reference) != set(actual):
        return {"passed": False, "field_inventory_mismatch": sorted(set(reference) ^ set(actual))}
    # Arithmetic on bool arrays is undefined and uint8 differences wrap.
    left, right, empty = {}, {}, {}
    for key, value in reference.items():
        expected, observed = np.asarray(value, dtype=np.float64), np.asarray(actual[key], dtype=np.float64)
        if expected.size:
            left[key], right[key] = expected, observed
        else:
            empty[key] = {"passed": expected.shape == observed.shape,
                          "empty": True, "max_abs": 0., "rms": 0.}
    report = prefix_comparison(left, right, tolerance)
    report["fields"].update(empty)
    report["passed"] = all(row["passed"] for row in report["fields"].values())
    return report


def physical_sample(model, measured, slot, origin, simulator_joint_names):
    """Scene-world axes, with env_origin removed from all translations."""
    root, cube = _array(measured["root"])[slot], _array(measured["cube_root"])[slot]
    root[:3] -= origin
    cube[:3] -= origin
    q, qdot = _array(measured["q"])[slot], _array(measured["qdot"])[slot]
    mapping = [simulator_joint_names.index(name) for name in model.joint_names]
    q_mjcf = q[mapping]
    positions, matrices = model.transforms(q_mjcf, root[:3], root[3:7])
    soles = np.stack([positions[model.body_names.index(name)]
        + matrices[model.body_names.index(name)] @ offset
        for name, offset in model.sole_offsets.items()])
    return {"actual_tcp": _array(measured["tcp"])[slot] - origin,
        "actual_R": Rotation.from_quat(_array(measured["wrist_quat_wxyz"])[slot][[1, 2, 3, 0]]).as_matrix(),
        "cube_root": cube, "root": root, "q": q, "qdot": qdot,
        "gripper": _array(measured["gripper"])[slot],
        "waist": np.asarray(q_mjcf[model.joint_names.index("waist_pitch_joint")]),
        "soles": soles}


def nominal_sample(model, native, index):
    q, roots, quats, _ = native
    if not 0 <= index < len(q):
        raise ValueError("Delivered native reference does not cover the sampled clock")
    positions, matrices = model.transforms(q[index], roots[index], quats[index])
    matrix = matrices[model.right_wrist_index]
    return {"nominal_tcp": positions[model.right_wrist_index] + matrix @ [.085, 0., 0.],
            "nominal_R": matrix}


def bridge_installed_prefix(patcher, motion_id, cursor, native, bridge_steps=15):
    """Smoothly bridge old installed references to a new common Servo window.

    The Servo updates at 10-step intervals but blends over 15 frames. Its old
    correction is therefore not necessarily the reference already installed at
    this cursor. Frame zero must come from that installed reference, and a
    one-frame replacement would introduce a velocity jump. The same bridged
    baseline is supplied to every factor cell, including the default cell.
    """
    from .candidates import NativeCommand
    from scipy.spatial.transform import Slerp
    q, roots, quats, fingers = (np.array(value, copy=True) for value in native)
    if bridge_steps < 1 or len(q) <= bridge_steps:
        raise ValueError("Bridge must fit within the common native window")
    start = int(_array(patcher.lib.length_starts)[motion_id]) + cursor
    old_q, old_root, old_quat = patcher._existing_inputs(start, start + bridge_steps + 1)
    old_q, old_root, old_quat = map(_array, (old_q, old_root, old_quat))
    fraction = np.arange(bridge_steps + 1, dtype=float) / bridge_steps
    weight = fraction**3 * (10. - 15.*fraction + 6.*fraction**2)
    q[:bridge_steps + 1] = old_q * (1. - weight[:, None]) + q[:bridge_steps + 1] * weight[:, None]
    roots[:bridge_steps + 1] = old_root * (1. - weight[:, None]) + roots[:bridge_steps + 1] * weight[:, None]
    for index, value in enumerate(weight):
        rotations = Rotation.from_quat(np.stack((old_quat[index], quats[index]))[:, [1, 2, 3, 0]])
        quats[index] = Slerp([0., 1.], rotations)([value]).as_quat()[0][[3, 0, 1, 2]]
    # Preserve the old frame's exact float values, including quaternion sign.
    q[0], roots[0], quats[0] = old_q[0], old_root[0], old_quat[0]
    bridged = NativeCommand(q, roots, quats, fingers)
    return bridged, {"bridge_steps": bridge_steps, "bridge_seconds": bridge_steps / 50.,
        "rule": "quintic old-installed to common new Servo path; quaternion slerp",
        "frame_zero_q_max_abs_error": float(np.max(np.abs(q[0] - old_q[0]))),
        "old_installed_prefix_sha256": _native_digest((old_q, old_root, old_quat)),
        "common_baseline_sha256": _native_digest(bridged)}


def sample_arrays(anchor_step, values):
    """Missing physical samples remain NaN with an explicit validity mask."""
    widths = {"actual_tcp": (3,), "actual_R": (3, 3), "cube_root": (13,),
              "root": (13,), "q": (31,), "qdot": (31,), "gripper": (2,),
              "waist": (), "soles": (2, 3), "nominal_tcp": (3,), "nominal_R": (3, 3)}
    result = {"offset_steps": np.asarray(SAMPLE_OFFSETS, np.int64),
              "offset_seconds": np.asarray(SAMPLE_OFFSETS, float) / 50.,
              "global_steps": anchor_step + np.asarray(SAMPLE_OFFSETS, np.int64),
              "valid": np.asarray([offset in values for offset in SAMPLE_OFFSETS], bool)}
    for name, shape in widths.items():
        result[name] = np.stack([np.asarray(values[offset][name], float)
            if offset in values else np.full(shape, np.nan) for offset in SAMPLE_OFFSETS])
    result["nominal_command_step"] = np.asarray([
        values[offset].get("nominal_command_step", -1) if offset in values else -1
        for offset in SAMPLE_OFFSETS], np.int64)
    return result


def truncate_trace(trace, slot, first_termination_step):
    """Include the termination boundary; do not use post-termination motion."""
    end = len(trace["root"]) - 1 if first_termination_step is None else first_termination_step
    result = {key: value[:end + 1] for key, value in trace.items() if key != "env_origins"}
    result["env_origins"] = trace["env_origins"]
    return result, end


def _outcome(metrics_fn, trace, slot, origin, first_termination_step, task):
    observed, end = truncate_trace(trace, slot, first_termination_step)
    # Root supplies the geometry-aware metric hook. The legacy metric is safe
    # only after an explicit original-geometry equivalence check by the caller.
    outcome = metrics_fn(observed, slot=slot, origin=origin,
                         terminated=first_termination_step is not None,
                         control_dt=.02, geometry=task["geometry"])
    if first_termination_step is not None:
        outcome["success"] = False
    return outcome, end


def collect_factor_branches(wrapper, policy, request, *, candidate_factory=None,
                            metrics_fn=None):
    """Run an explicit M2 capsule list, keeping protocols separate from outcomes.

    Root integration must provide a geometry-aware metrics_fn, or expose
    m2_geometry.episode_outcome with the same keyword interface. request keeps
    existing sonic/reference/task/asset/output/cameras fields and adds
    m2_collection_config. One request is one already-built scene condition.
    """
    from .candidates import NativeCommand, encode_native
    from .kinematics import G1Kinematics
    from .native_reference import NativeReferencePatcher, install_reference_chunk
    from .perception import render_observations
    from .servo import ReferenceServo
    from .simulation import rebuild_physics

    if candidate_factory is None:
        from .m2_candidates import generate_factor_candidates
        candidate_factory = generate_factor_candidates
    if metrics_fn is None:
        raise ValueError("M2 collector requires an explicit geometry-aware outcome function")
    env = wrapper.env
    if not request.cameras or not np.isclose(env.step_dt, .02):
        raise ValueError("M2 requires real RGB-D cameras and the audited 50-Hz clock")
    config_path = Path(request.m2_collection_config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    total_steps = round(float(request.seconds) / env.step_dt)
    batches = validate_batches(config, env.num_envs, total_steps)
    tolerance = config.get("prefix_tolerance", {"default": 1e-6})
    task_path = request.task or Path(request.reference) / "task.json"
    task = json.loads(Path(task_path).read_text(encoding="utf-8"))
    planned = dict(np.load(Path(request.reference) / "planned.npz", allow_pickle=False))
    model = G1Kinematics(request.sonic)
    schedule = task["gripper_schedule"]
    command = env.command_manager.get_term("motion")
    lib = command.motion_lib
    if int(lib._num_motions) < env.num_envs:
        raise ValueError("Each physical slot needs an independent native timeline")
    original_fields = {name: value.clone() for name, value in NativeReferencePatcher(lib).fields.items()}
    origins = _array(env.scene.env_origins)
    joint_names = list(env.scene["robot"].joint_names)
    output = Path(request.output) / "m2-collection"
    output.mkdir(parents=True, exist_ok=False)
    provenance = {"configuration_sha256": sha256(config_path), "robot_asset_sha256": sha256(request.asset),
        "task_sha256": sha256(task_path), "reference_sha256": {name: sha256(Path(request.reference) / name)
            for name in ("motion.pkl", "planned.npz", "reference.json")},
        "paths": {"configuration": str(config_path.resolve()), "task": str(Path(task_path).resolve()),
            "asset": str(Path(request.asset).resolve()), "reference": str(Path(request.reference).resolve()),
            "request": str((Path(request.output) / "request.json").resolve()),
            "interface": str((Path(request.output) / "interface.json").resolve())},
        "sources": {str(path): sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))}}
    request_path = Path(request.output) / "request.json"
    provenance["request_sha256"] = sha256(request_path) if request_path.is_file() else None
    interface_path = Path(request.output) / "interface.json"
    provenance["interface_sha256"] = sha256(interface_path) if interface_path.is_file() else None
    manifest = {"schema": SCHEMA, "completed": False, "provenance": provenance,
        "configuration": config, "groups": [], "capsules": [], "passes": [], "cell_factors": CELL_FACTORS,
        "completed_physical_passes": 0, "physical_total_slot_rollouts": 0,
        "geometry": copy.deepcopy(task["geometry"]),
        "simulator_joint_names": joint_names, "mjcf_joint_names": model.joint_names,
        "sample_offsets_steps": SAMPLE_OFFSETS, "control_hz": 50, "execution_steps": EXECUTION,
        "native_horizon": HORIZON, "statistical_unit": "complete initial episode seed",
        "rendering": "Sensing 3.4..6.6 seconds every ten steps plus anchor/four evidence clocks; identical all-cell cadence",
        "cached_observation_replay": "Actual baseline RGB-D cache copied with each prefix patch; anchor cache copied after that clock's sensing and before any intervention; post-anchor sensing remains branch-specific",
        "coordinates": "world axes with environment-origin translation removed",
        "physical_contact_claim": "No contact-force measurement; kinematic pregrasp subset only"}
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)
    started = time.monotonic()
    torch.set_num_threads(1)

    for batch_spec in batches:
        capsule_id = batch_spec["capsule_id"]
        capsule_dir = output / capsule_id
        capsule_dir.mkdir()
        anchor_step = batch_spec["anchors"][0]["anchor_step"]
        active_count = len(batch_spec["anchors"])
        seeds = [item["seed"] for item in batch_spec["anchors"]] + batch_spec["companion_seeds"]
        group_rows = [{"id": f"seed-{item['seed']}-anchor-{anchor_step}",
            "condition": batch_spec["condition"], "family": batch_spec["family"],
            "seed": item["seed"], "anchor_step": anchor_step, "slot": slot,
            "capsule_id": capsule_id, "noise_repeat": bool(item.get("noise_repeat", False)),
            "noise_anchor": bool(item.get("noise_repeat", False)),
            "accepted": True, "rejections": [], "runs": []}
            for slot, item in enumerate(batch_spec["anchors"])]
        manifest["groups"].extend(group_rows)
        baseline_actions, baseline_grippers, baseline_patches = [], [], {}
        canonical, anchor_servo, anchor_estimates, baseline_initial, baseline_anchor = {}, {}, {}, {}, {}
        capsule_row = {"id": capsule_id, "seeds": seeds, "active_count": active_count,
            "condition": batch_spec["condition"], "family": batch_spec["family"],
            "anchor_step": anchor_step, "repeat_full_batch": any(g["noise_repeat"] for g in group_rows)}
        manifest["capsules"].append(capsule_row)
        repeats = range(3) if capsule_row["repeat_full_batch"] else range(1)

        for repeat in repeats:
            for candidate_id, (u, b) in enumerate(CELL_FACTORS):
                is_baseline = repeat == 0 and candidate_id == 0
                pass_dir = capsule_dir / f"repeat-{repeat}-cell-{candidate_id}"
                pass_dir.mkdir()
                servos = [ReferenceServo(request.sonic, planned, env.reference,
                                         task["geometry"], model=model) for _ in seeds]
                estimates = [None] * env.num_envs
                for name, value in original_fields.items():
                    getattr(lib, name).copy_(value)
                env.episode_seeds = seeds.copy()
                rebuild_physics(env)
                policy.init_rollout()
                observations = wrapper.process_raw_obs(env.reset_nominal(), True)
                if len(set(command.motion_ids.tolist())) != env.num_envs:
                    raise ValueError("Reset assigned a native timeline to multiple slots")
                done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
                first_termination = [None] * env.num_envs
                trace_records, delivery_rows, delivery_native = [env.capture()], [], []
                delivered_actions, delivered_grippers = [], []
                samples = [{} for _ in group_rows]
                latest_delivered = {}
                branch = {}
                run_rows = [{"candidate": candidate_id, "cell": {
                    "u_index": candidate_id // 3, "b_index": candidate_id % 3,
                    "u_sign": u, "b_sign": b},
                    "repeat": repeat, "analysis_included": repeat == 0 or group["noise_repeat"],
                    "condition": group["condition"], "family": group["family"],
                    "seed": group["seed"], "anchor_step": anchor_step, "slot": slot,
                    "capsule_id": capsule_id, "valid": True, "native_executed": False}
                    for slot, group in enumerate(group_rows)]
                for row in run_rows:
                    row.update(initial_archive=None, initial_snapshot_archive=None,
                        initial_snapshot_sha256=None, prefix_archive=None,
                        prefix_snapshot_archive=None, prefix_snapshot_sha256=None,
                        anchor_waist_rad=None)
                initial = trace_records[0]
                all_initial_snapshots = []
                for slot in range(env.num_envs):
                    state = history_snapshot(policy, observations, servos[slot], initial,
                                             slot, origins[slot], lib, int(command.motion_ids[slot]), 0,
                                             cached_estimates=estimates[slot])
                    initial_path = pass_dir / f"initial-slot-{slot}.npz"
                    _save_npz(initial_path, **state)
                    initial_hash = sha256(initial_path)
                    all_initial_snapshots.append({"slot": slot,
                        "archive": str(initial_path.relative_to(output)), "sha256": initial_hash})
                    if slot < active_count:
                        run_rows[slot].update(initial_archive=str(initial_path.relative_to(output)),
                            initial_snapshot_archive=str(initial_path.relative_to(output)),
                            initial_snapshot_sha256=initial_hash)
                    if is_baseline:
                        baseline_initial[slot] = state
                    else:
                        comparison = compare_history(baseline_initial[slot], state, tolerance)
                        if slot < active_count:
                            run_rows[slot]["initial_comparison"] = comparison
                        if not comparison["passed"]:
                            write_json(pass_dir / "protocol-error.json", {"slot": slot,
                                "reason": "initial history mismatch", "comparison": comparison})
                            raise ValueError(f"M2 {capsule_id} initial replay mismatch in slot {slot}")

                try:
                    for step in range(total_steps + 1):
                        elapsed = step * env.step_dt
                        measured = trace_records[-1]
                        if step == anchor_step:
                            for slot, group in enumerate(group_rows):
                                if bool(done[slot]):
                                    if is_baseline:
                                        group.update(anchor_not_reached=True, accepted=False)
                                        group["rejections"].append("anchor_not_reached: physical termination before intervention")
                                    run_rows[slot].update(valid=False, anchor_observable=False,
                                        invalid_reason="anchor_not_reached", native_executed=False)
                                elif group.get("anchor_not_reached", False):
                                    # A dead baseline has no common causal anchor.
                                    # A later branch that reaches it cannot be
                                    # substituted for the absent baseline state.
                                    raise ValueError("Baseline pre-anchor termination was not reproduced")
                        # Record +1 BEFORE the 50-step pulse is replaced by the
                        # continuation policy's newly observed feedback target.
                        offset = step - anchor_step
                        if offset in SAMPLE_OFFSETS:
                            for slot in range(active_count):
                                if first_termination[slot] is None or step <= first_termination[slot]:
                                    if slot in latest_delivered:
                                        delivered_step, native = latest_delivered[slot]
                                        samples[slot][offset] = {**physical_sample(model, measured, slot,
                                            origins[slot], joint_names), **nominal_sample(model, native, step - delivered_step),
                                            "nominal_command_step": delivered_step}
                        if step == total_steps:
                            break
                        replay_prefix = not is_baseline and step < anchor_step
                        # Keep every companion's complete baseline controls,
                        # not only active rows or the anchors selected for noise.
                        replay_slots = set(range(env.num_envs)) if replay_prefix else set()
                        if not is_baseline:
                            replay_slots.update(range(active_count, env.num_envs))
                        sensing = step % 10 == 0 and 3.4 <= elapsed <= 6.6
                        render_now = sensing or step == anchor_step or offset in SAMPLE_OFFSETS
                        images = (render_observations(env, list(range(env.num_envs)), refresh=True)
                                  if render_now else {})
                        if sensing:
                            for slot in range(env.num_envs):
                                if slot not in replay_slots:
                                    estimates[slot] = _local_estimates(images[slot]["estimates"], origins[slot])
                        if step % 10 == 0:
                            for slot, servo in enumerate(servos):
                                cursor = int(command.time_steps[slot] + command.motion_start_time_steps[slot])
                                if cursor != step:
                                    raise ValueError("M2 native cursor and physical control time disagree")
                                if slot in replay_slots:
                                    cached = baseline_patches.get((step, slot))
                                    if cached is None:
                                        # Original inactive slot may have terminated;
                                        # its already delivered references stay fixed.
                                        if bool(done[slot]):
                                            continue
                                        raise ValueError("Missing exact baseline native prefix")
                                    native, execution = cached["native"], cached["execution_steps"]
                                    _restore_servo(servo, cached["servo_state"])
                                    estimates[slot] = copy.deepcopy(cached["cached_estimates"])
                                elif bool(done[slot]):
                                    continue
                                elif slot < active_count and anchor_step < step < anchor_step + EXECUTION:
                                    # Includes default cell: no re-observation or
                                    # patching inside the same frozen intervention.
                                    continue
                                else:
                                    is_anchor = slot < active_count and step == anchor_step
                                    if is_anchor:
                                        # Current sensing is pre-intervention. A
                                        # renderer microdifference cannot create
                                        # a second initial policy cache: replay
                                        # the baseline's actual same-clock cache,
                                        # then let future sensing branch normally.
                                        if is_baseline:
                                            anchor_estimates[slot] = copy.deepcopy(estimates[slot])
                                        else:
                                            estimates[slot] = copy.deepcopy(anchor_estimates[slot])
                                        state = history_snapshot(policy, observations, servo, measured, slot,
                                            origins[slot], lib, int(command.motion_ids[slot]), cursor,
                                            cached_estimates=estimates[slot])
                                        prefix_path = pass_dir / f"prefix-slot-{slot}.npz"
                                        _save_npz(prefix_path, **state)
                                        run_rows[slot].update(prefix_archive=str(prefix_path.relative_to(output)),
                                            prefix_snapshot_archive=str(prefix_path.relative_to(output)),
                                            prefix_snapshot_sha256=sha256(prefix_path),
                                            anchor_waist_rad=float(state["q"][joint_names.index("waist_pitch_joint")]))
                                        if is_baseline:
                                            baseline_anchor[slot] = state
                                            anchor = physical_sample(model, measured, slot, origins[slot], joint_names)
                                            cube_heights = [_array(item["cube"])[slot, 2] - origins[slot, 2]
                                                            for item in trace_records]
                                            anchor.update(prefix_cube_initial_z=np.asarray(cube_heights[0]),
                                                          prefix_max_cube_z=np.asarray(max(cube_heights)))
                                            anchor_path = capsule_dir / f"anchor-slot-{slot}.npz"
                                            _save_npz(anchor_path, **anchor)
                                            group_rows[slot]["anchor_sample_archive"] = str(anchor_path.relative_to(output))
                                            group_rows[slot]["anchor_sample_sha256"] = sha256(anchor_path)
                                        else:
                                            comparison = compare_history(baseline_anchor[slot], state, tolerance)
                                            run_rows[slot]["prefix_comparison"] = comparison
                                            if not comparison["passed"]:
                                                raise ValueError(f"M2 {capsule_id}/{slot}: pre-intervention history mismatch")
                                    if is_anchor and not is_baseline:
                                        _restore_servo(servo, anchor_servo[slot])
                                        estimates[slot] = copy.deepcopy(anchor_estimates[slot])
                                        candidates = canonical[slot]
                                    else:
                                        servo.observe_targets(estimates[slot], elapsed)
                                        previous, correction, _ = servo.update(cursor,
                                            _array(measured["tcp"])[slot] - origins[slot],
                                            _array(measured["wrist_quat_wxyz"])[slot])
                                        frames = HORIZON if is_anchor else 75
                                        q, roots, quats = servo.trajectory(cursor, previous, correction, frames=frames)
                                        fingers = _fingers(schedule, cursor, frames=frames)
                                        native = NativeCommand(q, roots, quats, fingers)
                                        if is_anchor:
                                            native, bridge_report = bridge_installed_prefix(
                                                NativeReferencePatcher(lib), int(command.motion_ids[slot]), cursor, native)
                                            group_rows[slot]["common_baseline_bridge"] = bridge_report
                                            candidates = candidate_factory(model, *native,
                                                amplitude_m=float(config.get("amplitude_m", .010)),
                                                body_pitch_rad=float(config.get("body_pitch_rad", .03)),
                                                max_halvings=int(config.get("max_amplitude_halvings", 3)))
                                            if len(candidates.native) != 9 or candidates.actions.shape != (9, HORIZON, 38):
                                                raise ValueError("M2 factor candidate API must supply K9/H105")
                                            canonical[slot], anchor_servo[slot] = candidates, _servo_state(servo)
                                            canonical_path = capsule_dir / f"canonical-slot-{slot}.npz"
                                            _save_npz(canonical_path, actions=candidates.actions,
                                                feasible=candidates.feasible,
                                                qtraj_mjcf=np.stack([n[0] for n in candidates.native]),
                                                roots=np.stack([n[1] for n in candidates.native]),
                                                root_quaternion_wxyz=np.stack([n[2] for n in candidates.native]),
                                                finger_joint_targets=np.stack([n[3] for n in candidates.native]),
                                                targets_tcp=candidates.targets_tcp,
                                                targets_quaternion_wxyz=candidates.targets_quaternion_wxyz,
                                                nominal_tcp=candidates.nominal_tcp,
                                                nominal_quaternion_wxyz=candidates.nominal_quaternion_wxyz)
                                            group_rows[slot].update(canonical_archive=str(canonical_path.relative_to(output)),
                                                canonical_sha256=sha256(canonical_path),
                                                candidate_metadata=candidates.metadata,
                                                candidate_feasible=candidates.feasible.tolist())
                                    execution = EXECUTION if is_anchor else 10
                                    if is_anchor:
                                        grid_feasible = bool(candidates.feasible.all())
                                        run_rows[slot]["cell"].update(
                                            u_m=u * float(candidates.metadata["effective_amplitude_m"]),
                                            b_rad=b * float(candidates.metadata["effective_body_pitch_rad"]),
                                            common_scale=float(candidates.metadata["common_scale"]))
                                        if not grid_feasible:
                                            group_rows[slot]["accepted"] = False
                                            if "canonical grid infeasible" not in group_rows[slot]["rejections"]:
                                                group_rows[slot]["rejections"].append("canonical grid infeasible")
                                            # Preserve the seed as an explicit
                                            # coverage failure; no candidate is
                                            # presented as an executed grid row.
                                            # The same default fallback runs in
                                            # every cell to retain companions.
                                            run_rows[slot].update(valid=False,
                                                invalid_reason="canonical grid infeasible; diagnostic default continuation",
                                                candidate_grid_feasible=False)
                                        executed = candidate_id if grid_feasible else 0
                                        native = candidates.native[executed]
                                        if not np.array_equal(encode_native(native), candidates.actions[executed]):
                                            raise ValueError("M2 native/action codec identity mismatch")
                                        branch[slot] = native
                                        run_rows[slot].update(native_executed=grid_feasible,
                                            executed_candidate=executed,
                                            native_sha256=_native_digest(native),
                                            canonical_native_sha256=_native_digest(candidates.native[executed]))
                                patch = install_reference_chunk(lib, int(command.motion_ids[slot]), cursor,
                                    *native[:3], execution_steps=execution, preview_frames=45)
                                latest_delivered[slot] = (step, native)
                                if is_baseline:
                                    baseline_patches[(step, slot)] = {"native": tuple(np.array(x, copy=True) for x in native),
                                        "servo_state": _servo_state(servo),
                                        "cached_estimates": copy.deepcopy(estimates[slot]),
                                        "execution_steps": execution}
                                delivery_rows.append({"step": step, "slot": slot, "execution_steps": execution,
                                    "native_index": len(delivery_native),
                                    "native_sha256": _native_digest(native), "patch": patch,
                                    "servo_state": _servo_state(servo),
                                    "cached_estimates": copy.deepcopy(estimates[slot]),
                                    "replayed_prefix": slot in replay_slots})
                                delivery_native.append(native)
                            observations = wrapper.process_raw_obs(
                                env.observation_manager.compute(update_history=False), True)
                        for slot in range(env.num_envs):
                            if slot in replay_slots:
                                env.gripper_target[slot] = env.gripper_target.new_tensor(baseline_grippers[step][slot])
                            elif slot in branch and anchor_step <= step < anchor_step + EXECUTION:
                                env.gripper_target[slot] = env.gripper_target.new_tensor(branch[slot][3][step - anchor_step])
                            else:
                                env.gripper_target[slot] = env.gripper_target.new_tensor(_fingers(schedule, step, frames=1)[0])
                        with torch.no_grad():
                            action = policy.act_inference(obs_dict=observations, cur_dones=done,
                                                          skip_episode_attnmask=True).clone()
                            for slot in replay_slots:
                                action[slot] = baseline_actions[step][slot].to(action.device)
                            delivered_actions.append(_array(action))
                            delivered_grippers.append(_array(env.gripper_target))
                            if is_baseline:
                                baseline_actions.append(action.detach().clone())
                                baseline_grippers.append(_array(env.gripper_target))
                            observations, _, newly_done, _ = wrapper.step({"actions": action})
                        newly_done = newly_done.reshape(-1).bool()
                        for slot in range(env.num_envs):
                            if bool(newly_done[slot]) and first_termination[slot] is None:
                                first_termination[slot] = step + 1
                        done |= newly_done
                        trace_records.append(env.capture())
                except Exception as error:
                    for group in group_rows:
                        group["accepted"] = False
                        group["rejections"].append(f"protocol error: {type(error).__name__}: {error}")
                    write_json(pass_dir / "protocol-error.json", {"type": type(error).__name__, "message": str(error)})
                    write_json(manifest_path, manifest)
                    raise
                finally:
                    trace = {key: np.stack([_array(record[key]) for record in trace_records])
                             for key in trace_records[0]}
                    trace["env_origins"] = origins
                    trace["physical_valid"] = np.stack([np.arange(len(trace_records)) <= (
                        total_steps if end is None else end) for end in first_termination], axis=1)
                    trace_path = pass_dir / "trace.npz"
                    _save_npz(trace_path, **trace)
                    write_json(pass_dir / "commands.json", delivery_rows)
                    # Variable 75/105-frame commands are individual arrays in a
                    # shared archive, preserving dtype and avoiding object NPZs.
                    command_fields = {}
                    for index, native in enumerate(delivery_native):
                        for name, value in zip(("q", "roots", "quaternions", "fingers"), native):
                            command_fields[f"command_{index:05d}_{name}"] = value
                    _save_npz(pass_dir / "commands.npz", **command_fields)
                    _save_npz(pass_dir / "controls.npz",
                        actions=np.stack(delivered_actions) if delivered_actions else np.zeros((0, env.num_envs, 0)),
                        gripper_targets=np.stack(delivered_grippers) if delivered_grippers else np.zeros((0, env.num_envs, 2)))
                if is_baseline:
                    capsule_row["control_archive"] = str((pass_dir / "controls.npz").relative_to(output))
                    capsule_row["control_archive_sha256"] = sha256(pass_dir / "controls.npz")
                    write_json(capsule_dir / "servo-prefix.json", [{"step": step, "slot": slot,
                        "execution_steps": row["execution_steps"],
                        "cached_estimates": row["cached_estimates"], **row["servo_state"]}
                        for (step, slot), row in sorted(baseline_patches.items())])
                    capsule_row["servo_prefix_archive"] = str((capsule_dir / "servo-prefix.json").relative_to(output))
                    capsule_row["servo_prefix_sha256"] = sha256(capsule_dir / "servo-prefix.json")
                shared_hashes = {"trace_sha256": sha256(trace_path),
                    "controls_sha256": sha256(pass_dir / "controls.npz"),
                    "commands_sha256": sha256(pass_dir / "commands.npz"),
                    "commands_json_sha256": sha256(pass_dir / "commands.json")}
                for slot, group in enumerate(group_rows):
                    row = run_rows[slot]
                    sample_path = pass_dir / f"samples-slot-{slot}.npz"
                    _save_npz(sample_path, **sample_arrays(anchor_step, samples[slot]))
                    outcome, end = _outcome(metrics_fn, trace, slot, origins[slot], first_termination[slot], task)
                    row.update(trace_archive=str(trace_path.relative_to(output)),
                        samples_archive=str(sample_path.relative_to(output)),
                        commands_archive=str((pass_dir / "commands.npz").relative_to(output)),
                        controls_archive=str((pass_dir / "controls.npz").relative_to(output)),
                        commands_json=str((pass_dir / "commands.json").relative_to(output)),
                        first_termination_step=first_termination[slot], outcome=outcome,
                        outcome_observation_end_step=end, d24_valid=first_termination[slot] is None,
                        measured_sample_offsets_steps=sorted(samples[slot]),
                        outcome_budget_seconds=end / 50.,
                        samples_sha256=sha256(sample_path), **shared_hashes)
                    if first_termination[slot] is not None and first_termination[slot] <= anchor_step:
                        row.update(anchor_observable=False, native_executed=False)
                    else:
                        row["anchor_observable"] = True
                    group["runs"].append(row)
                manifest["completed_physical_passes"] += 1
                manifest["physical_total_slot_rollouts"] += env.num_envs
                write_json(pass_dir / "branches.json", run_rows)
                manifest["passes"].append({"capsule_id": capsule_id, "repeat": repeat,
                    "candidate": candidate_id, "active_count": active_count,
                    "physical_slot_rollouts": env.num_envs,
                    "all_initial_snapshots": all_initial_snapshots,
                    "branches_archive": str((pass_dir / "branches.json").relative_to(output)),
                    "branches_sha256": sha256(pass_dir / "branches.json"), **shared_hashes})
                write_json(manifest_path, manifest)
                print(f"G1_M2 capsule={capsule_id} repeat={repeat} cell={candidate_id} complete", flush=True)
        capsule_row["canonical_reused_across_cells_and_repeats"] = True
        capsule_row["all_companions_replayed_from_common_baseline"] = True
        expected_runs = 27 if capsule_row["repeat_full_batch"] else 9
        for group in group_rows:
            expected_ids = {(repeat, candidate) for repeat in repeats for candidate in range(9)}
            actual_ids = {(row["repeat"], row["candidate"]) for row in group["runs"]}
            if len(group["runs"]) != expected_runs or actual_ids != expected_ids:
                raise ValueError("M2 incomplete run inventory is a protocol failure, not physical censoring")
        write_json(manifest_path, manifest)
    manifest.update(completed=True, wall_seconds=time.monotonic() - started,
        attempted_anchors=len(manifest["groups"]),
        valid_grid_anchors=sum(group["accepted"] for group in manifest["groups"]),
        analysis_branches=sum(row["analysis_included"] for group in manifest["groups"] for row in group["runs"]),
        physical_active_branch_rows=sum(len(group["runs"]) for group in manifest["groups"]),
        physical_branches=manifest["physical_total_slot_rollouts"],
        physical_padding_slot_rollouts=manifest["physical_total_slot_rollouts"] - sum(len(group["runs"]) for group in manifest["groups"]),
        physical_extra_active_companion_rollouts=sum(not row["analysis_included"]
            for group in manifest["groups"] for row in group["runs"]))
    write_json(manifest_path, manifest)
    return {"completed": True, "mode": "m2_collect", "manifest": str(manifest_path),
        "attempted_anchors": manifest["attempted_anchors"],
        "analysis_branches": manifest["analysis_branches"],
        "physical_active_branch_rows": manifest["physical_active_branch_rows"],
        "physical_total_slot_rollouts": manifest["physical_total_slot_rollouts"],
        "physical_branches": manifest["physical_branches"]}
