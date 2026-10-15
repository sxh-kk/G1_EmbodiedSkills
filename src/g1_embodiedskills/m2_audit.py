"""Independent CPU verification of M2 physical records, not effect estimates.

The auditor reads the actual archives and reconstructs every gate. No recorded
``passed`` flag substitutes for a state, reference, or outcome comparison.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


SCHEMA = "g1_embodiedskills.m2_collection.v1"
FACTORS = tuple((u, b) for u in (0, 1, -1) for b in (0, 1, -1))
OFFSETS = (10, 25, 50, 150)
PARTS = ("q", "roots", "quaternions", "fingers")
TRACE_SHAPES = {"root": (13,), "q": (31,), "qdot": (31,), "tcp": (3,),
    "wrist_quat_wxyz": (4,), "reference_tcp": (3,), "cube": (3,),
    "cube_root": (13,), "gripper": (2,), "time": (), "terminated": (), "physical_valid": ()}
REFERENCE_FIELDS = ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w",
    "body_pos_w_full", "body_quat_w_full", "body_lin_vel_w_full", "body_ang_vel_w_full",
    "body_pos_b", "root_linv_vel_w", "root_ang_vel_w", "dof_pos", "dof_vel", "feet_l", "feet_r")
SNAPSHOT_FIELDS = ("root", "q", "qdot", "actor_obs", "tokenizer", "servo_correction",
    "cube_offset", "tray_offset", "cube_root", "servo_target_valid", "servo_target_time",
    "policy_steps", "policy_history_keys", "policy_dones_present", "policy_dones",
    "cached_estimates_present", "cached_estimates_json")
SAMPLE_SHAPES = {"actual_tcp": (3,), "actual_R": (3, 3), "cube_root": (13,),
    "root": (13,), "q": (31,), "qdot": (31,), "gripper": (2,), "waist": (),
    "soles": (2, 3), "nominal_tcp": (3,), "nominal_R": (3, 3)}
_UNSET = object()
RIGHT_ARM_NAMES = tuple("right_"+name+"_joint" for name in (
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"))


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def integer(value, label, *, minimum=0, maximum=None):
    require(isinstance(value, int) and not isinstance(value, bool), f"{label}: expected integer")
    require(value >= minimum and (maximum is None or value <= maximum), f"{label}: integer outside bounds")
    return value


def boolean(value, label):
    require(type(value) is bool, f"{label}: expected JSON boolean")
    return value


def close(left, right, label, *, atol=1e-8):
    left, right = np.asarray(left), np.asarray(right)
    require(left.shape == right.shape, f"{label}: shape differs: {left.shape} vs {right.shape}")
    require(np.isfinite(left).all() and np.isfinite(right).all(), f"{label}: non-finite comparison")
    maximum = float(np.max(np.abs(left.astype(float)-right.astype(float)), initial=0))
    require(maximum <= atol, f"{label}: max error {maximum:g} exceeds {atol:g}")
    return maximum


def bitwise(left, right, label):
    left, right = np.asarray(left), np.asarray(right)
    require(left.shape == right.shape and left.dtype == right.dtype
        and left.tobytes() == right.tobytes(), f"{label}: dtype/shape/bytes differ")


def _canonical_json(value, label):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise AuditError(f"{label}: cache is not finite, complete JSON: {error}") from error


def validate_cached_estimates(value, label):
    """Validate actual cached perception contents, including absence and flags.

    Every original field remains in the canonical payload. Known detector
    fields are checked independently, so a hash or empty placeholder cannot
    stand in for the mutable planner input that will be used at continuation.
    """
    _canonical_json(value, label)
    if value is None:
        return value
    require(type(value) is dict and set(value) == {"cube", "tray"}, f"{label}: cached target inventory is incomplete")
    required = {"target", "found", "position", "surface_position", "mask_count",
        "valid_depth_count", "confidence", "reason", "pixel_centroid", "method"}
    for target in ("cube", "tray"):
        item = value[target]
        require(type(item) is dict and required <= set(item), f"{label}/{target}: incomplete cached detection")
        require(item["target"] == target, f"{label}/{target}: cached target name differs")
        found = boolean(item["found"], f"{label}/{target}/found")
        masks = integer(item["mask_count"], f"{label}/{target}/mask_count")
        depths = integer(item["valid_depth_count"], f"{label}/{target}/valid_depth_count")
        require(depths <= masks, f"{label}/{target}: cached depth count exceeds mask count")
        confidence = item["confidence"]
        require(type(confidence) in (int, float) and 0 <= confidence <= 1,
            f"{label}/{target}: invalid cached confidence")
        for key in ("reason", "method"):
            require(type(item[key]) is str and bool(item[key]), f"{label}/{target}/{key}: missing detector semantics")
        for key, width in (("position", 3), ("surface_position", 3), ("pixel_centroid", 2)):
            if found:
                array = np.asarray(item[key])
                require(array.shape == (width,) and array.dtype.kind in "fiu" and np.isfinite(array).all(),
                    f"{label}/{target}/{key}: cached detected value missing/nonfinite")
            else:
                require(item[key] is None, f"{label}/{target}/{key}: undetected target has an invented pose")
    return value


def snapshot_cached_estimates(snapshot, label):
    presence = np.asarray(snapshot["cached_estimates_present"])
    payload = np.asarray(snapshot["cached_estimates_json"])
    require(presence.shape == () and presence.dtype == np.uint8 and int(presence) in (0, 1),
        f"{label}: invalid cached-estimate presence flag")
    require(payload.ndim == 1 and payload.dtype == np.uint8, f"{label}: cached estimates must contain actual UTF8 JSON bytes")
    try:
        value = json.loads(payload.tobytes().decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise AuditError(f"{label}: invalid cached-estimate JSON content") from error
    validate_cached_estimates(value, label)
    require(bool(presence) == (value is not None), f"{label}: cached-estimate presence/content differ")
    require(payload.tobytes() == _canonical_json(value, label), f"{label}: cached-estimate payload is not canonical complete JSON")
    return value


def same_cached_estimates(left, right, label):
    validate_cached_estimates(left, f"{label}/left")
    validate_cached_estimates(right, f"{label}/right")
    require(_canonical_json(left, label) == _canonical_json(right, label), f"{label}: actual cached detection contents differ")


def validate_delivery_servo_state(value, label):
    require(type(value) is dict and set(value) == {"correction", "cube_offset", "tray_offset", "last_estimate_time"},
        f"{label}: incomplete actual delivery Servo state")
    for key, width in (("correction", 29), ("cube_offset", 3), ("tray_offset", 3)):
        array = np.asarray(value[key])
        require(array.shape == (width,) and array.dtype.kind in "fiu" and np.isfinite(array).all(), f"{label}/{key}: invalid Servo state")
    require(type(value["last_estimate_time"]) is dict and set(value["last_estimate_time"]) == {"cube", "tray"},
        f"{label}: incomplete Servo perception times")
    for time in value["last_estimate_time"].values():
        require(time is None or (type(time) in (int, float) and np.isfinite(time) and time >= 0), f"{label}: invalid Servo perception time")


def native_digest(native):
    digest = hashlib.sha256()
    for value in native:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def schedule_targets(schedule, steps=1200):
    require(isinstance(schedule, list) and schedule, "Missing gripper schedule")
    targets = np.full((steps, 2), .035, float)
    previous = -np.inf
    for event in schedule:
        time, value = float(event["time"]), float(event["joint_target_m"])
        require(np.isfinite([time, value]).all() and time >= previous, "Invalid gripper schedule")
        require(0 <= value <= .035, "Gripper target outside physical bounds")
        targets[np.arange(steps)/50 >= time] = value
        previous = time
    return targets


def compare_snapshots(baseline, actual, tolerance, label):
    require(set(baseline) == set(actual), f"{label}: snapshot field inventory differs")
    same_cached_estimates(snapshot_cached_estimates(baseline, f"{label}/baseline cache"),
        snapshot_cached_estimates(actual, f"{label}/actual cache"), f"{label}/replayed perception cache")
    maximum = {}
    for key in baseline:
        limit = tolerance.get(key, tolerance.get("default", 1e-6)) if isinstance(tolerance, dict) else tolerance
        require(np.isfinite(limit) and 0 <= limit <= 1e-3, f"{label}/{key}: invalid or excessive prefix tolerance")
        maximum[key] = close(baseline[key], actual[key], f"{label}/{key}", atol=float(limit))
    return maximum


def validate_trace(trace, controls, count, label):
    for field, shape in TRACE_SHAPES.items():
        require(field in trace and trace[field].shape == (1201, count, *shape), f"{label}/{field}: incomplete physical trace")
        require(np.isfinite(trace[field]).all(), f"{label}/{field}: non-finite trace")
    require(trace["env_origins"].shape == (count, 3), f"{label}: invalid environment origins")
    for key in ("terminated", "physical_valid"):
        require(trace[key].dtype == np.bool_, f"{label}/{key}: expected bool array")
    require(controls["actions"].shape == (1200, count, 29), f"{label}: incomplete delivered controls")
    require(controls["gripper_targets"].shape == (1200, count, 2), f"{label}: incomplete delivered gripper targets")
    require(all(np.isfinite(value).all() for value in controls.values()), f"{label}: non-finite delivered controls")
    close(trace["time"], np.repeat((np.arange(1201)/50)[:, None], count, axis=1), f"{label}/physical clock", atol=5e-5)
    close(trace["cube"], trace["cube_root"][..., :3], f"{label}/cube state identity", atol=0)
    endings = []
    for slot in range(count):
        flags = trace["terminated"][:, slot]
        require(not flags[0], f"{label}/{slot}: terminated reset")
        ending = int(np.flatnonzero(flags)[0]) if flags.any() else None
        endings.append(ending)
        expected = np.arange(1201) <= (1200 if ending is None else ending)
        bitwise(trace["physical_valid"][:, slot], expected, f"{label}/{slot}/termination validity")
    return endings


class Reader:
    def __init__(self):
        self.hashes = {}
        self.collection_pinned = set()

    def pin(self, path, expected=_UNSET):
        path = Path(path).resolve()
        require(path.is_file(), f"Missing artifact: {path}")
        if expected is not _UNSET:
            require(type(expected) is str and len(expected) == 64 and set(expected) <= set("0123456789abcdef"), f"Missing/malformed collection-time SHA256: {path}")
            self.collection_pinned.add(str(path))
        if str(path) in self.hashes:
            require(expected is _UNSET or self.hashes[str(path)] == expected, f"Conflicting pinned SHA256: {path}")
            return path
        digest = file_digest(path)
        require(expected is _UNSET or digest == expected, f"SHA256 mismatch: {path}")
        previous = self.hashes.get(str(path))
        require(previous is None or previous == digest, f"Artifact changed during read: {path}")
        self.hashes[str(path)] = digest
        return path

    def artifact(self, base, value, expected=_UNSET):
        require(expected is not _UNSET, "Recorded evidence artifact has no collection-time SHA256")
        path = Path(value)
        path = path if path.is_absolute() else Path(base)/path
        path = path.resolve()
        require(path.is_relative_to(Path(base).resolve()), f"Artifact escapes collection directory: {path}")
        return self.pin(path, expected)

    def json(self, path, expected=_UNSET):
        path = self.pin(path, expected)
        return json.loads(path.read_text(encoding="utf-8"))

    def npz(self, path, expected=_UNSET):
        path = self.pin(path, expected)
        with np.load(path, allow_pickle=False) as source:
            return {key: source[key] for key in source.files}

    def finish(self):
        after = {path: file_digest(path) for path in self.hashes}
        require(after == self.hashes, "Source/evidence artifacts changed during independent audit")
        return {"unchanged": True, "artifact_count": len(after),
            "collection_time_pinned_count": len(self.collection_pinned), "before": dict(self.hashes), "after": after}


def _snapshot_shape(snapshot, label, *, step):
    require(all(key in snapshot for key in SNAPSHOT_FIELDS), f"{label}: incomplete actor/Servo/physical snapshot")
    cached = snapshot_cached_estimates(snapshot, f"{label}/cached estimates")
    require(step != 0 or cached is None, f"{label}: initial cached estimates must be absent")
    require(step < 170 or cached is not None, f"{label}: cached perception missing after mandatory sensing window")
    history_fields = sorted(key.removeprefix("policy_observation/") for key in snapshot if key.startswith("policy_observation/"))
    declared_history = bytes(snapshot["policy_history_keys"].astype(np.uint8)).decode()
    require(declared_history == "\n".join(history_fields), f"{label}: actor history key inventory differs")
    if step == 0:
        require(not history_fields and snapshot["policy_steps"] == 0
            and snapshot["policy_dones_present"] == 0 and snapshot["policy_dones"].size == 0,
            f"{label}: initial actor history differs from actual init_rollout semantics")
    else:
        require(bool(history_fields), f"{label}: missing actor temporal observations")
    require(all("reference/"+key in snapshot for key in REFERENCE_FIELDS), f"{label}: missing installed native reference state")
    require(snapshot["q"].shape == (31,) and snapshot["qdot"].shape == (31,), f"{label}: invalid joint snapshot")
    require(snapshot["root"].shape == (13,) and snapshot["cube_root"].shape == (13,), f"{label}: incomplete root/object snapshot")
    for key, shape in (("servo_correction", (29,)), ("cube_offset", (3,)), ("tray_offset", (3,)),
            ("servo_target_valid", (2,)), ("servo_target_time", (2,)), ("policy_steps", ()), ("policy_dones_present", ())):
        require(snapshot[key].shape == shape, f"{label}/{key}: incomplete history state")
    require(snapshot["policy_steps"].dtype.kind in "iu", f"{label}: policy counter is not integer")
    require(int(snapshot["policy_steps"]) == step, f"{label}: actor history count differs from actual delivered controls")
    for key in ("servo_target_valid", "policy_dones_present", "policy_dones"):
        require(np.isin(snapshot[key], [0, 1]).all(), f"{label}/{key}: nonbinary history flags")
    for field in REFERENCE_FIELDS:
        require(snapshot["reference/"+field].shape[0] == 105+min(step, 9), f"{label}/{field}: incomplete native preview/history")
    require(all(np.isfinite(value).all() for value in snapshot.values()), f"{label}: non-finite state snapshot")


def _snapshot_trace(snapshot, trace, slot, step, label):
    origin = trace["env_origins"][slot]
    for field in ("root", "cube_root"):
        measured = trace[field][step, slot].copy()
        measured[:3] -= origin
        if measured[3] < 0:
            measured[3:7] *= -1
        close(snapshot[field], measured, f"{label}/{field} vs trace", atol=1e-8)
    for field in ("q", "qdot"):
        close(snapshot[field], trace[field][step, slot], f"{label}/{field} vs trace", atol=0)


def _command_grid(rows, archive, count, anchor, endings, label):
    by_slot = {slot: {} for slot in range(count)}
    seen_indices = set()
    for row in rows:
        step = integer(row["step"], f"{label}/command step", maximum=1199)
        slot = integer(row["slot"], f"{label}/command slot", maximum=count-1)
        index = integer(row["native_index"], f"{label}/native_index")
        validate_cached_estimates(row["cached_estimates"], f"{label}/{slot}/{step}/actual delivered cache")
        validate_delivery_servo_state(row["servo_state"], f"{label}/{slot}/{step}/actual delivered Servo state")
        require((row["cached_estimates"] is not None) == (step >= 170), f"{label}/{slot}/{step}: actual cache presence differs from sensing protocol")
        require(index not in seen_indices and step not in by_slot[slot], f"{label}: duplicate command index or clock")
        seen_indices.add(index)
        require(step % 10 == 0, f"{label}: command off the declared replanning boundary")
        native = tuple(archive[f"command_{index:05d}_{part}"] for part in PARTS)
        frames = len(native[0])
        for value, shape in zip(native, ((frames, 29), (frames, 3), (frames, 4), (frames, 2))):
            require(value.shape == shape and np.issubdtype(value.dtype, np.floating) and np.isfinite(value).all(), f"{label}: malformed native command")
        require(native_digest(native) == row["native_sha256"], f"{label}: delivered native digest differs")
        execution = integer(row["execution_steps"], f"{label}/execution", minimum=1)
        require(execution in (10, 50) and frames >= execution+55, f"{label}: insufficient controller preview")
        patch = row["patch"]
        require(patch["cursor"] == step and patch["patched_frames"] == frames
            and patch["last_execution_preview_frame"] == step+execution+45, f"{label}: actual patch cursor/window differs")
        require(boolean(patch["prefix_preserved"], f"{label}/prefix_preserved"), f"{label}: patch reports prefix mutation")
        require(not boolean(patch["gripper_fields_modified"], f"{label}/gripper_fields_modified"), f"{label}: unexpected gripper patch")
        by_slot[slot][step] = native
    require(seen_indices == set(range(len(rows))), f"{label}: noncontiguous native indices")
    require(set(archive) == {f"command_{index:05d}_{part}" for index in seen_indices for part in PARTS}, f"{label}: extra/missing native arrays")
    return by_slot


def _canonical(reader, base, group, model, joint_names, label):
    path = reader.artifact(base, group["canonical_archive"], group["canonical_sha256"])
    grid = reader.npz(path)
    require(grid["actions"].shape == (9, 105, 38), f"{label}: expected canonical K9/H105/D38")
    native = tuple(tuple(grid[key][cell] for key in ("qtraj_mjcf", "roots", "root_quaternion_wxyz", "finger_joint_targets")) for cell in range(9))
    for cell, command in enumerate(native):
        encoded = np.concatenate(command, axis=-1).astype(np.float32)
        bitwise(grid["actions"][cell], encoded, f"{label}/{cell}/canonical encoding")
        require(all(np.isfinite(value).all() for value in command), f"{label}: non-finite canonical reference")
        for part in range(4):
            bitwise(command[part][0], native[0][part][0], f"{label}/{cell}/common frame zero")
        for part in (1, 2, 3):
            bitwise(command[part], native[0][part], f"{label}/{cell}/root/quaternion/gripper invariance")
        bitwise(command[0][50:], native[0][0][50:], f"{label}/{cell}/returned nominal tail")
    feasible = np.asarray(grid["feasible"])
    require(feasible.shape == (9,) and feasible.dtype == np.bool_, f"{label}: malformed feasibility mask")
    require(feasible.tolist() == group["candidate_feasible"], f"{label}: feasibility metadata differs")
    require(boolean(group["accepted"], f"{label}/accepted") == bool(feasible.all()), f"{label}: accepted grid feasibility mismatch")
    metadata = group["candidate_metadata"]
    require(metadata["factor_grid"] == [list(pair) for pair in FACTORS], f"{label}: altered factor order")
    scale = float(metadata["common_scale"])
    require(np.isfinite(scale) and 0 < scale <= 1, f"{label}: invalid common shrink")
    close(metadata["effective_amplitude_m"], scale*metadata["requested_amplitude_m"], f"{label}/shared hand shrink", atol=1e-12)
    close(metadata["effective_body_pitch_rad"], scale*metadata["requested_body_pitch_rad"], f"{label}/shared body shrink", atol=1e-12)
    waist = joint_names.index("waist_pitch_joint")
    arm = [joint_names.index(name) for name in RIGHT_ARM_NAMES]
    unchanged = np.setdiff1d(np.arange(29), [waist, *arm])
    times = np.arange(105)/50
    rising = np.clip(times/.4, 0, 1)
    falling = np.clip((1-times)/.4, 0, 1)
    smooth = lambda value: value**3*(10-15*value+6*value*value)
    pulse = np.minimum(smooth(rising), smooth(falling))
    for cell, (u, b) in enumerate(FACTORS):
        expected_waist = native[0][0][:, waist]+b*metadata["effective_body_pitch_rad"]*pulse
        close(native[cell][0][:, waist], expected_waist, f"{label}/{cell}/actual body pulse", atol=1e-6)
        bitwise(native[cell][0][:, unchanged], native[0][0][:, unchanged], f"{label}/{cell}/unrequested joints unchanged")
        expected_target = grid["nominal_tcp"][0].copy()
        expected_target[:, 0] += u*metadata["effective_amplitude_m"]*pulse
        close(grid["targets_tcp"][cell], expected_target, f"{label}/{cell}/same world hand pulse", atol=1e-10)
    require(grid["nominal_tcp"].shape == (9, 105, 3), f"{label}: missing canonical nominal TCPs")
    positions, rotations = None, None
    if model is not None:
        q = grid["qtraj_mjcf"]
        positions_all, matrices = model.transforms(q, grid["roots"], grid["root_quaternion_wxyz"])
        rotations = matrices[..., model.right_wrist_index, :, :]
        positions = positions_all[..., model.right_wrist_index, :] + np.einsum("...ij,j->...i", rotations, [.085, 0., 0.])
        close(grid["nominal_tcp"], positions, f"{label}/canonical official FK", atol=1e-7)
        recorded_R = Rotation.from_quat(grid["nominal_quaternion_wxyz"][..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(9, 105, 3, 3)
        close(recorded_R, rotations, f"{label}/canonical official orientation FK", atol=1e-7)
    else:
        positions = grid["nominal_tcp"]
        rotations = Rotation.from_quat(grid["nominal_quaternion_wxyz"][..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(9, 105, 3, 3)
    target_R = Rotation.from_quat(grid["targets_quaternion_wxyz"][..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(9, 105, 3, 3)
    close(target_R, np.broadcast_to(rotations[0], target_R.shape), f"{label}/common requested world orientation", atol=1e-7)
    target_position_error = float(np.linalg.norm(positions-grid["targets_tcp"], axis=-1).max())
    target_orientation_error = float(Rotation.from_matrix((rotations @ np.swapaxes(target_R, -1, -2)).reshape(-1, 3, 3)).magnitude().max())
    max_position, max_rotation = 0., 0.
    for hand in range(3):
        for left, right in ((0, 1), (0, 2), (1, 2)):
            a, b = 3*hand+left, 3*hand+right
            max_position = max(max_position, float(np.linalg.norm(positions[a]-positions[b], axis=-1).max()))
            relative = rotations[a] @ np.swapaxes(rotations[b], -1, -2)
            max_rotation = max(max_rotation, float(Rotation.from_matrix(relative).magnitude().max()))
    if group["accepted"]:
        require(max_position <= .002 and max_rotation <= .02, f"{label}: same-hand body conditions change nominal pose beyond tolerance")
        require(target_position_error <= .002 and target_orientation_error <= .02, f"{label}: actual native FK differs from requested world pose")
        require(len({value.tobytes() for value in grid["actions"]}) == 9, f"{label}: duplicate canonical factor cells")
    return grid, native, {"max_same_hand_position_m": max_position, "max_same_hand_orientation_rad": max_rotation}


def _sample_expected(trace, slot, step, joint_names, model):
    origin = trace["env_origins"][slot]
    root, cube = trace["root"][step, slot].copy(), trace["cube_root"][step, slot].copy()
    root[:3] -= origin
    cube[:3] -= origin
    q = trace["q"][step, slot]
    result = {"actual_tcp": trace["tcp"][step, slot]-origin,
        "actual_R": Rotation.from_quat(trace["wrist_quat_wxyz"][step, slot][[1, 2, 3, 0]]).as_matrix(),
        "root": root, "cube_root": cube, "q": q, "qdot": trace["qdot"][step, slot],
        "gripper": trace["gripper"][step, slot], "waist": np.asarray(q[joint_names.index("waist_pitch_joint")])}
    if model is not None:
        mapping = [joint_names.index(name) for name in model.joint_names]
        positions, matrices = model.transforms(q[mapping], root[:3], root[3:7])
        matrix = matrices[model.right_wrist_index]
        tcp = positions[model.right_wrist_index]+matrix @ [.085, 0., 0.]
        close(result["actual_tcp"], tcp, "Actual physical TCP vs official joint/root FK", atol=1e-4)
        orientation_error = float(Rotation.from_matrix(result["actual_R"] @ matrix.T).magnitude())
        require(orientation_error <= .001, "Actual wrist orientation differs from official joint/root FK")
        result["soles"] = np.stack([positions[model.body_names.index(name)]
            + matrices[model.body_names.index(name)] @ offset for name, offset in model.sole_offsets.items()])
    return result


def _samples(reader, path, expected_hash, trace, slot, anchor, ending, commands, native_cell,
             joint_names, model, label):
    sample = reader.npz(path, expected_hash)
    bitwise(sample["offset_steps"], np.asarray(OFFSETS, np.int64), f"{label}/sample offsets")
    bitwise(sample["global_steps"], anchor+np.asarray(OFFSETS, np.int64), f"{label}/sample clocks")
    close(sample["offset_seconds"], np.asarray(OFFSETS)/50, f"{label}/sample seconds", atol=0)
    require(sample["valid"].shape == (4,) and sample["valid"].dtype == np.bool_, f"{label}: invalid sample mask")
    expected_valid = np.asarray([ending is None or anchor+offset <= ending for offset in OFFSETS])
    bitwise(sample["valid"], expected_valid, f"{label}/sample censoring")
    for key, shape in SAMPLE_SHAPES.items():
        require(sample[key].shape == (4, *shape), f"{label}/{key}: invalid sample shape")
        require(np.isnan(sample[key][~expected_valid]).all(), f"{label}/{key}: post-termination value imputed")
        require(np.isfinite(sample[key][expected_valid]).all(), f"{label}/{key}: missing value in observed sample")
    for index, offset in enumerate(OFFSETS):
        step = anchor+offset
        if not expected_valid[index]:
            require(sample["nominal_command_step"][index] == -1, f"{label}: censored nominal command was invented")
            continue
        expected = _sample_expected(trace, slot, step, joint_names, model)
        for key, value in expected.items():
            close(sample[key][index], value, f"{label}/{offset}/{key} recomputed", atol=1e-7)
        close(sample["nominal_tcp"][index], trace["reference_tcp"][step, slot]-trace["env_origins"][slot],
              f"{label}/{offset}/actual installed reference TCP", atol=1e-4)
        candidates = [cursor for cursor in commands if cursor < step]
        # Samples occur before that boundary's new patch. The +1 endpoint must
        # still use the anchor's frozen reference, then normal feedback resumes.
        cursor = max(candidates)
        require(sample["nominal_command_step"][index] == cursor, f"{label}: nominal sample not tied to actually delivered command")
        require(offset != 50 or cursor == anchor, f"{label}: +1 sample uses post-intervention feedback")
        command = commands[cursor]
        require(step-cursor < len(command[0]), f"{label}: nominal sample lies outside delivered window")
        if model is not None:
            p, r = model.transforms(command[0][step-cursor], command[1][step-cursor], command[2][step-cursor])
            matrix = r[model.right_wrist_index]
            nominal_tcp = p[model.right_wrist_index]+matrix @ [.085, 0., 0.]
            close(sample["nominal_tcp"][index], nominal_tcp, f"{label}/{offset}/nominal delivered FK", atol=1e-7)
            close(sample["nominal_R"][index], matrix, f"{label}/{offset}/nominal delivered R", atol=1e-7)
        elif offset <= 50 and native_cell is not None:
            close(sample["nominal_tcp"][index], native_cell["nominal_tcp"][offset], f"{label}/{offset}/canonical nominal", atol=1e-7)
    return sample


def _outcome(trace, slot, ending, geometry, metrics_fn, recorded, label):
    end = 1200 if ending is None else ending
    truncated = {key: value[:end+1] for key, value in trace.items() if key != "env_origins"}
    outcome = metrics_fn(truncated, slot=slot, origin=trace["env_origins"][slot],
        terminated=ending is not None, control_dt=.02, geometry=geometry)
    if ending is not None:
        outcome["success"] = False
    require(set(recorded) == set(outcome), f"{label}: outcome inventory differs from metric")
    for key, value in outcome.items():
        if isinstance(value, bool):
            require(boolean(recorded[key], f"{label}/{key}") == value, f"{label}/{key}: actual physical outcome differs")
        else:
            close(recorded[key], value, f"{label}/{key}: physical outcome", atol=1e-7)
    return end


def audit_manifest(path, *, reader, metrics_fn, model=None):
    path = Path(path).resolve()
    manifest = reader.json(path)
    base = path.parent
    require(manifest.get("schema") == SCHEMA, f"{path}: unexpected collector schema")
    require(boolean(manifest.get("completed"), "manifest/completed"), "Collection is incomplete")
    require(manifest["cell_factors"] == [list(pair) for pair in FACTORS], "Factor order mismatch")
    require(manifest["control_hz"] == 50 and manifest["execution_steps"] == 50
        and manifest["native_horizon"] == 105 and manifest["sample_offsets_steps"] == list(OFFSETS), "M2 execution protocol mismatch")
    request_path = base.parent/"request.json"
    request = reader.json(request_path)
    require(request.get("seconds") == 24. and request.get("cameras") is True, "Physical request differs from M2 budget/camera protocol")
    count = integer(request["num_envs"], "num_envs", minimum=1)
    joint_names = manifest["simulator_joint_names"]
    require(len(joint_names) == 31 and len(set(joint_names)) == 31, "Simulator joint inventory differs")
    require(len(manifest["mjcf_joint_names"]) == 29, "MJCF body inventory differs")
    if model is not None:
        require(model.joint_names == manifest["mjcf_joint_names"], "Auditor official FK joint order differs")
    provenance = manifest["provenance"]
    require(isinstance(provenance["sources"], dict) and bool(provenance["sources"]), "Missing collection-time source hashes")
    reader.pin(request_path, provenance["request_sha256"])
    require(provenance["request_sha256"] is not None, "Request has no collection-time SHA256")
    for field, expected in (("request", request_path), ("configuration", request["m2_collection_config"]),
            ("task", request.get("task") or Path(request["reference"])/"task.json"),
            ("asset", request["asset"]), ("reference", request["reference"])):
        require(Path(provenance["paths"][field]).resolve() == Path(expected).resolve(), f"Provenance {field} path differs from actual request")
    for source, digest in provenance["sources"].items():
        reader.pin(source, digest)
    interface_path = Path(provenance["paths"].get("interface", base.parent/"interface.json"))
    interface = reader.json(interface_path, provenance["interface_sha256"])
    require(interface["all_joint_names"] == joint_names and interface["action_dim"] == 29,
        "Actual simulator interface differs from collection joint/action inventory")
    require(interface["floating_base"] is True and interface["control_dt"] == .02
        and interface["physics_dt"] == .005, "Actual simulator timing/floating-base interface differs")
    body_joint_names = interface["body_joint_names"]
    require(len(body_joint_names) == 29 and len(set(body_joint_names)) == 29
        and set(body_joint_names) == set(manifest["mjcf_joint_names"]), "Actual controller body joint mapping differs")
    body_to_mjcf = [body_joint_names.index(name) for name in manifest["mjcf_joint_names"]]
    reader.pin(request["asset"], provenance["robot_asset_sha256"])
    reference = Path(request["reference"])
    for name, digest in provenance["reference_sha256"].items():
        reader.pin(reference/name, digest)
    initial_reference = reader.json(reference/"reference.json")
    require(interface["reference_sha256"] == provenance["reference_sha256"]["motion.pkl"]
        and interface["asset_sha256"] == provenance["robot_asset_sha256"], "Actual simulator asset/reference differs from collector")
    task_path = Path(request.get("task") or reference/"task.json")
    task = reader.json(task_path, provenance["task_sha256"])
    require(task["geometry"] == manifest["geometry"], "Physical scene/task geometry metadata differs")
    require(interface["task_geometry"] == task["geometry"], "Actual scene task geometry differs from declared task")
    geometry = task["geometry"]
    scene = {"table_size": [*geometry["table_size_xy"], geometry.get("table_thickness_m", .04)],
        "table_position": [*geometry["table_center_xy"], geometry["table_top_z"]-geometry.get("table_thickness_m", .04)/2],
        "cube_size": [geometry["cube_size_m"]]*3,
        "cube_position": [*geometry["cube_position"][:2], geometry["cube_position"][2]+geometry.get("cube_spawn_clearance_m", .0005)],
        "tray_size": [*geometry["tray_size_xy"], geometry.get("tray_thickness_m", .012)],
        "tray_position": [*geometry["tray_position_xy"], geometry["tray_top_z"]-geometry.get("tray_thickness_m", .012)/2]}
    require(interface["scene_geometry"] == scene, "Actual scene spawn geometry differs from task prior")
    configuration = reader.json(request["m2_collection_config"], provenance["configuration_sha256"])
    require(configuration == manifest["configuration"], "Configuration differs from recorded batch layout")
    expected_schedule = schedule_targets(task["gripper_schedule"])
    capsules = manifest["capsules"]
    require(len({capsule["id"] for capsule in capsules}) == len(capsules), "Duplicate capsule")
    config_by_id = {batch["capsule_id"]: batch for batch in configuration["batches"]}
    require(set(config_by_id) == {capsule["id"] for capsule in capsules}, "Declared capsules differ from frozen config")
    groups = manifest["groups"]
    require(len({(group["seed"], group["anchor_step"]) for group in groups}) == len(groups), "Duplicate group identity")
    counters = Counter()
    pass_keys = [(item["capsule_id"], item["repeat"], item["candidate"]) for item in manifest["passes"]]
    require(len(pass_keys) == len(set(pass_keys)), "Duplicate physical pass record")
    pass_records = dict(zip(pass_keys, manifest["passes"]))
    audited_passes = set()
    maxima = Counter()
    prefix_maxima = {}
    for capsule in capsules:
        capsule_id = capsule["id"]
        require(Path(capsule_id).name == capsule_id, "Invalid capsule path")
        batch = config_by_id[capsule_id]
        active = integer(capsule["active_count"], "active_count", minimum=1, maximum=count)
        seeds = [item["seed"] for item in batch["anchors"]]+batch["companion_seeds"]
        require(capsule["seeds"] == seeds and len(seeds) == count, "Physical companion layout differs")
        anchor = integer(capsule["anchor_step"], "anchor_step", maximum=1050)
        require(anchor % 10 == 0, "Anchor off control boundary")
        local_groups = sorted([group for group in groups if group["capsule_id"] == capsule_id], key=lambda group: group["slot"])
        require(len(local_groups) == active and [group["slot"] for group in local_groups] == list(range(active)), "Capsule active slots missing or reordered")
        for slot, group in enumerate(local_groups):
            require(group["seed"] == seeds[slot] and group["anchor_step"] == anchor
                and group["condition"] == batch["condition"] and group["family"] == batch["family"], "Group seed/scene/clock differs from frozen capsule")
            boolean(group["accepted"], "group/accepted")
            require(boolean(group["noise_anchor"], "noise_anchor") == bool(batch["anchors"][slot].get("noise_repeat", False)), "Noise selection changed")
        repeats = 3 if any(group["noise_anchor"] for group in local_groups) else 1
        require(boolean(capsule["repeat_full_batch"], "repeat_full_batch") == (repeats == 3), "Repeat companion layout changed")
        run_maps = []
        for group in local_groups:
            runs = group["runs"]
            keys = [(integer(run["repeat"], "repeat", maximum=2), integer(run["candidate"], "candidate", maximum=8)) for run in runs]
            require(len(keys) == len(set(keys)) and set(keys) == {(r, c) for r in range(repeats) for c in range(9)}, "Incomplete/duplicate physical 9-cell repeat grid")
            for run in runs:
                expected_inclusion = run["repeat"] == 0 or group["noise_anchor"]
                require(boolean(run["analysis_included"], "analysis_included") == expected_inclusion, "Analysis repeat inclusion changed")
                boolean(run["valid"], "run/valid")
                boolean(run["native_executed"], "native_executed")
            run_maps.append({key: run for key, run in zip(keys, runs)})
        baseline_trace = baseline_controls = baseline_commands = baseline_command_states = None
        baseline_initial, baseline_prefix = {}, {}
        servo_path = reader.artifact(base, capsule["servo_prefix_archive"], capsule["servo_prefix_sha256"])
        servo_rows = reader.json(servo_path)
        servo_keys = [(entry["step"], entry["slot"]) for entry in servo_rows]
        require(len(servo_keys) == len(set(servo_keys)), f"{capsule_id}: duplicate saved Servo prefix state")
        servo_states = dict(zip(servo_keys, servo_rows))
        for entry in servo_rows:
            validate_cached_estimates(entry["cached_estimates"], f"{capsule_id}/saved actual cached estimates")
        canonical = {}
        for slot, group in enumerate(local_groups):
            if "canonical_archive" in group:
                canonical[slot] = _canonical(reader, base, group, model, manifest["mjcf_joint_names"], f"{capsule_id}/{slot}/canonical")
                close(canonical[slot][1][0][3], expected_schedule[anchor:anchor+105], f"{capsule_id}/{slot}/canonical gripper schedule", atol=0)
                maxima["same_hand_position_m"] = max(maxima["same_hand_position_m"], canonical[slot][2]["max_same_hand_position_m"])
                maxima["same_hand_orientation_rad"] = max(maxima["same_hand_orientation_rad"], canonical[slot][2]["max_same_hand_orientation_rad"])
            else:
                require(not group["accepted"], "Accepted grid has no canonical intervention")
        for repeat in range(repeats):
            for cell in range(9):
                label = f"{capsule_id}/repeat{repeat}/cell{cell}"
                rows = [mapping[repeat, cell] for mapping in run_maps]
                run = rows[0]
                pass_key = (capsule_id, repeat, cell)
                require(pass_key in pass_records, f"{label}: missing physical pass registry")
                pass_record = pass_records[pass_key]
                audited_passes.add(pass_key)
                require(pass_record["active_count"] == active and pass_record["physical_slot_rollouts"] == count, f"{label}: physical pass layout differs")
                paths = {}
                for field, hash_field in (("trace_archive", "trace_sha256"), ("controls_archive", "controls_sha256"),
                        ("commands_archive", "commands_sha256"), ("commands_json", "commands_json_sha256")):
                    require(all(row[field] == run[field] and row[hash_field] == run[hash_field] for row in rows), f"{label}: shared pass artifact identity differs")
                    paths[field] = reader.artifact(base, run[field], run[hash_field])
                    require(pass_record[hash_field] == run[hash_field], f"{label}: pass/run shared artifact hashes differ")
                trace, controls = reader.npz(paths["trace_archive"]), reader.npz(paths["controls_archive"])
                commands, command_rows = reader.npz(paths["commands_archive"]), reader.json(paths["commands_json"])
                endings = validate_trace(trace, controls, count, label)
                by_slot = _command_grid(command_rows, commands, count, anchor, endings, label)
                pass_dir = paths["trace_archive"].parent
                branches_path = reader.artifact(base, pass_record["branches_archive"], pass_record["branches_sha256"])
                require(branches_path == pass_dir/"branches.json", f"{label}: branches file bound to wrong pass")
                recorded_rows = reader.json(branches_path)
                require(recorded_rows == rows, f"{label}: per-pass branch records differ from manifest")
                if repeat == 0 and cell == 0:
                    baseline_trace, baseline_controls, baseline_commands = trace, controls, by_slot
                    baseline_command_states = {(row["slot"], row["step"]): row for row in command_rows}
                    reader.pin(base/capsule["control_archive"], capsule["control_archive_sha256"])
                    bitwise(baseline_controls["actions"], reader.npz(base/capsule["control_archive"])["actions"], f"{label}/baseline controls reference")
                    require(set(servo_states) == {(cursor, slot) for slot, slot_commands in by_slot.items() for cursor in slot_commands},
                        f"{label}: saved Servo replay inventory differs from actual baseline deliveries")
                    for row in command_rows:
                        saved = servo_states[row["step"], row["slot"]]
                        same_cached_estimates(row["cached_estimates"], saved["cached_estimates"], f"{label}/baseline delivery vs saved cache")
                        saved_state = {key: saved[key] for key in ("correction", "cube_offset", "tray_offset", "last_estimate_time")}
                        require(_canonical_json(row["servo_state"], label) == _canonical_json(saved_state, label),
                            f"{label}: actual delivered Servo state differs from saved baseline prefix state")
                initial_records = pass_record["all_initial_snapshots"]
                require(len(initial_records) == count and {entry["slot"] for entry in initial_records} == set(range(count)),
                    f"{label}: incomplete all-companion initial snapshot registry")
                initial_by_slot = {entry["slot"]: entry for entry in initial_records}
                for slot in range(count):
                    initial_path = pass_dir/f"initial-slot-{slot}.npz"
                    initial_record = initial_by_slot[slot]
                    require(reader.artifact(base, initial_record["archive"], initial_record["sha256"]) == initial_path,
                        f"{label}/{slot}: initial snapshot registry bound to wrong pass")
                    if slot < active:
                        require(reader.artifact(base, rows[slot]["initial_snapshot_archive"], rows[slot]["initial_snapshot_sha256"]) == initial_path,
                            f"{label}/{slot}: initial snapshot bound to wrong pass")
                    initial = reader.npz(initial_path)
                    _snapshot_shape(initial, f"{label}/{slot}/initial", step=0)
                    _snapshot_trace(initial, trace, slot, 0, f"{label}/{slot}/initial")
                    jitter = float(request.get("position_jitter", 0.))
                    delta_xy = np.random.default_rng(seeds[slot]).uniform(-jitter, jitter, 2)
                    expected_cube = np.asarray(scene["cube_position"])+np.r_[delta_xy, 0.]
                    close(trace["cube"][0, slot]-trace["env_origins"][slot], expected_cube, f"{label}/{slot}/actual seeded initial cube", atol=1e-6)
                    close(initial["qdot"], np.zeros(31), f"{label}/{slot}/reset joint velocities", atol=0)
                    close(initial["root"][:3], initial_reference["root_position"], f"{label}/{slot}/actual initial root", atol=1e-6)
                    close(initial["root"][3:7], initial_reference["root_quaternion_wxyz"], f"{label}/{slot}/actual initial root orientation", atol=1e-6)
                    initial_q = np.asarray([initial_reference["q"][initial_reference["joint_names_mjcf"].index(name)]
                        if name in initial_reference["joint_names_mjcf"] else .035 for name in joint_names])
                    close(initial["q"], initial_q, f"{label}/{slot}/actual initial joints", atol=1e-6)
                    if repeat == 0 and cell == 0:
                        baseline_initial[slot] = initial
                        for key in ("servo_correction", "cube_offset", "tray_offset", "servo_target_valid", "servo_target_time"):
                            close(initial[key], np.zeros_like(initial[key]), f"{label}/{slot}/reset Servo {key}", atol=0)
                    comparison = compare_snapshots(baseline_initial[slot], initial,
                        configuration.get("prefix_tolerance", {"default": 1e-6}), f"{label}/{slot}/initial replay")
                    for key, value in comparison.items():
                        prefix_maxima[key] = max(prefix_maxima.get(key, 0), value)
                    bitwise(controls["actions"][:anchor, slot], baseline_controls["actions"][:anchor, slot], f"{label}/{slot}/actual prefix controls")
                    bitwise(controls["gripper_targets"][:anchor, slot], baseline_controls["gripper_targets"][:anchor, slot], f"{label}/{slot}/actual prefix grippers")
                    if repeat == 0 and cell == 0:
                        expected_cursors = {step for step in range(0, 1200, 10)
                            if (slot >= active or not anchor < step < anchor+50)
                            and (endings[slot] is None or step < endings[slot])}
                    elif slot >= active:
                        expected_cursors = set(baseline_commands[slot])
                    else:
                        prefix_cursors = {cursor for cursor in baseline_commands[slot] if cursor < anchor}
                        after_cursors = {step for step in range(anchor, 1200, 10)
                            if not anchor < step < anchor+50
                            and (endings[slot] is None or step < endings[slot])}
                        expected_cursors = prefix_cursors | after_cursors
                    require(set(by_slot[slot]) == expected_cursors, f"{label}/{slot}: missing/extra actual reference delivery")
                    for cursor, command in by_slot[slot].items():
                        entry = next(entry for entry in command_rows if entry["slot"] == slot and entry["step"] == cursor)
                        anchor_delivery = slot < active and cursor == anchor
                        require(entry["execution_steps"] == (50 if anchor_delivery else 10)
                            and len(command[0]) == (105 if anchor_delivery else 75), f"{label}/{slot}: native execution/preview boundary differs")
                        replayed = (repeat != 0 or cell != 0) and (cursor < anchor or slot >= active)
                        require(boolean(entry["replayed_prefix"], "replayed_prefix") == replayed, f"{label}/{slot}: replay claim differs")
                        if replayed:
                            for actual, original in zip(command, baseline_commands[slot][cursor]):
                                bitwise(actual, original, f"{label}/{slot}/actual native replay")
                        if replayed or anchor_delivery:
                            baseline_entry = baseline_command_states[slot, cursor]
                            same_cached_estimates(entry["cached_estimates"], baseline_entry["cached_estimates"], f"{label}/{slot}/{cursor}/actual replay cache")
                            require(_canonical_json(entry["servo_state"], label) == _canonical_json(baseline_entry["servo_state"], label),
                                f"{label}/{slot}/{cursor}: actual replay Servo state differs")
                    if slot >= active:
                        bitwise(controls["actions"][:, slot], baseline_controls["actions"][:, slot], f"{label}/{slot}/companion controls")
                        bitwise(controls["gripper_targets"][:, slot], baseline_controls["gripper_targets"][:, slot], f"{label}/{slot}/companion grippers")
                    close(controls["gripper_targets"][:, slot], expected_schedule, f"{label}/{slot}/physical gripper schedule", atol=2e-9)
                    if slot < active:
                        row, group = rows[slot], local_groups[slot]
                        require(row["slot"] == slot and row["seed"] == seeds[slot] and row["anchor_step"] == anchor
                            and row["capsule_id"] == capsule_id, f"{label}/{slot}: physical row identity differs")
                        ending = endings[slot]
                        require(row["first_termination_step"] == ending, f"{label}/{slot}: first termination misreported")
                        require(boolean(row["d24_valid"], "d24_valid") == (ending is None), f"{label}: terminal d24 substituted")
                        observable = ending is None or ending > anchor
                        require(boolean(row["anchor_observable"], "anchor_observable") == observable, f"{label}: anchor availability misreported")
                        require(not row["valid"] or group["accepted"], f"{label}: rejected grid row marked valid")
                        require(row["valid"] == group["accepted"], f"{label}: valid accepted branch silently omitted")
                        factors = row["cell"]
                        require((factors["u_index"], factors["b_index"]) == (cell//3, cell%3)
                            and (factors["u_sign"], factors["b_sign"]) == FACTORS[cell], f"{label}: mislabeled factor cell")
                        if observable:
                            prefix_path = pass_dir/f"prefix-slot-{slot}.npz"
                            require(reader.artifact(base, row["prefix_snapshot_archive"], row["prefix_snapshot_sha256"]) == prefix_path,
                                f"{label}/{slot}: prefix snapshot bound to wrong pass")
                            prefix = reader.npz(prefix_path)
                            _snapshot_shape(prefix, f"{label}/{slot}/prefix", step=anchor)
                            _snapshot_trace(prefix, trace, slot, anchor, f"{label}/{slot}/prefix")
                            if repeat == 0 and cell == 0:
                                baseline_prefix[slot] = prefix
                                previous_cursor = max(cursor for cursor in baseline_commands[slot] if cursor < anchor)
                                saved_servo = servo_states[previous_cursor, slot]
                                for snapshot_key, servo_key in (("servo_correction", "correction"), ("cube_offset", "cube_offset"), ("tray_offset", "tray_offset")):
                                    close(prefix[snapshot_key], saved_servo[servo_key], f"{label}/{slot}/saved actual Servo {servo_key}", atol=0)
                                previous_times = [saved_servo["last_estimate_time"][key] for key in ("cube", "tray")]
                                close(prefix["servo_target_valid"], [value is not None for value in previous_times], f"{label}/saved Servo validity", atol=0)
                                close(prefix["servo_target_time"], [0. if value is None else value for value in previous_times], f"{label}/saved Servo time", atol=0)
                                # Cache was refreshed by this anchor's sensing
                                # before the snapshot; Servo correction still
                                # comes from the previous actual delivery.
                                same_cached_estimates(snapshot_cached_estimates(prefix, f"{label}/anchor cache"),
                                    servo_states[anchor, slot]["cached_estimates"], f"{label}/anchor snapshot vs saved live cache")
                                if slot in canonical:
                                    native_zero = canonical[slot][1][0]
                                    close(native_zero[0][0], prefix["reference/dof_pos"][9, body_to_mjcf], f"{label}/first frame from already installed joints", atol=0)
                                    close(native_zero[1][0], prefix["reference/body_pos_w_full"][9, 0], f"{label}/first frame from installed root", atol=0)
                                    close(native_zero[2][0], prefix["reference/body_quat_w_full"][9, 0], f"{label}/first frame from installed root quaternion", atol=0)
                            comparison = compare_snapshots(baseline_prefix[slot], prefix,
                                configuration.get("prefix_tolerance", {"default": 1e-6}), f"{label}/{slot}/anchor replay")
                            for key, value in comparison.items():
                                prefix_maxima[key] = max(prefix_maxima.get(key, 0), value)
                            require(anchor in by_slot[slot], f"{label}: missing actual anchor command")
                            require(not any(anchor < cursor < anchor+50 for cursor in by_slot[slot]), f"{label}: replanned inside frozen one-second intervention")
                            require(next(command["execution_steps"] for command in command_rows if command["slot"] == slot and command["step"] == anchor) == 50, f"{label}: incorrect intervention execution duration")
                            require(slot in canonical, f"{label}: observable anchor lacks canonical grid")
                            grid, native, _ = canonical[slot]
                            metadata = group["candidate_metadata"]
                            close(factors["u_m"], FACTORS[cell][0]*metadata["effective_amplitude_m"], f"{label}/actual hand amplitude", atol=1e-12)
                            close(factors["b_rad"], FACTORS[cell][1]*metadata["effective_body_pitch_rad"], f"{label}/actual body amplitude", atol=1e-12)
                            close(factors["common_scale"], metadata["common_scale"], f"{label}/actual common scale", atol=0)
                            close(row["anchor_waist_rad"], trace["q"][anchor, slot, joint_names.index("waist_pitch_joint")], f"{label}/actual anchor waist", atol=0)
                            executed = cell if group["accepted"] else 0
                            require(row["executed_candidate"] == executed, f"{label}: fallback candidate misreported")
                            require(row["native_executed"] == group["accepted"], f"{label}: grid validity/execution differs")
                            for actual, expected in zip(by_slot[slot][anchor], native[executed]):
                                bitwise(actual, expected, f"{label}/{slot}/actually installed canonical cell")
                            require(row["native_sha256"] == row["canonical_native_sha256"] == native_digest(native[executed]), f"{label}: recorded native identity differs")
                            canonical_nominal = {"nominal_tcp": grid["nominal_tcp"][executed]}
                        else:
                            require(not row["native_executed"], f"{label}: claims execution after unobservable anchor")
                            require(row["prefix_snapshot_archive"] is None and row["prefix_snapshot_sha256"] is None,
                                f"{label}: prefix snapshot invented for unobservable anchor")
                            canonical_nominal = None
                        sample_path = reader.artifact(base, row["samples_archive"], row["samples_sha256"])
                        sample = _samples(reader, sample_path, row["samples_sha256"], trace, slot, anchor, ending,
                            by_slot[slot], canonical_nominal, joint_names, model, f"{label}/{slot}/samples")
                        require(row["measured_sample_offsets_steps"] == [offset for offset, valid in zip(OFFSETS, sample["valid"]) if valid], f"{label}: measured sample inventory differs")
                        end = _outcome(trace, slot, ending, task["geometry"], metrics_fn, row["outcome"], f"{label}/{slot}/outcome")
                        require(row["outcome_observation_end_step"] == end and row["outcome_budget_seconds"] == end/50, f"{label}: outcome endpoint differs")
                        if observable and repeat == 0 and cell == 0:
                            anchor_path = reader.artifact(base, group["anchor_sample_archive"], group["anchor_sample_sha256"])
                            anchor_sample = reader.npz(anchor_path)
                            expected_anchor = _sample_expected(trace, slot, anchor, joint_names, model)
                            for key, value in expected_anchor.items():
                                close(anchor_sample[key], value, f"{label}/{slot}/common anchor {key}", atol=1e-7)
                            cube_z = trace["cube"][:anchor+1, slot, 2]-trace["env_origins"][slot, 2]
                            close(anchor_sample["prefix_cube_initial_z"], cube_z[0], f"{label}/initial height", atol=0)
                            close(anchor_sample["prefix_max_cube_z"], max(cube_z), f"{label}/prefix height", atol=0)
                        counters["physical_active_branches"] += 1
                        counters["analysis_branches"] += int(row["analysis_included"])
                        counters["valid_analysis_branches"] += int(row["analysis_included"] and row["valid"])
                        counters["physical_terminations"] += int(ending is not None)
                counters["physical_control_steps_all_slots"] += 1200*count
                counters["physical_passes"] += 1
        counters["capsules"] += 1
    require(manifest["attempted_anchors"] == len(groups), "Attempted anchor count differs")
    require(audited_passes == set(pass_records), "Physical pass registry contains extra/missing records")
    require(manifest["analysis_branches"] == counters["analysis_branches"]
        and manifest["physical_active_branch_rows"] == counters["physical_active_branches"], "Active branch totals differ from actual run records")
    physical_slots = counters["physical_passes"]*count
    require(manifest["physical_branches"] == manifest["physical_total_slot_rollouts"] == physical_slots
        and manifest["completed_physical_passes"] == counters["physical_passes"], "Actual all-slot physical budget misreported")
    require(manifest["physical_padding_slot_rollouts"] == physical_slots-counters["physical_active_branches"], "Padding physical rollout count differs")
    counters["physical_total_slot_rollouts"] = physical_slots
    return {"manifest_path": str(path), "manifest_sha256": reader.hashes[str(path)],
        "completed": True, "passed": True, "counts": dict(counters), "maxima": dict(maxima),
        "prefix_maximum_absolute_errors": prefix_maxima,
        "official_cpu_fk_verified": model is not None,
        "groups": groups}


def audit_manifests(manifests, *, metrics_fn=None, model=None, formal=False):
    if metrics_fn is None:
        from .metrics import episode_outcome
        metrics_fn = episode_outcome
    reader = Reader()
    reader.pin(__file__)
    started = datetime.now(timezone.utc).isoformat()
    reports = [audit_manifest(path, reader=reader, metrics_fn=metrics_fn, model=model) for path in manifests]
    groups = [group for report in reports for group in report.pop("groups")]
    if formal:
        from .m2_analysis import validate_formal_groups
        validate_formal_groups(groups)
    hashes = reader.finish()
    result = {"schema": "g1_embodiedskills.m2_independent_audit.v1", "completed": True, "passed": True,
        "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_200_anchor_roster_verified": bool(formal), "official_cpu_fk_verified": model is not None,
        "numpy_version": np.__version__, "collections": reports, "evidence_hashes_before_after": hashes,
        "auditor_source_sha256": reader.hashes[str(Path(__file__).resolve())],
        "verification_scope": "Actual archived histories, controls, canonical/delivered references, sampled physical states and independently recomputed geometry-aware outcomes; no effect estimates"}
    if len(reports) == 1:
        result.update(manifest_path=reports[0]["manifest_path"], manifest_sha256=reports[0]["manifest_sha256"])
    return result
