"""Synthetic integrity fixtures; these never represent experimental outcomes."""
import importlib
import importlib.util
import json
from pathlib import Path
import copy
import shutil
import sys
import types

import numpy as np
import pytest


ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("m2_audit_contract", ROOT/"src/g1_embodiedskills/m2_audit.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
geometry_root = ROOT/"src/g1_embodiedskills"
if not (geometry_root/"metrics.py").exists():
    geometry_root = ROOT.parent/"geometry/src/g1_embodiedskills"
package = types.ModuleType("_m2_integrity_geometry")
package.__path__ = [str(geometry_root)]
sys.modules[package.__name__] = package
geometry = importlib.import_module(package.__name__+".policies")
metric = importlib.import_module(package.__name__+".metrics").episode_outcome


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))
    return str(path)


def save(path, **data):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    return path


def envelope():
    time = np.arange(105)/50
    rise = np.clip(time/.4, 0, 1)
    fall = np.clip((1-time)/.4, 0, 1)
    smooth = lambda x: x**3*(10-15*x+6*x*x)
    return np.minimum(smooth(rise), smooth(fall))


def cache_fixture():
    """Structured sensor cache for integrity checks, never effect evidence."""
    values = {}
    for name, position, surface in (("cube", [.28, -.23, .7275], [.28, -.23, .745]),
            ("tray", [.28, -.03, .716], [.28, -.03, .722])):
        values[name] = {"target": name, "found": True, "position": position,
            "surface_position": surface, "confidence": .95, "mask_count": 100,
            "valid_depth_count": 95, "reason": "visible_top_approximation",
            "pixel_centroid": [128., 127.5], "method": "color_rgbd_visible_top_center"}
    return values


def cache_snapshot(value):
    return {"cached_estimates_present": np.asarray(value is not None, np.uint8),
        "cached_estimates_json": np.frombuffer(json.dumps(value, sort_keys=True,
            separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode(), np.uint8).copy()}


def servo_fixture():
    return {"correction": [0.]*29, "cube_offset": [0.]*3, "tray_offset": [0.]*3,
        "last_estimate_time": {"cube": None, "tray": None}}


@pytest.fixture
def physical_archive(tmp_path):
    """One complete nine-cell schema fixture with constant measured physics.

    It exercises evidence plumbing only. The measured cube never lifts and
    every recomputed task outcome is failure, so no method gain is invented.
    """
    run_root = tmp_path/"run"
    base = run_root/"m2-collection"
    base.mkdir(parents=True)
    anchor, capsule_id = 300, "fixture-anchor300"
    names = [f"joint_{i}" for i in range(29)]
    names[12] = "waist_pitch_joint"
    names[22:29] = list(m.RIGHT_ARM_NAMES)
    names += ["right_hand_index_0_joint", "right_hand_middle_0_joint"]
    schedule = [{"time": 0., "joint_target_m": .035}, {"time": 10., "joint_target_m": 0.}, {"time": 20., "joint_target_m": .035}]
    task = {"geometry": geometry.PickPlaceSpec().as_dict(), "gripper_schedule": schedule}
    task_path = tmp_path/"task.json"
    dump(task_path, task)
    # JSON normalizes dataclass tuples, just as real experiment manifests do.
    task = json.loads(task_path.read_text())
    reference = tmp_path/"reference"
    reference.mkdir()
    for name in ("motion.pkl", "planned.npz", "reference.json"):
        (reference/name).write_bytes(b"integrity-fixture-only")
    dump(reference/"reference.json", {"q": [0.]*29, "joint_names_mjcf": names[:29],
        "root_position": [0., 0., .76], "root_quaternion_wxyz": [1., 0., 0., 0.]})
    asset = tmp_path/"asset.urdf"
    asset.write_text("integrity-fixture-only")
    config = {"schema": "g1_embodiedskills.m2_collection_config.v1", "prefix_tolerance": {"default": 1e-6},
        "batches": [{"capsule_id": capsule_id, "condition": "low", "family": "height",
            "anchors": [{"seed": 920000, "anchor_step": anchor, "noise_repeat": False}], "companion_seeds": []}]}
    config_path = tmp_path/"config.json"
    dump(config_path, config)
    request = {"seconds": 24., "cameras": True, "num_envs": 1, "asset": str(asset),
        "reference": str(reference), "task": str(task_path), "m2_collection_config": str(config_path)}
    request_path = run_root/"request.json"
    dump(request_path, request)
    full_fingers = m.schedule_targets(schedule, steps=1305)
    q = np.zeros((9, 105, 29))
    nominal = np.broadcast_to([.2, -.23, .8], (9, 105, 3)).copy()
    for cell, (u, b) in enumerate(m.FACTORS):
        q[cell, :, 12] = b*.03*envelope()
        q[cell, :, 22] = u*.01*envelope()
        q[cell, envelope() == 0] = 0.
        nominal[cell, :, 0] += u*.01*envelope()
    roots = np.tile([0., 0., .76], (9, 105, 1))
    quats = np.tile([1., 0., 0., 0.], (9, 105, 1))
    fingers = np.tile(full_fingers[anchor:anchor+105], (9, 1, 1))
    native = [(q[c], roots[c], quats[c], fingers[c]) for c in range(9)]
    canonical_path = base/capsule_id/"canonical-slot-0.npz"
    save(canonical_path, qtraj_mjcf=q, roots=roots, root_quaternion_wxyz=quats,
        finger_joint_targets=fingers, actions=np.stack([np.concatenate(n, -1).astype(np.float32) for n in native]),
        feasible=np.ones(9, bool), nominal_tcp=nominal, targets_tcp=nominal,
        nominal_quaternion_wxyz=quats, targets_quaternion_wxyz=quats)
    group = {"id": "fixture", "condition": "low", "family": "height", "seed": 920000,
        "anchor_step": anchor, "slot": 0, "capsule_id": capsule_id, "noise_anchor": False, "noise_repeat": False,
        "accepted": True, "runs": [], "canonical_archive": str(canonical_path.relative_to(base)),
        "canonical_sha256": m.file_digest(canonical_path), "candidate_feasible": [True]*9,
        "candidate_metadata": {"factor_grid": [list(x) for x in m.FACTORS], "common_scale": 1.,
            "effective_amplitude_m": .01, "requested_amplitude_m": .01,
            "effective_body_pitch_rad": .03, "requested_body_pitch_rad": .03}}
    capsule = {"id": capsule_id, "seeds": [920000], "active_count": 1,
        "condition": "low", "family": "height", "anchor_step": anchor, "repeat_full_batch": False}
    manifest = {"schema": m.SCHEMA, "completed": True, "provenance": {
        "sources": {str(Path(m.__file__)): m.file_digest(m.__file__)}, "configuration_sha256": m.file_digest(config_path), "robot_asset_sha256": m.file_digest(asset),
        "task_sha256": m.file_digest(task_path), "reference_sha256": {name: m.file_digest(reference/name)
            for name in ("motion.pkl", "planned.npz", "reference.json")}, "request_sha256": m.file_digest(request_path),
        "paths": {"configuration": str(config_path), "task": str(task_path), "asset": str(asset),
            "reference": str(reference), "request": str(request_path)}},
        "configuration": config, "groups": [group], "capsules": [capsule], "passes": [],
        "geometry": task["geometry"], "simulator_joint_names": names, "mjcf_joint_names": names[:29],
        "cell_factors": [list(x) for x in m.FACTORS], "sample_offsets_steps": list(m.OFFSETS),
        "control_hz": 50, "execution_steps": 50, "native_horizon": 105, "attempted_anchors": 1,
        "analysis_branches": 9, "physical_active_branch_rows": 9, "physical_branches": 9,
        "physical_total_slot_rollouts": 9, "completed_physical_passes": 9, "physical_padding_slot_rollouts": 0}
    interface_path = run_root/"interface.json"
    dump(interface_path, {"all_joint_names": names, "body_joint_names": names[:29], "action_dim": 29,
        "floating_base": True, "control_dt": .02, "physics_dt": .005,
        "reference_sha256": m.file_digest(reference/"motion.pkl"), "asset_sha256": m.file_digest(asset),
        "task_geometry": task["geometry"], "scene_geometry": geometry.scene_geometry(task["geometry"])})
    manifest["provenance"]["paths"]["interface"] = str(interface_path)
    manifest["provenance"]["interface_sha256"] = m.file_digest(interface_path)
    for cell in range(9):
        directory = base/capsule_id/f"repeat-0-cell-{cell}"
        trace = {key: np.zeros((1201, 1, *shape)) for key, shape in m.TRACE_SHAPES.items()}
        trace["root"][:, 0, 2:4] = [.76, 1.]
        trace["cube"][:] = [.28, -.23, .728]
        trace["cube_root"][:, 0, :4] = [.28, -.23, .728, 1.]
        trace["tcp"][:] = [.2, -.23, .8]
        trace["reference_tcp"][:] = [.2, -.23, .8]
        trace["reference_tcp"][anchor:anchor+51, 0] = nominal[cell, :51]
        trace["wrist_quat_wxyz"][:] = [1., 0., 0., 0.]
        trace["gripper"][:, 0] = np.vstack(([.035, .035], full_fingers[:1200]))
        trace["q"][:, 0, -2:] = trace["gripper"][:, 0]
        trace["time"][:, 0] = np.arange(1201)/50
        trace["terminated"] = np.zeros((1201, 1), bool)
        trace["physical_valid"] = np.ones((1201, 1), bool)
        trace["env_origins"] = np.zeros((1, 3))
        trace_path = save(directory/"trace.npz", **trace)
        controls_path = save(directory/"controls.npz", actions=np.zeros((1200, 1, 29)),
            gripper_targets=full_fingers[:1200, None])
        commands, command_rows, indexed = {}, [], {}
        for step in range(0, 1200, 10):
            if anchor < step < anchor+50:
                continue
            command = native[cell] if step == anchor else (np.zeros((75, 29)), np.tile([0., 0., .76], (75, 1)),
                np.tile([1., 0., 0., 0.], (75, 1)), full_fingers[step:step+75])
            index = len(command_rows)
            for part, value in zip(m.PARTS, command):
                commands[f"command_{index:05d}_{part}"] = value
            execution = 50 if step == anchor else 10
            command_rows.append({"step": step, "slot": 0, "execution_steps": execution,
                "native_index": index, "native_sha256": m.native_digest(command), "replayed_prefix": cell != 0 and step < anchor,
                "servo_state": servo_fixture(), "cached_estimates": cache_fixture() if step >= 170 else None,
                "patch": {"cursor": step, "patched_frames": len(command[0]),
                    "last_execution_preview_frame": step+execution+45, "prefix_preserved": True, "gripper_fields_modified": False}})
            indexed[step] = command
        command_path = save(directory/"commands.npz", **commands)
        commands_json = directory/"commands.json"
        dump(commands_json, command_rows)
        snapshot = {key: np.zeros(1) for key in m.SNAPSHOT_FIELDS}
        snapshot.update(root=trace["root"][0, 0], cube_root=trace["cube_root"][0, 0],
            q=trace["q"][0, 0], qdot=trace["qdot"][0, 0], servo_correction=np.zeros(29),
            cube_offset=np.zeros(3), tray_offset=np.zeros(3), servo_target_valid=np.zeros(2, np.uint8),
            servo_target_time=np.zeros(2), policy_steps=np.asarray(0, np.int64),
            policy_dones_present=np.asarray(0, np.uint8), policy_dones=np.zeros(0, np.uint8),
            policy_history_keys=np.zeros(0, np.uint8))
        snapshot.update({"reference/"+key: np.zeros((105, 1)) for key in m.REFERENCE_FIELDS})
        snapshot.update(cache_snapshot(None))
        snapshot["reference/dof_pos"] = np.zeros((105, 29))
        snapshot["reference/body_pos_w_full"] = np.zeros((105, 30, 3))
        snapshot["reference/body_pos_w_full"][:, 0, 2] = .76
        snapshot["reference/body_quat_w_full"] = np.zeros((105, 30, 4))
        snapshot["reference/body_quat_w_full"][:, :, 0] = 1.
        initial_path = save(directory/"initial-slot-0.npz", **snapshot)
        prefix = {**snapshot, "q": trace["q"][anchor, 0]}
        prefix.update(policy_steps=np.asarray(anchor, np.int64),
            policy_history_keys=np.frombuffer(b"example", np.uint8), **{"policy_observation/example": np.zeros(2)})
        prefix.update(cache_snapshot(cache_fixture()))
        prefix.update({"reference/"+key: np.zeros((114, 1)) for key in m.REFERENCE_FIELDS})
        prefix["reference/dof_pos"] = np.zeros((114, 29))
        prefix["reference/body_pos_w_full"] = np.zeros((114, 30, 3))
        prefix["reference/body_pos_w_full"][:, 0, 2] = .76
        prefix["reference/body_quat_w_full"] = np.zeros((114, 30, 4))
        prefix["reference/body_quat_w_full"][:, :, 0] = 1.
        prefix_path = save(directory/"prefix-slot-0.npz", **prefix)
        sample = {key: np.zeros((4, *shape)) for key, shape in m.SAMPLE_SHAPES.items()}
        for i, offset in enumerate(m.OFFSETS):
            for key, value in m._sample_expected(trace, 0, anchor+offset, names, None).items():
                sample[key][i] = value
            sample["nominal_tcp"][i] = nominal[cell, offset] if offset <= 50 else [.2, -.23, .8]
            sample["nominal_R"][i] = np.eye(3)
        sample.update(offset_steps=np.asarray(m.OFFSETS, np.int64), offset_seconds=np.asarray(m.OFFSETS)/50,
            global_steps=anchor+np.asarray(m.OFFSETS, np.int64), valid=np.ones(4, bool),
            nominal_command_step=np.asarray([anchor, anchor, anchor, anchor+140], np.int64))
        sample_path = save(directory/"samples-slot-0.npz", **sample)
        outcome = metric(trace, geometry=task["geometry"])
        u, b = m.FACTORS[cell]
        row = {"candidate": cell, "repeat": 0, "analysis_included": True, "valid": True, "native_executed": True,
            "cell": {"u_index": cell//3, "b_index": cell%3, "u_sign": u, "b_sign": b,
                "u_m": u*.01, "b_rad": b*.03, "common_scale": 1.},
            "slot": 0, "seed": 920000, "anchor_step": anchor, "capsule_id": capsule_id,
            "first_termination_step": None, "d24_valid": True, "anchor_observable": True,
            "executed_candidate": cell, "anchor_waist_rad": 0.,
            "native_sha256": m.native_digest(native[cell]), "canonical_native_sha256": m.native_digest(native[cell]),
            "outcome": outcome, "outcome_observation_end_step": 1200, "outcome_budget_seconds": 24.,
            "measured_sample_offsets_steps": list(m.OFFSETS)}
        for field, hash_field, path in (("trace_archive", "trace_sha256", trace_path),
                ("controls_archive", "controls_sha256", controls_path), ("commands_archive", "commands_sha256", command_path),
                ("commands_json", "commands_json_sha256", commands_json), ("samples_archive", "samples_sha256", sample_path),
                ("initial_snapshot_archive", "initial_snapshot_sha256", initial_path),
                ("prefix_snapshot_archive", "prefix_snapshot_sha256", prefix_path)):
            row[field], row[hash_field] = str(path.relative_to(base)), m.file_digest(path)
        group["runs"].append(row)
        branches_path = directory/"branches.json"
        dump(branches_path, [row])
        manifest["passes"].append({"capsule_id": capsule_id, "repeat": 0, "candidate": cell,
            "active_count": 1, "physical_slot_rollouts": 1, "branches_archive": str(branches_path.relative_to(base)),
            "branches_sha256": m.file_digest(branches_path),
            "all_initial_snapshots": [{"slot": 0, "archive": str(initial_path.relative_to(base)), "sha256": m.file_digest(initial_path)}], **{key: row[key] for key in
                ("trace_sha256", "controls_sha256", "commands_sha256", "commands_json_sha256")}})
        if cell == 0:
            capsule.update(control_archive=str(controls_path.relative_to(base)), control_archive_sha256=m.file_digest(controls_path))
            common = m._sample_expected(trace, 0, anchor, names, None)
            common.update(prefix_cube_initial_z=np.asarray(.728), prefix_max_cube_z=np.asarray(.728), soles=np.zeros((2, 3)))
            anchor_path = save(base/capsule_id/"anchor-slot-0.npz", **common)
            group.update(anchor_sample_archive=str(anchor_path.relative_to(base)), anchor_sample_sha256=m.file_digest(anchor_path))
            servo_path = base/capsule_id/"servo-prefix.json"
            dump(servo_path, [{"step": step, "slot": 0, "execution_steps": 50 if step == anchor else 10,
                **servo_fixture(), "cached_estimates": cache_fixture() if step >= 170 else None} for step in indexed])
            capsule.update(servo_prefix_archive=str(servo_path.relative_to(base)), servo_prefix_sha256=m.file_digest(servo_path))
    manifest_path = base/"manifest.json"
    dump(manifest_path, manifest)
    return manifest_path, manifest


def test_complete_actual_artifact_plumbing_passes_and_strict_json(physical_archive):
    path, _ = physical_archive
    result = m.audit_manifests([path], metrics_fn=metric)
    assert result["passed"] and result["collections"][0]["counts"]["physical_active_branches"] == 9
    assert result["evidence_hashes_before_after"]["unchanged"]
    assert not result["official_cpu_fk_verified"]
    json.dumps(result, allow_nan=False)


def test_declared_complete_flag_cannot_hide_missing_cell(physical_archive):
    path, manifest = physical_archive
    manifest["groups"][0]["runs"].pop()
    dump(path, manifest)
    with pytest.raises(m.AuditError, match="Incomplete/duplicate physical"):
        m.audit_manifests([path], metrics_fn=metric)


def test_false_string_is_never_accepted_as_a_boolean(physical_archive):
    path, manifest = physical_archive
    manifest["completed"] = "false"
    dump(path, manifest)
    with pytest.raises(m.AuditError, match="JSON boolean"):
        m.audit_manifests([path], metrics_fn=metric)


def test_after_collection_archive_mutation_fails_hash(physical_archive):
    path, manifest = physical_archive
    artifact = path.parent/manifest["groups"][0]["runs"][3]["controls_archive"]
    artifact.write_bytes(artifact.read_bytes()+b"mutation")
    with pytest.raises(m.AuditError, match="SHA256 mismatch"):
        m.audit_manifests([path], metrics_fn=metric)


def test_bitwise_prefix_does_not_accept_close_but_changed_actions():
    first = np.zeros((300, 29), np.float32)
    other = first.copy()
    other[100, 2] = 1e-12
    with pytest.raises(m.AuditError, match="bytes differ"):
        m.bitwise(first, other, "physical prefix")


def test_validity_mask_censors_the_termination_boundary_without_imputation():
    trace = {key: np.zeros((1201, 1, *shape)) for key, shape in m.TRACE_SHAPES.items()}
    trace.update(env_origins=np.zeros((1, 3)), terminated=np.zeros((1201, 1), bool),
        physical_valid=(np.arange(1201) <= 350)[:, None])
    trace["terminated"][350, 0] = True
    trace["time"][:, 0] = np.arange(1201)/50
    controls = {"actions": np.zeros((1200, 1, 29)), "gripper_targets": np.zeros((1200, 1, 2))}
    assert m.validate_trace(trace, controls, 1, "fixture") == [350]
    trace["physical_valid"][351, 0] = True
    with pytest.raises(m.AuditError, match="termination validity"):
        m.validate_trace(trace, controls, 1, "fixture")


def test_reader_refuses_evidence_changed_during_audit(tmp_path):
    path = tmp_path/"artifact"
    path.write_bytes(b"before")
    reader = m.Reader()
    reader.pin(path)
    path.write_bytes(b"after")
    with pytest.raises(m.AuditError, match="changed during independent audit"):
        reader.finish()


def _repin_row(path, manifest, cell):
    """Update declared hashes to test checks beyond archive hash equality."""
    row = manifest["groups"][0]["runs"][cell]
    record = manifest["passes"][cell]
    for archive_field, hash_field in (("controls_archive", "controls_sha256"),
            ("samples_archive", "samples_sha256"), ("prefix_snapshot_archive", "prefix_snapshot_sha256"),
            ("commands_json", "commands_json_sha256")):
        row[hash_field] = m.file_digest(path.parent/row[archive_field])
        if hash_field in record:
            record[hash_field] = row[hash_field]
    branches = path.parent/record["branches_archive"]
    dump(branches, [row])
    record["branches_sha256"] = m.file_digest(branches)
    dump(path, manifest)


def test_rehashed_control_change_still_fails_actual_prefix_pairing(physical_archive):
    path, manifest = physical_archive
    row = manifest["groups"][0]["runs"][1]
    archive = path.parent/row["controls_archive"]
    with np.load(archive) as source:
        data = {key: source[key] for key in source.files}
    data["actions"][100, 0, 0] = 1e-12
    save(archive, **data)
    _repin_row(path, manifest, 1)
    with pytest.raises(m.AuditError, match="actual prefix controls"):
        m.audit_manifests([path], metrics_fn=metric)


def test_rehashed_actor_history_change_cannot_pass_physical_only_pairing(physical_archive):
    path, manifest = physical_archive
    row = manifest["groups"][0]["runs"][1]
    archive = path.parent/row["prefix_snapshot_archive"]
    with np.load(archive) as source:
        data = {key: source[key] for key in source.files}
    data["policy_observation/example"][0] = .01
    save(archive, **data)
    _repin_row(path, manifest, 1)
    with pytest.raises(m.AuditError, match="anchor replay/policy_observation/example"):
        m.audit_manifests([path], metrics_fn=metric)


def test_rehashed_one_second_sample_cannot_reference_new_feedback(physical_archive):
    path, manifest = physical_archive
    row = manifest["groups"][0]["runs"][1]
    archive = path.parent/row["samples_archive"]
    with np.load(archive) as source:
        data = {key: source[key] for key in source.files}
    data["nominal_command_step"][2] = 350
    save(archive, **data)
    _repin_row(path, manifest, 1)
    with pytest.raises(m.AuditError, match="nominal sample not tied"):
        m.audit_manifests([path], metrics_fn=metric)


def test_reported_success_must_match_recomputed_physics(physical_archive):
    path, manifest = physical_archive
    manifest["groups"][0]["runs"][1]["outcome"]["success"] = True
    _repin_row(path, manifest, 1)
    with pytest.raises(m.AuditError, match="actual physical outcome differs"):
        m.audit_manifests([path], metrics_fn=metric)


def test_cache_presence_and_complete_contents_are_not_a_hash_substitute():
    with pytest.raises(m.AuditError, match="target inventory is incomplete"):
        m.snapshot_cached_estimates(cache_snapshot({"sha256": "a"*64}), "fixture")
    missing_flag = cache_snapshot(None)
    missing_flag["cached_estimates_present"] = np.asarray(1, np.uint8)
    with pytest.raises(m.AuditError, match="presence/content differ"):
        m.snapshot_cached_estimates(missing_flag, "fixture")
    incomplete = cache_fixture()
    del incomplete["cube"]["surface_position"]
    with pytest.raises(m.AuditError, match="incomplete cached detection"):
        m.snapshot_cached_estimates(cache_snapshot(incomplete), "fixture")
    full_utf8 = cache_fixture()
    full_utf8["cube"]["reason"] = "可见顶面"
    assert m.snapshot_cached_estimates(cache_snapshot(full_utf8), "fixture") == full_utf8


@pytest.mark.parametrize("kind", ["position", "found", "confidence"])
def test_rehashed_cached_prefix_values_and_flags_must_match_common_state(physical_archive, kind):
    path, manifest = physical_archive
    row = manifest["groups"][0]["runs"][1]
    archive = path.parent/row["prefix_snapshot_archive"]
    with np.load(archive) as source:
        data = {key: source[key] for key in source.files}
    changed = cache_fixture()
    if kind == "position":
        changed["cube"]["position"][0] += 1e-10
    elif kind == "confidence":
        changed["cube"]["confidence"] -= 1e-10
    else:
        changed["cube"].update(found=False, position=None, surface_position=None,
            pixel_centroid=None, confidence=0., reason="missing_color")
    data.update(cache_snapshot(changed))
    save(archive, **data)
    _repin_row(path, manifest, 1)
    with pytest.raises(m.AuditError, match="actual cached detection contents differ"):
        m.audit_manifests([path], metrics_fn=metric)


def test_actual_delivery_cache_cannot_differ_even_with_a_correct_snapshot(physical_archive):
    path, manifest = physical_archive
    row = manifest["groups"][0]["runs"][1]
    commands_path = path.parent/row["commands_json"]
    commands = json.loads(commands_path.read_text())
    entry = next(entry for entry in commands if entry["step"] == 200)
    entry["cached_estimates"]["tray"]["position"][0] += 1e-10
    dump(commands_path, commands)
    _repin_row(path, manifest, 1)
    with pytest.raises(m.AuditError, match="actual replay cache.*actual cached detection contents differ"):
        m.audit_manifests([path], metrics_fn=metric)


def test_saved_baseline_servo_cache_is_checked_by_contents(physical_archive):
    path, manifest = physical_archive
    capsule = manifest["capsules"][0]
    archive = path.parent/capsule["servo_prefix_archive"]
    entries = json.loads(archive.read_text())
    next(entry for entry in entries if entry["step"] == 300)["cached_estimates"]["tray"]["confidence"] -= 1e-10
    dump(archive, entries)
    capsule["servo_prefix_sha256"] = m.file_digest(archive)
    dump(path, manifest)
    with pytest.raises(m.AuditError, match="baseline delivery vs saved cache.*actual cached detection contents differ"):
        m.audit_manifests([path], metrics_fn=metric)


def _expand_full_noise_capsule(path, manifest):
    group, capsule = manifest["groups"][0], manifest["capsules"][0]
    group.update(noise_anchor=True, noise_repeat=True)
    capsule["repeat_full_batch"] = True
    manifest["configuration"]["batches"][0]["anchors"][0]["noise_repeat"] = True
    configuration_path = Path(manifest["provenance"]["paths"]["configuration"])
    dump(configuration_path, manifest["configuration"])
    manifest["provenance"]["configuration_sha256"] = m.file_digest(configuration_path)
    original_rows, original_passes = copy.deepcopy(group["runs"]), copy.deepcopy(manifest["passes"])
    for repeat in (1, 2):
        for cell, original in enumerate(original_rows):
            old_fragment, new_fragment = f"repeat-0-cell-{cell}", f"repeat-{repeat}-cell-{cell}"
            directory = path.parent/Path(original["trace_archive"]).parent
            new_directory = directory.parent/new_fragment
            shutil.copytree(directory, new_directory)
            row = copy.deepcopy(original)
            row["repeat"] = repeat
            for field, value in list(row.items()):
                if isinstance(value, str) and old_fragment in value:
                    row[field] = value.replace(old_fragment, new_fragment)
            commands_path = path.parent/row["commands_json"]
            entries = json.loads(commands_path.read_text())
            for entry in entries:
                entry["replayed_prefix"] = entry["step"] < row["anchor_step"]
            dump(commands_path, entries)
            row["commands_json_sha256"] = m.file_digest(commands_path)
            dump(new_directory/"branches.json", [row])
            record = copy.deepcopy(original_passes[cell])
            record.update(repeat=repeat, branches_archive=str((new_directory/"branches.json").relative_to(path.parent)),
                branches_sha256=m.file_digest(new_directory/"branches.json"), commands_json_sha256=row["commands_json_sha256"])
            for initial in record["all_initial_snapshots"]:
                initial["archive"] = initial["archive"].replace(old_fragment, new_fragment)
            group["runs"].append(row)
            manifest["passes"].append(record)
    manifest.update(analysis_branches=27, physical_active_branch_rows=27, physical_branches=27,
        physical_total_slot_rollouts=27, completed_physical_passes=27)
    dump(path, manifest)


def test_complete_r3_actual_cache_replay_and_mutation_rejection(physical_archive):
    path, manifest = physical_archive
    _expand_full_noise_capsule(path, manifest)
    valid = m.audit_manifests([path], metrics_fn=metric)
    assert valid["passed"] and valid["collections"][0]["counts"]["analysis_branches"] == 27
    index = 18+3
    row = manifest["groups"][0]["runs"][index]
    commands_path = path.parent/row["commands_json"]
    entries = json.loads(commands_path.read_text())
    next(entry for entry in entries if entry["step"] == row["anchor_step"])["cached_estimates"]["cube"]["confidence"] -= 1e-10
    dump(commands_path, entries)
    _repin_row(path, manifest, index)
    with pytest.raises(m.AuditError, match="actual replay cache.*actual cached detection contents differ"):
        m.audit_manifests([path], metrics_fn=metric)
