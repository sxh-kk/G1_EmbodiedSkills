"""Failure-focused checks for contrasts/noise and the episode sampling unit."""
from pathlib import Path
import hashlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/"src"))
from g1_embodiedskills import m2_analysis as m


def test_additive_vectors_do_not_inherit_distance_norm_interaction():
    hand, body = [0., 1., -1.], [0., 2., -2.]
    grid = np.asarray([[hand[u], body[b], 0.] for u, b in m.CELLS])
    contrasts, observed = m.vector_interactions(grid)
    np.testing.assert_array_equal(contrasts, np.zeros((4, 3)))
    assert observed.all()
    # Taking norms first fabricates a nonadditivity in this additive example.
    false_effect, _ = m.vector_interactions(np.linalg.norm(grid, axis=1))
    assert np.max(np.abs(false_effect)) > .1


def test_censor_one_cell_only_where_that_cell_is_necessary():
    grid = np.zeros((9, 3))
    grid[m.CELL_INDEX[1, 1]] = np.nan
    contrasts, observed = m.vector_interactions(grid)
    assert observed.tolist() == [False, True, True, True]
    assert np.isnan(contrasts[0]).all()
    grid[0] = np.nan
    assert not m.vector_interactions(grid)[1].any()


def test_relative_pairs_of_additive_vectors_are_zero_before_norms():
    hand, body = [0., .01, -.01], [0., .02, -.02]
    grid = np.asarray([[hand[u], body[b], 0.] for u, b in m.CELLS])
    contrasts, observed = m.relative_pair_interactions(grid)
    assert contrasts.shape == (6, 3) and observed.all()
    np.testing.assert_allclose(contrasts, 0., atol=1e-17)


def test_relative_pair_mask_uses_exactly_four_cells_without_default_hand():
    grid = np.full((9, 3), np.nan)
    valid = np.zeros(9, bool)
    # The +/- contrast at b+ needs neither (0,0) nor (0,b+).
    required = ((1, 1), (2, 1), (1, 0), (2, 0))
    for value, cell in enumerate(required):
        grid[m.CELL_INDEX[cell]] = [float(value), 0., 0.]
        valid[m.CELL_INDEX[cell]] = True
    contrasts, observed = m.relative_pair_interactions(grid, valid)
    target = m.RELATIVE_PAIRS.index((1, 2, 1))
    assert observed.tolist() == [False, False, False, False, True, False]
    np.testing.assert_array_equal(contrasts[target], [0., 0., 0.])
    assert not m.vector_interactions(grid, valid)[1].any()
    valid[m.CELL_INDEX[1, 1]] = False
    assert not m.relative_pair_interactions(grid, valid)[1].any()
    with pytest.raises(ValueError, match="shape"):
        m.relative_pair_interactions(np.zeros((9, 3)), np.ones((9, 1), bool))


def test_centered_noise_does_not_absorb_true_nonzero_effect():
    repeated = np.array([[1., 0., 0.], [1.1, 0., 0.], [.9, 0., 0.]])
    assert m.centered_repeat_noise(repeated) == pytest.approx(.1)
    assert m.centered_repeat_noise(np.tile([2., 0., 0.], (3, 1))) == 0.
    repeated[1, 2] = np.nan
    assert m.centered_repeat_noise(repeated) is None


def test_pair_noise_cancels_common_grid_shifts_and_rejects_noisy_flips():
    utility = np.array([0., 0., 0., 1., -1., 1., 0., 0., 0.])
    # Correlated shifts cancel within the hand differences in every repeat.
    reference = np.stack([utility-100, utility, utility+100])
    rows = m.robust_rank_flips(utility, reference)
    target = next(row for row in rows if row["hand_pair"] == [0, 1] and row["body_pair"] == [0, 1])
    assert target["raw_flip"] and target["resolved_flip"]
    reference[0, 3] -= 10
    reference[2, 3] += 10
    rows = m.robust_rank_flips(utility, reference)
    target = next(row for row in rows if row["hand_pair"] == [0, 1] and row["body_pair"] == [0, 1])
    assert target["raw_flip"] and not target["resolved_flip"]


def test_bootstrap_rejects_branch_pseudoreplication_and_retains_missing():
    rows = [{"condition": "low", "seed": 1, "effect": 2.},
            {"condition": "low", "seed": 2, "effect": 2.},
            {"condition": "high", "seed": 3, "effect": None}]
    result = m.stratified_episode_bootstrap(rows, "effect", repetitions=50)
    assert result["episodes"] == 3 and result["observed_episodes"] == 2
    assert result["ci95"] == [2., 2.]
    with pytest.raises(ValueError, match="one row per original episode"):
        m.stratified_episode_bootstrap(rows+rows[:1], "effect")


def test_complete_grid_schema_and_rejected_grid_cannot_create_evidence(tmp_path):
    runs = []
    for repeat in range(3):
        for cell, (u, b) in enumerate(m.CELLS):
            us, bs = [0, 1, -1][u], [0, 1, -1][b]
            nominal = np.tile([us*.01, 0., 0.], (4, 1))
            actual = nominal+np.tile([us*bs*.005, 0., 0.], (4, 1))
            path = tmp_path/f"r{repeat}-c{cell}.npz"
            np.savez(path, offset_steps=m.OFFSETS, valid=np.ones(4, bool),
                actual_tcp=actual, nominal_tcp=nominal, actual_R=np.tile(np.eye(3), (4, 1, 1)),
                nominal_R=np.tile(np.eye(3), (4, 1, 1)), waist=np.full(4, bs*.03),
                cube_root=np.zeros((4, 13)), root=np.zeros((4, 13)), soles=np.zeros((4, 2, 3)))
            distance = .04 if u == 0 else .05 if u == 1 and b == 1 else .03
            runs.append({"candidate": cell, "repeat": repeat, "analysis_included": True,
                "anchor_waist_rad": 0.,
                "samples_archive": path.name, "valid": True, "d24_valid": True,
                "first_termination_step": None,
                "outcome": {"final_target_distance_m": distance, "return_value": 1.-distance,
                    "success": True, "lifted": True, "standing": True}})
    group = {"id": "test", "condition": "low", "family": "height", "seed": 920000,
        "anchor_step": 300, "noise_anchor": True, "accepted": True, "runs": runs, "_base": tmp_path}
    positive = m.summarize_m2([group])
    row = positive["anchors"][0]
    assert row["primary_gate"] and row["matched_distance_joint_gate"]
    assert positive["complete_primary_gamma_noise_references"] == 4
    rejected = m.summarize_m2([{**group, "accepted": False}])["anchors"][0]
    assert not rejected["primary_gate"] and not rejected["resolved_distance_rank_flip"]
    assert rejected["successful_candidate_count"] == 0
    assert rejected["primary_observed_combinations"] == 0
    with pytest.raises(ValueError, match="complete frozen"):
        m.summarize_m2([group], formal=True)


def _relative_only_group(tmp_path, *, missing_default_body_plus=False, pair_repeat_noise=0.):
    """u+ versus u- reverses; the default hand stays worse than both."""
    runs = []
    for repeat in range(3):
        for cell, (u, b) in enumerate(m.CELLS):
            us, bs = [0, 1, -1][u], [0, 1, -1][b]
            nominal = np.tile([us*.01, 0., 0.], (4, 1))
            displacement = us*.005 if b == 1 else 0.
            # Perturb one paired cell across complete repeats to test nT,
            # rather than reusing independent-branch noise estimates.
            if u == 1 and b == 1:
                displacement += (repeat-1)*pair_repeat_noise
            actual = nominal+np.tile([displacement, 0., 0.], (4, 1))
            valid = np.ones(4, bool)
            if missing_default_body_plus and (u, b) == (0, 1):
                valid[:] = False
            path = tmp_path/f"relative-r{repeat}-c{cell}.npz"
            np.savez(path, offset_steps=m.OFFSETS, valid=valid,
                actual_tcp=actual, nominal_tcp=nominal,
                actual_R=np.tile(np.eye(3), (4, 1, 1)), nominal_R=np.tile(np.eye(3), (4, 1, 1)),
                waist=np.full(4, bs*.03), cube_root=np.zeros((4, 13)),
                root=np.zeros((4, 13)), soles=np.zeros((4, 2, 3)))
            distance = .05 if u == 0 else (.03 if b == 1 else .01) if u == 1 else (.01 if b == 1 else .03)
            runs.append({"candidate": cell, "repeat": repeat, "analysis_included": True,
                "anchor_waist_rad": 0., "samples_archive": path.name,
                "valid": True, "d24_valid": True, "first_termination_step": None,
                "outcome": {"final_target_distance_m": distance, "return_value": 6.-distance,
                            "success": True, "lifted": True, "standing": True}})
    return {"id": "relative-only", "condition": "low", "family": "height", "seed": 920000,
        "anchor_step": 300, "noise_anchor": True, "accepted": True, "runs": runs, "_base": tmp_path,
        "candidate_metadata": {"requested_amplitude_m": .01, "requested_body_pitch_rad": .03,
                               "effective_amplitude_m": .01, "effective_body_pitch_rad": .03, "common_scale": 1.}}


def test_relative_joint_gate_detects_plus_minus_flip_missed_by_default_pairs(tmp_path):
    report = m.summarize_m2([_relative_only_group(tmp_path)])
    row = report["anchors"][0]
    assert len(row["samples"][1]["interactions"]) == 4  # Original primary stays unchanged.
    assert row["primary_gate"] and not row["matched_distance_joint_gate"]
    assert row["resolved_distance_rank_flip"]
    assert row["relative_pair_joint_gate"] and row["relative_pair_joint_gate_same_amplitudes"]
    assert len(row["relative_pair_interactions"]) == 6
    target = next(value for value in row["relative_pair_interactions"]
                  if value["hand_pair"] == [1, 2] and value["body_pair"] == [0, 1])
    np.testing.assert_allclose(target["residual_contrast_m"], [.01, 0., 0.])
    assert target["noise_m"] == 0. and target["body_executed_above_noise"]
    assert target["noise_actual_amplitudes_match"]
    for metric in ("distance", "return", "dense_return"):
        assert not row[f"matched_{metric}_joint_gate"]
        assert row[f"relative_pair_{metric}_joint_gate"]
        matched = next(value for value in row[f"relative_pair_{metric}_interaction_rank"]
                       if value["hand_pair"] == [1, 2] and value["body_pair"] == [0, 1])
        assert matched["resolved_joint_evidence"]
    assert report["complete_primary_gamma_noise_references"] == 4
    assert report["complete_relative_pair_noise_references"] == 6
    assert len(report["episodes"]) == 1
    interval = report["summaries"]["all"]["episode_cluster_intervals"]["relative_pair_distance_joint_anchor_fraction"]
    assert interval["episodes"] == 1  # Six pairs never become six clusters.
    assert report["summaries"]["all"]["relative_pair_distance_joint_episodes"] == 1


def test_relative_contrast_survives_missing_unused_default_but_body_gate_does_not(tmp_path):
    row = m.summarize_m2([_relative_only_group(tmp_path, missing_default_body_plus=True)])["anchors"][0]
    target = next(value for value in row["relative_pair_interactions"]
                  if value["hand_pair"] == [1, 2] and value["body_pair"] == [0, 1])
    assert target["observed"] and target["noise_m"] == 0.
    np.testing.assert_allclose(target["residual_contrast_m"], [.01, 0., 0.])
    assert not target["default_hand_body_response"]["observed"]
    assert not target["body_executed_above_noise"]
    assert not row["relative_pair_distance_joint_gate"]


def test_relative_noise_is_computed_from_the_repeated_pair_contrast(tmp_path):
    row = m.summarize_m2([_relative_only_group(tmp_path, pair_repeat_noise=.02)])["anchors"][0]
    target = next(value for value in row["relative_pair_interactions"]
                  if value["hand_pair"] == [1, 2] and value["body_pair"] == [0, 1])
    assert target["noise_m"] == pytest.approx(.02)
    assert target["engineering_threshold_m"] == pytest.approx(.06)
    assert not target["above_threshold"] and not row["relative_pair_joint_gate"]


def test_formal_relative_summary_retains_40_episode_clusters_with_all_grids_missing():
    groups = []
    for condition, seed, anchor in sorted(m.formal_roster()):
        family = m.FAMILIES[condition]
        selected = seed == m.NOISE_SEEDS[family][m.ANCHOR_STEPS.index(anchor)]
        runs = [{"repeat": repeat, "candidate": cell, "analysis_included": True,
                 "valid": False, "first_termination_step": None}
                for repeat in range(3 if selected else 1) for cell in range(9)]
        groups.append({"id": f"{condition}-{seed}-{anchor}", "condition": condition,
            "family": family, "seed": seed, "anchor_step": anchor,
            "noise_anchor": selected, "accepted": False, "runs": runs})
    with pytest.raises(ValueError, match="official CPU FK"):
        m.summarize_m2(groups, formal=True)
    # No accepted grid needs FK, but the formal aggregator still requires a
    # supplied backend. The actual publishing CLI constructs only G1Kinematics.
    report = m.summarize_m2(groups, formal=True, model=object())
    summary = report["summaries"]["all"]
    assert len(report["episodes"]) == 40
    assert summary["relative_pair_planned_contrasts"] == 1200
    assert summary["relative_pair_observed_contrasts"] == 0
    interval = summary["episode_cluster_intervals"]["relative_pair_distance_joint_anchor_fraction"]
    assert interval["episodes"] == 40 and interval["mean"] == 0.
    # This is the documented all-planned-anchor lower bound, not 1200
    # independent measurements and not evidence that missing effects are zero.
    assert "known lower bounds" in report["missingness"]


class _OracleFK:
    joint_names = [f"joint_{index}" for index in range(29)]
    joint_names[14] = "waist_pitch_joint"
    right_wrist_index = 0
    joint_limits = np.tile([-3., 3.], (29, 1))

    def transforms(self, q, roots, quaternions):
        from scipy.spatial.transform import Rotation
        q = np.asarray(q)
        rotations = Rotation.from_quat(np.asarray(quaternions)[..., [1, 2, 3, 0]]).as_matrix()
        local = np.stack([q[..., 0]**2+q[..., 1]**2, .1*q[..., 2], np.full(q.shape[:-1], .04)], axis=-1)
        positions = roots+np.einsum("...ij,...j->...i", rotations, local)
        return positions[..., None, :], rotations[..., None, :, :]


def _oracle_group(tmp_path, *, cross_extra=.0, cross_repeat_noise=.0, missing_joint_cell=None):
    model = _OracleFK()
    names = ["left_finger", *reversed(model.joint_names), "right_finger"]
    nominal_q = np.zeros((9, 29))
    error = np.zeros((9, 29))
    roots = np.tile([0., 0., .76], (9, 1))
    quats = np.tile([1., 0., 0., 0.], (9, 1))
    for cell, (u, b) in enumerate(m.CELLS):
        us, bs = (0, 1, -1)[u], (0, 1, -1)[b]
        nominal_q[cell, 0] = .3+.05*us+.1*bs
        nominal_q[cell, 1] = np.sqrt(.5+.02*us-nominal_q[cell, 0]**2)
        nominal_q[cell, 14] = .03*bs
        error[cell, 0] = .02+.03*us-.01*bs
    nominal_tcp = model.transforms(nominal_q, roots, quats)[0][:, 0]+[.085, 0., 0.]
    canonical = tmp_path/"oracle-canonical.npz"
    np.savez(canonical, qtraj_mjcf=np.repeat(nominal_q[:, None], 105, axis=1),
        roots=np.repeat(roots[:, None], 105, axis=1),
        root_quaternion_wxyz=np.repeat(quats[:, None], 105, axis=1),
        nominal_tcp=np.repeat(nominal_tcp[:, None], 105, axis=1))
    evidence = {str(canonical.resolve()): hashlib.sha256(canonical.read_bytes()).hexdigest()}
    runs = []
    excesses = []
    for repeat in range(3):
        q = nominal_q+error
        q[4, 0] += cross_extra+(repeat-1)*cross_repeat_noise
        actual_tcp = model.transforms(q, roots, quats)[0][:, 0]+[.085, 0., 0.]
        additive_tcp = model.transforms(nominal_q+error, roots, quats)[0][:, 0]+[.085, 0., 0.]
        excesses.append(actual_tcp[4]-additive_tcp[4])
        for cell, (u, b) in enumerate(m.CELLS):
            q31 = np.zeros(31)
            for index, name in enumerate(model.joint_names):
                q31[names.index(name)] = q[cell, index]
            if cell == missing_joint_cell:
                q31[names.index(model.joint_names[0])] = np.nan
            root = np.r_[roots[cell], quats[cell], np.zeros(6)]
            path = tmp_path/f"oracle-r{repeat}-c{cell}.npz"
            np.savez(path, offset_steps=m.OFFSETS, valid=np.ones(4, bool),
                actual_tcp=np.tile(actual_tcp[cell], (4, 1)), nominal_tcp=np.tile(nominal_tcp[cell], (4, 1)),
                actual_R=np.tile(np.eye(3), (4, 1, 1)), nominal_R=np.tile(np.eye(3), (4, 1, 1)),
                q=np.tile(q31, (4, 1)), root=np.tile(root, (4, 1)), cube_root=np.zeros((4, 13)),
                waist=np.full(4, q[cell, 14]), soles=np.zeros((4, 2, 3)))
            evidence[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
            distance = .05 if u == 0 else .01 if u == 1 else .03
            runs.append({"candidate": cell, "repeat": repeat, "analysis_included": True,
                "anchor_waist_rad": 0., "samples_archive": path.name,
                "samples_sha256": evidence[str(path.resolve())], "valid": True,
                "d24_valid": True, "first_termination_step": None,
                "outcome": {"final_target_distance_m": distance, "return_value": 6.-distance,
                            "success": True, "lifted": True, "standing": True}})
    group = {"id": "oracle", "condition": "low", "family": "height", "seed": 920000,
        "anchor_step": 300, "noise_anchor": True, "accepted": True, "runs": runs, "_base": tmp_path,
        "_simulator_joint_names": names, "_mjcf_joint_names": model.joint_names,
        "_audited_evidence": evidence, "canonical_archive": canonical.name,
        "canonical_sha256": evidence[str(canonical.resolve())],
        "candidate_metadata": {"effective_amplitude_m": .01, "effective_body_pitch_rad": .03, "common_scale": 1.}}
    return group, model, np.asarray(excesses)


def test_oracle_null_explains_nonlinear_fk_without_replacing_original_gamma_or_pairs(tmp_path):
    group, model, _ = _oracle_group(tmp_path)
    original = m.summarize_m2([group])
    report = m.summarize_m2([group], model=model)
    row = report["anchors"][0]
    assert row["samples"] == original["anchors"][0]["samples"]
    assert row["relative_pair_interactions"] == original["anchors"][0]["relative_pair_interactions"]
    assert max(entry["residual_norm_m"] for entry in row["samples"][1]["interactions"]) > .002
    for variant in m.KINEMATIC_NULL_VARIANTS:
        null = row["kinematic_oracle_null"][variant]
        assert null["observed_combinations"] == 4 and not null["gate"]
        for entry in null["interactions"]:
            np.testing.assert_allclose(entry["gamma_excess_m"], 0., atol=1e-15)
            assert entry["noise_actual_amplitudes_match"]
        interval = report["summaries"]["all"]["episode_cluster_intervals"][f"kinematic_null_{variant}_observed_fraction"]
        assert interval["episodes"] == 1


def test_oracle_excess_uses_paired_repeat_noise_and_keeps_null_mask_separate(tmp_path):
    group, model, excesses = _oracle_group(tmp_path, cross_extra=.03, cross_repeat_noise=.01, missing_joint_cell=8)
    row = m.summarize_m2([group], model=model)["anchors"][0]
    for variant in m.KINEMATIC_NULL_VARIANTS:
        null = row["kinematic_oracle_null"][variant]
        first = null["interactions"][0]
        assert first["observed"]
        np.testing.assert_allclose(first["gamma_excess_m"], excesses[0], atol=1e-15)
        assert first["noise_m"] == pytest.approx(m.centered_repeat_noise(excesses))
        assert first["engineering_threshold_m"] == pytest.approx(max(3*m.centered_repeat_noise(excesses), .002))
        assert not null["interactions"][3]["observed"]
        assert null["interactions"][3]["gamma_excess_m"] is None
    assert row["samples"][1]["interactions"][3]["observed"]
    assert row["relative_pair_interactions"][-1]["observed"]


def test_oracle_canonical_bytes_must_match_collection_and_passed_audit(tmp_path):
    group, model, _ = _oracle_group(tmp_path)
    canonical = tmp_path/group["canonical_archive"]
    canonical.write_bytes(canonical.read_bytes()+b"altered")
    with pytest.raises(ValueError, match="collection-time SHA256"):
        m.summarize_m2([group], model=model)


def test_oracle_backend_must_match_the_audited_physical_and_nominal_geometry(tmp_path):
    group, _, _ = _oracle_group(tmp_path)
    class WrongModel(_OracleFK):
        def transforms(self, *args):
            positions, rotations = super().transforms(*args)
            return positions+[.001, 0., 0.], rotations
    with pytest.raises(ValueError, match="audited canonical/physical geometry"):
        m.summarize_m2([group], model=WrongModel())


def test_load_groups_retains_manifest_joint_mapping_for_the_oracle(tmp_path):
    import json
    names = [f"joint_{index}" for index in range(29)]
    simulator = ["left_finger", *reversed(names), "right_finger"]
    path = tmp_path/"manifest.json"
    path.write_text(json.dumps({"schema": "g1_embodiedskills.m2_collection.v1", "completed": True,
        "simulator_joint_names": simulator, "mjcf_joint_names": names, "groups": [{"id": "group"}]}))
    group = m.load_groups([path])[0]
    assert group["_simulator_joint_names"] == simulator
    assert group["_mjcf_joint_names"] == names


def test_publishing_cli_pins_official_fk_sources_and_model_assets(tmp_path):
    import importlib.util
    path = Path(__file__).parents[1]/"scripts/analyze_m2.py"
    spec = importlib.util.spec_from_file_location("analyze_m2_entrypoint", path)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    sonic = tmp_path/"sonic"
    sources = ("gear_sonic/utils/motion_lib/torch_humanoid_batch.py",
        "gear_sonic/isaac_utils/rotations.py", "gear_sonic/trl/utils/torch_transform.py",
        "gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml",
        "gear_sonic/data/assets/robot_description/urdf/g1/main.urdf",
        "gear_sonic/envs/manager_env/robots/g1.py")
    for relative in sources:
        source = sonic/relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("fixture model source")
    pinned = entry.official_fk_input_hashes(sonic)
    assert len(pinned) == 9
    entry.verify_evidence(pinned)
    (sonic/sources[3]).write_text("changed model")
    with pytest.raises(ValueError, match="changed"):
        entry.verify_evidence(pinned)
