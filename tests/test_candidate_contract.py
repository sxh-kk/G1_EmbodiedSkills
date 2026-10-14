"""Candidate/native action alignment, immutability and real-FK diagnostics."""
import os

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from g1_embodiedskills.candidates import encode_native, generate_candidates


class AnalyticArm:
    """A transparent full-rank TCP model for codec and rejection checks."""
    right_arm_indices = np.arange(22, 29)
    right_wrist_index = 0
    joint_limits = np.tile([-3., 3.], (29, 1))

    def transforms(self, q, roots, quaternions):
        q, roots, quaternions = np.asarray(q), np.asarray(roots), np.asarray(quaternions)
        shape = np.broadcast_shapes(q.shape[:-1], roots.shape[:-1], quaternions.shape[:-1])
        q = np.broadcast_to(q, shape + (29,))
        roots = np.broadcast_to(roots, shape + (3,))
        local = Rotation.from_rotvec(q[..., 25:28].reshape(-1, 3).copy())
        root = Rotation.from_quat(np.broadcast_to(quaternions, shape + (4,))[..., [1, 2, 3, 0]].reshape(-1, 4))
        matrices = (root * local).as_matrix().reshape(shape + (3, 3))
        position = roots + np.einsum("...ij,...j->...i", root.as_matrix().reshape(shape + (3, 3)), .2 * q[..., 22:25])
        return position[..., None, :], matrices[..., None, :, :]


def previews(dtype=np.float64):
    q = np.zeros((75, 29), dtype=dtype)
    roots = np.tile([0., 0., .75], (75, 1)).astype(dtype)
    quaternions = np.tile([1., 0., 0., 0.], (75, 1)).astype(dtype)
    fingers = np.full((75, 2), .035, dtype=dtype)
    return q, roots, quaternions, fingers


def test_native_codec_preserves_baseline_and_never_mutates_inputs():
    inputs = previews()
    inputs[0][:, 1] = np.linspace(0, .1, 75)
    snapshots = [value.copy() for value in inputs]
    batch = generate_candidates(AnalyticArm(), *inputs)
    assert isinstance(batch.native, tuple) and len(batch.native) == 5
    assert batch.actions.shape == (5, 75, 38) and batch.actions.dtype == np.float32
    assert batch.all_feasible
    for original, snapshot, baseline in zip(inputs, snapshots, batch.native[0]):
        assert original.tobytes() == snapshot.tobytes() == baseline.tobytes()
        assert original.dtype == baseline.dtype
        assert not np.shares_memory(original, baseline)
    for index, command in enumerate(batch.native):
        assert np.array_equal(batch.actions[index], encode_native(command))
        assert np.array_equal(command.qtraj_mjcf[0], inputs[0][0])
        assert np.array_equal(command.qtraj_mjcf[:, :22], inputs[0][:, :22])
        for value, original in zip(command[1:], inputs[1:]):
            assert np.array_equal(value, original)
        assert all(not value.flags.writeable for value in command)


def test_actual_fk_offsets_have_the_requested_world_directions_and_ramp():
    batch = generate_candidates(AnalyticArm(), *previews(np.float32))
    requested = np.asarray([[.012, 0, 0], [-.012, 0, 0], [0, .012, 0], [0, -.012, 0]])
    actual = np.asarray([row["final_actual_tcp_offset_m"] for row in batch.diagnostics[1:]])
    # DLS intentionally regularizes the inverse; on this full-rank arm its
    # small attenuation is bounded to 0.15mm rather than claimed to be exact.
    assert np.allclose(actual, requested, atol=.00015)
    assert np.allclose(batch.native[1].qtraj_mjcf[10:], batch.native[1].qtraj_mjcf[10])
    assert np.linalg.norm(batch.native[1].qtraj_mjcf[5]) < np.linalg.norm(batch.native[1].qtraj_mjcf[10])


def test_limit_failures_return_status_without_clipping_or_duplicate_substitution():
    model = AnalyticArm()
    model.joint_limits = np.tile([-.02, .02], (29, 1))
    batch = generate_candidates(model, *previews())
    assert batch.feasible[0] and not batch.feasible[1:].any()
    assert batch.native[1].qtraj_mjcf[:, 22].max() > .02
    for row in batch.diagnostics[1:]:
        assert "joint_limit_violation" in row["reasons"]
        assert row["clipped"] is False
    assert len({row.tobytes() for row in batch.actions}) == 5


def test_invalid_quaternions_and_shape_are_not_silently_repaired():
    inputs = list(previews())
    inputs[2] *= 2
    with pytest.raises(ValueError, match="normalized"):
        generate_candidates(AnalyticArm(), *inputs)
    with pytest.raises(ValueError, match="shape"):
        generate_candidates(AnalyticArm(), previews()[0][:-1], *previews()[1:])


def test_degenerate_jacobian_is_marked_infeasible_instead_of_five_fake_choices():
    class DisconnectedArm(AnalyticArm):
        def transforms(self, q, roots, quaternions):
            return super().transforms(np.zeros_like(q), roots, quaternions)
    batch = generate_candidates(DisconnectedArm(), *previews())
    assert batch.feasible.tolist() == [True, False, False, False, False]
    assert all("duplicate_encoded_candidate" in row["reasons"] for row in batch.diagnostics[1:])
    assert all("offset_direction_not_realized" in row["reasons"] for row in batch.diagnostics[1:])


def test_offset_slew_is_uniformly_scaled_and_effective_target_is_explicit():
    batch = generate_candidates(AnalyticArm(), *previews(), amplitude_m=.12)
    assert batch.all_feasible
    for row in batch.diagnostics[1:]:
        assert 0 < row["uniform_slew_scale"] < 1
        assert row["maximum_offset_slew_rad_s"] <= 2.00001
        assert row["maximum_joint_speed_rad_s"] <= 6.
        assert np.allclose(row["effective_world_offset_m"],
                           np.asarray(row["requested_world_offset_m"]) * row["uniform_slew_scale"])
        assert row["clipped"] is False


def _singular_pilot_preview(model):
    """Saved direct-reference knots from debug seed1024/anchor990.

    This kinematic regression fixture contains no task outcome or held-out
    evaluation data. The near-singular shoulder/elbow reference used to make
    the current-pose inverse reverse +X later in its 75-frame preview.
    """
    q, root, _ = model.default_stand()
    knots = np.asarray([
        [-1.1442303229793014, -.049149241942389635, 2.1866647720336916, 1.4187018394470214, .543422057705171, 1.055675556546203, .9045114704386559],
        [-1.1290205473024344, -.0766418252646939, 2.1869961738586428, 1.4235826849937439, .540552917630339, 1.0462935314717288, .9252968519820209],
        [-1.1240181445113622, -.08701219563506427, 2.1869961738586428, 1.4235826849937439, .5399199512329701, 1.043843161194732, .9320242847413303],
        [-1.1263795613280736, -.09039906007371726, 2.186733675003052, 1.4156851291656494, .5419628408280018, 1.0480736604623102, .9307221020669223],
    ])
    q = np.repeat(q[None], 75, axis=0)
    q[:, model.right_arm_indices] = np.stack([np.interp(np.arange(75), [0, 10, 60, 74], knots[:, column])
                                             for column in range(7)], axis=-1)
    return q, np.repeat(root[None], 75, axis=0), np.tile([1., 0., 0., 0.], (75, 1)), np.full((75, 2), .035)


@pytest.mark.skipif(not os.environ.get("SONIC_ROOT"), reason="Set SONIC_ROOT to run real G1 FK integration")
def test_real_near_singular_pilot_has_no_direction_reversal_or_unbounded_slew():
    from g1_embodiedskills.kinematics import G1Kinematics
    model = G1Kinematics(os.environ["SONIC_ROOT"])
    inputs = _singular_pilot_preview(model)
    legacy = generate_candidates(model, *inputs, amplitude_m=.00078125, damping=.001,
        finite_difference_rad=1e-4, jacobian_mode="current", max_delta_speed_rad_s=None,
        max_joint_speed_rad_s=None)
    assert "offset_direction_not_realized" in legacy.diagnostics[1]["reasons"]
    batch = generate_candidates(model, *inputs, amplitude_m=.012)
    assert batch.all_feasible, batch.diagnostics
    assert len({row.tobytes() for row in batch.actions}) == 5
    assert np.array_equal(batch.native[0].qtraj_mjcf, inputs[0])
    for row in batch.diagnostics[1:]:
        assert row["jacobian_pose_count"] == 75 and row["damping"] == .02
        assert row["maximum_offset_slew_rad_s"] <= 2.00001
        assert row["maximum_joint_speed_rad_s"] <= 6.
        assert row["minimum_joint_limit_margin_rad"] >= 0
        assert np.asarray(row["final_actual_tcp_offset_m"]) @ row["requested_world_offset_m"] > 0


@pytest.mark.skipif(not os.environ.get("SONIC_ROOT"), reason="Set SONIC_ROOT to run real G1 FK integration")
def test_real_sonic_fk_direction_amplitude_and_native_contract():
    from g1_embodiedskills.kinematics import G1Kinematics
    model = G1Kinematics(os.environ["SONIC_ROOT"])
    q, root, _ = model.default_stand()
    inputs = (np.repeat(q[None], 75, axis=0), np.repeat(root[None], 75, axis=0),
              np.tile([1., 0., 0., 0.], (75, 1)), np.full((75, 2), .035))
    batch = generate_candidates(model, *inputs)
    assert batch.all_feasible, batch.diagnostics
    for row in batch.diagnostics[1:]:
        actual = np.asarray(row["final_actual_tcp_offset_m"])
        desired = np.asarray(row["requested_world_offset_m"])
        assert actual @ desired > 0
        assert np.linalg.norm(actual - desired) < .003
        assert row["maximum_orientation_change_rad"] < .035
    assert np.array_equal(batch.native[0].qtraj_mjcf, inputs[0])
    assert all(np.array_equal(batch.actions[i], encode_native(command)) for i, command in enumerate(batch.native))
