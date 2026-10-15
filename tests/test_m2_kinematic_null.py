"""Counterexamples separating nonlinear FK artifacts from excess joint errors."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from g1_embodiedskills.m2_kinematic_null import (
    CELLS, kinematic_additive_error_null, root_rotations_from_wxyz, simulator_to_mjcf,
)


class NonlinearKinematics:
    right_wrist_index = 0
    joint_limits = np.tile([-3., 3.], (29, 1))

    def transforms(self, q, roots, quaternions):
        q, roots = np.asarray(q), np.asarray(roots)
        rotations = Rotation.from_quat(np.asarray(quaternions)[..., [1, 2, 3, 0]]).as_matrix()
        local = np.stack((q[..., 0]**3, q[..., 1]**2, .8+.2*np.sin(q[..., 2])), axis=-1)
        position = roots+np.einsum("...ij,...j->...i", rotations, local)
        return position[..., None, :], rotations[..., None, :, :]


def input_grid(model, *, constant=False):
    nominal = np.zeros((9, 29))
    error = np.zeros((9, 29))
    roots = np.tile([0., 0., .76], (9, 1))
    rotations = np.tile(np.eye(3), (9, 1, 1))
    for cell, (u, b) in enumerate(CELLS):
        us, bs = (0, 1, -1)[u], (0, 1, -1)[b]
        nominal[cell, 0] = .3+.1*us+.15*bs
        nominal[cell, 1] = -.1+.1*us+.05*bs
        error[cell, 0] = .05 if constant else .02+.03*us-.01*bs
    actual = nominal+error
    measured = model.transforms(actual, roots, np.tile([1., 0., 0., 0.], (9, 1)))[0][:, 0]
    return {"actual_q": actual, "actual_roots": roots.copy(), "actual_root_rotations": rotations.copy(),
        "nominal_q": nominal, "nominal_roots": roots, "nominal_root_rotations": rotations,
        "actual_tcp": measured, "tcp_offset": (0., 0., 0.)}


@pytest.mark.parametrize("constant", [True, False])
def test_nonlinear_fk_creates_tracking_interaction_but_no_excess(constant):
    model = NonlinearKinematics()
    args = input_grid(model, constant=constant)
    result = kinematic_additive_error_null(model, **args)
    assert np.linalg.norm(result["gamma_tracking_actual"], axis=-1).max() > .001
    for version in ("unclipped", "joint_clipped"):
        np.testing.assert_allclose(result[version]["gamma_excess"], 0., atol=1e-15)
        np.testing.assert_allclose(result[version]["gamma_null"], result["gamma_actual"], atol=1e-15)
        assert result[version]["observed"].all()


def test_nonadditive_cross_joint_error_is_not_used_to_fit_the_oracle():
    model = NonlinearKinematics()
    args = input_grid(model, constant=True)
    original = kinematic_additive_error_null(model, **args)
    args["actual_q"][4, 0] += .05
    args["actual_tcp"] = model.transforms(args["actual_q"], args["actual_roots"],
        np.tile([1., 0., 0., 0.], (9, 1)))[0][:, 0]
    changed = kinematic_additive_error_null(model, **args)
    np.testing.assert_array_equal(original["q_null_unclipped"], changed["q_null_unclipped"])
    assert changed["unclipped"]["gamma_excess"][0, 0] > .01
    np.testing.assert_allclose(changed["unclipped"]["gamma_excess"][1:], 0., atol=1e-15)


def test_additive_world_root_errors_use_left_so3_composition():
    model = NonlinearKinematics()
    args = input_grid(model, constant=True)
    for cell, (u, b) in enumerate(CELLS):
        us, bs = (0, 1, -1)[u], (0, 1, -1)[b]
        nominal_R = (Rotation.from_euler("z", .3*us)*Rotation.from_euler("y", .2*bs)).as_matrix()
        eta = np.array([.03+.01*us, -.02+.02*bs, .01*us-.01*bs])
        args["nominal_root_rotations"][cell] = nominal_R
        args["actual_root_rotations"][cell] = Rotation.from_rotvec(eta).as_matrix() @ nominal_R
        args["actual_roots"][cell] += [.01+.02*us, -.01+.03*bs, .005*us-.004*bs]
    quaternions = Rotation.from_matrix(args["actual_root_rotations"]).as_quat()[:, [3, 0, 1, 2]]
    args["actual_tcp"] = model.transforms(args["actual_q"], args["actual_roots"], quaternions)[0][:, 0]
    result = kinematic_additive_error_null(model, **args)
    np.testing.assert_allclose(result["unclipped"]["gamma_excess"], 0., atol=2e-15)


def test_measured_axis_tcps_are_preserved_and_excess_is_cross_cell_difference():
    model = NonlinearKinematics()
    args = input_grid(model)
    args["actual_tcp"][[0, 1, 2, 3, 6], 1] += np.array([.001, -.003, .002, .004, -.002])
    result = kinematic_additive_error_null(model, **args)
    np.testing.assert_array_equal(result["unclipped"]["tcp_null"][[0, 1, 2, 3, 6]], args["actual_tcp"][[0, 1, 2, 3, 6]])
    np.testing.assert_allclose(result["unclipped"]["gamma_excess"], 0., atol=1e-15)
    np.testing.assert_allclose(result["gamma_actual"]-result["unclipped"]["gamma_null"],
        result["unclipped"]["gamma_excess"], atol=1e-15)


def test_four_necessary_cell_censoring_never_fills_zero():
    model = NonlinearKinematics()
    args = input_grid(model)
    args["actual_q"][3] = np.nan
    result = kinematic_additive_error_null(model, **args)
    assert result["observed"].tolist() == [False, False, True, True]
    assert np.isnan(result["unclipped"]["gamma_excess"][:2]).all()
    args["actual_tcp"][0] = np.nan
    result = kinematic_additive_error_null(model, **args)
    assert not result["observed"].any()
    assert np.isnan(result["unclipped"]["gamma_excess"]).all()


def test_joint_limit_extrapolation_and_fixed_clip_are_both_reported():
    model = NonlinearKinematics()
    model.joint_limits = np.tile([-.2, .2], (29, 1))
    args = input_grid(model)
    args["nominal_q"][:] = 0.
    args["actual_q"][:] = 0.
    args["actual_q"][[1, 3], 0] = .18
    args["actual_q"][4, 0] = .2
    args["actual_tcp"] = model.transforms(args["actual_q"], args["actual_roots"], np.tile([1., 0., 0., 0.], (9, 1)))[0][:, 0]
    result = kinematic_additive_error_null(model, **args)
    assert result["null_joint_limit_exceeded"][0]
    assert result["null_max_joint_limit_violation_rad"][0] == pytest.approx(.16)
    assert result["unclipped"]["q_null"][0, 0] == pytest.approx(.36)
    assert result["joint_clipped"]["q_null"][0, 0] == pytest.approx(.2)
    assert abs(result["unclipped"]["gamma_excess"][0, 0]) > .03
    np.testing.assert_allclose(result["joint_clipped"]["gamma_excess"][0], 0., atol=1e-15)


def test_named_q_mapping_and_missing_root_quaternions():
    body_names = [f"body{i}" for i in range(29)]
    sim_names = ["finger_left", *reversed(body_names), "finger_right"]
    q = np.arange(31.)[None]
    mapped = simulator_to_mjcf(q, sim_names, body_names)
    np.testing.assert_array_equal(mapped[0], np.arange(29., 0., -1))
    with pytest.raises(ValueError, match="contained MJCF"):
        simulator_to_mjcf(q, sim_names, [*body_names[:-1], "missing_joint"])
    quaternions = np.array([[1., 0., 0., 0.], [np.nan]*4])
    rotations = root_rotations_from_wxyz(quaternions)
    np.testing.assert_array_equal(rotations[0], np.eye(3))
    assert np.isnan(rotations[1]).all()


def test_invalid_observed_rotation_is_not_silently_projected():
    model = NonlinearKinematics()
    args = input_grid(model)
    args["actual_root_rotations"][4, 0, 0] = 2.
    with pytest.raises(ValueError, match="non-SO"):
        kinematic_additive_error_null(model, **args)
