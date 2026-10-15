"""Speed limits are optimization constraints, including the inactive return."""
import numpy as np

from g1_embodiedskills.m2_kinematics import solve_compensated_trajectory, tcp_transforms
from g1_embodiedskills.m2_candidates import generate_factor_candidates


class TranslationModel:
    joint_names = [f"joint_{i}" for i in range(29)]
    joint_names[14] = "waist_pitch_joint"
    right_arm_indices = np.arange(22, 29)
    right_wrist_index = 0
    joint_limits = np.tile([-3., 3.], (29, 1))

    def transforms(self, q, root, quaternion):
        q, root, quaternion = np.asarray(q), np.asarray(root), np.asarray(quaternion)
        shape = np.broadcast_shapes(q.shape[:-1], root.shape[:-1], quaternion.shape[:-1])
        q, root = np.broadcast_to(q, shape+(29,)), np.broadcast_to(root, shape+(3,))
        return (root+q[..., 22:25])[..., None, :], np.broadcast_to(np.eye(3), shape+(1, 3, 3))


def solve(offset, qbase=None, *, max_joint_speed_rad_s=6., max_offset_speed_rad_s=2.):
    frames = len(offset)
    qbase = np.zeros((frames, 29)) if qbase is None else np.asarray(qbase, float)
    roots = np.zeros((frames, 3))
    quat = np.tile([1., 0., 0., 0.], (frames, 1))
    model = TranslationModel()
    positions, rotations = tcp_transforms(model, qbase, roots, quat, tcp_offset=(0., 0., 0.))
    positions[:, 0] += offset
    active = np.r_[False, np.ones(frames-2, bool), False]
    return solve_compensated_trajectory(model, qbase, roots, quat, positions, rotations,
        qbase[:, 14], active, waist_index=14, tcp_offset=(0., 0., 0.),
        max_joint_speed_rad_s=max_joint_speed_rad_s, max_offset_speed_rad_s=max_offset_speed_rad_s)


def test_continuous_feasible_solution_and_bitwise_inactive_endpoints():
    result = solve([0., .02, .03, .02, 0.])
    assert result.frame_passed.all()
    assert result.velocity_feasible.all()
    assert np.abs(np.diff(result.q, axis=0)).max()*50. <= 2.+1e-8
    np.testing.assert_array_equal(result.q[[0, -1]], np.zeros((2, 29)))
    np.testing.assert_allclose(result.q[:, 22], [0., .02, .03, .02, 0.], atol=1e-6)


def test_unreachable_pose_is_not_repaired_by_final_clipping():
    result = solve([0., .06, .06, .06, 0.])
    # First/last active poses cannot reach .06 under 2 rad/s in one step.
    assert result.velocity_feasible.all()
    assert not result.frame_passed[1]
    assert not result.frame_passed[-2]
    assert result.position_errors_m[-2] >= .019999
    assert result.q[-2, 22] <= .04+1e-10
    assert np.abs(np.diff(result.q, axis=0)).max()*50. <= 2.+1e-8
    assert result.q[-1, 22] == 0.


def test_total_speed_bound_and_relative_offset_bound_both_apply():
    qbase = np.zeros((5, 29))
    qbase[:, 22] = np.arange(5)*.1
    result = solve([0., .035, .05, .035, 0.], qbase)
    assert result.velocity_feasible.all()
    assert np.abs(np.diff(result.q, axis=0)).max()*50. <= 6.+1e-8
    assert np.abs(np.diff(result.q-qbase, axis=0)).max()*50. <= 2.+1e-8
    assert not result.frame_passed[1]
    np.testing.assert_array_equal(result.q[-1], qbase[-1])


def test_empty_speed_intersection_explicitly_invalidates_frame():
    qbase = np.zeros((3, 29))
    qbase[:, 22] = [0., .5, 1.]
    result = solve([0., 0., 0.], qbase)
    assert not result.velocity_feasible[1]
    assert not result.frame_passed[1]
    # Diagnostic fallback remains visibly the unchanged base, not a fabricated
    # supposedly valid endpoint or a clipped executable command.
    np.testing.assert_array_equal(result.q[1], qbase[1])


def test_exactly_singleton_speed_boundary_is_a_valid_fixed_coordinate():
    qbase = np.zeros((3, 29))
    qbase[:, 22:] = np.array([0., .12, .24])[:, None]
    result = solve([0., 0., 0.], qbase)
    assert result.velocity_feasible.all()
    assert result.frame_passed.all()
    np.testing.assert_array_equal(result.q, qbase)
    # Seven exactly determined arm coordinates need no optimization variables.
    assert result.solver_evaluations[1] == 0


def test_candidate_grid_preserves_explicit_velocity_infeasibility():
    qbase = np.zeros((105, 29))
    qbase[1, 22] = .5
    qbase[2:, 22] = 1.
    result = generate_factor_candidates(TranslationModel(),qbase,np.zeros((105,3)),
        np.tile([1.,0.,0.,0.],(105,1)),np.zeros((105,2)),max_halvings=0)
    assert not result.all_feasible
    assert any('velocity_constraint_intersection_infeasible' in row['reasons']
               and row['solver']['velocity_infeasible_frames'] > 0 for row in result.diagnostics)
