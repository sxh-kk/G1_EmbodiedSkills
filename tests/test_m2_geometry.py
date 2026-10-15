"""Geometry/metric regressions before any new M2 physics collection."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation, Slerp

from g1_embodiedskills.metrics import episode_outcome
from g1_embodiedskills.policies import (
    NominalPickPlacePolicy, PickPlaceSpec, scene_geometry, spec_from_geometry,
    translated_spec,
)


def _old_geometry():
    geometry = PickPlaceSpec().as_dict()
    for key in ("table_thickness_m", "tray_thickness_m", "cube_spawn_clearance_m",
                "approach_tcp_x", "approach_tcp_z", "retreat_tcp_z"):
        geometry.pop(key)
    return geometry


def test_old_manifest_and_default_scene_preserve_spawn_gap_exactly():
    assert spec_from_geometry(_old_geometry()) == PickPlaceSpec()
    assert translated_spec() == PickPlaceSpec()
    assert scene_geometry(_old_geometry()) == {
        "table_size": (.55, .65, .04), "table_position": (.5, -.12, .69),
        "cube_size": (.035, .035, .035), "cube_position": (.28, -.23, .728),
        "tray_size": (.16, .16, .012), "tray_position": (.28, -.03, .716),
    }
    assert PickPlaceSpec().cube_position == (.28, -.23, .7275)


@pytest.mark.parametrize("change", [
    {"typo": 0.}, {"cube_position": [0., 0.]}, {"table_top_z": float("nan")},
    {"cube_size_m": True}, {"tray_size_xy": [0., .1]},
    {"cube_spawn_clearance_m": -.001}, {"table_top_z": .8},
    {"tray_top_z": .70}, {"duration": 23.},
])
def test_invalid_or_inconsistent_geometry_fails_explicitly(change):
    geometry = _old_geometry()
    geometry.update(change)
    with pytest.raises(ValueError):
        spec_from_geometry(geometry)


@pytest.mark.parametrize("value", [True, "0.03", None, float("nan"), [.03]])
def test_invalid_displacement_fails_explicitly(value):
    with pytest.raises(ValueError):
        translated_spec(dz=value)


def test_default_policy_matches_legacy_waypoints_and_all_50hz_targets():
    initial = np.array([.15654515166271082, -.2275724668766297, .6727760053453533])
    quaternion = np.array([.916776035627039, -.08970305217279476,
                           .38643645804247834, -.04628095533195908])
    down = (Rotation.from_euler("z", np.pi / 2) * Rotation.from_euler("y", np.pi / 2)).as_quat()[[3, 0, 1, 2]]
    times = [0., 1., 4., 7., 10., 11., 14., 17., 20., 21., 24.]
    positions = [initial, initial, [.15, -.23, .85], [.28, -.23, .83],
                 [.28, -.23, .747], [.28, -.23, .747], [.28, -.23, .83],
                 [.28, -.03, .83], [.28, -.03, .759], [.28, -.03, .759],
                 [.28, -.03, .84]]
    quaternions = [quaternion] * 3 + [down] * 8
    policy = NominalPickPlacePolicy(initial, quaternion, _old_geometry())
    for row, time, position, rotation in zip(policy.keyframes, times, positions, quaternions):
        assert row[0] == time
        np.testing.assert_array_equal(row[2], position)
        np.testing.assert_array_equal(row[3], rotation)
    for time in np.arange(1301) / 50:
        if time >= 24.:
            expected_p, expected_q = np.asarray(positions[-1]), quaternions[-1]
        else:
            i = max(i for i, left in enumerate(times[:-1]) if left <= max(time, 0.))
            fraction = np.clip((time-times[i])/(times[i+1]-times[i]), 0., 1.)
            fraction = fraction**3 * (10-15*fraction+6*fraction**2)
            expected_p = (1-fraction)*np.asarray(positions[i])+fraction*np.asarray(positions[i+1])
            rotations = Rotation.from_quat(np.asarray(quaternions[i:i+2])[:, [1, 2, 3, 0]])
            expected_q = Slerp([0, 1], rotations)(fraction).as_quat()[[3, 0, 1, 2]]
        actual_p, actual_q = policy.target_pose(time)
        np.testing.assert_array_equal(actual_p, expected_p)
        np.testing.assert_array_equal(actual_q, expected_q)
        assert policy.gripper_target(time) == (0. if 10. <= time < 20. else .035)


@pytest.mark.parametrize("changes,shift", [
    ({"dz": -.03}, [0., 0., -.03]), ({"dz": .03}, [0., 0., .03]),
    ({"object_dx": .03}, [.03, 0., 0.]), ({"object_dy": -.03}, [0., -.03, 0.]),
])
def test_condition_translates_only_declared_geometry_and_operation_targets(changes, shift):
    base, modified = PickPlaceSpec(), translated_spec(**changes)
    shift = np.asarray(shift)
    np.testing.assert_allclose(modified.cube_position, np.asarray(base.cube_position)+shift, atol=1e-15, rtol=0)
    np.testing.assert_allclose(modified.tray_position_xy, np.asarray(base.tray_position_xy)+shift[:2], atol=1e-15, rtol=0)
    assert modified.table_center_xy == base.table_center_xy
    assert modified.table_size_xy == base.table_size_xy
    assert modified.table_top_z == base.table_top_z+shift[2]
    initial, quaternion = [.15, -.23, .67], [1., 0., 0., 0.]
    before = NominalPickPlacePolicy(initial, quaternion, base)
    after = NominalPickPlacePolicy(initial, quaternion, modified)
    for old, new in zip(before.keyframes[:2], after.keyframes[:2]):
        np.testing.assert_array_equal(old[2], new[2])
    for old, new in zip(before.keyframes[2:], after.keyframes[2:]):
        np.testing.assert_allclose(new[2], np.asarray(old[2])+shift, atol=1e-15, rtol=0)
    for key in ("cube_position", "tray_position"):
        np.testing.assert_allclose(scene_geometry(modified)[key], np.asarray(scene_geometry(base)[key])+shift, atol=1e-15, rtol=0)


def _trace(z=.7395):
    count = 1201
    cube = np.tile([.28, -.03, z], (count, 1, 1))
    cube[0, 0] = [.28, -.23, .728]
    cube[1, 0, 2] = .85
    return {"cube": cube, "tcp": np.tile([.10, -.03, .84], (count, 1, 1)),
            "root": np.tile([0., 0., .76, 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], (count, 1, 1)),
            "gripper": np.full((count, 1, 2), .035)}


@pytest.mark.parametrize("z,expected", [
    (.733, False), (np.nextafter(.733, np.inf), True),
    (.75, False), (np.nextafter(.75, -np.inf), True), (.7395, True),
])
def test_legacy_strict_placement_height_endpoints_are_unchanged(z, expected):
    trace = _trace(z)
    default = episode_outcome(trace)
    explicit = episode_outcome(trace, geometry=_old_geometry())
    assert default == explicit
    assert default["placed"] is expected
    assert default["success"] is expected
    assert default["lifted"] and default["released"] and default["retreated"] and default["standing"]


@pytest.mark.parametrize("changes,shift", [
    ({"dz": -.03}, [0., 0., -.03]), ({"dz": .03}, [0., 0., .03]),
    ({"object_dx": .03}, [.03, 0., 0.]), ({"object_dy": -.03}, [0., -.03, 0.]),
])
def test_relative_outcomes_are_invariant_to_matching_task_displacement(changes, shift):
    trace = _trace()
    original = episode_outcome(trace)
    moved = {key: value.copy() for key, value in trace.items()}
    moved["cube"] += shift
    moved["tcp"] += shift
    outcome = episode_outcome(moved, geometry=translated_spec(**changes))
    for key in ("lifted", "placed", "released", "retreated", "standing", "terminated", "success"):
        assert outcome[key] == original[key]
    assert outcome["return_value"] == original["return_value"]
    assert outcome["final_target_distance_m"] == original["final_target_distance_m"]
    np.testing.assert_allclose(outcome["final_cube_position"], np.asarray(original["final_cube_position"])+shift, atol=1e-15, rtol=0)
    assert outcome["maximum_cube_height_m"] == original["maximum_cube_height_m"]+shift[2]


def test_env_origin_translation_is_distinct_from_task_geometry():
    trace = _trace()
    original = episode_outcome(trace)
    origin = np.array([5., -10., 0.])
    moved = {key: value.copy() for key, value in trace.items()}
    moved["cube"] += origin
    moved["tcp"] += origin
    moved["root"][..., :3] += origin
    outcome = episode_outcome(moved, origin=origin)
    for key in ("lifted", "placed", "released", "retreated", "standing", "terminated", "success"):
        assert outcome[key] == original[key]
    assert outcome["return_value"] == pytest.approx(original["return_value"], abs=1e-14)
