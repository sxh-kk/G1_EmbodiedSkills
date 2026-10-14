"""Synthetic RGB-D tests; no Isaac application or object scene state required."""
import unittest
from types import SimpleNamespace
import numpy as np

from g1_embodiedskills.perception import estimate_target, color_mask, render_observations


class PerceptionTest(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        self.rgb[27:38, 27:38] = [210, 12, 8]
        self.depth = np.full((64, 64), np.inf)
        self.depth[27:38, 27:38] = .5
        self.K = np.array([[100., 0., 32.], [0., 100., 32.], [0., 0., 1.]])
        # ROS +Z looks straight down; the visible plane is world z=.75.
        self.camera_pos = [.3, -.2, 1.25]
        self.camera_q = [0., 1., 0., 0.]

    def estimate(self, **kwargs):
        return estimate_target(self.rgb, self.depth, self.K,
                               self.camera_pos, self.camera_q, **kwargs)

    def test_top_plane_center_and_half_height(self):
        result = self.estimate()
        self.assertTrue(result["found"])
        np.testing.assert_allclose(result["position"], [.3, -.2, .7325], atol=1e-9)
        self.assertEqual(result["mask_count"], 121)
        self.assertEqual(result["confidence"], 1.)

    def test_actual_translation_rotation_and_intrinsics(self):
        # 90 degree rotation about world Z composed with downward ROS camera.
        self.camera_q = [0., 2**-.5, 2**-.5, 0.]
        self.rgb.fill(0)
        self.rgb[27:38, 37:48] = [210, 12, 8]
        self.depth[:] = .5
        result = self.estimate()
        # Ten horizontal pixels -> camera X=.05 -> world Y=.05.
        np.testing.assert_allclose(result["position"], [.3, -.15, .7325], atol=1e-8)

    def test_green_tray_normalized_rgba_and_depth_channel(self):
        rgb = np.zeros((64, 64, 4), dtype=float)
        rgb[27:38, 27:38] = [.03, .65, .08, 1.]
        result = estimate_target(rgb, self.depth[..., None], self.K,
                                  self.camera_pos, self.camera_q, target="tray")
        self.assertTrue(result["found"])
        np.testing.assert_allclose(result["position"], [.3, -.2, .744], atol=1e-9)

    def test_missing_or_invalid_depth_never_falls_back(self):
        self.rgb.fill(0)
        result = self.estimate()
        self.assertFalse(result["found"])
        self.assertIsNone(result["position"])
        self.rgb[27:38, 27:38] = [210, 12, 8]
        self.depth[:] = np.nan
        result = self.estimate()
        self.assertFalse(result["found"])
        self.assertEqual(result["reason"], "insufficient_valid_depth")

    def test_disconnected_color_noise_removed_and_dark_rgb_supported(self):
        self.rgb[0, 0] = [255, 0, 0]
        self.assertEqual(int(color_mask(self.rgb).sum()), 121)
        self.rgb[27:38, 27:38] = [45, 3, 2]
        self.assertTrue(self.estimate()["found"])

    def test_rejects_invalid_calibration(self):
        self.K[0, 0] = 0.
        with self.assertRaises(ValueError):
            self.estimate()


class BatchedCameraTest(unittest.TestCase):
    def setUp(self):
        rgb = np.zeros((3, 64, 64, 4), dtype=np.uint8)
        rgb[:, 27:38, 27:38] = [210, 12, 8, 255]
        rgb[:, 27:38, 8:19] = [8, 180, 16, 255]
        # One camera genuinely cannot see the cube.
        rgb[1, 27:38, 27:38] = 0
        intrinsic = np.tile([[100., 0., 32.], [0., 100., 32.], [0., 0., 1.]], (3, 1, 1))
        origins = np.array([[0., 0., 0.], [5., 0., 0.], [0., -5., 0.]])
        self.positions = origins + [.3, -.2, 1.25]
        self.rgb = rgb
        data = SimpleNamespace(output={"rgb": rgb,
                                       "distance_to_image_plane": np.full((3, 64, 64, 1), .5)},
                               intrinsic_matrices=intrinsic,
                               pos_w=self.positions,
                               # Lab 3 is XYZW; this is a downward looking ROS camera.
                               quat_w_ros=np.tile([1., 0., 0., 0.], (3, 1)))
        self.render_count = 0
        self.update_count = 0

        def render():
            self.render_count += 1

        def update(dt, force_recompute):
            self.assertEqual(dt, 0.)
            self.assertTrue(force_recompute)
            self.update_count += 1

        self.env = SimpleNamespace(sim=SimpleNamespace(render=render),
                                   scene={"head_camera": SimpleNamespace(data=data, update=update)})

    def test_batch_refresh_once_and_world_origin_not_removed(self):
        observations = render_observations(self.env, slots=[2, 0])
        self.assertEqual(list(observations), [2, 0])
        self.assertEqual(self.render_count, 1)
        self.assertEqual(self.update_count, 1)
        for slot in (2, 0):
            self.assertEqual(observations[slot]["rgb"].shape, (64, 64, 3))
            self.assertEqual(observations[slot]["rgb"].dtype, np.uint8)
            self.assertEqual(observations[slot]["depth"].shape, (64, 64, 1))
            np.testing.assert_allclose(observations[slot]["estimates"]["cube"]["position"],
                                       self.positions[slot] + [0., 0., -.5175], atol=1e-8)
            self.assertTrue(observations[slot]["estimates"]["tray"]["found"])
        # Buffered simulator updates cannot mutate the captured host observation.
        self.rgb.fill(0)
        self.assertGreater(int(observations[0]["rgb"].sum()), 0)

    def test_all_slots_missing_stays_missing_and_no_refresh(self):
        observations = render_observations(self.env, refresh=False)
        self.assertEqual(list(observations), [0, 1, 2])
        self.assertEqual(self.render_count, 0)
        self.assertEqual(self.update_count, 0)
        self.assertFalse(observations[1]["estimates"]["cube"]["found"])
        self.assertIsNone(observations[1]["estimates"]["cube"]["position"])
        self.assertTrue(observations[2]["estimates"]["cube"]["found"])

    def test_invalid_slot_rejected(self):
        with self.assertRaises(IndexError):
            render_observations(self.env, slots=[3], refresh=False)
        with self.assertRaises(ValueError):
            render_observations(self.env, slots=[.5], refresh=False)


if __name__ == "__main__":
    unittest.main()
