"""Pure CPU contracts for M2 collection bookkeeping, not efficacy trials."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
import json
import os
import sys
import tempfile
from unittest.mock import patch
import unittest

import numpy as np
import torch

PROJECT = Path(os.environ.get("G1_PROJECT_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(PROJECT / "src"))
SOURCE = Path(__file__).resolve().parents[1] / "src/g1_embodiedskills/m2_collection.py"
spec = spec_from_file_location("g1_embodiedskills.m2_collection", SOURCE)
m2 = module_from_spec(spec)
spec.loader.exec_module(m2)


class CollectionContracts(unittest.TestCase):
    def batch(self):
        return {"schema": "g1_embodiedskills.m2_collection_config.v1", "batches": [{
            "capsule_id": "low-anchor300-batch0", "condition": "lower", "family": "height",
            "anchors": [{"seed": 920000, "anchor_step": 300, "noise_repeat": True}],
            "companion_seeds": [920000]}]}

    def test_explicit_companions_and_common_anchor_time(self):
        self.assertEqual(len(m2.validate_batches(self.batch(), 2)), 1)
        config = self.batch()
        config["batches"][0]["companion_seeds"] = []
        with self.assertRaisesRegex(ValueError, "companion"):
            m2.validate_batches(config, 2)
        config = self.batch()
        config["batches"][0]["anchors"].append({"seed": 920001, "anchor_step": 310})
        config["batches"][0]["companion_seeds"] = []
        with self.assertRaisesRegex(ValueError, "anchor time"):
            m2.validate_batches(config, 2)

    def test_missing_posttermination_sample_is_nan_and_masked(self):
        fields = {"actual_tcp": (3,), "actual_R": (3, 3), "cube_root": (13,),
            "root": (13,), "q": (31,), "qdot": (31,), "gripper": (2,),
            "waist": (), "soles": (2, 3), "nominal_tcp": (3,), "nominal_R": (3, 3)}
        values = {10: {key: np.zeros(shape) for key, shape in fields.items()}}
        values[10]["nominal_command_step"] = 300
        samples = m2.sample_arrays(300, values)
        np.testing.assert_array_equal(samples["valid"], [True, False, False, False])
        self.assertTrue(np.isnan(samples["actual_tcp"][1:]).all())
        np.testing.assert_array_equal(samples["global_steps"], [310, 325, 350, 450])
        np.testing.assert_array_equal(samples["nominal_command_step"], [300, -1, -1, -1])

    def test_trace_is_cut_at_physical_termination_boundary(self):
        trace = {"root": np.ones((20, 2, 13)), "cube": np.ones((20, 2, 3)),
                 "env_origins": np.zeros((2, 3))}
        result, end = m2.truncate_trace(trace, 0, 7)
        self.assertEqual(end, 7)
        self.assertEqual(result["root"].shape, (8, 2, 13))
        self.assertEqual(result["env_origins"].shape, (2, 3))

    def test_boolean_history_fields_are_compared_without_unsigned_wrap(self):
        left = {"done": np.array([True, False]), "key": np.array([0, 255], np.uint8)}
        right = {"done": np.array([False, False]), "key": np.array([255, 0], np.uint8)}
        report = m2.compare_history(left, right, 0.)
        self.assertFalse(report["passed"])
        self.assertEqual(report["fields"]["done"]["max_abs"], 1.)
        self.assertEqual(report["fields"]["key"]["max_abs"], 255.)

    def test_policy_history_includes_temporal_state_and_servo_validity(self):
        policy = SimpleNamespace(steps=3,
            obs_dict_buffer={"actor_obs": torch.ones((2, 3, 930)), "tokenizer": torch.ones((2, 3, 1761))},
            dones_buffer=torch.zeros((2, 2), dtype=torch.bool))
        servo = SimpleNamespace(correction=np.zeros(29), cube_offset=np.zeros(3), tray_offset=np.zeros(3),
                                last_estimate_time={"cube": 5., "tray": None})
        measured = {"root": np.tile(np.r_[np.zeros(3), 1., np.zeros(9)], (2, 1)),
                    "cube_root": np.tile(np.r_[np.zeros(3), 1., np.zeros(9)], (2, 1)),
                    "q": np.zeros((2, 31)), "qdot": np.zeros((2, 31))}
        observations = {"actor_obs": torch.zeros((2, 930)), "tokenizer": torch.zeros((2, 1761))}
        state = m2.history_snapshot(policy, observations, servo, measured, 0, np.zeros(3))
        self.assertEqual(state["policy_observation/actor_obs"].shape, (3, 930))
        np.testing.assert_array_equal(state["servo_target_valid"], [1, 0])
        self.assertEqual(int(state["policy_steps"]), 3)
        self.assertEqual(int(state["cached_estimates_present"]), 0)
        self.assertEqual(state["cached_estimates_json"].tobytes(), b"null")

    def test_cached_estimates_snapshot_preserves_exact_complete_values(self):
        cache = {"tray": {"found": True, "confidence": .9, "position": np.array([.3, -.03, .716]),
                           "surface_position": np.array([.3, -.03, .722]), "reason": "visible_top_approximation"},
                 "cube": {"found": False, "confidence": 0., "position": None,
                           "surface_position": None, "reason": "missing_color"}}
        serialized = m2.serialize_cached_estimates(cache)
        self.assertEqual(json.loads(serialized)["tray"]["position"], [.3, -.03, .716])
        reordered = {key: cache[key] for key in reversed(cache)}
        self.assertEqual(serialized, m2.serialize_cached_estimates(reordered))
        before = np.frombuffer(serialized.encode(), dtype=np.uint8)
        cache["tray"]["confidence"] = .900000000000001
        after = np.frombuffer(m2.serialize_cached_estimates(cache).encode(), dtype=np.uint8)
        self.assertFalse(m2.compare_history({"cached_estimates_json": before},
                                          {"cached_estimates_json": after}, 1e-6)["passed"])

    def test_mock_full_grid_records_frozen_default_and_sampling_order(self):
        self._mock_full_grid()

    def test_late_anchor_default_noise_repeats_reuse_rgbd_cache_and_continuation(self):
        self._mock_full_grid(anchor_step=480, consume_cache=True)

    def test_bridge_starts_at_installed_reference_and_preserves_tail(self):
        from g1_embodiedskills.candidates import NativeCommand
        q, root, quat = np.ones((105, 29)), np.ones((105, 3)), np.tile([1., 0., 0., 0.], (105, 1))
        fingers = np.full((105, 2), .035)
        patcher = SimpleNamespace(lib=SimpleNamespace(length_starts=np.array([0])),
            _existing_inputs=lambda start, end: (np.zeros((end-start, 29)),
                np.zeros((end-start, 3)), np.tile([1., 0., 0., 0.], (end-start, 1))))
        native, report = m2.bridge_installed_prefix(patcher, 0, 300, NativeCommand(q, root, quat, fingers))
        np.testing.assert_array_equal(native[0][0], 0.)
        np.testing.assert_array_equal(native[0][15:], q[15:])
        np.testing.assert_array_equal(native[3], fingers)
        self.assertEqual(report["frame_zero_q_max_abs_error"], 0.)

    def _mock_full_grid(self, anchor_step=300, consume_cache=False):
        """CPU state-machine smoke only; does not model robot dynamics."""
        from g1_embodiedskills.candidates import NativeCommand, encode_native

        class Model:
            joint_names = ["waist_pitch_joint"] + [f"joint-{index}" for index in range(28)]
            body_names = ["wrist", "left-foot", "right-foot"]
            right_wrist_index = 0
            sole_offsets = {"left-foot": np.zeros(3), "right-foot": np.zeros(3)}
            def __init__(self, root):
                pass
            def transforms(self, q, root, quat):
                return np.tile(root, (3, 1)), np.tile(np.eye(3), (3, 1, 1))

        class Servo:
            def __init__(self, *args, **kwargs):
                self.correction, self.cube_offset, self.tray_offset = np.zeros(29), np.zeros(3), np.zeros(3)
                self.last_estimate_time = {"cube": None, "tray": None}
            def observe_targets(self, estimates, elapsed):
                if consume_cache and estimates is not None and estimates["tray"]["found"]:
                    value = np.asarray(estimates["tray"]["position"]) - [.28, -.03, .716]
                    self.tray_offset = .5*self.tray_offset + .5*value
                    self.last_estimate_time["tray"] = elapsed
            def update(self, *args):
                if consume_cache:
                    self.correction[0] = self.tray_offset[0]
                return self.correction.copy(), self.correction.copy(), {}
            def trajectory(self, cursor, previous, correction, frames):
                return np.broadcast_to(correction, (frames, 29)).copy(), np.zeros((frames, 3)), np.tile([1., 0., 0., 0.], (frames, 1))

        class Patcher:
            def __init__(self, lib):
                self.lib, self.fields = lib, {"dof_pos": lib.dof_pos}
            def _existing_inputs(self, start, end):
                return self.lib.dof_pos[start:end].clone(), np.zeros((end-start, 3)), np.tile([1., 0., 0., 0.], (end-start, 1))

        patch_calls = []
        def install(lib, motion_id, cursor, q, roots, quat, **kwargs):
            start = int(lib.length_starts[motion_id]) + cursor
            lib.dof_pos[start:start+len(q)] = torch.as_tensor(q)
            patch_calls.append((cursor, motion_id, kwargs["execution_steps"]))
            return {"cursor": cursor, "patched_frames": len(q), **kwargs}

        class Policy:
            def init_rollout(self):
                self.obs_dict_buffer, self.dones_buffer, self.steps = {}, None, 0
            def act_inference(self, obs_dict, cur_dones, **kwargs):
                self.obs_dict_buffer = {key: value[:, None] for key, value in obs_dict.items()}
                self.steps += 1
                if consume_cache:
                    return lib.dof_pos[int(cmd.time_steps[0]), 0].expand(1, 29).clone()
                return torch.zeros((1, 29))

        lib = SimpleNamespace(_num_motions=1, length_starts=torch.tensor([0]), dof_pos=torch.zeros((2000, 29)))
        cmd = SimpleNamespace(motion_lib=lib, motion_ids=torch.tensor([0]),
            time_steps=torch.tensor([0]), motion_start_time_steps=torch.tensor([0]))
        terminal_step = None

        class Env:
            num_envs, step_dt, device = 1, .02, "cpu"
            reference = {}
            gripper_target = torch.full((1, 2), .035)
            scene = {"robot": SimpleNamespace(joint_names=Model.joint_names + ["finger-left", "finger-right"])}
            scene = type("Scene", (dict,), {})(scene)
            scene.env_origins = torch.zeros((1, 3))
            command_manager = SimpleNamespace(get_term=lambda name: cmd)
            def __init__(self):
                self.observation_manager = SimpleNamespace(compute=lambda **kw: self.obs())
            def obs(self):
                return {"actor_obs": torch.zeros((1, 930)), "tokenizer": torch.zeros((1, 1761))}
            def reset_nominal(self):
                cmd.time_steps.zero_()
                return self.obs()
            def capture(self):
                step = int(cmd.time_steps[0])
                return {"root": np.array([[0., 0., .8, 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]),
                    "cube_root": np.array([[.28, -.03, .74, 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]),
                    "cube": np.array([[.28, -.03, .74]]), "tcp": np.zeros((1, 3)),
                    "q": np.zeros((1, 31)), "qdot": np.zeros((1, 31)),
                    "wrist_quat_wxyz": np.array([[1., 0., 0., 0.]]),
                    "gripper": self.gripper_target.numpy().copy(),
                    "terminated": np.array([terminal_step is not None and step >= terminal_step]),
                    "time": np.array([step / 50.])}

        env = Env()
        class Wrapper:
            def process_raw_obs(self, obs, *args):
                return obs
            def step(self, action):
                cmd.time_steps.add_(1)
                done = terminal_step is not None and int(cmd.time_steps[0]) >= terminal_step
                return env.obs(), None, torch.tensor([done]), None
        wrapper = Wrapper()
        wrapper.env = env
        def candidates(model, q, roots, quat, fingers, **kwargs):
            native = tuple(NativeCommand(q.copy(), roots.copy(), quat.copy(), fingers.copy()) for _ in range(9))
            return SimpleNamespace(native=native, actions=np.stack([encode_native(n) for n in native]),
                feasible=np.ones(9, bool), targets_tcp=np.zeros((9, 105, 3)),
                targets_quaternion_wxyz=np.tile([1., 0., 0., 0.], (9, 105, 1)),
                nominal_tcp=np.zeros((9, 105, 3)),
                nominal_quaternion_wxyz=np.tile([1., 0., 0., 0.], (9, 105, 1)),
                metadata={"effective_amplitude_m": .01, "effective_body_pitch_rad": .03, "common_scale": 1.})
        render_counter = 0
        def rendered(env, slots, **kwargs):
            nonlocal render_counter
            render_counter += 1
            elapsed = int(cmd.time_steps[0]) / 50.
            position = [.28+.001*elapsed+render_counter*1e-8, -.03, .716]
            estimates = {"cube": {"found": False, "confidence": 0., "position": None,
                "surface_position": None, "reason": "missing_color"},
                "tray": {"found": True, "confidence": .9, "position": position,
                "surface_position": np.asarray(position)+[0., 0., .006],
                "reason": "visible_top_approximation"}}
            return {slot: {"estimates": estimates} for slot in slots}
        modules = {
            "g1_embodiedskills.kinematics": SimpleNamespace(G1Kinematics=Model),
            "g1_embodiedskills.servo": SimpleNamespace(ReferenceServo=Servo),
            "g1_embodiedskills.native_reference": SimpleNamespace(NativeReferencePatcher=Patcher, install_reference_chunk=install),
            "g1_embodiedskills.perception": SimpleNamespace(render_observations=rendered),
            "g1_embodiedskills.simulation": SimpleNamespace(rebuild_physics=lambda env: None)}
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules, modules):
            root = Path(folder)
            reference = root / "reference"
            reference.mkdir()
            for name in ("motion.pkl", "reference.json", "asset.usd"):
                (reference / name).write_text("{}")
            np.savez(reference / "planned.npz", q=np.zeros((1300, 29)))
            (reference / "task.json").write_text(json.dumps({"geometry": {},
                "gripper_schedule": [{"time": 10., "joint_target_m": 0.}]}))
            config = self.batch()
            config["batches"][0]["companion_seeds"] = []
            config["batches"][0]["capsule_id"] = f"low-anchor{anchor_step}-batch0"
            config["batches"][0]["anchors"][0].update(anchor_step=anchor_step, noise_repeat=consume_cache)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            request = SimpleNamespace(cameras=True, seconds=24., m2_collection_config=config_path,
                task=None, reference=reference, sonic=root, asset=reference / "asset.usd", output=root / "output")
            outcome = lambda trace, **kwargs: {"success": False, "return_value": 0., "terminated": kwargs["terminated"]}
            result = m2.collect_factor_branches(wrapper, Policy(), request,
                candidate_factory=candidates, metrics_fn=outcome)
            expected_runs = 27 if consume_cache else 9
            self.assertEqual(result["analysis_branches"], expected_runs)
            manifest = json.loads(Path(result["manifest"]).read_text())
            group = manifest["groups"][0]
            self.assertEqual(len(group["runs"]), expected_runs)
            for row in group["runs"]:
                with np.load(Path(result["manifest"]).parent / row["samples_archive"]) as sample:
                    np.testing.assert_array_equal(sample["valid"], [True] * 4)
                    self.assertEqual(sample["nominal_command_step"][2], anchor_step)
            # Every default/nondefault pass installs exactly one 50-step pulse
            # and no replacement in its interior, including cell zero.
            self.assertEqual(sum(execution == 50 for _, _, execution in patch_calls), expected_runs)
            self.assertFalse(any(anchor_step < cursor < anchor_step+50 for cursor, _, _ in patch_calls))
            if consume_cache:
                base = Path(result["manifest"]).parent
                defaults = sorted((row for row in group["runs"] if row["candidate"] == 0),
                                  key=lambda row: row["repeat"])
                self.assertEqual(len(defaults), 3)
                baseline_commands = json.loads((base/defaults[0]["commands_json"]).read_text())
                baseline_resume = next(row for row in baseline_commands if row["step"] == anchor_step+50)
                self.assertGreater(np.linalg.norm(baseline_resume["servo_state"]["tray_offset"]), 0.)
                self.assertEqual(baseline_resume["servo_state"]["last_estimate_time"]["tray"], (anchor_step+50)/50.)
                with np.load(base/defaults[0]["controls_archive"]) as control:
                    baseline_control = control["actions"].copy()
                with np.load(base/defaults[0]["prefix_snapshot_archive"]) as snapshot:
                    baseline_cache = snapshot["cached_estimates_json"].copy()
                    self.assertEqual(int(snapshot["cached_estimates_present"]), 1)
                for row in defaults[1:]:
                    commands = json.loads((base/row["commands_json"]).read_text())
                    resumed = next(entry for entry in commands if entry["step"] == anchor_step+50)
                    self.assertEqual(resumed["servo_state"], baseline_resume["servo_state"])
                    self.assertEqual(resumed["cached_estimates"], baseline_resume["cached_estimates"])
                    with np.load(base/row["prefix_snapshot_archive"]) as snapshot:
                        np.testing.assert_array_equal(snapshot["cached_estimates_json"], baseline_cache)
                    with np.load(base/row["controls_archive"]) as control:
                        np.testing.assert_array_equal(control["actions"], baseline_control)
            # A physical baseline failure before the intervention keeps all
            # 27 noise-repeat rows and has no fabricated canonical grid.
            terminal_step = anchor_step - 10
            config["batches"][0]["anchors"][0]["noise_repeat"] = True
            config_path.write_text(json.dumps(config))
            request.output = root / "pre-anchor-termination"
            censored = m2.collect_factor_branches(wrapper, Policy(), request,
                candidate_factory=candidates, metrics_fn=outcome)
            censored_manifest = json.loads(Path(censored["manifest"]).read_text())
            dead_group = censored_manifest["groups"][0]
            self.assertFalse(dead_group["accepted"])
            self.assertTrue(dead_group["anchor_not_reached"])
            self.assertNotIn("canonical_archive", dead_group)
            self.assertEqual(len(dead_group["runs"]), 27)
            self.assertEqual(censored["analysis_branches"], 27)
            self.assertEqual(censored["physical_total_slot_rollouts"], 27)
            for row in dead_group["runs"]:
                self.assertFalse(row["valid"])
                self.assertFalse(row["anchor_observable"])
                self.assertIsNone(row["prefix_archive"])
                self.assertEqual(row["first_termination_step"], terminal_step)
                self.assertEqual(row["outcome_observation_end_step"], terminal_step)
                with np.load(Path(censored["manifest"]).parent / row["samples_archive"]) as sample:
                    self.assertFalse(sample["valid"].any())
                    self.assertTrue(np.isnan(sample["actual_tcp"]).all())


if __name__ == "__main__":
    unittest.main()
