"""Isaac Lab integration, imported only after the simulator application starts."""
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.sim as sim_utils
from gear_sonic.envs.manager_env.mdp.commands import TrackingCommand
from gear_sonic.envs.env_utils.joint_utils import G1_ISAACLab_ORDER, get_body_joint_indices
from gear_sonic.envs.wrapper.manager_env_wrapper import ManagerEnvWrapper
from gear_sonic.trl.utils.common import custom_instantiate
from gear_sonic.utils import lab3_wxyz

from .assets import GRIPPER_JOINTS, TCP_OFFSET
from .io import write_json, sha256, RunFinished


def tensor(value):
    return value.torch if hasattr(value, "torch") else value


def body_position(env):
    robot = env.scene["robot"]
    ids = get_body_joint_indices(robot)
    return robot.data.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids]


def body_velocity(env):
    robot = env.scene["robot"]
    ids = get_body_joint_indices(robot)
    return robot.data.joint_vel[:, ids] - robot.data.default_joint_vel[:, ids]


class FixedTrackingCommand(TrackingCommand):
    def _resample_command(self, env_ids):
        self.is_evaluating = True
        super()._resample_command(env_ids)

    def _update_command(self):
        if torch.any(self.time_steps + self.motion_start_time_steps + 1 >= self.motion_lib.get_time_step_total(self.motion_ids)):
            raise RuntimeError("Reference exhausted; implicit motion switching is disabled")
        super()._update_command()


def create_env(config, device, args_cli, request):
    with open_dict(config):
        config.manager_env.events = {"_target_": config.manager_env.events._target_}
        config.manager_env.curriculum = {"_target_": config.manager_env.curriculum._target_}
        config.trainer.schedule_dict = {}
        config.manager_env.config.episode_length_s = request.seconds + 1.
        config.manager_env.config.render_results = False
        config.manager_env.config.terrain_type = "plane"
        config.manager_env.config.experiment_dir = str(request.output)
        for group in config.manager_env.observations.values():
            if OmegaConf.is_dict(group):
                group.enable_corruption = False
        motion = config.manager_env.commands.motion
        motion.encoder_sample_probs = dict(g1=1., teleop=0., smpl=0.)
        motion.start_from_first_frame = True
        motion.randomize_heading = False
        motion.randomize_wrist_poses = False
        motion.variable_frames_enabled = False
        motion.freeze_frame_aug = False
        motion.cat_upper_body_poses = False
        motion.pose_range = {k: [0., 0.] for k in ["x", "y", "z", "roll", "pitch", "yaw"]}
        motion.velocity_range = {k: [0., 0.] for k in ["x", "y", "z", "roll", "pitch", "yaw"]}
        motion.joint_position_range = [0., 0.]
        motion.joint_velocity_range = [0., 0.]
        motion.motion_lib_num_dof = 29
        motion.hand_default_positions = [.035, .035]
        motion.hand_default_velocities = [0., 0.]
        motion.motion_lib_cfg.adaptive_sampling.enable = False
        motion.motion_lib_cfg.multi_thread = False
        motion.motion_lib_cfg.upper_body_augment_prefixes = []
        motion.debug_vis = False
        for name in list(config.manager_env.commands):
            if name != "motion" and not name.startswith("_"):
                del config.manager_env.commands[name]
    cfg = custom_instantiate(config.manager_env)
    cfg.seed = config.seed
    cfg.sim.device = device
    cfg.config["headless"] = True
    cfg.commands.motion.class_type = FixedTrackingCommand
    cfg.scene.robot.spawn.asset_path = str(request.asset)
    cfg.scene.robot.spawn.force_usd_conversion = True
    cfg.scene.robot.spawn.usd_dir = str(request.output / "converted_asset")
    cfg.scene.robot.init_state.joint_pos.update({name: .035 for name in GRIPPER_JOINTS})
    cfg.scene.robot.actuators["parallel_gripper"] = ImplicitActuatorCfg(
        joint_names_expr=GRIPPER_JOINTS, effort_limit_sim=20., velocity_limit_sim=.15,
        stiffness=1200., damping=15., armature=.001)
    cfg.actions.joint_pos.joint_names = list(G1_ISAACLab_ORDER)
    cfg.actions.joint_pos.preserve_order = True
    for group in [cfg.observations.policy, cfg.observations.critic]:
        group.joint_pos.func, group.joint_pos.params = body_position, {}
        group.joint_vel.func, group.joint_vel.params = body_velocity, {}
    # The generic sensor is unused; one rigid body avoids nested-contact ambiguity.
    cfg.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/Geometry/pelvis"
    cfg.scene.table = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(size=(.55, .65, .04), collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.42, .34, .24))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(.5, -.12, .69)))
    cfg.scene.cube = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(size=(.035, .035, .035),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.),
            mass_props=sim_utils.MassPropertiesCfg(mass=.025),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.001, rest_offset=0.),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.5, dynamic_friction=1.3, restitution=0.),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.9, .025, .025))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(.28, -.23, .728)))
    cfg.scene.tray = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Tray",
        spawn=sim_utils.CuboidCfg(size=(.16, .16, .012), collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.025, .65, .08))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(.28, -.03, .716)))
    if request.cameras:
        from .perception import make_head_camera_cfg
        cfg.scene.head_camera = make_head_camera_cfg()
    env = MigrationEnv(cfg=cfg)
    env.request = request
    env.reference = json.loads((request.reference / "reference.json").read_text(encoding="utf-8"))
    robot = env.scene["robot"]
    env.body_ids = get_body_joint_indices(robot)
    env.gripper_ids = robot.find_joints(GRIPPER_JOINTS, preserve_order=True)[0]
    assert len(env.body_ids) == 29 and len(env.gripper_ids) == 2
    assert [robot.joint_names[i] for i in env.body_ids.tolist()] == G1_ISAACLab_ORDER
    env.gripper_target = torch.full((env.num_envs, 2), .035, device=device)
    env.step_records = []
    env.physics_elapsed = 0.
    import omni.physx
    def physical_step(dt):
        env.physics_elapsed += float(dt)
    env.physics_subscription = omni.physx.get_physx_interface().subscribe_physics_step_events(physical_step)
    write_json(request.output / "interface.json", dict(
        body_joint_names=[robot.joint_names[i] for i in env.body_ids.tolist()],
        gripper_joint_names=[robot.joint_names[i] for i in env.gripper_ids],
        all_joint_names=robot.joint_names, body_names=robot.body_names,
        physics_dt=env.physics_dt, control_dt=env.step_dt, floating_base=True,
        action_dim=env.action_manager.total_action_dim,
        observation_shapes={k: str(v) for k, v in env.observation_space.items()},
        reference_sha256=sha256(request.reference / "motion.pkl"), asset_sha256=sha256(request.asset)))
    return ManagerEnvWrapper(env, cfg.config)


class MigrationEnv(ManagerBasedRLEnv):
    """Keep terminal observations intact and let the caller own episode resets."""
    def step(self, action):
        self.action_manager.process_action(action.to(self.device))
        assert not self._physics_handles_decimation
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            if hasattr(self, "gripper_target"):
                self.scene["robot"].set_joint_position_target(self.gripper_target, joint_ids=self.gripper_ids)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated.clone()
        self.reset_time_outs = self.termination_manager.time_outs.clone()
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        self.command_manager.compute(dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute(update_history=True)
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def reset_nominal(self):
        self.reset()
        cmd = self.command_manager.get_term("motion")
        dtype = cmd.motion_num_steps.dtype
        cmd.set_motion_state(torch.arange(self.num_envs, device=self.device) % cmd.motion_lib._num_motions,
            torch.zeros(self.num_envs, dtype=torch.long, device=self.device))
        cmd.motion_num_steps = cmd.motion_num_steps.to(dtype)
        robot = self.scene["robot"]
        ref = self.reference
        q = torch.zeros_like(tensor(robot.data.joint_pos))
        for i, name in enumerate(robot.joint_names):
            q[:, i] = ref["q"][ref["joint_names_mjcf"].index(name)] if name in ref["joint_names_mjcf"] else .035
        state = torch.zeros((self.num_envs, 13), device=self.device)
        state[:, :3] = state.new_tensor(ref["root_position"]) + tensor(self.scene.env_origins)
        state[:, 3:7] = state.new_tensor(ref["root_quaternion_wxyz"])
        ids = torch.arange(self.num_envs, device=self.device)
        robot.reset(ids)
        robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
        lab3_wxyz.write_root_state(robot, state, env_ids=ids)
        robot.set_joint_position_target(q, env_ids=ids)
        robot.set_joint_velocity_target(torch.zeros_like(q), env_ids=ids)
        robot.set_joint_effort_target(torch.zeros_like(q), env_ids=ids)
        cube = self.scene["cube"]
        cube_state = tensor(cube.data.default_root_state).clone()
        cube_state[:, :3] += tensor(self.scene.env_origins)
        jitter = getattr(self.request, "position_jitter", 0.)
        if jitter:
            seeds = getattr(self, "episode_seeds", [getattr(self, "episode_seed", 0) + i for i in range(self.num_envs)])
            offsets = np.stack([np.random.default_rng(seed).uniform(-jitter, jitter, 2) for seed in seeds])
            cube_state[:, :2] += cube_state.new_tensor(offsets)
        cube.reset(ids)
        cube.write_root_state_to_sim(cube_state, env_ids=ids)
        for name in ["permanent_wrench_composer", "instantaneous_wrench_composer"]:
            getattr(robot, name).reset(env_ids=ids)
        self.gripper_target.fill_(.035)
        self.action_manager.reset(ids)
        self.observation_manager.reset(ids)
        self.termination_manager.reset(ids)
        for name in ["_term_dones", "_last_episode_dones", "_truncated_buf", "_terminated_buf"]:
            getattr(self.termination_manager, name)[ids] = False
        self.episode_length_buf.zero_()
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.reset_terminated = torch.zeros_like(self.reset_buf)
        self.reset_time_outs = torch.zeros_like(self.reset_buf)
        cmd.running_ref_root_height[:] = cmd.anchor_pos_w[:, 2]
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=0.)
        for name, value in vars(robot.data).items():
            if name.startswith("_root_") and name.endswith("_b") and hasattr(value, "timestamp"):
                value.timestamp = -1.
        self.obs_buf = self.observation_manager.compute(update_history=True)
        self.episode_start_time = self.physics_elapsed
        return self.obs_buf

    def capture(self):
        robot = self.scene["robot"]
        d = robot.data
        root = torch.cat([tensor(d.root_pos_w), lab3_wxyz.from_lab(d.root_quat_w),
            tensor(d.root_lin_vel_w), tensor(d.root_ang_vel_w)], -1)
        index = robot.body_names.index("right_wrist_yaw_link")
        quat = lab3_wxyz.from_lab(d.body_quat_w)[:, index]
        pos = tensor(d.body_pos_w)[:, index]
        tcp = pos + lab3_wxyz.quat_apply(quat, pos.new_tensor(TCP_OFFSET).expand_as(pos))
        command = self.command_manager.get_term("motion")
        reference_index = command.cfg.body_names.index("right_wrist_yaw_link")
        reference_quat = command.body_quat_w[:, reference_index]
        reference_tcp = command.body_pos_w[:, reference_index] + lab3_wxyz.quat_apply(
            reference_quat, pos.new_tensor(TCP_OFFSET).expand_as(pos))
        return {"root": root.detach().clone(), "q": tensor(d.joint_pos).detach().clone(),
            "qdot": tensor(d.joint_vel).detach().clone(), "tcp": tcp.detach().clone(),
            "wrist_quat_wxyz": quat.detach().clone(),
            "reference_tcp": reference_tcp.detach().clone(),
            "cube": tensor(self.scene["cube"].data.root_pos_w).detach().clone(),
            "gripper": tensor(d.joint_pos)[:, self.gripper_ids].detach().clone(),
            "time": torch.full((self.num_envs,), self.physics_elapsed - self.episode_start_time, device=self.device)}


class SimulationCallback:
    def on_step_end(self, args, state, control, **kwargs):
        wrapper, model = kwargs["env"], kwargs["model"]
        env, policy = wrapper.env, model.policy
        request = env.request
        wrapper.set_is_evaluating(True)
        model.eval()
        if hasattr(policy, "eval_mode"):
            policy.eval_mode()
        env.command_manager.get_term("motion").motion_lib.load_motions(
            random_sample=False, num_motions_to_load=env.num_envs, is_evaluation=True)
        if request.mode != "interface":
            if request.mode == "collect":
                from .collection import collect_branches
                result = collect_branches(wrapper, policy, request)
            elif request.mode == "closed_loop":
                from .online import run_closed_loop
                result = run_closed_loop(wrapper, policy, request)
            else:
                from .task_runtime import run_task
                result = run_task(wrapper, policy, request)
            write_json(request.output / "result.json", result)
            raise RunFinished(0 if result["completed"] else 2)
        episodes = []
        started = time.monotonic()
        for repeat in range(request.repeats):
            policy.init_rollout()
            obs = wrapper.process_raw_obs(env.reset_nominal(), True)
            if request.cameras:
                from .task_runtime import camera_observation
                from PIL import Image
                rgb, depth, estimates = camera_observation(env)
                Image.fromarray(rgb).save(request.output / f"camera-{repeat:02d}.png")
                write_json(request.output / f"camera-{repeat:02d}.json", estimates)
            cmd = env.command_manager.get_term("motion")
            write_json(request.output / "preview.json", dict(
                offsets_seconds=cmd.future_time_steps_init[0] / cmd.motion_lib.target_fps,
                policy_history_shape=list(obs["actor_obs"].shape), tokenizer_shape=list(obs["tokenizer"].shape)))
            records = [env.capture()]
            dones = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            for step in range(round(request.seconds / env.step_dt)):
                # Close and reopen while the complete body remains controlled by SONIC.
                elapsed = step * env.step_dt
                env.gripper_target.fill_(0. if 2. <= elapsed < 4. else .035)
                with torch.no_grad():
                    action = policy.act_inference(obs_dict=obs, cur_dones=dones, skip_episode_attnmask=True)
                    obs, _, new_done, _ = wrapper.step(dict(actions=action))
                dones |= new_done.reshape(-1).bool()
                records.append(env.capture())
                if step % 100 == 99:
                    print(f"G1_INTERFACE repeat={repeat} step={step+1} done={dones.tolist()}", flush=True)
                if dones.any():
                    break
            fields = {key: torch.stack([r[key] for r in records]).cpu().numpy() for key in records[0]}
            np.savez_compressed(request.output / f"trace-{repeat:02d}.npz", **fields)
            for slot in range(env.num_envs):
                root = fields["root"][:, slot]
                xy_drift = np.linalg.norm(root[:, :2] - root[:1, :2], axis=-1).max()
                tilt = 2*np.arccos(np.clip(np.sqrt(root[:, 3]**2 + root[:, 6]**2), 0, 1))
                gripper = fields["gripper"][:, slot]
                closed = gripper[(fields["time"][:, slot] > 3) & (fields["time"][:, slot] < 4)]
                reopened = gripper[fields["time"][:, slot] > 5]
                row = dict(repeat=repeat, slot=slot, steps=len(records)-1,
                    duration=float(fields["time"][-1, slot]), terminated=bool(dones[slot]),
                    minimum_root_height=float(root[:, 2].min()), maximum_tilt_rad=float(tilt.max()),
                    maximum_xy_drift_m=float(xy_drift),
                    mean_tcp_reference_error_m=float(np.linalg.norm(fields["tcp"][:, slot]-fields["reference_tcp"][:, slot], axis=-1).mean()),
                    maximum_tcp_reference_error_m=float(np.linalg.norm(fields["tcp"][:, slot]-fields["reference_tcp"][:, slot], axis=-1).max()),
                    closed_joint_max_m=float(closed.max()) if closed.size else None,
                    reopened_joint_min_m=float(reopened.min()) if reopened.size else None)
                row["standing_pass"] = bool(not row["terminated"] and row["minimum_root_height"] > .5 and row["maximum_tilt_rad"] < .6)
                row["gripper_pass"] = bool(closed.size and reopened.size and closed.max() < .004 and reopened.min() > .03)
                episodes.append(row)
            write_json(request.output / "episodes.json", episodes)
        result = dict(completed=True, mode="interface", episodes=episodes,
            all_passed=all(r["standing_pass"] and r["gripper_pass"] for r in episodes),
            wall_seconds=time.monotonic()-started)
        write_json(request.output / "result.json", result)
        raise RunFinished(0 if result["all_passed"] else 2)
