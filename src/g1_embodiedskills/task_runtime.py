"""Task rollout diagnostics; observation-driven planner hooks build on this loop."""
import json
from pathlib import Path
import time
import subprocess
import numpy as np
import torch
from .io import write_json
from .simulation import tensor
from .metrics import episode_outcome


def camera_observation(env):
    from .perception import estimate_target
    env.sim.render()
    camera = env.scene["head_camera"]
    camera.update(0., force_recompute=True)
    data = camera.data
    rgb = tensor(data.output["rgb"])[0, ..., :3].cpu().numpy().copy()
    depth = tensor(data.output["distance_to_image_plane"])[0].cpu().numpy().copy()
    intrinsic = tensor(data.intrinsic_matrices)[0].cpu().numpy()
    position = tensor(data.pos_w)[0].cpu().numpy()
    quaternion = tensor(data.quat_w_ros)[0].cpu().numpy()[[3, 0, 1, 2]]
    estimates = {target: estimate_target(rgb, depth, intrinsic, position, quaternion, target=target)
                 for target in ["cube", "tray"]}
    return rgb, depth, estimates


def run_task(wrapper, policy, request):
    env = wrapper.env
    task_path = request.task or request.reference / "task.json"
    task = json.loads(Path(task_path).read_text(encoding="utf-8"))
    schedule = task["gripper_schedule"]
    cmd = env.command_manager.get_term("motion")
    lib = cmd.motion_lib
    original_fields = {name: value.clone() for name, value in vars(lib).items()
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == len(lib.dof_pos)}
    planned = dict(np.load(request.reference / "planned.npz"))
    torch.set_num_threads(1)
    episodes = []
    for repeat in range(request.repeats):
        for name, value in original_fields.items():
            getattr(lib, name).copy_(value)
        if request.hard_reset:
            from .simulation import rebuild_physics
            rebuild_physics(env)
        policy.init_rollout()
        env.episode_seed = request.seed_base + repeat
        obs = wrapper.process_raw_obs(env.reset_nominal(), True)
        reset_audit = {key: value.detach().cpu().numpy().copy() for key, value in obs.items()}
        records, images, estimates_log = [env.capture()], [], []
        servo_records = []
        estimates = None
        if request.servo:
            from .servo import ReferenceServo
            servo = ReferenceServo(request.sonic, planned, env.reference, task["geometry"])
            from .native_reference import install_reference_chunk
            if env.num_envs != 1:
                raise ValueError("Feedback pilot currently requires one environment")
        done = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        started = time.monotonic()
        for step in range(round(request.seconds / env.step_dt)):
            elapsed = step * env.step_dt
            if request.servo and step % 10 == 0:
                servo.observe_targets(estimates, elapsed)
                measured = env.capture()
                cursor = int(cmd.time_steps[0] + cmd.motion_start_time_steps[0])
                previous, correction, servo_row = servo.update(cursor,
                    measured["tcp"][0].cpu().numpy(), measured["wrist_quat_wxyz"][0].cpu().numpy())
                q, roots, quaternions = servo.trajectory(cursor, previous, correction)
                servo_row["patch"] = install_reference_chunk(lib, int(cmd.motion_ids[0]), cursor, q, roots, quaternions)
                raw = env.observation_manager.compute(update_history=False)
                obs = wrapper.process_raw_obs(raw, True)
                servo_records.append(servo_row)
            value = .035
            for event in schedule:
                if elapsed >= event["time"]:
                    value = event["joint_target_m"]
            env.gripper_target.fill_(value)
            with torch.no_grad():
                action = policy.act_inference(obs_dict=obs, cur_dones=done, skip_episode_attnmask=True)
                if step == 0:
                    reset_audit["first_action"] = action.detach().cpu().numpy().copy()
                obs, _, new_done, _ = wrapper.step(dict(actions=action))
            done |= new_done.reshape(-1).bool()
            records.append(env.capture())
            sensing_frame = request.record_video or 3.4 <= elapsed <= 6.6 or step == 0 or step == round(request.seconds/env.step_dt)-1
            if request.cameras and step % 10 == 0 and sensing_frame:
                rgb, depth, estimates = camera_observation(env)
                images.append(rgb)
                estimates_log.append(dict(time=elapsed+env.step_dt, **estimates))
                if len(images) == 1:
                    from PIL import Image
                    Image.fromarray(rgb).save(request.output / f"initial-{repeat:02d}.png")
                    np.savez_compressed(request.output / f"initial-camera-{repeat:02d}.npz", rgb=rgb, depth=depth)
            if step % 100 == 99:
                sample = records[-1]
                print(f"G1_TASK repeat={repeat} time={elapsed+env.step_dt:.2f} tcp={sample['tcp'][0].tolist()} cube={sample['cube'][0].tolist()} done={done.tolist()}", flush=True)
            if done.any():
                break
        fields = {key: torch.stack([record[key] for record in records]).cpu().numpy() for key in records[0]}
        np.savez_compressed(request.output / f"trace-{repeat:02d}.npz", **fields)
        np.savez_compressed(request.output / f"reset-audit-{repeat:02d}.npz", **reset_audit)
        if request.cameras:
            write_json(request.output / f"perception-{repeat:02d}.json", estimates_log)
            if images and request.record_video:
                encode_video(request.output / f"head-{repeat:02d}.mp4", images, fps=5)
        if request.servo:
            write_json(request.output / f"servo-{repeat:02d}.json", servo_records)
        for slot in range(env.num_envs):
            outcome = episode_outcome(fields, slot=slot, origin=tensor(env.scene.env_origins)[slot].cpu().numpy(),
                terminated=bool(done[slot]), control_dt=env.step_dt)
            row = dict(repeat=repeat, slot=slot, seed=env.episode_seed, **outcome,
                duration=float(fields["time"][-1, slot]), wall_seconds=time.monotonic()-started,
                controller="frozen_sonic", planner="reference_feedback" if request.servo else "reference_diagnostic")
            episodes.append(row)
        write_json(request.output / "episodes.json", episodes)
    return dict(completed=True, mode=request.mode, episodes=episodes,
        success_rate=sum(row["success"] for row in episodes)/len(episodes),
        protocol="reference_feedback" if request.servo else "reference_diagnostic; vision recorded but not yet used for control")


def encode_video(path, frames, fps):
    height, width = frames[0].shape[:2]
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
        "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "-", "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(path)]
    process = subprocess.run(command, input=np.stack(frames).astype(np.uint8).tobytes(), capture_output=True)
    if process.returncode:
        raise RuntimeError("Video encoding failed: " + process.stderr.decode("utf-8"))
