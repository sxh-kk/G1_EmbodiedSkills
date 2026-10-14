"""Patch an already-loaded SONIC G1 motion slot with a short native reference.

Inputs use 29-DOF MJCF hinge order, metre/radian units and root WXYZ. Root
positions are in the motion library's reference frame, before command/env-origin
offsets. The 31-DOF simulator's two extra gripper joints are never touched.

The official FK computes body linear velocity with np.gradient and a sigma=2,
truncate=4 Gaussian (8-frame radius); angular velocity uses forward quaternion
differences and the same filter. Nine context frames cover the gradient plus
filter. Existing frames before the cursor remain bitwise unchanged, including
their velocity targets: causal prefix preservation intentionally differs from
retroactively reloading a whole edited motion's pre-cursor velocities.
"""

from __future__ import annotations

import math
import time

import torch


class NativeReferencePatcher:
    """Reuse ``motion_lib.mesh_parsers``; no MotionLib construction or pickle IO."""

    FILTER_CONTEXT = 9
    BODY_FIELDS = (
        "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w",
        "body_pos_w_full", "body_quat_w_full", "body_lin_vel_w_full", "body_ang_vel_w_full",
    )

    def __init__(self, motion_lib):
        self.lib = motion_lib
        self.fk = motion_lib.mesh_parsers
        cfg = motion_lib.m_cfg
        if self.fk is None or self.fk.num_dof != 29 or len(self.fk.body_names) != 30:
            raise ValueError("Native patching requires the official 29-DOF/30-body G1 FK")
        if len(cfg.get("extend_config", [])):
            raise ValueError("Extended-body references are not supported by this G1 adapter")
        # ``smpl_motion_file=dummy`` produces a list of None plus zero feature
        # tensors so inactive observation terms can still run. These are valid
        # for a G1-only encoder; inspect per-slot availability below instead of
        # rejecting the existence of a placeholder container.
        if getattr(motion_lib, "has_action", False):
            raise ValueError("Action-labelled MotionLib slots require a separate action adapter")
        if float(motion_lib.target_fps) != 50 or float(motion_lib.motion_fps_scale) != 1:
            raise ValueError("Native G1 patching requires unscaled 50-Hz references")
        if not math.isclose(float(motion_lib._sim_fps), 50, abs_tol=1e-5):
            raise ValueError("time_step must address 50-Hz reference frames")
        self.axes = self.fk.dof_axis.detach().cpu().to(torch.float32)
        if self.axes.shape != (29, 3) or not torch.allclose(self.axes.sum(-1), torch.ones(29)):
            raise ValueError("Expected G1 positive unit hinge axes in MJCF traversal order")
        self.mapped = "mujoco_to_isaaclab_body" in cfg
        self.body_order = torch.as_tensor(cfg.mujoco_to_isaaclab_body if self.mapped else list(range(30)), dtype=torch.long)
        self.dof_order = torch.as_tensor(cfg.mujoco_to_isaaclab_dof if self.mapped else list(range(29)), dtype=torch.long)
        if sorted(self.body_order.tolist()) != list(range(30)) or sorted(self.dof_order.tolist()) != list(range(29)):
            raise ValueError("Invalid MJCF-to-IsaacLab G1 permutations")
        self.inverse_dof_order = self.dof_order.argsort()
        self.body_indexes = list(motion_lib.body_indexes) if self.mapped else list(range(30))
        if not self.body_indexes or self.body_indexes[0] != 0 or int(self.body_order[0]) != 0:
            raise ValueError("The motion anchor must remain the pelvis at index 0")
        required = (*self.BODY_FIELDS, "body_pos_b", "root_linv_vel_w", "root_ang_vel_w", "dof_pos", "dof_vel", "feet_l", "feet_r")
        self.fields = {name: getattr(motion_lib, name) for name in required}
        if self.fields["dof_pos"].shape[1:] != (29,):
            raise ValueError("Loaded reference DOFs must contain only the G1 body")

    def _existing_inputs(self, start: int, end: int):
        q = self.lib.dof_pos[start:end].detach().cpu()[:, self.inverse_dof_order].clone()
        root = self.lib.body_pos_w_full[start:end, 0].detach().cpu().clone()
        quat = self.lib.body_quat_w_full[start:end, 0].detach().cpu().clone()
        if not self.mapped:
            quat = quat[:, [3, 0, 1, 2]]  # Unmapped loader retains FK XYZW.
        return q, root, quat

    @torch.no_grad()
    def patch(self, motion_id, time_step, qtraj_mjcf, roottraj, rootquat_wxyz, *, execution_steps=10, preview_frames=45, synchronize=True) -> dict:
        """Install a candidate from the current *absolute motion-frame* cursor.

        ``qtraj_mjcf[0]`` corresponds to ``time_step``. Supply at least
        execution_steps + preview_frames + 1 + 9 frames (65 at defaults), so
        all controller queries during execution have their own future filter
        context. Root quaternion may be [4] or [T,4]. No hold padding is invented.
        Pass ``cmd.time_steps + cmd.motion_start_time_steps``, not episode time
        alone. Pause stepping while patching; refresh the command's cached
        reference observations after installing, without advancing its cursor.
        """
        started = time.perf_counter()
        motion_id, cursor = int(motion_id), int(time_step)
        if execution_steps < 1 or preview_frames < 0:
            raise ValueError("execution_steps must be positive; preview_frames nonnegative")
        if not 0 <= motion_id < int(self.lib._num_motions):
            raise ValueError("motion_id is not a loaded MotionLib slot")
        for modality in ("smpl", "soma", "object"):
            availability = getattr(self.lib, f"motion_has_{modality}", None)
            if availability is not None and bool(availability[motion_id]):
                raise ValueError(f"Selected slot has active {modality} motion; body-only patching would leave it stale")
        if not math.isclose(float(self.lib._motion_fps[motion_id]), 50, abs_tol=1e-5):
            raise ValueError("Selected motion is not sampled at 50 Hz")
        q = torch.as_tensor(qtraj_mjcf, dtype=torch.float32, device="cpu").detach().clone()
        root = torch.as_tensor(roottraj, dtype=torch.float32, device="cpu").detach().clone()
        quat = torch.as_tensor(rootquat_wxyz, dtype=torch.float32, device="cpu").detach().clone()
        if q.ndim != 2 or q.shape[1] != 29 or root.shape != (len(q), 3):
            raise ValueError("Expected qtraj_mjcf[T,29] and roottraj[T,3]")
        minimum = execution_steps + preview_frames + 1 + self.FILTER_CONTEXT
        if len(q) < minimum:
            raise ValueError(f"Need at least {minimum} reference frames including execution, preview and filter context")
        if quat.shape == (4,):
            quat = quat.expand(len(q), 4).clone()
        if quat.shape != (len(q), 4) or any(not bool(value.isfinite().all()) for value in (q, root, quat)):
            raise ValueError("Reference inputs must be finite; root quaternion must be WXYZ [4] or [T,4]")
        norms = quat.norm(dim=-1, keepdim=True)
        if bool((norms < 1e-8).any()):
            raise ValueError("Root quaternions cannot have zero norm")
        quat /= norms
        frames = int(self.lib._motion_num_frames[motion_id])
        if cursor < 0 or cursor + len(q) > frames:
            raise ValueError("Candidate exceeds preallocated motion slot; allocate a sufficiently long reference before rollout")
        slot_start = int(self.lib.length_starts[motion_id])
        patch_start, patch_end = slot_start + cursor, slot_start + cursor + len(q)
        context_start = max(slot_start, patch_start - self.FILTER_CONTEXT)
        context_end = min(slot_start + frames, patch_end + self.FILTER_CONTEXT)
        context_q, context_root, context_quat = self._existing_inputs(context_start, context_end)
        local_start = patch_start - context_start
        local_end = local_start + len(q)
        context_q[local_start:local_end] = q
        context_root[local_start:local_end] = root
        context_quat[local_start:local_end] = quat

        from gear_sonic.trl.utils.torch_transform import quaternion_to_angle_axis
        pose = torch.zeros((len(context_q), 30, 3), dtype=torch.float32)
        pose[:, 0] = quaternion_to_angle_axis(context_quat)
        pose[:, 1:] = context_q[:, :, None] * self.axes
        fk_start = time.perf_counter()
        fk = self.fk.fk_batch(pose[None], context_root[None], return_full=True, fps=50, target_fps=50, interpolate_data=False, use_parallel_fk=self.lib.use_parallel_fk)
        fk_seconds = time.perf_counter() - fk_start
        native = {key: value[0] for key, value in fk.items() if isinstance(value, torch.Tensor)}
        feet_l, feet_r = self.lib.foot_detect(native["global_translation"], 0.0005, 0.05)
        full = {
            "body_pos_w_full": native["global_translation"][:, self.body_order],
            "body_quat_w_full": native["global_rotation"][:, self.body_order],
            "body_lin_vel_w_full": native["global_velocity"][:, self.body_order],
            "body_ang_vel_w_full": native["global_angular_velocity"][:, self.body_order],
        }
        if self.mapped:
            full["body_quat_w_full"] = full["body_quat_w_full"][:, :, [3, 0, 1, 2]]
        values = {
            **full,
            **{name.removesuffix("_full"): value[:, self.body_indexes] for name, value in full.items()},
            "body_pos_b": native["local_rotation"],
            "root_linv_vel_w": native["global_root_velocity"],
            "root_ang_vel_w": native["global_root_angular_velocity"],
            "dof_pos": native["dof_pos"][:, self.dof_order],
            "dof_vel": native["dof_vels"][:, self.dof_order],
            "feet_l": feet_l, "feet_r": feet_r,
        }
        # Validate every field before the first mutation; no partial patch on shape errors.
        prepared = {}
        for name, target in self.fields.items():
            value = values[name][local_start:local_end]
            if value.shape != target[patch_start:patch_end].shape or not bool(value.isfinite().all()):
                raise ValueError(f"Native reference field contract mismatch: {name}")
            prepared[name] = value.to(device=target.device, dtype=target.dtype)
        for name, value in prepared.items():
            self.fields[name][patch_start:patch_end].copy_(value)
        if synchronize and self.lib.dof_pos.is_cuda:
            torch.cuda.synchronize(self.lib.dof_pos.device)
        return {
            "motion_id": motion_id, "cursor": cursor, "patched_frames": len(q),
            "context_frames": len(context_q), "left_context_frames": local_start,
            "right_context_frames": context_end - patch_end,
            "last_execution_preview_frame": cursor + execution_steps + preview_frames,
            "reference_fps": 50, "fk_seconds": fk_seconds,
            "total_seconds": time.perf_counter() - started,
            "prefix_preserved": True, "gripper_fields_modified": False,
            "auxiliary_encoder_fields_modified": False,
        }


def install_reference_chunk(motion_lib, motion_id, time_step, qtraj_mjcf, roottraj, rootquat_wxyz, **kwargs):
    """Convenience wrapper; retains one lightweight patcher on the loaded library."""
    patcher = getattr(motion_lib, "_g1_native_reference_patcher", None)
    if patcher is None:
        patcher = NativeReferencePatcher(motion_lib)
        motion_lib._g1_native_reference_patcher = patcher
    return patcher.patch(motion_id, time_step, qtraj_mjcf, roottraj, rootquat_wxyz, **kwargs)
