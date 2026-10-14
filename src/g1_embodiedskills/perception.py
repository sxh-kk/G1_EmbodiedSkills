"""Small RGB-D perception baseline for the first tabletop task.

Only camera observations, calibration and measured camera pose enter this module.
The colors and dimensions are task priors, not object transforms from the scene.
The result is a visible-top-surface center approximation for an upright object;
it is deliberately not a six-DoF pose estimator or an occlusion completion model.
"""
from __future__ import annotations

import math
import numpy as np


def make_head_camera_cfg(robot_prim="{ENV_REGEX_NS}/Robot", *,
                         mount="Geometry/torso_link", width=256, height=256):
    """Create the local Lab 3 camera config after the simulator has started.

    The head-height mount looks forward/right/down at the nearby tabletop.
    Its initial field of view must be checked against actual rendered frames.
    Lab 3 uses XYZW quaternions in OffsetCfg and CameraData, unlike this module's
    explicit WXYZ estimate_target input. Update the world pose each frame because
    the torso is floating. RGB values may depend on renderer tonemapping, so the
    color thresholds below use channel dominance rather than exact RGB equality.
    """
    from isaaclab.sensors import CameraCfg
    from isaaclab.sim import PinholeCameraCfg

    # World-camera convention: +X forward, +Z up. Pitch > 0 looks downward.
    yaw, pitch = -.35, 1.0
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    quaternion_xyzw = (-sy * sp, cy * sp, sy * cp, cy * cp)
    return CameraCfg(
        prim_path=f"{robot_prim.rstrip('/')}/{mount.strip('/')}/HeadCamera",
        update_period=0., width=width, height=height,
        data_types=["rgb", "distance_to_image_plane"],
        update_latest_camera_pose=True,
        spawn=PinholeCameraCfg(focal_length=16., horizontal_aperture=24.,
                               clipping_range=(.01, 3.0)),
        offset=CameraCfg.OffsetCfg(pos=(.08, 0., .32), rot=quaternion_xyzw,
                                    convention="world"),
    )


def _numpy(value):
    """Accept NumPy, torch tensors and Isaac Lab ProxyArray without torch import."""
    if hasattr(value, "torch"):
        value = value.torch
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def quaternion_wxyz_rotation(quaternion):
    """Return camera-to-world rotation for a normalized WXYZ quaternion."""
    q = _numpy(quaternion).astype(float).reshape(-1)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError("camera_quat_wxyz_ros must be a finite nonzero WXYZ quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def _largest_component(mask):
    """Remove isolated color noise using four-connected components."""
    unseen = mask.copy()
    height, width = mask.shape
    best = []
    for row, col in zip(*np.nonzero(mask)):
        if not unseen[row, col]:
            continue
        unseen[row, col] = False
        pending, component = [(int(row), int(col))], []
        while pending:
            y, x = pending.pop()
            component.append((y, x))
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and unseen[ny, nx]:
                    unseen[ny, nx] = False
                    pending.append((ny, nx))
        if len(component) > len(best):
            best = component
    result = np.zeros_like(mask)
    if best:
        ys, xs = np.asarray(best).T
        result[ys, xs] = True
    return result


def color_mask(rgb, target="cube"):
    """Segment the red cube or green tray in uint8 or [0,1] RGB/RGBA."""
    image = _numpy(rgb)
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError("rgb must have unbatched shape [H,W,3] or [H,W,4]")
    if not np.isfinite(image).all():
        raise ValueError("rgb contains non-finite values")
    image = image[..., :3].astype(float)
    if np.max(image, initial=0.) > 1.:
        image /= 255.
    if np.min(image, initial=0.) < 0. or np.max(image, initial=0.) > 1.:
        raise ValueError("rgb must be uint8-like [0,255] or normalized [0,1]")
    red, green, blue = np.moveaxis(image, -1, 0)
    if target == "cube":
        mask = (red > .12) & (red > 1.6 * green) & (red > 1.6 * blue) & (red - green > .07)
    elif target == "tray":
        mask = (green > .12) & (green > 1.5 * red) & (green > 1.35 * blue) & (green - red > .07)
    else:
        raise ValueError("target must be 'cube' or 'tray'")
    return _largest_component(mask)


def estimate_target(rgb, depth, K, camera_pos_w, camera_quat_wxyz_ros, *,
                    target="cube", half_height=None, min_pixels=12,
                    depth_range=(.02, 3.0)):
    """Estimate an upright object's center from segmented top-surface RGB-D.

    ``depth`` must be distance_to_image_plane in meters, not Euclidean ray range.
    ``K`` is the *actual* camera intrinsic matrix. ROS axes are +X right, +Y down,
    +Z forward. Pass ``camera.data.quat_w_ros.torch[i][[3,0,1,2]]`` for Lab 3.
    All arguments are unbatched. A missing object returns position=None; there is
    no fallback to privileged object pose. Confidence is a heuristic quality
    indicator, not a calibrated probability, and cannot detect hidden-face bias.

    Defaults use the task's 35 mm cube and 12 mm tray. The point is the median of
    the visible top band, shifted down by the known half-height. Severe partial
    occlusion can bias XY; callers should retain an earlier valid observation or
    stop/reobserve, rather than interpreting this as a full shape reconstruction.
    """
    if min_pixels < 1:
        raise ValueError("min_pixels must be positive")
    mask = color_mask(rgb, target)
    depth = _numpy(depth).astype(float)
    if depth.shape == (*mask.shape, 1):
        depth = depth[..., 0]
    if depth.shape != mask.shape:
        raise ValueError("depth must have shape [H,W] or [H,W,1] matching rgb")
    intrinsic = _numpy(K).astype(float)
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError("K must be a finite 3x3 intrinsic matrix")
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0 or abs(np.linalg.det(intrinsic)) < 1e-10:
        raise ValueError("K must be an invertible pinhole intrinsic matrix")
    camera_pos = _numpy(camera_pos_w).astype(float)
    if camera_pos.shape != (3,) or not np.isfinite(camera_pos).all():
        raise ValueError("camera_pos_w must have shape [3] and finite values")
    rotation = quaternion_wxyz_rotation(camera_quat_wxyz_ros)
    if half_height is None:
        half_height = .0175 if target == "cube" else .006
    if half_height < 0 or not np.isfinite(half_height):
        raise ValueError("half_height must be finite and nonnegative")
    valid = mask & np.isfinite(depth) & (depth > depth_range[0]) & (depth < depth_range[1])
    result = dict(target=target, found=False, position=None, surface_position=None,
                  mask_count=int(mask.sum()), valid_depth_count=int(valid.sum()),
                  confidence=0., reason="missing_color", pixel_centroid=None,
                  method="color_rgbd_visible_top_center")
    if result["mask_count"] < min_pixels:
        result["reason"] = "missing_color" if not mask.any() else "insufficient_color_pixels"
        return result
    if result["valid_depth_count"] < min_pixels:
        result["reason"] = "insufficient_valid_depth"
        return result
    ys, xs = np.nonzero(valid)
    # Integer pixel coordinates match Isaac Lab's intrinsic-matrix projection.
    rays = np.stack((xs, ys, np.ones_like(xs)), axis=1) @ np.linalg.inv(intrinsic).T
    points_camera = rays * depth[ys, xs, None]
    points_world = points_camera @ rotation.T + camera_pos
    # Reject a tiny number of high depth outliers before selecting the top band.
    top_z = float(np.quantile(points_world[:, 2], .95))
    top = (points_world[:, 2] >= top_z - .003) & (points_world[:, 2] <= top_z + .003)
    if np.count_nonzero(top) < min_pixels:
        result["reason"] = "insufficient_top_surface"
        return result
    surface = np.median(points_world[top], axis=0)
    position = surface - np.array([0., 0., half_height])
    border = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
    count_quality = min(1., result["valid_depth_count"] / 80.)
    depth_quality = result["valid_depth_count"] / result["mask_count"]
    result.update(found=True, position=position, surface_position=surface,
                  confidence=float(count_quality * depth_quality * (.5 if border else 1.)),
                  reason="image_border_truncation" if border else "visible_top_approximation",
                  pixel_centroid=[float(np.mean(xs[top])), float(np.mean(ys[top]))])
    return result


def render_observations(env, slots=None, refresh=True):
    """Return independent RGB-D observations for selected head-camera slots.

    A refresh renders the scene exactly once and refreshes the batched camera
    once, irrespective of the number of slots. ``refresh=False`` reads the last
    available camera frame without rendering or advancing simulation. Camera
    data are copied to host arrays before estimation, so subsequent renderer
    updates do not change the returned observations.

    Each value is ``{'rgb': uint8[H,W,3], 'depth': [H,W,1], 'estimates': ...}``.
    Estimated positions stay in the actual world frame, including environment
    origins. The caller owns any conversion to environment-local coordinates.
    This function never reads object transforms or environment origins.
    """
    camera = env.scene["head_camera"]
    if refresh:
        env.sim.render()
        camera.update(0., force_recompute=True)
    data = camera.data
    rgb = _numpy(data.output["rgb"]).copy()
    depth = _numpy(data.output["distance_to_image_plane"]).copy()
    intrinsic = _numpy(data.intrinsic_matrices).copy()
    camera_pos = _numpy(data.pos_w).copy()
    quaternion_xyzw = _numpy(data.quat_w_ros).copy()
    if rgb.ndim != 4 or rgb.shape[-1] not in (3, 4):
        raise ValueError("head_camera RGB must have batched shape [N,H,W,3 or 4]")
    count, height, width = rgb.shape[:3]
    if depth.shape == (count, height, width):
        depth = depth[..., None]
    if depth.shape != (count, height, width, 1):
        raise ValueError("head_camera plane depth must have shape [N,H,W,1]")
    if intrinsic.shape != (count, 3, 3) or camera_pos.shape != (count, 3) or quaternion_xyzw.shape != (count, 4):
        raise ValueError("head_camera image and calibration/pose batch sizes must match")
    rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        if not np.isfinite(rgb).all() or np.min(rgb, initial=0.) < 0.:
            raise ValueError("head_camera RGB contains invalid pixel values")
        maximum = np.max(rgb, initial=0.)
        if maximum > 255.:
            raise ValueError("head_camera RGB must be uint8-like or normalized [0,1]")
        rgb = np.rint(rgb * (255. if maximum <= 1. else 1.)).astype(np.uint8)
    if slots is None:
        selected = np.arange(count, dtype=int)
    else:
        selected = _numpy(slots)
        if selected.ndim == 0:
            selected = selected.reshape(1)
        if selected.ndim != 1 or selected.dtype.kind not in "iu":
            raise ValueError("slots must be a one-dimensional sequence of integer camera indices")
        if np.any(selected < 0) or np.any(selected >= count):
            raise IndexError("requested camera slot is outside the image batch")
        if len(np.unique(selected)) != len(selected):
            raise ValueError("slots must not contain duplicate indices")
    observations = {}
    for slot in selected:
        slot = int(slot)
        image, plane_depth = rgb[slot].copy(), depth[slot].copy()
        quaternion = quaternion_xyzw[slot][[3, 0, 1, 2]]
        estimates = {
            target: estimate_target(image, plane_depth, intrinsic[slot], camera_pos[slot],
                                    quaternion, target=target)
            for target in ("cube", "tray")
        }
        observations[slot] = dict(rgb=image, depth=plane_depth, estimates=estimates)
    return observations
