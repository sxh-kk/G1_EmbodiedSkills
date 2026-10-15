"""Shared physical task outcomes, evaluated from recorded states."""
import numpy as np
from .policies import spec_from_geometry


def episode_outcome(trace, slot=0, origin=(0., 0., 0.), terminated=False, control_dt=.02,
                    geometry=None):
    spec = spec_from_geometry(geometry)
    target_xy = np.asarray(spec.tray_position_xy)
    # Literal legacy endpoints at the default geometry preserve strict boundary
    # behavior; offsets express the same window relative to the tray surface.
    tray_height_offset = spec.tray_top_z - .722
    placement_min_z, placement_max_z = .733 + tray_height_offset, .75 + tray_height_offset
    origin = np.asarray(origin)
    cube = np.asarray(trace["cube"])[:, slot] - origin
    tcp = np.asarray(trace["tcp"])[:, slot] - origin
    root = np.asarray(trace["root"])[:, slot]
    gripper = np.asarray(trace["gripper"])[:, slot]
    hold_samples = int(np.ceil(1./control_dt)) + 1
    lift = bool(cube[:, 2].max()-cube[0, 2] >= .03)
    last = cube[-hold_samples:]
    placed = bool(len(last) == hold_samples
        and np.max(np.linalg.norm(last[:, :2]-target_xy, axis=-1)) < .06
        and np.min(last[:, 2]) > placement_min_z and np.max(last[:, 2]) < placement_max_z
        and np.max(np.linalg.norm(last-last[-1], axis=-1)) < .004)
    released = bool(gripper[-1].min() > .03)
    retreated = bool(np.linalg.norm(tcp[-1]-cube[-1]) >= .06)
    tilt = 2*np.arccos(np.clip(np.sqrt(root[:,3]**2+root[:,6]**2), 0, 1))
    standing = bool(not terminated and root[:,2].min() > .5 and tilt.max() < .6)
    success = bool(lift and placed and released and retreated and standing)
    distance = float(np.linalg.norm(cube[-1,:2]-target_xy))
    physical_return = (4.*success + float(lift) + min(max(float(cube[:,2].max()-cube[0,2]),0.)/.1,1.)
        - min(distance/.25,2.) - 2.*float(not standing))
    return dict(lifted=lift, placed=placed, released=released, retreated=retreated,
        standing=standing, terminated=bool(terminated), success=success,
        maximum_cube_height_m=float(cube[:,2].max()), final_cube_position=cube[-1].tolist(),
        final_target_distance_m=distance, return_value=float(physical_return))
