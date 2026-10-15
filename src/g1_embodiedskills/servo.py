"""Bounded task-space feedback applied to references of the frozen controller."""
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .kinematics import G1Kinematics
from .policies import spec_from_geometry


class ReferenceServo:
    def __init__(self, sonic, planned, reference, geometry, model=None):
        self.model = model if model is not None else G1Kinematics(sonic)
        self.q = np.asarray(planned["q"], dtype=float)
        self.root = np.asarray(reference["root_position"], dtype=float)
        self.root_quaternion = np.asarray(reference["root_quaternion_wxyz"], dtype=float)
        self.spec = spec_from_geometry(geometry)
        self.geometry = self.spec.as_dict()
        self.correction = np.zeros(29)
        self.cube_offset = np.zeros(3)
        self.tray_offset = np.zeros(3)
        self.last_estimate_time = {"cube": None, "tray": None}

    def observe_targets(self, estimates, elapsed):
        if estimates is None:
            return
        cube = estimates["cube"]
        # Estimate during the clear pre-approach view, before fingers occlude the cube.
        if cube["found"] and cube["confidence"] >= .8 and 3.5 <= elapsed < 6.5:
            value = np.asarray(cube["position"]) - np.asarray(self.geometry["cube_position"])
            if np.linalg.norm(value[:2]) < .06 and abs(value[2]) < .03:
                self.cube_offset = .5*self.cube_offset + .5*value
                self.last_estimate_time["cube"] = elapsed
        tray = estimates["tray"]
        if tray["found"]:
            center = np.r_[self.spec.tray_position_xy, self.spec.tray_top_z-self.spec.tray_thickness_m/2]
            value = np.asarray(tray["position"]) - center
            if np.linalg.norm(value[:2]) < .06 and abs(value[2]) < .03:
                self.tray_offset = .5*self.tray_offset + .5*value
                self.last_estimate_time["tray"] = elapsed

    def target(self, cursor):
        q = self.q[min(cursor, len(self.q)-1)]
        positions, matrices = self.model.transforms(q, self.root, self.root_quaternion)
        matrix = matrices[self.model.right_wrist_index]
        position = positions[self.model.right_wrist_index] + matrix @ [.085, 0., 0.]
        elapsed = cursor / 50
        if 4. <= elapsed < 14.:
            fraction = min(1., (elapsed-4.)/3.)
            position += fraction*self.cube_offset
        elif 14. <= elapsed:
            fraction = min(1., (elapsed-14.)/3.)
            position += (1-fraction)*self.cube_offset + fraction*self.tray_offset
        return position, matrix

    def update(self, cursor, actual_tcp, actual_wrist_quaternion_wxyz):
        started = time.monotonic()
        target, target_matrix = self.target(cursor)
        actual_tcp = np.asarray(actual_tcp)
        actual_matrix = Rotation.from_quat(np.asarray(actual_wrist_quaternion_wxyz)[[1,2,3,0]]).as_matrix()
        position_error = target - actual_tcp
        orientation_error = Rotation.from_matrix(target_matrix @ actual_matrix.T).as_rotvec()
        previous = self.correction.copy()
        q = self.q[min(cursor, len(self.q)-1)] + self.correction
        positions, matrices = self.model.transforms(q, self.root, self.root_quaternion)
        base_matrix = matrices[self.model.right_wrist_index]
        base_tcp = positions[self.model.right_wrist_index] + base_matrix @ [.085,0.,0.]
        jacobian = np.zeros((6, 7))
        for column, index in enumerate(self.model.right_arm_indices):
            probe = q.copy()
            probe[index] += .002
            p, r = self.model.transforms(probe, self.root, self.root_quaternion)
            matrix = r[self.model.right_wrist_index]
            tcp = p[self.model.right_wrist_index] + matrix @ [.085,0.,0.]
            jacobian[:3, column] = (tcp-base_tcp)/.002
            jacobian[3:, column] = .12*Rotation.from_matrix(matrix @ base_matrix.T).as_rotvec()/.002
        error = np.r_[.45*position_error, .12*.3*orientation_error]
        update = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + .02**2*np.eye(6), error)
        if cursor >= 50:
            self.correction[self.model.right_arm_indices] += np.clip(update, -.04, .04)
            self.correction = np.clip(self.correction, -.45, .45)
            nominal = self.q[min(cursor, len(self.q)-1)]
            self.correction = np.clip(self.correction,
                self.model.joint_limits[:,0]+.015-nominal,
                self.model.joint_limits[:,1]-.015-nominal)
        return previous, self.correction.copy(), dict(cursor=cursor, target=target,
            actual_tcp=actual_tcp, position_error=position_error, orientation_error=orientation_error,
            correction=self.correction, wall_seconds=time.monotonic()-started,
            perceived_cube_at=self.last_estimate_time["cube"], perceived_tray_at=self.last_estimate_time["tray"])

    def trajectory(self, cursor, previous, correction, frames=75):
        indices = np.minimum(np.arange(cursor, cursor+frames), len(self.q)-1)
        blend = np.minimum(np.arange(frames)/15., 1.)
        blend = blend*blend*(3-2*blend)
        q = self.q[indices] + previous[None] + blend[:,None]*(correction-previous)[None]
        q = np.clip(q, self.model.joint_limits[:,0]+.015, self.model.joint_limits[:,1]-.015)
        roots = np.repeat(self.root[None], frames, axis=0)
        quaternions = np.repeat(self.root_quaternion[None], frames, axis=0)
        return q, roots, quaternions
