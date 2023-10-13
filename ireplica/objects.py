from typing import List

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation
from sklearn.neighbors import KDTree

from .utils import find_closest_before, find_closest_after


class ObjectLocation:
    def __init__(self, translation, quaternion, time=None):
        self._translation = np.asarray(translation)
        self._quaternion = np.asarray(quaternion)
        if time is None:
            self._time = None
        else:
            self._time = float(time)

    def to_dict(self):
        return {"position": self._translation.tolist(),
                "quaternion": self._quaternion.tolist()}

    @property
    def translation(self) -> np.ndarray:
        return self._translation

    @property
    def position(self) -> np.ndarray:
        return self._translation

    @property
    def quaternion(self) -> np.ndarray:
        return self._quaternion

    @property
    def time(self) -> float:
        return self._time

    def __getitem__(self, item):
        if item in ["position", "translation"]:
            return self.translation
        elif item == "quaternion":
            return self.quaternion
        elif item == "time":
            return self.time
        else:
            raise IndexError(f"No such index '{item}'")


class ObjectTrajectory:
    def __init__(self, traj_poses, traj_quats, traj_times):
        self._translations = np.asarray(traj_poses)
        self._quaternions = np.asarray(traj_quats)
        self._times = np.asarray(traj_times)

    @property
    def translations(self) -> np.ndarray:
        return self._translations

    @property
    def positions(self) -> np.ndarray:
        return self._translations

    @property
    def quaternions(self) -> np.ndarray:
        return self._quaternions

    @property
    def times(self) -> np.ndarray:
        return self._times

    def __getitem__(self, item: int):
        return ObjectLocation(self.translations[item], self.quaternions[item], time=self.times[item])

    def __len__(self):
        return len(self._translations)

    @classmethod
    def cat_trajectories(cls, traj_list: List["ObjectTrajectory"]):
        transls = []
        quats = []
        times = []
        for traj in traj_list:
            transls.append(traj.translations)
            quats.append(traj.quaternions)
            times.append(traj.times)
        res = cls(np.concatenate(transls, axis=0), np.concatenate(quats, axis=0), np.concatenate(times, axis=0))
        return res


class ObjectInfo:
    def __init__(self, obj_model, obj_positions_info, obj_name=None):
        self.name = obj_name
        self.model = obj_model
        self.positions_info = obj_positions_info
        self.vertices = np.asarray(obj_model.vertices)
        self.center = np.asarray(obj_positions_info["center"])
        self.hinge = np.asarray(obj_positions_info["hinge"]) if "hinge" in obj_positions_info else None
        self.anchor_positions = np.asarray([x["position"] for x in obj_positions_info["locations"]])
        self.anchor_quaternions = np.asarray([x["quaternion"] for x in obj_positions_info["locations"]])
        self.anchor_timestamps = np.asarray([x["time"] for x in obj_positions_info["locations"]])

    @staticmethod
    def _apply_location(vertices, location_dict):
        obj_rot = Rotation.from_quat(np.roll(location_dict["quaternion"], -1))
        obj_pos = np.array(location_dict["position"])
        obj_model_rotated = obj_rot.apply(vertices) + obj_pos[None, :]
        return obj_model_rotated

    @staticmethod
    def _detect_contact_point(obj_vertices, body_point, obj_tree=None):
        assert body_point.ndim == 1, f"{body_point.shape}"
        if obj_tree is None:
            obj_tree = KDTree(obj_vertices)
        hand_closets_dists, hand_closest_inds = obj_tree.query(body_point[None, :], k=20)
        contact_point = np.array(obj_vertices)[hand_closest_inds.flatten()].mean(axis=0)
        return contact_point

    def detect_contact_point(self, body_point, location=None):
        obj_vertices = self.vertices if location is None else self.get_localized_verts(location)
        contact_point = self._detect_contact_point(obj_vertices, body_point)
        return contact_point

    def get_obj_location(self, timestamp, search_before=True):
        if search_before:
            obj_loc_ind = find_closest_before(self.anchor_timestamps, timestamp)
            if obj_loc_ind is None:
                logger.warning("No known position before the start, searching after")
                obj_loc_ind = find_closest_after(self.anchor_timestamps, timestamp)
                if obj_loc_ind is None:
                    raise ValueError("No anchor object positions to start")
        else:
            obj_loc_ind = find_closest_after(self.anchor_timestamps, timestamp)
            if obj_loc_ind is None:
                logger.warning("No known position after the end, searching before")
                obj_loc_ind = find_closest_before(self.anchor_timestamps, timestamp)
        return ObjectLocation(self.anchor_positions[obj_loc_ind], self.anchor_quaternions[obj_loc_ind])

    def get_localized_verts(self, location: ObjectLocation):
        positioned_obj_vertices = self._apply_location(self.vertices, location)
        return positioned_obj_vertices
