from typing import Union, Optional

import numpy as np
import torch
from .smpl import parts_map as smpl_joints_map
from smplx import SMPL, SMPLH, SMPLX
from torch import Tensor
from loguru import logger

HAND_NAMES2INDS = {"left_hand": 0, "right_hand": 1}

SMPL_HANDS_JOINTS_ID = {"left_hand": smpl_joints_map['SMPLH_left_middle1'],  # SMPL_LeftWrist
                        "right_hand": smpl_joints_map['SMPLH_right_middle1']}  # SMPL_RightWrist
SMPL_HANDS_ROTJOINTS_ID = {"left_hand": smpl_joints_map['SMPL_LeftWrist'],  # SMPL_LeftWrist
                           "right_hand": smpl_joints_map['SMPL_RightWrist']}  # SMPL_RightWrist

HAND_INDS2NAMES = {v: k for k, v in HAND_NAMES2INDS.items()}


def centrify_smplx_root_joint(smpl_model: Union[SMPL, SMPLH, SMPLX]):
    def centrifying_forward(
            betas: Optional[Tensor] = None,
            body_pose: Optional[Tensor] = None,
            global_orient: Optional[Tensor] = None,
            transl: Optional[Tensor] = None,
            return_verts=True,
            return_full_pose: bool = False,
            pose2rot: bool = True,
            **kwargs
    ):
        smpl_output = old_forward(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl, return_verts=return_verts,
                                  return_full_pose=return_full_pose, pose2rot=pose2rot, **kwargs)
        apply_trans = transl is not None or hasattr(smpl_model, 'transl')
        if transl is None and hasattr(smpl_model, 'transl'):
            transl = smpl_model.transl
        diff = -smpl_output.joints[0, 0, :]
        if apply_trans:
            diff = diff + transl
        smpl_output.joints = smpl_output.joints + diff.view(1, 1, 3)
        smpl_output.vertices = smpl_output.vertices + diff.view(1, 1, 3)
        return smpl_output

    old_forward = smpl_model.forward
    smpl_model.forward = centrifying_forward
    return smpl_model


def find_closest_before(arr, val):
    start_diff = val - arr
    starts_happened_before = start_diff >= 0
    if np.count_nonzero(starts_happened_before) == 0:
        return None
    closest_start_ind = np.argmin(start_diff[starts_happened_before])
    return closest_start_ind


def find_closest_after(arr, val):
    start_diff = val - arr
    starts_happened_after = start_diff < 0
    if np.count_nonzero(starts_happened_after) == 0:
        return None
    closest_start_ind = np.argmax(start_diff[starts_happened_after]) + np.count_nonzero(~starts_happened_after)
    return closest_start_ind


def parse_seqname(seqname):
    take_num = None
    res = {}
    dotsplit = seqname.split(".")
    if len(dotsplit) > 1:
        try:
            take_num = int(dotsplit[1])
        except ValueError:
            take_num = None
            logger.warning(f"Cannot parse take number '{dotsplit[1]}' in sequence '{seqname}'")
    if take_num is not None:
        res['take_id'] = take_num
    namesplit = seqname.split(".")[0].split("_")
    if seqname.startswith("SUB"):
        res["subject_id"] = int(namesplit[0][3:])
        namesplit = namesplit[1:]
    if namesplit[1] == "BIB":
        res["scene_name"] = "_".join(namesplit[:3])
        namesplit = namesplit[3:]
    else:
        res["scene_name"] = "_".join(namesplit[:2])
        namesplit = namesplit[2:]
    if len(namesplit) > 0:
        res["sequence_name"] = "_".join(namesplit)
    return res


def add_zero(pts):
    pts = torch.cat([torch.zeros_like(pts[0, None, ...]), pts], dim=0)
    return pts
