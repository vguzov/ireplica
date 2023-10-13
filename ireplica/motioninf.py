import numpy as np
import torch
from .utils import add_zero
from scipy.spatial.transform import Rotation
from tqdm import trange


def compute_1st_derivative(vct):
    return vct[1:] - vct[:-1]


def restore_from_1st_deriv(derivs, starting_point=None):
    if starting_point is None:
        starting_point = torch.zeros_like(derivs[0])
    derivs = add_zero(derivs)
    vct = torch.cumsum(derivs, dim=0) + starting_point
    return vct


def pts_to_anglen(pts):
    start_vct = (1, 0)
    vcts = pts[1:] - pts[:-1]
    lengths = torch.norm(vcts, dim=1)
    norm_vcts = vcts / lengths[:, None]
    cos_a = norm_vcts[0]
    angles = torch.arccos(norm_vcts[:, 0]) * torch.sign(norm_vcts[:, 1])
    angles[norm_vcts[:, 0] > 1] = 0
    angles[norm_vcts[:, 0] < -1] = np.pi
    return angles, lengths


def anglen_to_pts(angles, lengths):
    start_vct = (1, 0)
    norm_vcts = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    vcts = lengths[:, None] * norm_vcts
    vcts = add_zero(vcts)
    pts = torch.cumsum(vcts, dim=0)
    return pts


def optimize_flex_tape(orig_pts, control_pts, rigid_delta=1e4, lr=1e-4, iters_count=1000):
    def compute_loss(derivs):
        angles = restore_from_1st_deriv(derivs)[1:]
        pts = anglen_to_pts(angles, lengths)[1:]
        diffs = pts[control_pts_mask] - control_pts_masked
        target_loss = torch.norm(diffs, dim=1).mean()
        reg_diff = derivs_orig - derivs
        reg_loss = torch.abs(reg_diff).mean()
        return target_loss + rigid_delta * reg_loss

    pts = add_zero(orig_pts)
    control_pts_mask = torch.isfinite(control_pts[:, 0])
    control_pts_masked = control_pts[control_pts_mask]
    orig_angles, lengths = pts_to_anglen(pts)
    angles = add_zero(orig_angles)
    derivs_orig = compute_1st_derivative(angles)
    derivs = derivs_orig.clone().requires_grad_(True)
    optim = torch.optim.Adam([derivs], lr=lr)
    losshist = []
    for i in trange(iters_count):
        optim.zero_grad()
        loss = compute_loss(derivs)
        losshist.append(loss.item())
        loss.backward()
        optim.step()
    angles = restore_from_1st_deriv(derivs.detach())[1:]
    pts = anglen_to_pts(angles, lengths)[1:]
    angles_diff = angles - orig_angles
    pts_diff = pts - orig_pts
    return pts, (angles_diff, pts_diff), losshist


def restrict_to_hinge_rotation(loc_dict, hinge_xyz):
    quat = np.array(loc_dict['quaternion'])
    rot = Rotation.from_quat(np.roll(quat, -1))
    new_position = -rot.apply(hinge_xyz) + hinge_xyz
    return {"position": new_position.tolist(), "quaternion": quat.tolist()}
