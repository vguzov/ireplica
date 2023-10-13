import json
import os
import shutil
import numpy as np
import smplx
import termplotlib as tpl
import torch
import trimesh
import zipjson
from collections import defaultdict
from pathlib import Path
from loguru import logger
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
from sklearn.neighbors import KDTree
from smplx.lbs import batch_rodrigues, blend_shapes, vertices2joints, batch_rigid_transform
from .config import IReplicaConfig
from .objects import ObjectInfo, ObjectLocation, ObjectTrajectory
from .timelines import InteractionTimeline, MultiobjectInteractionTimeline, InteractionInterval
from .utils import HAND_NAMES2INDS, HAND_INDS2NAMES, find_closest_before, find_closest_after, SMPL_HANDS_JOINTS_ID
from .motioninf import optimize_flex_tape, restrict_to_hinge_rotation


class IReplicaTracker:
    MODEL_PARAM_NAMES = {
        "smpl": ["betas", "body_pose", "global_orient", "transl"],
        "smplh": ["betas", "body_pose", "global_orient", "transl", "left_hand_pose", "right_hand_pose"],
        "smplx": ["betas", "body_pose", "global_orient", "transl", "left_hand_pose", "right_hand_pose", "expression", "jaw_pose", "leye_pose",
                  "reye_pose"],
    }

    def __init__(self, args: "IReplicaConfig", device=None):
        self.args = args
        self.straight_hands = args.straight_hands
        self.contact_smoothing_thresh = args.contact_smoothing_thresh
        self.relocate_body_to_start = args.relocate_body_to_start
        self.smpl_root = args.smpl_root
        self.device = torch.device(device if device is not None else "cpu")
        self.center_root_joint = True
        self.motion_format_version = args.motion_format_version
        self.model_type = "smplh"
        self.use_gt_contact_intervals = args.use_gt_contact_intervals
        self.start_reloc_params = args.start_reloc_params
        self.contact_labeling_thresh = args.contact_labeling_thresh
        self.compute_endpoint_metrics = args.compute_endpoint_metrics
        self.config = args

    def _init_model(self, model_type="smplh", gender='neutral', smpl_compatible=False, flat_hand_mean=True, template=None):
        self.model_layer = smplx.create(self.smpl_root, model_type=model_type, gender=gender, use_pca=False, flat_hand_mean=flat_hand_mean).to(
            self.device)
        self.model_layer.requires_grad_(False)
        if smpl_compatible:
            smpl_model = smplx.create(self.smpl_root, model_type="smpl", gender=gender)
            self.model_layer.shapedirs[:] = smpl_model.shapedirs.detach().to(self.device)
        if template is not None:
            self.model_layer.v_template[:] = torch.tensor(template, dtype=self.model_layer.v_template.dtype,
                                                          device=self.device)
        self.gender = gender
        self.smpl_compatible = smpl_compatible
        self.available_smpl_params = self.MODEL_PARAM_NAMES[model_type]

    @staticmethod
    def center_output(smpl_model, params, smpl_output):
        if 'transl' in params and params['transl'] is not None:
            transl = params['transl']
        else:
            transl = None
        apply_trans = transl is not None or hasattr(smpl_model, 'transl')
        if transl is None and hasattr(smpl_model, 'transl'):
            transl = smpl_model.transl
        diff = -smpl_output.joints[:, 0, :]
        if apply_trans:
            diff = diff + transl
        smpl_output.joints = smpl_output.joints + diff.view(-1, 1, 3)
        smpl_output.vertices = smpl_output.vertices + diff.view(-1, 1, 3)
        return smpl_output

    def _preprocess_smpl_param(self, param):
        if not isinstance(param, torch.Tensor):
            param = torch.tensor(param, dtype=torch.float32)
        param = param.to(self.device)
        return param

    def get_smpl_verts_joints(self, params_dict):
        batch_params = {k: self._preprocess_smpl_param(params_dict[k]).unsqueeze(0) for k in self.available_smpl_params if k in params_dict}
        smpl_output = self._torch_call_smpl_model(batch_params)
        return smpl_output.vertices[0].cpu().numpy(), smpl_output.joints[0].cpu().numpy()

    def _torch_get_smpl_rots(self, betas, pose):
        v_shaped = self.model_layer.v_template + blend_shapes(betas, self.model_layer.shapedirs)

        # Get the joints
        # NxJx3 array
        J = vertices2joints(self.model_layer.J_regressor, v_shaped)

        batch_size = pose.shape[0]
        dtype = betas.dtype
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view(
            [batch_size, -1, 3, 3])

        J_transformed, A = batch_rigid_transform(rot_mats, J, self.model_layer.parents, dtype=dtype)

        return A[:, :, :3, :3]

    def _torch_call_smpl_model(self, batch_params, return_rotations=False):
        smpl_output = self.model_layer(**batch_params, return_full_pose=return_rotations)
        if return_rotations:
            full_pose = smpl_output.full_pose
            betas = batch_params["betas"]
            smpl_output.global_rotation_mats = self._torch_get_smpl_rots(betas, full_pose)
        if self.center_root_joint:
            return self.center_output(self.model_layer, batch_params, smpl_output)
        else:
            return smpl_output

    @staticmethod
    def smooth_contact_pred(preds, timestamps, thresh=0.5):
        if thresh <= 0.:
            logger.info("Smoothing thresh <=0, skipping smoothing step...")
            return preds
        last_true_state_ts = -np.inf
        new_preds = []
        buf_ts = []
        for ts, pred in zip(timestamps, preds):
            buf_ts.append(ts)
            if pred:
                if ts - last_true_state_ts < thresh:
                    for _ in buf_ts:
                        new_preds.append(True)
                else:
                    for _ in buf_ts:
                        new_preds.append(False)
                last_true_state_ts = ts
                buf_ts = []
        for _ in buf_ts:
            new_preds.append(False)
        return np.asarray(new_preds)

    @staticmethod
    def draw_contacts(contact_ts, contact_preds, width=None, height=15):
        if width is None:
            termsize = shutil.get_terminal_size((80, 20))
            width = termsize.columns
        for ind in range(2):
            counts_all, bin_edges = np.histogram(contact_ts, bins=width)
            counts, _ = np.histogram(contact_ts[contact_preds[:, ind]], bins=bin_edges)
            fig = tpl.figure()
            fig.hist(counts / counts_all, bin_edges, grid=[height, width], force_ascii=False)
            logger.debug(("Left" if ind == 0 else "Right") + " hand\n" + fig.get_string())

    @staticmethod
    def convert_to_new_motion_format(motion_seq, motion_ts, model_type="smpl"):
        # Converting to the new format
        rebuilt_motion_seq = []
        for old_style_params, timestamp in zip(motion_seq, motion_ts):
            new_params = {"time": float(timestamp)}
            if 'pose' in old_style_params:
                new_params["global_orient"] = old_style_params["pose"][:3]
                new_params["body_pose"] = old_style_params["pose"][3:]
            if 'translation' in old_style_params:
                new_params["transl"] = old_style_params["translation"]
            if 'shape' in old_style_params:
                new_params["betas"] = old_style_params["shape"]
            rebuilt_motion_seq.append(new_params)
        motion_seq = rebuilt_motion_seq
        if model_type in ["smplh", "smplx"] and len(motion_seq[0]["body_pose"]) == 69:
            motion_seq = [{k: v if k != "body_pose" else v[:63] for k, v in x.items()} for x in motion_seq]
        elif model_type == "smpl" and len(motion_seq[0]["body_pose"]) == 63:
            fbx_motion_seq = [{k: v if k != "body_pose" else np.concatenate([np.array(v), np.zeros(6)]) for k, v in x.items()} for x in
                              motion_seq]
        return motion_seq

    def compute_interaction_intervals(self, sequence_interval: np.ndarray, contact_preds, contact_ts, interaction_type,
            contact_labeling_thresh) -> InteractionTimeline:
        sequence_interval_frames = np.asarray((find_closest_before(contact_ts, sequence_interval[0]),
                                               find_closest_before(contact_ts, sequence_interval[1])))

        contact_preds = contact_preds > contact_labeling_thresh

        logger.debug("Before smoothing")
        self.draw_contacts(contact_ts[sequence_interval_frames[0]:sequence_interval_frames[1]],
                           contact_preds[sequence_interval_frames[0]:sequence_interval_frames[1]])
        sm_contact_preds = np.stack(
            [self.smooth_contact_pred(contact_preds[:, hand_ind], contact_ts, thresh=self.contact_smoothing_thresh) for hand_ind in range(2)], axis=1)
        logger.debug("After smoothing")
        self.draw_contacts(contact_ts[sequence_interval_frames[0]:sequence_interval_frames[1]],
                           sm_contact_preds[sequence_interval_frames[0]:sequence_interval_frames[1]])

        int_timeline = InteractionTimeline.from_predicted_contacts(sequence_interval, sm_contact_preds, contact_ts, interaction_type=interaction_type)
        return int_timeline

    def get_smpl_output_interval(self, cropped_smpl_motion_seq, return_rotations=False):
        batch_smpl_params_interval = {k: torch.stack(
            [self._preprocess_smpl_param(params_dict[k]) if k in params_dict else getattr(self.model_layer, k)[0, :] for params_dict in
             cropped_smpl_motion_seq]) for k in self.available_smpl_params}
        logger.debug("Calling SMPL model")
        smpl_output_interval = self._torch_call_smpl_model(batch_smpl_params_interval, return_rotations=return_rotations)
        return smpl_output_interval

    def infer_traj_single_interaction_interval(self, interaction_interval: InteractionInterval, smpl_motion_seq, smpl_timestamps,
            last_obj_location=None, infer_rotation_from_hand=False, snap_obj_to_hand=False):
        inferred_traj_positions = []
        inferred_traj_quats = []
        # inferred_traj_timestamps = []

        smpl_interval_start = find_closest_before(smpl_timestamps, interaction_interval.start_time)
        smpl_interval_end = find_closest_before(smpl_timestamps, interaction_interval.end_time)
        if smpl_interval_start == smpl_interval_end:
            logger.info("Interval is too short to process, returning empty traj")
            inferred_traj_positions = np.zeros((0, 3))
            inferred_traj_quats = np.zeros((0, 4))
            inferred_traj_timestamps = np.zeros(0)
            hands_traj = {"left_hand": None, "right_hand": None}
            return inferred_traj_positions, inferred_traj_quats, inferred_traj_timestamps, hands_traj

        cropped_smpl_timestamps = smpl_timestamps[smpl_interval_start:smpl_interval_end]
        cropped_smpl_motion_seq = smpl_motion_seq[smpl_interval_start:smpl_interval_end]
        inferred_traj_timestamps = cropped_smpl_timestamps

        logger.debug("Preparing SMPL params")
        smpl_output_interval = self.get_smpl_output_interval(cropped_smpl_motion_seq, return_rotations=(
                                                                                                               len(interaction_interval.interacting_hands_labels) == 1) and infer_rotation_from_hand)
        logger.debug("Processing hand trajectories")
        smpl_hands_traj = {hand_name: smpl_output_interval.joints[:, SMPL_HANDS_JOINTS_ID[hand_name]].cpu().numpy() for hand_name in
                           HAND_NAMES2INDS.keys()}  # 0 - left hand, 1 - right hand

        if last_obj_location is None:
            last_obj_location = {"quaternion": np.array([1., 0, 0, 0]), "position": np.zeros(3)}

        first_obj_location = last_obj_location
        hands_traj = {"left_hand": None, "right_hand": None}

        if len(interaction_interval.interacting_hands_labels) == 2:
            hands_traj = {"left_hand": [], "right_hand": []}
            logger.debug("Two-hand motion: inferring poses and rotations")
            last_mvnx_left_hand = smpl_hands_traj["left_hand"][0]
            last_mvnx_right_hand = smpl_hands_traj["right_hand"][0]
            first_mvnx_right_hand = smpl_hands_traj["right_hand"][0]
            first_mvnx_left_hand = smpl_hands_traj["left_hand"][0]
            first_vct = first_mvnx_left_hand - first_mvnx_right_hand
            first_vct[2] = 0
            # first_vct = first_vct[:2]
            inferred_traj_positions.append(last_obj_location["position"])
            inferred_traj_quats.append(last_obj_location["quaternion"])
            last_rotation_delta_ang = 0
            rotation_delta_ang_list = [last_rotation_delta_ang]
            last_cross = np.zeros(3)
            hands_traj["left_hand"].append(first_mvnx_left_hand)
            hands_traj["right_hand"].append(first_mvnx_right_hand)
            for frame_time, smpl_frame_ind, global_frame_ind in zip(cropped_smpl_timestamps, range(1, smpl_interval_end - smpl_interval_start),
                                                                    range(smpl_interval_start + 1, smpl_interval_end)):
                curr_mvnx_left_hand = smpl_hands_traj["left_hand"][smpl_frame_ind]
                curr_mvnx_right_hand = smpl_hands_traj["right_hand"][smpl_frame_ind]
                hands_traj["left_hand"].append(curr_mvnx_left_hand)
                hands_traj["right_hand"].append(curr_mvnx_right_hand)
                curr_vct = curr_mvnx_left_hand - curr_mvnx_right_hand
                curr_vct[2] = 0
                cross = np.cross(first_vct / np.linalg.norm(first_vct), curr_vct / np.linalg.norm(curr_vct))
                rotation_delta_ang_list.append(cross)
                if np.abs(cross - last_cross)[2] > 0.1:
                    logger.debug(cross - last_cross)
                last_cross = cross
            first_quat = first_obj_location['quaternion']
            first_pos = first_obj_location['position']
            first_rot = Rotation.from_quat(np.roll(first_quat, -1))
            for frame_time, smpl_frame_ind, global_frame_ind in zip(cropped_smpl_timestamps, range(1, smpl_interval_end - smpl_interval_start),
                                                                    range(smpl_interval_start + 1, smpl_interval_end)):
                curr_mvnx_right_hand = smpl_hands_traj["right_hand"][smpl_frame_ind]
                a = rotation_delta_ang_list[smpl_frame_ind]
                rot = Rotation.from_rotvec(a)

                res_rot = rot * first_rot
                res_pos = rot.apply(first_pos - first_mvnx_right_hand) + curr_mvnx_right_hand
                res_quat = np.roll(res_rot.as_quat(), 1)
                inferred_traj_positions.append(res_pos)
                inferred_traj_quats.append(res_quat)

            inferred_traj_positions = np.stack(inferred_traj_positions, axis=0)
            inferred_traj_quats = np.stack(inferred_traj_quats, axis=0)
            for x in HAND_NAMES2INDS.keys():
                hands_traj[x] = np.stack(hands_traj[x], axis=0)
        elif len(interaction_interval.interacting_hands_labels) == 1:
            interacting_hand_label = interaction_interval.interacting_hands_labels[0]
            inferred_traj_positions = smpl_hands_traj[interacting_hand_label]
            hands_traj[interacting_hand_label] = inferred_traj_positions
            if not snap_obj_to_hand:
                inferred_traj_positions = inferred_traj_positions - inferred_traj_positions[0:1]

                inferred_traj_positions = inferred_traj_positions + last_obj_location["position"]

            if infer_rotation_from_hand:
                logger.debug("One-hand motion: inferring rotations from hand")
                rotmats = smpl_output_interval.global_rotation_mats[:, SMPL_HANDS_JOINTS_ID[interacting_hand_label], ...].cpu().numpy()
                logger.debug(f"Rotmats size: {rotmats.shape}")
                smpl_rots = Rotation.from_matrix(rotmats)
                first_obj_rot = Rotation.from_quat(np.roll(first_obj_location["quaternion"], -1))

                rots = smpl_rots * smpl_rots[0].inv()
                smpl_hand_rotjoint = smpl_output_interval.joints[:, SMPL_HANDS_JOINTS_ID[interacting_hand_label], :].cpu().numpy()
                inferred_traj_positions = rots.apply(inferred_traj_positions - smpl_hand_rotjoint) + smpl_hand_rotjoint
                res_rots = rots * first_obj_rot
                quats = np.roll(res_rots.as_quat(), 1, axis=1)
                inferred_traj_quats = quats
            else:
                logger.debug("One-hand motion: inferring poses only")
                inferred_traj_quats = np.tile(np.asarray(last_obj_location["quaternion"]).reshape(1, 4), (len(inferred_traj_timestamps), 1))
        else:
            logger.debug("No hand is it contact: no motion inferred")
            inferred_traj_positions = np.tile(np.asarray(last_obj_location["position"]).reshape(1, 3), (len(inferred_traj_timestamps), 1))
            inferred_traj_quats = np.tile(np.asarray(last_obj_location["quaternion"]).reshape(1, 4), (len(inferred_traj_timestamps), 1))

        return inferred_traj_positions, inferred_traj_quats, inferred_traj_timestamps, hands_traj

    def generate_object_hinge_positions_no_endpos(self, contact_point, interred_hands_traj, start_obj_loc, hinge_xyz):
        start_position = np.array(start_obj_loc['position'])
        start_quat = np.array(start_obj_loc['quaternion'])
        start_rot = Rotation.from_quat(np.roll(start_quat, -1))

        start_hands_positions = {x: interred_hands_traj[x][0] if interred_hands_traj[x] is not None else None for x in interred_hands_traj.keys()}
        traj_arr_list = [interred_hands_traj[HAND_INDS2NAMES[i]] for i in range(2)]
        if all([x is not None for x in traj_arr_list]):
            contact_point_traj = np.stack(traj_arr_list, axis=0)
            # logger.info(contact_point_traj.shape)
            contact_point_traj = contact_point_traj.mean(axis=0)
            logger.debug("2-hand hinge interaction - computing the mean traj")
        else:
            if traj_arr_list[0] is not None:
                contact_point_traj = traj_arr_list[0]
            else:
                contact_point_traj = traj_arr_list[1]

        contact_point_traj = torch.from_numpy(contact_point_traj)
        start_ind = 0
        contact_point_start = contact_point_traj[start_ind]
        traj_vcts = (contact_point_traj[:, :2] - hinge_xyz[:2])
        traj_lens = torch.norm(traj_vcts, dim=1)
        traj_cos_a = traj_vcts[:, 0] / traj_lens
        traj_ang = torch.arccos(traj_cos_a) * torch.sign(traj_vcts[:, 1])
        start_pt_vct = contact_point_start[:2] - hinge_xyz[:2]
        start_pt_vct_len = np.linalg.norm(start_pt_vct)
        start_pt_ang = np.arccos(start_pt_vct[0] / start_pt_vct_len) * np.sign(start_pt_vct[1])
        ang_diffs = traj_ang - start_pt_ang
        diff_rots = Rotation.from_euler("z", ang_diffs)
        res_rots = diff_rots * start_rot
        res_locs = []
        res_quats = np.roll(res_rots.as_quat(), 1, axis=1)

        rotmats = torch.stack([torch.stack([torch.cos(ang_diffs), -torch.sin(ang_diffs)], dim=1),
                               torch.stack([torch.sin(ang_diffs), torch.cos(ang_diffs)], dim=1)], dim=1)

        res_hand_traj = {}
        for hand_name, start_hand_pt in start_hands_positions.items():
            if start_hand_pt is not None:
                hand_traj = interred_hands_traj[hand_name]
                start_pt_vct = start_hand_pt[:2] - hinge_xyz[:2]
                mapped_traj = torch.matmul(rotmats, torch.tensor(start_pt_vct[None, :, None])).squeeze(-1) + hinge_xyz[:2]
                mapped_traj_3d = np.concatenate([mapped_traj.cpu().numpy(), hand_traj[:, 2:3]], axis=1)
                res_hand_traj[hand_name] = mapped_traj_3d
            else:
                res_hand_traj[hand_name] = None

        for smpl_frame_ind, res_quat in enumerate(res_quats):
            last_loc = (restrict_to_hinge_rotation({"quaternion": res_quat.tolist(), "position": None},
                                                   hinge_xyz))
            res_locs.append(last_loc)

        return res_locs, res_hand_traj

    def adapt_traj_to_motion_type(self, inferred_traj_positions, inferred_traj_quats, inferred_traj_timestamps, motion_type="free", hinge_xyz=None,
            contact_point=None, hands_traj=None):
        res_hands_traj = hands_traj
        if len(inferred_traj_timestamps) == 0:
            return inferred_traj_positions, inferred_traj_quats, inferred_traj_timestamps, res_hands_traj
        if motion_type == "along_floor":
            inferred_traj_positions[:, 2] = inferred_traj_positions[0, 2]
        elif motion_type == "hinge":
            inferred_traj_positions[:, 2] = inferred_traj_positions[0, 2]
            if hands_traj is None:
                for frame_ind, res_quat in enumerate(inferred_traj_quats):
                    last_loc = (restrict_to_hinge_rotation({"quaternion": res_quat.tolist(), "position": None},
                                                           hinge_xyz))
                    inferred_traj_positions[frame_ind, :] = last_loc["position"]
                    inferred_traj_quats[frame_ind, :] = last_loc["quaternion"]
            else:
                hinged_dicttraj, res_hands_traj = self.generate_object_hinge_positions_no_endpos(contact_point, hands_traj,
                                                                                                 {"position": inferred_traj_positions[0],
                                                                                                  "quaternion": inferred_traj_quats[0]}, hinge_xyz)
                inferred_traj_positions = np.asarray([x["position"] for x in hinged_dicttraj])
                inferred_traj_quats = np.asarray([x["quaternion"] for x in hinged_dicttraj])
        elif motion_type == "free":
            pass
        else:
            logger.error(f"Unknown motion type '{motion_type}', passing with no changes")

        return inferred_traj_positions, inferred_traj_quats, inferred_traj_timestamps, res_hands_traj

    def detect_contact_point(self, obj_vertices, body_point, obj_tree=None):
        assert body_point.ndim == 1, f"{body_point.shape}"
        if obj_tree is None:
            obj_tree = KDTree(obj_vertices)
        hand_closets_dists, hand_closest_inds = obj_tree.query(body_point[None, :], k=20)
        contact_point = np.array(obj_vertices)[hand_closest_inds.flatten()].mean(axis=0)
        return contact_point

    @staticmethod
    def apply_location(vertices, location_dict):
        obj_rot = Rotation.from_quat(np.roll(location_dict["quaternion"], -1))
        obj_pos = np.array(location_dict["position"])
        obj_model_rotated = obj_rot.apply(vertices) + obj_pos[None, :]
        return obj_model_rotated

    def relocate_body_to_object(self, positioned_obj_vertices, smpl_motion_seq, smpl_timestamps, interacting_hands_names, timestamp,
            traj_mod_type="translation", traj_mod_window=100, iters_count=300, flextape_opt_params=None):
        assert traj_mod_type in ["translation", "optimization"]
        if len(interacting_hands_names) == 0:
            logger.warning("No interaction, won't relocate the body")
            return smpl_motion_seq
        positioned_obj_tree = KDTree(positioned_obj_vertices)
        smpl_frame_ind = find_closest_before(smpl_timestamps, timestamp)
        smpl_verts, smpl_joints = self.get_smpl_verts_joints(smpl_motion_seq[smpl_frame_ind])
        hand_joints = smpl_joints[[SMPL_HANDS_JOINTS_ID[HAND_INDS2NAMES[hand_ind]] for hand_ind in range(2)], :]
        translation_vct = np.zeros(3)
        interactive_hand_inds = sorted([HAND_NAMES2INDS[hand_name] for hand_name in interacting_hands_names])
        for iter_ind in range(iters_count):
            hand_closets_dists, hand_closest_inds = positioned_obj_tree.query(hand_joints[interactive_hand_inds] + translation_vct, k=20)
            closest_verts = positioned_obj_vertices[hand_closest_inds.flatten()].reshape(hand_closest_inds.shape + (3,))
            diffs = closest_verts - (hand_joints[interactive_hand_inds] + translation_vct)[:, None, :]
            mean_diffs = diffs.mean(axis=1)
            translation_vct = translation_vct + mean_diffs.mean(axis=0) * np.array([1, 1, 0])
        logger.debug(f"Suggested translation is: {translation_vct}")

        if traj_mod_type == "optimization":
            dtype = getattr(torch, flextape_opt_params["dtype"])  # default: torch.float64
            free_window_radius = traj_mod_window
            tr_params = torch.from_numpy(np.asarray([params["transl"] for params in smpl_motion_seq]))
            orig_pts = torch.tensor(tr_params[:, :2], dtype=dtype, device=self.device)
            control_pts = torch.ones_like(orig_pts) * np.inf
            control_pts[smpl_frame_ind] = torch.tensor(orig_pts[smpl_frame_ind] + translation_vct[:2], dtype=dtype)
            new_pts, (angles_diff, pts_diff), losshist = optimize_flex_tape(orig_pts, control_pts,
                                                                            rigid_delta=flextape_opt_params["rigid_delta"],
                                                                            lr=flextape_opt_params["lr"],
                                                                            iters_count=flextape_opt_params["iters_count"])
            new_tr_params = tr_params.clone()
            new_tr_params[:, :2] = new_pts
            new_tr_params = new_tr_params.cpu().numpy()
        elif traj_mod_type == "translation":
            tr_params = np.asarray([params["transl"] for params in smpl_motion_seq])
            new_tr_params = tr_params + translation_vct
        else:
            raise NotImplementedError(f"Unknown traj_mod_type '{traj_mod_type}'")
        new_smpl_motion_seq = []
        for i, params_dict in enumerate(smpl_motion_seq):
            new_params_dict = {}
            for k, v in params_dict.items():
                if k != "transl":
                    new_params_dict[k] = v
                else:
                    new_params_dict[k] = new_tr_params[i, :]
            new_smpl_motion_seq.append(new_params_dict)
        return new_smpl_motion_seq

    def compute_metrics_objverts(self, pred_positioned_obj_vertices, gt_scene_vertices):
        res_metrics = {}
        # For every point of the object, find the closest point in the scene, compute mean distance
        gt_tree = KDTree(gt_scene_vertices)
        closest_dists, closest_inds = gt_tree.query(pred_positioned_obj_vertices, k=1)
        res_metrics["mean_dist_to_gt"] = float(closest_dists.mean())
        return res_metrics

    def compute_metrics_objloc(self, obj_vertices, predicted_loc, gt_loc):
        res_metrics = {}
        # Angle diff
        pred_rot = Rotation.from_quat(np.roll(predicted_loc["quaternion"], -1))
        gt_rot = Rotation.from_quat(np.roll(gt_loc["quaternion"], -1))
        diff_rotvec = Rotation.as_rotvec(pred_rot * gt_rot.inv())
        diff_ang = np.linalg.norm(diff_rotvec)
        res_metrics.update({"rotation_diff": float(diff_ang), "rotation_diff_deg": float(np.rad2deg(diff_ang))})

        pred_positioned_obj_vertices = self.apply_location(obj_vertices, predicted_loc)
        gt_positioned_obj_vertices = self.apply_location(obj_vertices, gt_loc)

        # Center diff
        center_diff = pred_positioned_obj_vertices.mean(0) - gt_positioned_obj_vertices.mean(0)
        assert (center_diff.ndim == 1) and (center_diff.shape[0] == 3)
        center_dist = np.linalg.norm(center_diff)
        res_metrics["center_dist"] = float(center_dist)

        # Obj-scene metrics
        res_metrics.update(self.compute_metrics_objverts(pred_positioned_obj_vertices, gt_positioned_obj_vertices))

        return res_metrics

    def linear_objtraj_interpolation(self, smpl_timestamps, interaction_interval: InteractionInterval, start_loc, ending_loc, obj_rot_center):

        smpl_interval_start = find_closest_after(smpl_timestamps, interaction_interval.start_time)
        smpl_interval_end = find_closest_before(smpl_timestamps, interaction_interval.end_time)
        if smpl_interval_start == smpl_interval_end:
            logger.warning("Interpolation interval is too small, skipping...")
            return None, None, None, None

        cropped_smpl_timestamps = smpl_timestamps[smpl_interval_start:smpl_interval_end]
        inferred_traj_timestamps = cropped_smpl_timestamps

        position_start = np.asarray(start_loc["position"])
        position_end = np.asarray(ending_loc["position"])
        quat_start = np.asarray(start_loc["quaternion"])
        quat_end = np.asarray(ending_loc["quaternion"])
        rot_start = Rotation.from_quat(np.roll(quat_start, -1))
        rot_end = Rotation.from_quat(np.roll(quat_end, -1))
        interval_list = [interaction_interval.start_time, interaction_interval.end_time]

        interpolator = interp1d(interval_list, [position_start + rot_start.apply(obj_rot_center) - obj_rot_center,
                                                position_end + rot_end.apply(obj_rot_center) - obj_rot_center], axis=0, kind='linear')
        interp_result = interpolator(inferred_traj_timestamps)
        vis_inttraj_positions = interp_result

        loc_rots = Rotation.from_quat(np.roll(np.stack([quat_start, quat_end], 0), -1, axis=1))
        slerp = Slerp(interval_list, loc_rots)
        vis_inttraj_rotations = slerp(inferred_traj_timestamps)
        vis_inttraj_quats = np.roll(vis_inttraj_rotations.as_quat(), 1, axis=1)
        vis_inttraj_positions = vis_inttraj_positions + obj_rot_center - vis_inttraj_rotations.apply(obj_rot_center)

        # Adding start and end positions
        vis_inttraj_quats = np.concatenate([quat_start[None, :], vis_inttraj_quats, quat_end[None, :]], axis=0)
        vis_inttraj_positions = np.concatenate([position_start[None, :], vis_inttraj_positions, position_end[None, :]], axis=0)
        inferred_traj_timestamps = np.concatenate([[interval_list[0]], inferred_traj_timestamps, [interval_list[1]]], axis=0)

        return vis_inttraj_positions, vis_inttraj_quats, inferred_traj_timestamps, None

    def compute_contact_point2hand_mean_dist_single_interval_objwise_timing(self, smpl_motion_seq, smpl_timestamps, overall_traj_poses,
            overall_traj_quats, overall_traj_times, obj_verts, interaction_interval: InteractionInterval):
        traj_time_start_ind = find_closest_before(overall_traj_times, interaction_interval.start_time)
        traj_time_end_ind = find_closest_before(overall_traj_times, interaction_interval.end_time)

        logger.debug("finding interval SMPL params")
        cropped_smpl_motion_seq = []
        cropped_overall_traj_times = overall_traj_times[traj_time_start_ind:traj_time_end_ind]
        cropped_overall_traj_poses = overall_traj_poses[traj_time_start_ind:traj_time_end_ind]
        cropped_overall_traj_quats = overall_traj_quats[traj_time_start_ind:traj_time_end_ind]
        inds = []
        for ts in cropped_overall_traj_times:
            smpl_ind = find_closest_before(smpl_timestamps, ts)
            inds.append(smpl_ind)
            cropped_smpl_motion_seq.append(smpl_motion_seq[smpl_ind])
        cropped_smpl_timestamps = smpl_timestamps[inds]

        logger.debug("Preparing SMPL params")
        smpl_output_interval = self.get_smpl_output_interval(cropped_smpl_motion_seq)
        logger.debug("Processing hand trajectories")
        smpl_hands_traj = {hand_name: smpl_output_interval.joints[:, SMPL_HANDS_JOINTS_ID[hand_name]].cpu().numpy() for hand_name in
                           HAND_NAMES2INDS.keys()}  # 0 - left hand, 1 - right hand

        logger.debug("Computing contact point traj and diff")
        start_obj_loc = {"position": cropped_overall_traj_poses[0], "quaternion": cropped_overall_traj_quats[0]}
        start_positioned_obj_verts = self.apply_location(obj_verts, start_obj_loc)
        all_dists = []
        for hand_label in interaction_interval.interacting_hands_labels:
            contact_point = self.detect_contact_point(start_positioned_obj_verts, smpl_hands_traj[hand_label][0])
            cropped_overall_traj_rots = Rotation.from_quat(np.roll(cropped_overall_traj_quats, -1, axis=1))
            contact_point_traj = cropped_overall_traj_rots.apply(contact_point) + cropped_overall_traj_poses

            dists = np.linalg.norm(contact_point_traj - smpl_hands_traj[hand_label], axis=1)
            all_dists.append(dists)

        return np.stack(all_dists, axis=1).mean(1)

    def compute_contact_point2hand_mean_dist_single_interval(self, smpl_motion_seq, smpl_timestamps, overall_traj_poses, overall_traj_quats,
            overall_traj_times, obj_verts, interaction_interval: InteractionInterval):
        smpl_interval_start = find_closest_before(smpl_timestamps, interaction_interval.start_time)
        smpl_interval_end = find_closest_before(smpl_timestamps, interaction_interval.end_time)
        cropped_smpl_motion_seq = smpl_motion_seq[smpl_interval_start:smpl_interval_end]
        cropped_smpl_timestamps = smpl_timestamps[smpl_interval_start:smpl_interval_end]
        inds = []
        for ts in cropped_smpl_timestamps:
            ind = find_closest_before(overall_traj_times, ts)
            if ind is None:
                logger.warning(f"Couldn't find object traj index before ts {ts}, searching after")
                ind = find_closest_after(overall_traj_times, ts)
            inds.append(ind)
        inds = np.asarray(inds)
        cropped_overall_traj_times = overall_traj_times[inds]
        cropped_overall_traj_poses = overall_traj_poses[inds]
        cropped_overall_traj_quats = overall_traj_quats[inds]

        logger.debug("Preparing SMPL params")
        smpl_output_interval = self.get_smpl_output_interval(cropped_smpl_motion_seq)
        logger.debug("Processing hand trajectories")
        smpl_hands_traj = {hand_name: smpl_output_interval.joints[:, SMPL_HANDS_JOINTS_ID[hand_name]].cpu().numpy() for hand_name in
                           HAND_NAMES2INDS.keys()}  # 0 - left hand, 1 - right hand

        logger.debug("Computing contact point traj and diff")
        start_obj_loc = {"position": cropped_overall_traj_poses[0], "quaternion": cropped_overall_traj_quats[0]}
        start_positioned_obj_verts = self.apply_location(obj_verts, start_obj_loc)
        all_dists = []
        for hand_label in interaction_interval.interacting_hands_labels:
            if self.config.contact_metric_recompute_contact_at_each_step:
                logger.debug("Running contact point redetection at each step")
                contact_point_traj = np.zeros((len(cropped_overall_traj_poses), 3))
                for traj_ind, (traj_pose, traj_quat) in enumerate(zip(cropped_overall_traj_poses, cropped_overall_traj_quats)):
                    loc = {"position": traj_pose, "quaternion": traj_quat}
                    pos_obj_verts = self.apply_location(obj_verts, loc)
                    contact_point = self.detect_contact_point(pos_obj_verts, smpl_hands_traj[hand_label][traj_ind])
                    contact_point_traj[traj_ind, :] = contact_point
            else:
                contact_point = self.detect_contact_point(start_positioned_obj_verts, smpl_hands_traj[hand_label][0])
                contact_point_unposed = Rotation.from_quat(np.roll(cropped_overall_traj_quats[0], -1)).inv().apply(
                    contact_point - cropped_overall_traj_poses[0])
                cropped_overall_traj_rots = Rotation.from_quat(np.roll(cropped_overall_traj_quats, -1, axis=1))
                contact_point_traj = cropped_overall_traj_rots.apply(contact_point_unposed) + cropped_overall_traj_poses

            dists = np.linalg.norm(contact_point_traj - smpl_hands_traj[hand_label], axis=1)
            all_dists.append(dists)

        return np.stack(all_dists, axis=1).mean(1)

    def compute_contact_point2hand_mean_dist(self, smpl_motion_seq, smpl_timestamps, overall_traj_poses, overall_traj_quats, overall_traj_times,
            obj_verts, interaction_timeline: InteractionTimeline):
        all_dists = []
        for interaction_interval in interaction_timeline:
            if not interaction_interval.is_noop:
                dists = self.compute_contact_point2hand_mean_dist_single_interval(smpl_motion_seq, smpl_timestamps, overall_traj_poses,
                                                                                  overall_traj_quats, overall_traj_times, obj_verts,
                                                                                  interaction_interval)
                all_dists.append(dists)
        all_dists = np.concatenate(all_dists, axis=0)
        logger.debug(f"Computed contact dists shape: {all_dists.shape}")
        return float(np.mean(all_dists))

    def infer_multi_motion(self, sequence_interval, smpl_motions_path: Path, predicted_contacts_pathdir: Path, object_positions_pathdir: Path,
            smpl_betas, smpl_gender, obj_models_dir: Path, gt_contacts_pathdir: Path = None):
        sequence_interval = np.asarray(sequence_interval)
        smpl_betas = np.asarray(smpl_betas)
        # Building all interaction intervals
        obj_names = [x.stem for x in object_positions_pathdir.glob("*.json") if x.stem in self.config.object_motion_params]
        start_time = sequence_interval[0]
        objects_info = {}
        curr_obj_locations = {}
        logger.debug(f"Loading {len(obj_names)} objects")
        for objname in obj_names:
            obj_model_path = obj_models_dir / f"{objname}.ply"
            object_positions_path = object_positions_pathdir / f"{objname}.json"

            obj_model = trimesh.load(obj_model_path)
            obj_positions_info = json.load(object_positions_path.open())
            obj_info = ObjectInfo(obj_model, obj_positions_info, obj_name=objname)
            objects_info[objname] = obj_info
            curr_obj_locations[objname] = obj_info.get_obj_location(start_time)

        objectwise_gt_interaction_timelines = {}
        for objname in obj_names:
            gt_contacts_path = gt_contacts_pathdir / f"{objname}.json"
            if gt_contacts_path.is_file():
                gt_contacts_info = json.load(gt_contacts_path.open())
                gt_contacts = gt_contacts_info["interactions"]
                gt_interaction_timeline = InteractionTimeline.from_gt_contacts(gt_contacts)
                objectwise_gt_interaction_timelines[objname] = gt_interaction_timeline
            else:
                logger.warning(f"No GT interaction intervals for {objname}")
                objectwise_gt_interaction_timelines[objname] = None
        objectwise_gt_interaction_multitimeline = MultiobjectInteractionTimeline(objectwise_gt_interaction_timelines)

        if self.use_gt_contact_intervals:
            logger.critical("Using GT contacts")
            interaction_multitimeline = objectwise_gt_interaction_multitimeline
        else:

            predictor_type_names = [os.path.splitext(x.stem)[0] for x in predicted_contacts_pathdir.glob("*.json.zip")]
            if self.config.contact_predictor_classes is not None:
                predictor_type_names = [x for x in predictor_type_names if x in self.config.contact_predictor_classes]
            logger.debug(f"Loading {len(predictor_type_names)} predicted contacts")
            intwise_interaction_timeline = {}
            for predictor_type_name in predictor_type_names:
                predicted_contacts_path = predicted_contacts_pathdir / f"{predictor_type_name}.json.zip"
                contact_preds_dict = zipjson.load(open(predicted_contacts_path, "rb"))
                contact_preds = np.stack([contact_preds_dict["sequence"]["left_hand"], contact_preds_dict["sequence"]["right_hand"]], axis=1)
                contact_ts = np.asarray(contact_preds_dict["timestamps"])
                logger.info("Processing contacts")

                interaction_timeline = self.compute_interaction_intervals(sequence_interval, contact_preds, contact_ts,
                                                                          interaction_type="any", contact_labeling_thresh=
                                                                          self.config.contact_predictor_classes[predictor_type_name]["thresh"])
                intwise_interaction_timeline[predictor_type_name] = interaction_timeline
            interaction_multitimeline = MultiobjectInteractionTimeline(intwise_interaction_timeline)

        logger.debug("Loading SMPL motions")
        _, smpl_motion_seq, smpl_timestamps, smpl_gender, smpl_betas = self.load_smpl_motion(smpl_motions_path, smpl_gender, smpl_betas)

        if len(smpl_motion_seq) > 0 and ('pose' in smpl_motion_seq[0] or 'shape' in smpl_motion_seq[0]):
            smpl_motion_seq = self.convert_to_new_motion_format(smpl_motion_seq, smpl_timestamps, model_type=self.model_type)

        logger.debug("Initializing model")
        self._init_model(self.model_type, smpl_gender, smpl_compatible=True, flat_hand_mean=self.straight_hands)

        logger.info("Processing multitimeline")
        curr_time = sequence_interval[0]
        total_processed_intervals = 0
        objects_traj_lists = defaultdict(list)
        for objname in obj_names:
            objloc = curr_obj_locations[objname]
            objects_traj_lists[objname].append(ObjectTrajectory([objloc.translation], [objloc.quaternion], [curr_time]))
        ignore_set = set()
        prev_smpl_frame_ind = None
        while curr_time < sequence_interval[1]:
            curr_interaction_interval, curr_interaction_timeline_name = interaction_multitimeline.find_closest_active_interval_after(curr_time,
                                                                                                                                     ignore_set=ignore_set)
            if curr_interaction_interval is None:
                break
            curr_time = curr_interaction_interval.start_time
            if self.use_gt_contact_intervals:
                object_name = curr_interaction_timeline_name
            else:
                predictor_class = curr_interaction_timeline_name

                smpl_frame_ind = find_closest_before(smpl_timestamps, curr_time)
                object_name = self.get_object_from_predictorclass(predictor_class, curr_obj_locations, objects_info, smpl_motion_seq[smpl_frame_ind],
                                                                  curr_interaction_interval.interacting_hands_labels,
                                                                  dist_thresh=self.config.object_interaction_dist_thresh if total_processed_intervals > 0 else self.config.first_object_interaction_dist_thresh)

            if object_name is None:
                logger.info(
                    f"No object for interval ({curr_interaction_interval.start_time}, {curr_interaction_interval.end_time}), type '{curr_interaction_timeline_name}', hands {curr_interaction_interval.interacting_hands_labels}, skipping")
                ignore_set.add(curr_interaction_interval)
                continue
            else:
                logger.info(
                    f"Processing interval ({curr_interaction_interval.start_time}, {curr_interaction_interval.end_time}), type '{curr_interaction_timeline_name}' with object {object_name}, hands {curr_interaction_interval.interacting_hands_labels}")

            prev_smpl_motion_seq = smpl_motion_seq

            smpl_motion_seq, res_curr_object_traj = self.infer_motion_single_interval(smpl_motion_seq, smpl_timestamps, curr_interaction_interval,
                                                                                      curr_obj_locations[object_name], objects_info[object_name],
                                                                                      relocate_body_to_start=((
                                                                                                                      total_processed_intervals == 0) or self.config.relocate_body_to_each_object))
            if self.config.relocate_body_to_each_object:
                smpl_frame_ind = find_closest_before(smpl_timestamps, curr_time)
                if prev_smpl_frame_ind is not None:
                    sep_ind = (smpl_frame_ind + prev_smpl_frame_ind) // 2
                    smpl_motion_seq = prev_smpl_motion_seq[:sep_ind] + smpl_motion_seq[sep_ind:]

            if len(res_curr_object_traj) > 0:
                curr_obj_locations[object_name] = res_curr_object_traj[-1]
                objects_traj_lists[object_name].append(res_curr_object_traj)
            curr_time = curr_interaction_interval.end_time
            prev_smpl_frame_ind = find_closest_after(smpl_timestamps, curr_time)
            logger.debug(f"Curr time is {curr_time:.2f}")
            total_processed_intervals += 1

        logger.info(f"Processed {total_processed_intervals} intervals")

        objwise_traj = {}

        for objname in obj_names:
            cat_traj = ObjectTrajectory.cat_trajectories(objects_traj_lists[objname])
            res_obj_traj = {"version": 2, "type": "sequential", "translations": cat_traj.translations, "quaternions": cat_traj.quaternions,
                            "timestamps": cat_traj.times}
            objwise_traj[objname] = res_obj_traj

        res_smpl_motions = {"version": 2, "global": {"betas": smpl_betas.tolist(), "gender": smpl_gender}, "sequence": None}
        res_smpl_motions["sequence"] = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in params.items()} for params in
                                        smpl_motion_seq]
        return res_smpl_motions, objwise_traj, None

    def get_object_from_predictorclass(self, predictor_class, curr_obj_locations, objects_info, body_params, interacting_hands_labels, dist_thresh):
        closest_obj_name = None
        closest_obj_dist = None
        for object_name, curr_object_loc in curr_obj_locations.items():
            obj_motion_params = self.config.object_motion_params[object_name]
            if obj_motion_params["contact_predictor_class"] != predictor_class:
                continue
            if obj_motion_params["interaction_type"] == "two_handed" and len(interacting_hands_labels) < 2:
                continue
            obj_info = objects_info[object_name]

            smpl_verts, smpl_joints = self.get_smpl_verts_joints(body_params)
            hand_joints = smpl_joints[[SMPL_HANDS_JOINTS_ID[HAND_INDS2NAMES[hand_ind]] for hand_ind in range(2)], :]

            curr_contact_points_both_hands = np.asarray([obj_info.detect_contact_point(x, curr_object_loc) for x in hand_joints])
            curr_contact_points_dist = np.linalg.norm(hand_joints - curr_contact_points_both_hands, axis=1)
            if len(interacting_hands_labels) == 2:
                obj_in_contact = np.all(curr_contact_points_dist <= dist_thresh)
                obj_contact_dist = np.mean(curr_contact_points_dist)
            else:
                obj_contact_dist = curr_contact_points_dist[HAND_NAMES2INDS[interacting_hands_labels[0]]]
                obj_in_contact = obj_contact_dist <= dist_thresh
            if obj_in_contact and (closest_obj_dist is None or closest_obj_dist > obj_contact_dist):
                closest_obj_dist = obj_contact_dist
                closest_obj_name = object_name
        return closest_obj_name

    def load_smpl_motion(self, smpl_motions_path, smpl_gender, smpl_betas):
        if self.motion_format_version < 2:
            smpl_motions = json.load(open(smpl_motions_path))
            smpl_timestamps = np.array([x['time'] for x in smpl_motions])
            smpl_motion_seq = [{k: np.array(v) for k, v in x.items()} for x in smpl_motions]
        else:
            smpl_motions = zipjson.load(open(smpl_motions_path, 'rb'))

            if not np.allclose(seq_smpl_betas := np.asarray(smpl_motions["global"]["betas"]), smpl_betas):
                logger.warning(f"Betas stored in the sequence are different than supplied: expected {smpl_betas}, got {seq_smpl_betas}")
            smpl_betas = seq_smpl_betas
            if smpl_gender != (seq_smpl_gender := smpl_motions["global"]["gender"]):
                logger.warning(
                    f"Gender stored in the sequence is different than supplied: expected {smpl_gender}, got {seq_smpl_gender}")
            smpl_gender = seq_smpl_gender

            smpl_motion_seq = [{k: np.array(v) for k, v in x.items()} for x in smpl_motions["sequence"]]
            smpl_timestamps = np.array([x['time'] for x in smpl_motion_seq])
            hand_info_present = "right_hand_pose" in smpl_motion_seq[0]
            logger.debug(f"Hand info is {hand_info_present}")
        return smpl_motions, smpl_motion_seq, smpl_timestamps, smpl_gender, smpl_betas

    def infer_motion_single_interval(self, smpl_motion_seq, smpl_timestamps, interaction_interval, starting_object_loc: ObjectLocation,
            obj_info: ObjectInfo, relocate_body_to_start=False):
        motion_params = self.config.object_motion_params[obj_info.name]
        motion_type = motion_params["motion_type"]
        # interaction_type = motion_params["interaction_type"]
        obj_hinge = obj_info.hinge

        if not self.config.no_traj_inference:

            if relocate_body_to_start:
                logger.info("Relocating the body to contact at start")
                positioned_obj_vertices = obj_info.get_localized_verts(starting_object_loc)
                smpl_motion_seq = self.relocate_body_to_object(positioned_obj_vertices, smpl_motion_seq, smpl_timestamps,
                                                               interaction_interval.interacting_hands_labels,
                                                               interaction_interval.start_time, self.start_reloc_params["traj_mod_type"],
                                                               self.start_reloc_params["traj_mod_window"],
                                                               self.start_reloc_params["iters_count"], self.start_reloc_params["flextape_opt_params"])
            last_predicted_objloc = None
            logger.debug("Processing interval")
            curr_traj_positions, curr_traj_quats, curr_traj_timestamps, hands_traj_dict = self.infer_traj_single_interaction_interval(
                interaction_interval,
                smpl_motion_seq, smpl_timestamps,
                last_obj_location=starting_object_loc, infer_rotation_from_hand=motion_type == "free", snap_obj_to_hand=motion_type == "free")
            if not interaction_interval.is_noop and len(curr_traj_positions) > 0:
                contact_point = None
                # if motion_type == "hinge":
                #     contact_point = self.detect_contact_point(positioned_obj_vertices, hands_traj_dict)
                curr_traj_positions, curr_traj_quats, curr_traj_timestamps, res_hands_traj = \
                    self.adapt_traj_to_motion_type(curr_traj_positions, curr_traj_quats, curr_traj_timestamps, motion_type, hinge_xyz=obj_hinge,
                                                   contact_point=contact_point, hands_traj=hands_traj_dict)
                last_obj_location = ObjectLocation(curr_traj_positions[-1], curr_traj_quats[-1])
                positioned_obj_vertices = obj_info.get_localized_verts(last_obj_location)
                last_predicted_objloc = last_obj_location
            else:
                logger.debug("Interval is NO OP, skipping additional procedures")
            overall_traj_poses = curr_traj_positions
            overall_traj_quats = curr_traj_quats
            overall_traj_times = curr_traj_timestamps
        else:
            logger.warning("no_traj_inference is on, skipping inference")
            if self.config.linear_obj_interpolation:
                raise NotImplementedError()
            else:
                overall_traj_poses = np.asarray(starting_object_loc["position"]).reshape((1, 3))
                overall_traj_quats = np.asarray(starting_object_loc["quaternion"]).reshape((1, 4))
                overall_traj_times = np.zeros(1)
                last_predicted_objloc = starting_object_loc

        return smpl_motion_seq, ObjectTrajectory(overall_traj_poses, overall_traj_quats, overall_traj_times)

    def infer_motion(self, sequence_interval, smpl_motions_path, predicted_contacts_path, object_positions_path, smpl_betas, smpl_gender,
            interaction_type, motion_type, obj_model_path, gt_contacts_path=None, predicted_object_positions_path=None):
        sequence_interval = np.asarray(sequence_interval)
        smpl_betas = np.asarray(smpl_betas)
        logger.debug("Loading object")
        obj_model = trimesh.load(obj_model_path)
        obj_vertices = np.asarray(obj_model.vertices)
        obj_position_info = json.load(object_positions_path.open())
        obj_center = np.asarray(obj_position_info["center"])
        obj_hinge = np.asarray(obj_position_info["hinge"]) if "hinge" in obj_position_info else None
        obj_anchor_positions = np.asarray([x["position"] for x in obj_position_info["locations"]])
        obj_anchor_quaternions = np.asarray([x["quaternion"] for x in obj_position_info["locations"]])
        obj_anchor_timestamps = np.asarray([x["time"] for x in obj_position_info["locations"]])

        if gt_contacts_path is not None and gt_contacts_path.is_file():
            gt_contacts_info = json.load(gt_contacts_path.open())
            gt_contacts = gt_contacts_info["interactions"]
            gt_interaction_timeline = InteractionTimeline.from_gt_contacts(gt_contacts)
        else:
            logger.warning("No GT interaction intervals for the object, contact metrics computation will fail")
            gt_interaction_timeline = None

        if predicted_object_positions_path is not None:
            object_predicted_positions_info = json.load(predicted_object_positions_path.open())
            object_predicted_positions = np.asarray([x["position"] for x in object_predicted_positions_info["locations"]])
            object_predicted_quaternions = np.asarray([x["quaternion"] for x in object_predicted_positions_info["locations"]])
            object_predicted_timestamps = np.asarray([x["time"] for x in object_predicted_positions_info["locations"]])
        else:
            object_predicted_positions_info = None

        if self.use_gt_contact_intervals:
            logger.critical("Using GT contacts")
            interaction_timeline = gt_interaction_timeline
        else:

            logger.debug("Loading contacts")
            contact_preds_dict = zipjson.load(open(predicted_contacts_path, "rb"))
            contact_preds = np.stack([contact_preds_dict["sequence"]["left_hand"], contact_preds_dict["sequence"]["right_hand"]], axis=1)
            contact_ts = np.asarray(contact_preds_dict["timestamps"])
            logger.info("Processing contacts")

            interaction_timeline = self.compute_interaction_intervals(sequence_interval, contact_preds, contact_ts,
                                                                      interaction_type=interaction_type,
                                                                      contact_labeling_thresh=self.contact_labeling_thresh)
        logger.info(f"Working with {len(interaction_timeline)} interaction intervals")

        smpl_motions, smpl_motion_seq, smpl_timestamps, smpl_gender, smpl_betas = self.load_smpl_motion(smpl_motions_path, smpl_gender, smpl_betas)

        if len(smpl_motion_seq) > 0 and ('pose' in smpl_motion_seq[0] or 'shape' in smpl_motion_seq[0]):
            smpl_motion_seq = self.convert_to_new_motion_format(smpl_motion_seq, smpl_timestamps, model_type=self.model_type)

        logger.debug("Initializing model")
        self._init_model(self.model_type, smpl_gender, smpl_compatible=True, flat_hand_mean=self.straight_hands)

        starting_interaction_interval = interaction_timeline[0]
        final_interaction_inverval = interaction_timeline[-1]

        starting_obj_loc_ind = find_closest_before(obj_anchor_timestamps, starting_interaction_interval.start_time)
        if starting_obj_loc_ind is None:
            logger.warning("No known position before the start, searching after")
            starting_obj_loc_ind = find_closest_after(obj_anchor_timestamps, starting_interaction_interval.start_time)
            if starting_obj_loc_ind is None:
                raise ValueError("No anchor object positions to start")
        ending_obj_loc_ind = find_closest_after(obj_anchor_timestamps, final_interaction_inverval.end_time)
        if ending_obj_loc_ind is None:
            logger.warning("No known position after the end, searching before")
            ending_obj_loc_ind = find_closest_before(obj_anchor_timestamps, final_interaction_inverval.end_time)
        if starting_obj_loc_ind == ending_obj_loc_ind:
            logger.critical("Object localization uses the same object location at the start and in the end of interaction")
            raise Exception("Same objloc")

        starting_obj_location = last_obj_location = {"position": obj_anchor_positions[starting_obj_loc_ind],
                                                     "quaternion": obj_anchor_quaternions[starting_obj_loc_ind]}
        ending_gt_obj_location = {"position": obj_anchor_positions[ending_obj_loc_ind],
                                  "quaternion": obj_anchor_quaternions[ending_obj_loc_ind]}
        positioned_obj_vertices = self.apply_location(obj_vertices, starting_obj_location)

        if not self.config.no_traj_inference:

            if self.relocate_body_to_start:
                logger.info("Relocating the body at start")
                smpl_motion_seq = self.relocate_body_to_object(positioned_obj_vertices, smpl_motion_seq, smpl_timestamps,
                                                               starting_interaction_interval.interacting_hands_labels,
                                                               starting_interaction_interval.start_time, self.start_reloc_params["traj_mod_type"],
                                                               self.start_reloc_params["traj_mod_window"],
                                                               self.start_reloc_params["iters_count"], self.start_reloc_params["flextape_opt_params"])

            overall_traj_poses = []
            overall_traj_quats = []
            overall_traj_times = []
            last_predicted_objloc = None
            for interaction_interval_ind, interaction_interval in enumerate(interaction_timeline):
                logger.info(f"Processing interaction interval {interaction_interval_ind + 1}/{len(interaction_timeline)}")
                curr_traj_positions, curr_traj_quats, curr_traj_timestamps, hands_traj_dict = self.infer_traj_single_interaction_interval(
                    interaction_interval,
                    smpl_motion_seq, smpl_timestamps,
                    last_obj_location=last_obj_location)
                if not interaction_interval.is_noop and len(curr_traj_positions) > 0:
                    contact_point = None
                    curr_traj_positions, curr_traj_quats, curr_traj_timestamps, res_hands_traj = \
                        self.adapt_traj_to_motion_type(curr_traj_positions, curr_traj_quats, curr_traj_timestamps, motion_type, hinge_xyz=obj_hinge,
                                                       contact_point=contact_point, hands_traj=hands_traj_dict)
                    last_obj_location = {"position": curr_traj_positions[-1], "quaternion": curr_traj_quats[-1]}
                    positioned_obj_vertices = self.apply_location(obj_vertices, last_obj_location)
                    last_predicted_objloc = last_obj_location
                else:
                    logger.debug("Interval is NO OP, skipping additional procedures")
                overall_traj_poses.append(curr_traj_positions)
                overall_traj_quats.append(curr_traj_quats)
                overall_traj_times.append(curr_traj_timestamps)
            overall_traj_poses = np.concatenate(overall_traj_poses, axis=0)
            overall_traj_quats = np.concatenate(overall_traj_quats, axis=0)
            overall_traj_times = np.concatenate(overall_traj_times, axis=0)
        else:
            logger.warning("no_traj_inference is on, skipping inference")
            if self.config.linear_obj_interpolation:
                logger.info("Performing linear int")
                if self.config.use_predicted_object_positions:
                    assert object_predicted_positions_info is not None, "No predicted object positions supplied"
                    object_predicted_positions, object_predicted_quaternions, object_predicted_timestamps, _ = \
                        self.adapt_traj_to_motion_type(object_predicted_positions, object_predicted_quaternions, object_predicted_timestamps,
                                                       motion_type,
                                                       hinge_xyz=obj_hinge, contact_point=None, hands_traj=None)
                    prev_ts = starting_interaction_interval.start_time
                    last_predicted_objloc = starting_obj_location
                    overall_traj_poses = []
                    overall_traj_quats = []
                    overall_traj_times = []
                    for ind, object_predicted_ts in enumerate(object_predicted_timestamps):
                        if object_predicted_ts <= starting_interaction_interval.start_time:
                            continue
                        if object_predicted_ts >= final_interaction_inverval.end_time:
                            break
                        interaction_interval = InteractionInterval(prev_ts, object_predicted_ts, [])

                        curr_obj_location = {"position": object_predicted_positions[ind],
                                             "quaternion": object_predicted_quaternions[ind]}
                        curr_traj_positions, curr_traj_quats, curr_traj_timestamps, res_hands_traj = \
                            self.linear_objtraj_interpolation(smpl_timestamps, interaction_interval, last_predicted_objloc, curr_obj_location,
                                                              obj_center if (motion_type != "hinge") else obj_hinge)
                        prev_ts = object_predicted_ts
                        last_predicted_objloc = curr_obj_location
                        if curr_traj_positions is None:
                            continue
                        curr_traj_positions, curr_traj_quats, curr_traj_timestamps, res_hands_traj = \
                            self.adapt_traj_to_motion_type(curr_traj_positions, curr_traj_quats, curr_traj_timestamps, motion_type,
                                                           hinge_xyz=obj_hinge,
                                                           contact_point=None, hands_traj=res_hands_traj)
                        overall_traj_poses.append(curr_traj_positions)
                        overall_traj_quats.append(curr_traj_quats)
                        overall_traj_times.append(curr_traj_timestamps)

                    if len(overall_traj_poses) == 0:
                        logger.warning("No valid interaction intervals detected")
                        overall_traj_poses = np.asarray(starting_obj_location["position"]).reshape((1, 3))
                        overall_traj_quats = np.asarray(starting_obj_location["quaternion"]).reshape((1, 4))
                        overall_traj_times = np.zeros(1)
                    else:
                        overall_traj_poses = np.concatenate(overall_traj_poses, axis=0)
                        overall_traj_quats = np.concatenate(overall_traj_quats, axis=0)
                        overall_traj_times = np.concatenate(overall_traj_times, axis=0)
                else:
                    interaction_interval = InteractionInterval(starting_interaction_interval.start_time, final_interaction_inverval.end_time, [])
                    overall_traj_poses, overall_traj_quats, overall_traj_times, res_hands_traj = \
                        self.linear_objtraj_interpolation(smpl_timestamps, interaction_interval, starting_obj_location, ending_gt_obj_location,
                                                          obj_center if (motion_type != "hinge") else obj_hinge)
                    last_predicted_objloc = ending_gt_obj_location
            else:
                if self.config.use_predicted_object_positions:
                    raise NotImplementedError()
                overall_traj_poses = np.asarray(starting_obj_location["position"]).reshape((1, 3))
                overall_traj_quats = np.asarray(starting_obj_location["quaternion"]).reshape((1, 4))
                overall_traj_times = np.zeros(1)
                last_predicted_objloc = starting_obj_location

        res_metrics = {}
        if self.config.compute_endpoint_metrics:
            logger.info("Computing endpoint metrics")
            res_metrics.update({"endpoint": self.compute_metrics_objloc(obj_vertices, last_predicted_objloc, ending_gt_obj_location)})
        if self.config.compute_contact_metrics:
            if gt_interaction_timeline is None:
                logger.warning("No GT interaction intervals, skipping contact metrics")
            else:
                logger.info("Computing contact metrics")
                res_metrics.update({"contact": {
                    "distance2hand": self.compute_contact_point2hand_mean_dist(smpl_motion_seq, smpl_timestamps, overall_traj_poses, overall_traj_quats,
                                                                               overall_traj_times, obj_vertices, gt_interaction_timeline)}})

        res_obj_traj = {"version": 2, "type": "sequential", "translations": overall_traj_poses, "quaternions": overall_traj_quats,
                        "timestamps": overall_traj_times}

        res_smpl_motions = {"version": 2, "global": {"betas": smpl_betas.tolist(), "gender": smpl_gender}, "sequence": None}
        res_smpl_motions["sequence"] = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in params.items()} for params in
                                        smpl_motion_seq]

        return res_smpl_motions, res_obj_traj, res_metrics
