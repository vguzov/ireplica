import toml
import dataclasses
import dacite
import json
import zipjson
import numpy as np
from pathlib import Path
from loguru import logger
from argparse import ArgumentParser
from ireplica import IReplicaTracker, IReplicaConfig, config_to_dict
from ireplica.utils import parse_seqname

if __name__ == "__main__":
    # Adding arguments from config file to allow overriding them from command line
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='config file path')
    parser.add_argument("--smpl_motions_dir", type=Path)
    parser.add_argument("--predicted_contacts_dir", type=Path)
    parser.add_argument("--object_positions_dir", type=Path)
    parser.add_argument("--gt_contacts_dir", type=Path)
    parser.add_argument("--object_models_dir", type=Path)
    parser.add_argument("--output_smpl_motions_dir", type=Path)
    parser.add_argument("--output_object_positions_dir", type=Path)
    parser.add_argument("--output_configs_dir", type=Path)
    parser.add_argument("--output_metrics_dir", type=Path)
    parser.add_argument("--object_name")
    parser.add_argument("--seqname")
    parser.add_argument("--expname")
    parser.add_argument("--use_end_objpose", action="store_true", default=None)
    parser.add_argument("--use_gt_contact_intervals", action="store_true", default=None)
    parser.add_argument("--relocate_body_to_start", action="store_true", default=None)
    parser.add_argument("--no_relocate_body_to_start", action="store_false", dest="relocate_body_to_start")
    parser.add_argument("--straight_hands", action="store_true", default=None)
    parser.add_argument("--contact_smoothing_thresh", type=float)
    parser.add_argument("--smpl_root", type=Path)
    parser.add_argument("--sequence_interval", nargs=2, type=float)
    parser.add_argument("--smpl_shapes_dir", type=Path)
    parser.add_argument("--interaction_type", choices=["two_handed", "single_handed"])
    parser.add_argument("--motion_type", choices=["along_floor", "hinge", "free"])
    parser.add_argument("--motion_format_version", type=int)
    parser.add_argument("--no_traj_inference", action="store_true", default=None)
    parser.add_argument("--linear_obj_interpolation", action="store_true", default=None)
    parser.add_argument("--use_predicted_object_positions", action="store_true", default=None)

    args = parser.parse_args()

    config_args_dict = toml.load(open(args.config))
    for param_group in ["object_motion_params", "contact_predictor_classes"]:
        if param_group not in config_args_dict:
            config_args_dict[param_group] = {}
    config_args = dacite.from_dict(data_class=IReplicaConfig, data=config_args_dict, config=dacite.Config(cast=[Path]))
    for config_key in dataclasses.asdict(config_args).keys():
        if hasattr(args, config_key) and getattr(args, config_key) is not None:
            setattr(config_args, config_key, getattr(args, config_key))
    args = config_args

    logger.info(args)

    tracker = IReplicaTracker(args)

    smpl_motions_path = args.smpl_motions_dir / f"{args.seqname}{'.json' if args.motion_format_version < 2 else '.json.zip'}"
    parsed_name = parse_seqname(args.seqname)
    body_params = json.load(open(args.smpl_shapes_dir / f"SUB{parsed_name['subject_id']}.json"))
    gender = body_params["gender"]
    smpl_betas = np.asarray(body_params["betas"])

    args.output_smpl_motions_dir.mkdir(parents=True, exist_ok=True)
    outname = f"{args.seqname}" if (args.expname is None or len(args.expname) == 0) else f"{args.seqname}.{args.expname}"
    output_smpl_motions_path = args.output_smpl_motions_dir / f"{outname}.json.zip"
    output_config_path = args.output_configs_dir / f"{outname}.toml"
    args.output_configs_dir.mkdir(parents=True, exist_ok=True)

    if args.sequence_type == "single_action":

        predicted_contacts_path = args.predicted_contacts_dir / f"{args.seqname}.json.zip"
        object_positions_path = args.object_positions_dir / f"{args.seqname}.json"
        predicted_object_positions_path = args.predicted_object_positions_dir / f"{args.seqname}.json"
        if not predicted_object_positions_path.is_file():
            predicted_object_positions_path = None
        gt_contacts_path = args.gt_contacts_dir / f"{args.seqname}.json"
        object_model_path = args.object_models_dir / f"{args.object_name}.ply"

        args.output_object_positions_dir.mkdir(parents=True, exist_ok=True)

        output_object_positions_path = args.output_object_positions_dir / f"{outname}.npz"

        if args.output_metrics_dir is not None:
            args.output_metrics_dir.mkdir(parents=True, exist_ok=True)
            output_metrics_path = args.output_metrics_dir / f"{outname}.json"
        else:
            output_metrics_path = None

        res_smpl_motions, res_obj_traj, metrics = tracker.infer_motion(args.sequence_interval, smpl_motions_path, predicted_contacts_path,
                                                                       object_positions_path, smpl_betas, gender,
                                                                       args.interaction_type, args.motion_type, object_model_path, gt_contacts_path,
                                                                       predicted_object_positions_path=predicted_object_positions_path)

        logger.info("Saving results to disk")
        np.savez_compressed(output_object_positions_path, **res_obj_traj)
        if output_metrics_path is not None:
            json.dump(metrics, output_metrics_path.open("w"))
    elif args.sequence_type == "multi_action":
        predicted_contacts_pathdir = args.predicted_contacts_dir / f"{args.seqname}"
        object_positions_pathdir = args.object_positions_dir / f"{args.seqname}"
        gt_contacts_pathdir = args.gt_contacts_dir / f"{args.seqname}"
        object_models_dir = args.object_models_dir

        output_object_positions_pathdir = args.output_object_positions_dir / f"{outname}"
        output_object_positions_pathdir.mkdir(parents=True, exist_ok=True)

        res_smpl_motions, objwise_traj, _ = tracker.infer_multi_motion(args.sequence_interval, smpl_motions_path, predicted_contacts_pathdir,
                                                                       object_positions_pathdir, smpl_betas, gender, object_models_dir,
                                                                       gt_contacts_pathdir)
        logger.info("Saving results to disk")
        objinfos_dict = {}
        for objname, res_obj_traj in objwise_traj.items():
            output_object_positions_path = output_object_positions_pathdir / f"{objname}.npz"
            np.savez_compressed(output_object_positions_path, **res_obj_traj)
            objinfos_dict[objname] = {"relpath": f"{objname}.npz"}
        for objname in args.absent_objects:
            objinfos_dict[objname] = {"relpath": f"{objname}.npz", "present": False}
        output_object_positions_json_path = output_object_positions_pathdir / f"info.json"
        json.dump({"version": 3, "objects": objinfos_dict}, output_object_positions_json_path.open("w"), indent=2)
    else:
        raise NotImplementedError(f"Unknown sequence type '{args.sequence_type}'")
    zipjson.dump(res_smpl_motions, output_smpl_motions_path.open("wb"))
    logger.info("Saving config")
    toml.dump(config_to_dict(args), output_config_path.open("w"))
    logger.info(f"Done, saved as {outname}")