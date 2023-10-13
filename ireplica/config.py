import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class IReplicaConfig:
    smpl_motions_dir: Path
    predicted_contacts_dir: Path
    object_positions_dir: Path
    predicted_object_positions_dir: Path
    gt_contacts_dir: Path
    object_models_dir: Path
    output_smpl_motions_dir: Path
    output_object_positions_dir: Path
    output_metrics_dir: Path
    output_configs_dir: Path
    seqname: str
    expname: str
    use_end_objpose: bool
    use_gt_contact_intervals: bool
    relocate_body_to_start: bool
    straight_hands: bool
    contact_smoothing_thresh: float
    smpl_root: Path
    sequence_interval: List[float]
    smpl_shapes_dir: Path
    start_reloc_params: dict
    object_motion_params: dict
    object_interaction_dist_thresh: float = 0.2
    first_object_interaction_dist_thresh: float = 0.5
    compute_endpoint_metrics: bool = True
    compute_contact_metrics: bool = True
    contact_metric_recompute_contact_at_each_step: bool = True
    motion_format_version: int = 2
    contact_labeling_thresh: float = 0.5
    # HPS and HPS + Interpolation mode switches
    no_traj_inference: bool = False         # HPS
    linear_obj_interpolation: bool = False  # + Interpolation
    sequence_type: str = "single_action"
    interaction_type: str = None
    motion_type: str = None
    object_name: str = None
    contact_predictor_classes: dict = None
    relocate_body_to_each_object: bool = False
    use_predicted_object_positions: bool = False
    absent_objects: List[str] = ()


def config_to_dict(config: IReplicaConfig):
    res_dict = {}
    config_dict = dataclasses.asdict(config)
    for param_name, param_val in config_dict.items():
        if isinstance(param_val, Path):
            param_val = str(param_val)
        res_dict[param_name] = param_val
    return res_dict
