"""
    Class to represent the HOV-SG graph
"""

import os
import copy
from typing import Any, Dict, List, Set, Tuple, Union
from pathlib import Path

import cv2
from omegaconf import DictConfig
import numpy as np
import open3d as o3d

import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree

from scipy.ndimage import binary_erosion, median_filter
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d

from sklearn.cluster import DBSCAN
from tqdm import tqdm
import networkx as nx

import torch
import open_clip
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from hovsg.graph.object import Object
from hovsg.graph.room import Room
from hovsg.graph.floor import Floor

from hovsg.dataloader.hm3dsem import HM3DSemDataset
from hovsg.dataloader.scannet import ScannetDataset
from hovsg.dataloader.replica import ReplicaDataset
from hovsg.dataloader.real_robot import RealRobotDataset

from hovsg.utils.clip_utils import get_img_feats, get_text_feats_multiple_templates
from hovsg.models.sam_clip_feats_extractor import extract_feats_per_pixel
from hovsg.utils.graph_utils import (
    seq_merge,
    pcd_denoise_dbscan,
    feats_denoise_dbscan,
    distance_transform,
    map_grid_to_point_cloud,
    compute_room_embeddings,
    find_intersection_share,
    hierarchical_merge,
)

from hovsg.graph.navigation_graph import NavigationGraph
from hovsg.utils.label_feats import get_label_feats
from hovsg.utils.constants import MATTERPORT_GT_LABELS, CLIP_DIM
from hovsg.utils.runtime_status import print_status
from hovsg.utils.llm_utils import (
    parse_floor_room_object_gpt35,
    parse_hier_query,
    infer_floor_id_from_query,
)

# pylint: disable=all


class Graph:
    """
    Class to represent the HOV-SG graph
    :param cfg: Config file
    :param inf_params: Inference parameters
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.full_pcd = o3d.geometry.PointCloud()
        self.mask_feats = []
        self.mask_feats_d = []
        self.mask_pcds = []
        self.mask_weights = []
        self.objects = []
        self.rooms = []
        self.floors = []
        self.full_feats_array = []
        self.frame_indices = []
        self.semantic_frame_indices = []
        self.graph = nx.Graph()
        self.graph.add_node(0, name="building", type="building")
        self.room_masks = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print_status(
            "RUNTIME_DEVICE",
            "OK",
            configured=self.cfg.main.get("device", "auto"),
            selected=self.device,
            cuda_available=torch.cuda.is_available(),
        )

        # load CLIP model
        if self.cfg.models.clip.type == "ViT-L/14@336px":
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14",
                pretrained=str(self.cfg.models.clip.checkpoint),
                device=self.device,
            )
            self.clip_feat_dim = CLIP_DIM["ViT-L-14"]
            # self.clip_feat_dim = constants.clip_feat_dim[self.cfg.models.clip.type]
        elif self.cfg.models.clip.type == "ViT-H-14":
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-H-14",
                pretrained=str(self.cfg.models.clip.checkpoint),
                device=self.device,
            )
            self.clip_feat_dim = CLIP_DIM["ViT-H-14"]
        self.clip_model.eval()
        print_status(
            "CLIP_MODEL",
            "OK",
            model=self.cfg.models.clip.type,
            checkpoint=self.cfg.models.clip.checkpoint,
            feature_dim=self.clip_feat_dim,
            device=self.device,
        )
        if not hasattr(self.cfg, "pipeline"):
            print("-- entering querying and evaluation mode")
            print_status("GRAPH_MODE", "INFO", mode="query_or_evaluation")
            return

        self.graph_tmp_folder = os.path.join(cfg.main.save_path, "tmp")
        if not os.path.exists(self.graph_tmp_folder):
            os.makedirs(self.graph_tmp_folder)
        print_status(
            "GRAPH_TEMP_DIRECTORY",
            "OK",
            path=self.graph_tmp_folder,
            exists=os.path.isdir(self.graph_tmp_folder),
        )

        # Loading a saved feature map only needs CLIP for room/object labels;
        # SAM is not used again.  Skipping it makes graph/navigation resume
        # substantially faster and avoids several GB of unnecessary VRAM.
        skip_sam_initialization = bool(
            self.cfg.pipeline.get("skip_sam_initialization", False)
        )
        self.sam = None
        self.mask_generator = None
        if skip_sam_initialization:
            print_status(
                "SAM_MODEL",
                "SKIP",
                reason="loading_saved_feature_map",
            )
        else:
            model_type = self.cfg.models.sam.type
            self.sam = sam_model_registry[model_type](
                checkpoint=str(self.cfg.models.sam.checkpoint)
            )
            self.sam.to(device=self.device)
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.sam,
                points_per_side=self.cfg.models.sam.points_per_side,
                pred_iou_thresh=self.cfg.models.sam.pred_iou_thresh,
                points_per_batch=self.cfg.models.sam.points_per_batch,
                stability_score_thresh=self.cfg.models.sam.stability_score_thresh,
                crop_n_layers=self.cfg.models.sam.crop_n_layers,
                min_mask_region_area=self.cfg.models.sam.min_mask_region_area,
            )
            self.sam.eval()
            print_status(
                "SAM_MODEL",
                "OK",
                model=model_type,
                checkpoint=self.cfg.models.sam.checkpoint,
                device=self.device,
            )
        # load the dataset
        dataset_cfg = {"root_dir": self.cfg.main.dataset_path, "transforms": None}
        if self.cfg.main.dataset == "hm3dsem":
            self.dataset = HM3DSemDataset(dataset_cfg)
        elif self.cfg.main.dataset == "scannet":
            self.dataset = ScannetDataset(dataset_cfg)
        elif self.cfg.main.dataset == "replica":
            self.dataset = ReplicaDataset(dataset_cfg)
        elif self.cfg.main.dataset == "real_robot":
            # RealRobotDataset needs metric depth limits in addition to the
            # common RGB-D dataset path. camera.json remains the fallback.
            for key in (
                "min_depth_m",
                "max_depth_m",
                "pose_mode",
                "planar_pose_height_m",
                "pose_warn_tilt_deg",
                "pose_warn_height_drift_m",
            ):
                value = self.cfg.pipeline.get(key)
                if value is not None:
                    dataset_cfg[key] = value
            self.dataset = RealRobotDataset(dataset_cfg)
        else:
            raise ValueError(f"Dataset not supported: {self.cfg.main.dataset}")
        print_status(
            "DATASET",
            "OK",
            name=self.cfg.main.dataset,
            class_name=type(self.dataset).__name__,
            root=self.cfg.main.dataset_path,
            frames=len(self.dataset),
        )

    @staticmethod
    def _validate_pose(pose):
        """Validate a rigid 4x4 camera-to-map transform."""
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (4, 4):
            return False, "pose_shape"
        if not np.isfinite(pose).all():
            return False, "pose_contains_nan_or_inf"
        if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
            return False, "pose_last_row"

        rotation = pose[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-2):
            return False, "rotation_not_orthonormal"
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-2):
            return False, "rotation_determinant"
        return True, None

    @staticmethod
    def _pose_delta(previous_pose, current_pose):
        """Return translation (m) and rotation (deg) between two poses."""
        translation_m = np.linalg.norm(
            current_pose[:3, 3] - previous_pose[:3, 3]
        )
        relative_rotation = previous_pose[:3, :3].T @ current_pose[:3, :3]
        cosine = np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
        rotation_deg = np.degrees(np.arccos(cosine))
        return float(translation_m), float(rotation_deg)

    def _processing_frame_indices(self):
        """Return geometry keyframes, or the configured stride as a fallback."""
        if self.frame_indices:
            return self.frame_indices
        legacy_skip = int(self.cfg.pipeline.get("skip_frames", 1))
        geometry_skip = max(
            1,
            int(self.cfg.pipeline.get("geometry_skip_frames", legacy_skip)),
        )
        return list(range(0, len(self.dataset), geometry_skip))

    def _semantic_processing_frame_indices(self):
        """Return the SAM/CLIP subset of accepted geometry keyframes."""
        if self.semantic_frame_indices:
            return self.semantic_frame_indices
        geometry_indices = self._processing_frame_indices()
        semantic_skip = max(
            1,
            int(self.cfg.pipeline.get("semantic_skip_frames", 1)),
        )
        selected = geometry_indices[::semantic_skip]
        if geometry_indices and selected[-1] != geometry_indices[-1]:
            selected.append(geometry_indices[-1])
        return selected

    def create_feature_map(self, save_path=None):
        """
        Create the feature map of the HOV-SG (full point cloud + feature map point level + feature map mask level)
        :param save_path : str, optional, The path to save the feature map

        """

        if self.dataset is None:
            print("No dataset loaded")
            return

        legacy_skip_frames = int(self.cfg.pipeline.get("skip_frames", 1))
        geometry_skip_frames = int(
            self.cfg.pipeline.get(
                "geometry_skip_frames",
                legacy_skip_frames,
            )
        )
        semantic_skip_frames = int(
            self.cfg.pipeline.get("semantic_skip_frames", 1)
        )
        if geometry_skip_frames < 1:
            raise ValueError("pipeline.geometry_skip_frames must be positive")
        if semantic_skip_frames < 1:
            raise ValueError("pipeline.semantic_skip_frames must be positive")

        candidate_frame_indices = list(
            range(0, len(self.dataset), geometry_skip_frames)
        )
        if not candidate_frame_indices:
            raise RuntimeError("The dataset contains no frames to process")

        # ========== 新增: 进度估算 ==========
        estimated_time_per_frame = 4.0  # 秒/帧 (SAM+CLIP平均处理时间)
        estimated_semantic_frames = max(
            1,
            int(np.ceil(len(candidate_frame_indices) / semantic_skip_frames)),
        )
        total_estimated_time = estimated_semantic_frames * estimated_time_per_frame

        print("=" * 70)
        print("📊 FEATURE MAP CREATION - PROGRESS ESTIMATION")
        print("=" * 70)
        print(f"  Total dataset frames: {len(self.dataset)}")
        print(f"  Geometry skip frames: {geometry_skip_frames}")
        print(f"  Geometry candidate frames: {len(candidate_frame_indices)}")
        print(f"  Semantic keyframe stride: {semantic_skip_frames}")
        print(f"  Estimated semantic frames: {estimated_semantic_frames}")
        print(f"  Estimated time per frame: {estimated_time_per_frame:.1f}s")
        print(f"  ⏱️  Total estimated time: {total_estimated_time/60:.1f} minutes ({total_estimated_time/3600:.2f} hours)")
        print(
            "  Tip: increase semantic_skip_frames to reduce SAM/CLIP time "
            "without reducing geometry density"
        )
        print("=" * 70)
        print()

        voxel_size = float(self.cfg.pipeline.voxel_size)
        fusion_batch_size = max(
            1,
            int(self.cfg.pipeline.get("pointcloud_fusion_batch_size", 10)),
        )
        keyframe_translation_m = max(
            0.0,
            float(self.cfg.pipeline.get("keyframe_translation_m", 0.0)),
        )
        keyframe_rotation_deg = max(
            0.0,
            float(self.cfg.pipeline.get("keyframe_rotation_deg", 0.0)),
        )
        max_pose_translation_jump_m = float(
            self.cfg.pipeline.get("max_pose_translation_jump_m", np.inf)
        )
        max_pose_rotation_jump_deg = float(
            self.cfg.pipeline.get("max_pose_rotation_jump_deg", np.inf)
        )
        if max_pose_translation_jump_m <= 0:
            raise ValueError(
                "pipeline.max_pose_translation_jump_m must be positive"
            )
        if max_pose_rotation_jump_deg <= 0:
            raise ValueError(
                "pipeline.max_pose_rotation_jump_deg must be positive"
            )
        association_distance_m = float(
            self.cfg.pipeline.get(
                "association_distance_m",
                max(0.05, 2.0 * voxel_size),
            )
        )
        if association_distance_m <= 0:
            raise ValueError("pipeline.association_distance_m must be positive")

        log_every_n_frames = max(
            1,
            int(self.cfg.pipeline.get("log_every_n_frames", 10)),
        )
        print_status(
            "FEATURE_MAP",
            "START",
            dataset_frames=len(self.dataset),
            candidate_frames=len(candidate_frame_indices),
            geometry_skip_frames=geometry_skip_frames,
            semantic_skip_frames=semantic_skip_frames,
            voxel_size=voxel_size,
            keyframe_translation_m=keyframe_translation_m,
            keyframe_rotation_deg=keyframe_rotation_deg,
            association_distance_m=association_distance_m,
            fusion_batch_size=fusion_batch_size,
            log_every_n_frames=log_every_n_frames,
        )

        # create the RGB-D point cloud
        self.full_pcd = o3d.geometry.PointCloud()
        self.frame_indices = []
        self.semantic_frame_indices = []
        last_accepted_pose = None
        rejected_invalid_pose = 0
        rejected_pose_jump = 0
        rejected_non_keyframe = 0
        rejected_empty_pointcloud = 0
        raw_accumulated_points = 0
        frames_since_downsample = 0
        for candidate_position, i in enumerate(
            tqdm(candidate_frame_indices, desc="Creating RGB-D point cloud")
        ):
            rgb_image, depth_image, pose, _, depth_intrinsics = self.dataset[i]
            pose = np.asarray(pose, dtype=np.float64)
            pose_valid, pose_error = self._validate_pose(pose)
            if not pose_valid:
                rejected_invalid_pose += 1
                print_status(
                    "POSE_FRAME",
                    "WARN",
                    frame_index=i,
                    reason=pose_error,
                )
                continue

            translation_delta_m = 0.0
            rotation_delta_deg = 0.0
            if last_accepted_pose is not None:
                translation_delta_m, rotation_delta_deg = self._pose_delta(
                    last_accepted_pose,
                    pose,
                )
                if (
                    translation_delta_m > max_pose_translation_jump_m
                    or rotation_delta_deg > max_pose_rotation_jump_deg
                ):
                    rejected_pose_jump += 1
                    print_status(
                        "POSE_FRAME",
                        "WARN",
                        frame_index=i,
                        reason="pose_jump",
                        translation_delta_m=translation_delta_m,
                        rotation_delta_deg=rotation_delta_deg,
                    )
                    continue

                keyframe_required = (
                    keyframe_translation_m > 0.0 or keyframe_rotation_deg > 0.0
                )
                translation_reached = (
                    keyframe_translation_m > 0.0
                    and translation_delta_m >= keyframe_translation_m
                )
                rotation_reached = (
                    keyframe_rotation_deg > 0.0
                    and rotation_delta_deg >= keyframe_rotation_deg
                )
                if keyframe_required and not (
                    translation_reached or rotation_reached
                ):
                    rejected_non_keyframe += 1
                    continue

            frame_pcd = self.dataset.create_pcd(rgb_image, depth_image, pose)
            if frame_pcd.is_empty():
                rejected_empty_pointcloud += 1
                print_status(
                    "POINT_CLOUD_FRAME",
                    "WARN",
                    frame_index=i,
                    reason="empty_point_cloud",
                )
                continue
            self.frame_indices.append(i)
            last_accepted_pose = pose.copy()
            raw_accumulated_points += len(frame_pcd.points)
            self.full_pcd += frame_pcd
            frames_since_downsample += 1

            if frames_since_downsample >= fusion_batch_size:
                self.full_pcd = self.full_pcd.voxel_down_sample(
                    voxel_size=voxel_size
                )
                frames_since_downsample = 0

            should_log = (
                len(self.frame_indices) == 1
                or candidate_position == len(candidate_frame_indices) - 1
                or len(self.frame_indices) % log_every_n_frames == 0
            )
            if should_log:
                depth_array = np.asarray(depth_image)
                if isinstance(self.dataset, RealRobotDataset):
                    _, valid_depth = self.dataset.get_valid_depth_mask(depth_array)
                else:
                    valid_depth = np.isfinite(depth_array) & (depth_array > 0)
                valid_depth_count = int(np.count_nonzero(valid_depth))
                print_status(
                    "POINT_CLOUD_FRAME",
                    "OK",
                    frame_index=i,
                    progress=(
                        f"{candidate_position + 1}/{len(candidate_frame_indices)}"
                    ),
                    accepted_keyframes=len(self.frame_indices),
                    rgb_size=getattr(rgb_image, "size", np.asarray(rgb_image).shape),
                    depth_shape=depth_array.shape,
                    valid_depth_pixels=valid_depth_count,
                    valid_depth_ratio=(
                        valid_depth_count / valid_depth.size if valid_depth.size else 0.0
                    ),
                    frame_points=len(frame_pcd.points),
                    raw_accumulated_points=raw_accumulated_points,
                    resident_points=len(self.full_pcd.points),
                    pose_translation=np.asarray(pose)[:3, 3].round(4).tolist(),
                    translation_delta_m=translation_delta_m,
                    rotation_delta_deg=rotation_delta_deg,
                )

        if not self.frame_indices:
            raise RuntimeError(
                "No frames passed pose validation and keyframe selection"
            )
        self.semantic_frame_indices = self.frame_indices[::semantic_skip_frames]
        if self.semantic_frame_indices[-1] != self.frame_indices[-1]:
            self.semantic_frame_indices.append(self.frame_indices[-1])
        print_status(
            "FRAME_SELECTION",
            "OK",
            candidate_frames=len(candidate_frame_indices),
            accepted_keyframes=len(self.frame_indices),
            semantic_keyframes=len(self.semantic_frame_indices),
            semantic_skip_frames=semantic_skip_frames,
            rejected_invalid_pose=rejected_invalid_pose,
            rejected_pose_jump=rejected_pose_jump,
            rejected_non_keyframe=rejected_non_keyframe,
            rejected_empty_pointcloud=rejected_empty_pointcloud,
        )

        # filter point cloud
        resident_points_before_final_voxel = len(self.full_pcd.points)
        self.full_pcd = self.full_pcd.voxel_down_sample(
            voxel_size=voxel_size
        )
        voxel_points = len(self.full_pcd.points)
        outlier_nb_neighbors = max(
            2,
            int(self.cfg.pipeline.get("outlier_nb_neighbors", 20)),
        )
        outlier_std_ratio = float(
            self.cfg.pipeline.get("outlier_std_ratio", 2.0)
        )
        if outlier_std_ratio <= 0:
            raise ValueError("pipeline.outlier_std_ratio must be positive")
        if voxel_points > outlier_nb_neighbors:
            self.full_pcd, _ = self.full_pcd.remove_statistical_outlier(
                nb_neighbors=outlier_nb_neighbors,
                std_ratio=outlier_std_ratio,
            )
        full_pcd_empty = self.full_pcd.is_empty()
        print_status(
            "FULL_POINT_CLOUD",
            "ERROR" if full_pcd_empty else "OK",
            raw_accumulated_points=raw_accumulated_points,
            resident_points_before_final_voxel=resident_points_before_final_voxel,
            voxel_points=voxel_points,
            denoised_points=len(self.full_pcd.points),
            denoise_method="statistical_outlier",
            outlier_nb_neighbors=outlier_nb_neighbors,
            outlier_std_ratio=outlier_std_ratio,
            min_bound=(
                []
                if full_pcd_empty
                else np.asarray(self.full_pcd.get_min_bound()).round(4).tolist()
            ),
            max_bound=(
                []
                if full_pcd_empty
                else np.asarray(self.full_pcd.get_max_bound()).round(4).tolist()
            ),
        )
        if full_pcd_empty:
            raise RuntimeError(
                "No valid 3D points were produced. Check the real-robot depth "
                "units, depth range, intrinsics, and pose files."
            )

        # create tree from full point cloud
        locs_in = np.array(self.full_pcd.points)
        tree_pcd = cKDTree(locs_in)
        n_points = locs_in.shape[0]
        counter = torch.zeros((n_points, 1), device="cpu")
        sum_features = torch.zeros((n_points, self.clip_feat_dim), device="cpu")

        # extract features for each frame
        frames_pcd = []
        frames_feats = []
        association_attempts = 0
        association_accepted = 0
        for frame_position, i in enumerate(
            tqdm(self.semantic_frame_indices, desc="Extracting features")
        ):
            # frame_position is the index of the frame in the selected frames
            # i is the index of the frame in the dataset
            rgb_image, depth_image, pose, _, _ = self.dataset[i]
            if rgb_image.size != depth_image.size:
                rgb_image = rgb_image.resize(depth_image.size)
            feature_result = extract_feats_per_pixel(
                np.array(rgb_image),
                self.mask_generator,
                self.clip_model,
                self.preprocess,
                clip_feat_dim=self.clip_feat_dim,
                bbox_margin=self.cfg.pipeline.clip_bbox_margin,
                maskedd_weight=self.cfg.pipeline.clip_masked_weight,
            )
            if feature_result is None or len(feature_result) != 4:
                print_status(
                    "SEMANTIC_FRAME",
                    "WARN",
                    frame_index=i,
                    reason="feature_extractor_returned_no_complete_result",
                )
                continue
            F_2D, F_masks, masks, F_g = feature_result
            # F_2D: pixel-level features, shape (H*W, D), feature for each pixel
            # F_masks: mask-level features, shape (N_masks, D), feature for each mask(item)
            # masks: list of masks, each mask is a binary mask of shape (H, W)
            # F_g: global feature, shape (D,), feature for the whole image
            if F_2D is None or F_masks is None or masks is None:
                print_status(
                    "SEMANTIC_FRAME",
                    "WARN",
                    frame_index=i,
                    reason="no_sam_masks",
                )
                continue
            F_2D = F_2D.cpu()
            pcd = self.dataset.create_pcd(rgb_image, depth_image, pose)
            masks_3d = self.dataset.create_3d_masks(
                masks,
                depth_image,
                self.full_pcd,
                tree_pcd,
                pose,
                down_size=self.cfg.pipeline.voxel_size,
                filter_distance=self.cfg.pipeline.max_mask_distance,
                association_distance=association_distance_m,
            )
            masks_3d_before_filter = len(masks_3d)
            valid_mask_indices = [
                mask_index
                for mask_index, mask_pcd in enumerate(masks_3d)
                if not mask_pcd.is_empty() and len(mask_pcd.points) > 0
            ]
            masks_3d = [masks_3d[index] for index in valid_mask_indices]
            # change 2D mask to 3D mask, each mask is a point cloud
            # frames_pcd is a list of 3D masks for each frame, each mask is a point cloud
            if masks_3d:
                frames_pcd.append(masks_3d)
                # Keep the mask features aligned with the retained 3D masks.
                # frames_feats is currently diagnostic state, but preserving
                # alignment avoids future regressions if it is reused.
                frames_feats.append(F_masks[valid_mask_indices])
            if masks_3d_before_filter != len(masks_3d):
                print_status(
                    "EMPTY_3D_MASK_FILTER",
                    "OK",
                    frame_index=i,
                    masks_before=masks_3d_before_filter,
                    masks_after=len(masks_3d),
                    removed=masks_3d_before_filter - len(masks_3d),
                )
            # fuse features for each point in the full pcd
            if isinstance(self.dataset, RealRobotDataset):
                _, mask = self.dataset.get_valid_depth_mask(depth_image)
            else:
                mask = np.array(depth_image) > 0
            mask = torch.from_numpy(mask)
            F_2D = F_2D[mask].to(dtype=torch.float32)
            # using cKdtree to find the closest point in the full pcd for each point in frame pcd
            frame_points = np.asarray(pcd.points)
            if frame_points.shape[0] != F_2D.shape[0]:
                raise RuntimeError(
                    "Point/feature count mismatch at frame "
                    f"{i}: points={frame_points.shape[0]}, features={F_2D.shape[0]}"
                )
            if frame_points.shape[0] > 0:
                dis, idx = tree_pcd.query(frame_points, k=1, workers=-1)
                finite_feature = torch.isfinite(F_2D).all(dim=1)
                nonzero_feature = torch.linalg.vector_norm(F_2D, dim=1) > 0
                association_mask = (
                    np.isfinite(dis) & (dis <= association_distance_m)
                )
                association_mask &= (finite_feature & nonzero_feature).numpy()
                association_attempts += int(frame_points.shape[0])
                association_accepted += int(np.count_nonzero(association_mask))

                if np.any(association_mask):
                    accepted_indices = torch.from_numpy(
                        np.asarray(idx[association_mask], dtype=np.int64)
                    )
                    accepted_features = F_2D[
                        torch.from_numpy(association_mask)
                    ]
                    sum_features.index_add_(
                        0,
                        accepted_indices,
                        accepted_features,
                    )
                    counter.index_add_(
                        0,
                        accepted_indices,
                        torch.ones(
                            (accepted_indices.numel(), 1),
                            dtype=counter.dtype,
                        ),
                    )

            should_log = (
                frame_position == 0
                or frame_position == len(self.semantic_frame_indices) - 1
                or (frame_position + 1) % log_every_n_frames == 0
            )
            if should_log:
                mask_point_count = sum(len(mask_pcd.points) for mask_pcd in masks_3d)
                print_status(
                    "SEMANTIC_FRAME",
                    "OK",
                    frame_index=i,
                    progress=(
                        f"{frame_position + 1}/"
                        f"{len(self.semantic_frame_indices)}"
                    ),
                    sam_masks=len(masks),
                    mask_feature_shape=tuple(F_masks.shape),
                    pixel_feature_shape=tuple(F_2D.shape),
                    frame_points=len(pcd.points),
                    associated_feature_points=(
                        int(np.count_nonzero(association_mask))
                        if frame_points.shape[0] > 0
                        else 0
                    ),
                    masks_3d=len(masks_3d),
                    masks_3d_points=mask_point_count,
                )
        if not frames_pcd:
            print_status("SEMANTIC_EXTRACTION", "ERROR", reason="no_frames_processed")
            raise RuntimeError("No frames produced SAM/CLIP features")
        print_status(
            "SEMANTIC_EXTRACTION",
            "OK",
            processed_frames=len(frames_pcd),
            total_selected_frames=len(self.semantic_frame_indices),
            association_attempts=association_attempts,
            association_accepted=association_accepted,
            association_ratio=(
                association_accepted / association_attempts
                if association_attempts
                else 0.0
            ),
        )
        # compute the average features
        sum_features = sum_features / counter.clamp_min(1.0)
        sum_features = torch.nn.functional.normalize(sum_features, p=2, dim=-1)
        self.full_feats_array = sum_features.cpu().numpy()
        self.full_feats_array: np.ndarray
        
        # free memory
        del sum_features, counter
        torch.cuda.empty_cache() 

        # merging the masks
        masks_before_merge = sum(len(frame_masks) for frame_masks in frames_pcd)
        if self.cfg.pipeline.merge_type == "hierarchical":
            tqdm.write("Merging 3d masks hierarchically")
            self.mask_pcds = hierarchical_merge(
                frames_pcd, 
                self.cfg.pipeline.init_overlap_thresh, 
                self.cfg.pipeline.overlap_thresh_factor, 
                self.cfg.pipeline.voxel_size, 
                self.cfg.pipeline.iou_thresh,
            )
        elif self.cfg.pipeline.merge_type == "sequential":
            tqdm.write("Merging 3d masks sequentially") 
            self.mask_pcds = seq_merge(
                frames_pcd, 
                self.cfg.pipeline.init_overlap_thresh, 
                self.cfg.pipeline.voxel_size, 
                self.cfg.pipeline.iou_thresh
            )
        print_status(
            "MASK_MERGE",
            "OK",
            merge_type=self.cfg.pipeline.merge_type,
            masks_before=masks_before_merge,
            masks_after=len(self.mask_pcds),
        )

        # remove any small pcds
        masks_before_size_filter = len(self.mask_pcds)
        self.mask_pcds = [
            pcd
            for pcd in self.mask_pcds
            if not pcd.is_empty() and len(pcd.points) >= 100
        ]
        print_status(
            "MASK_SIZE_FILTER",
            "OK",
            masks_before=masks_before_size_filter,
            masks_after=len(self.mask_pcds),
            minimum_points=100,
        )
        # fuse point features in every 3d mask
        masks_feats = []
        for i, mask_3d in tqdm(enumerate(self.mask_pcds), desc="Fusing features"):
            # find the points in the mask
            mask_3d = mask_3d.voxel_down_sample(self.cfg.pipeline.voxel_size * 2)
            points = np.asarray(mask_3d.points)
            dist, idx = tree_pcd.query(points, k=1, workers=-1)
            feats = self.full_feats_array[idx]
            feats = np.nan_to_num(feats)
            # filter feats with dbscan
            if feats.shape[0] == 0:
                masks_feats.append(
                    np.zeros((1, self.clip_feat_dim), dtype=self.full_feats_array.dtype)
                )
                continue
            feats = feats_denoise_dbscan(feats, eps=0.01, min_points=100)
            masks_feats.append(feats)
        self.mask_feats = masks_feats
        print("number of masks: ", len(self.mask_feats))
        print("number of pcds in hovsg: ", len(self.mask_pcds))
        assert len(self.mask_pcds) == len(self.mask_feats)
        print_status(
            "FEATURE_MAP",
            "OK",
            full_points=len(self.full_pcd.points),
            full_feature_shape=tuple(self.full_feats_array.shape),
            object_masks=len(self.mask_pcds),
            mask_features=len(self.mask_feats),
        )

    def _robust_single_floor_bounds(self, points, floor_peak=None):
        """Estimate single-floor bounds while ignoring vertical outliers."""
        heights = np.asarray(points, dtype=np.float64)[:, 1]
        finite_heights = heights[np.isfinite(heights)]
        if finite_heights.size == 0:
            raise RuntimeError("Cannot estimate a floor from an empty point cloud")

        lower_quantile = float(
            self.cfg.pipeline.get("single_floor_lower_quantile", 0.02)
        )
        upper_quantile = float(
            self.cfg.pipeline.get("single_floor_upper_quantile", 0.98)
        )
        if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
            raise ValueError(
                "single-floor quantiles must satisfy "
                "0 <= lower < upper <= 1"
            )

        robust_lower = float(np.quantile(finite_heights, lower_quantile))
        robust_upper = float(np.quantile(finite_heights, upper_quantile))
        minimum_height = max(
            0.1,
            float(self.cfg.pipeline.get("min_floor_height", 0.5)),
        )

        floor_zero_level = robust_lower
        if floor_peak is not None and np.isfinite(floor_peak):
            highest_valid_floor = robust_upper - minimum_height
            if highest_valid_floor >= robust_lower:
                floor_zero_level = float(
                    np.clip(floor_peak, robust_lower, highest_valid_floor)
                )

        floor_upper_level = max(
            robust_upper,
            floor_zero_level + minimum_height,
        )
        return floor_zero_level, floor_upper_level

    def _create_single_floor(
        self,
        floor_zero_level,
        floor_upper_level,
        reason,
    ):
        """Create a single Floor object from robust vertical bounds."""
        points = np.asarray(self.full_pcd.points)
        lower_margin = max(
            0.0,
            float(self.cfg.pipeline.get("single_floor_lower_margin_m", 0.15)),
        )
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)
        min_bound[1] = floor_zero_level - lower_margin
        max_bound[1] = floor_upper_level
        floor_pcd = self.full_pcd.crop(
            o3d.geometry.AxisAlignedBoundingBox(
                min_bound=min_bound,
                max_bound=max_bound,
            )
        )
        if floor_pcd.is_empty():
            raise RuntimeError(
                "Single-floor fallback produced an empty point cloud: "
                f"lower={floor_zero_level:.3f}, upper={floor_upper_level:.3f}"
            )

        floor = Floor("0", name="floor_0")
        floor.pcd = floor_pcd
        floor.floor_zero_level = float(floor_zero_level)
        floor.floor_height = float(floor_upper_level - floor_zero_level)
        floor.vertices = np.asarray(
            floor_pcd.get_axis_aligned_bounding_box().get_box_points()
        )
        self.floors.append(floor)
        print_status(
            "FLOOR_DETECTION",
            "OK",
            mode="single_floor_fallback",
            reason=reason,
            floor_zero_level_m=floor.floor_zero_level,
            floor_upper_level_m=float(floor_upper_level),
            floor_height_m=floor.floor_height,
            retained_points=len(floor.pcd.points),
        )
        return [floor]

    def segment_floors(self, path, flip_zy=False):
        """
        Segment the floors from the full point cloud
        :param path: str, The path to save the intermediate results

        ✅ 修改: 添加单楼层检测fallback，适配室内移动机器人
        """
        # ========== 新增: 单楼层检测 ==========
        points = np.asarray(self.full_pcd.points)
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        y_range = y_max - y_min

        print_status(
            "FLOOR_DETECTION",
            "INFO",
            y_min=f"{y_min:.3f}m",
            y_max=f"{y_max:.3f}m",
            y_range=f"{y_range:.3f}m",
        )

        # 检查配置或自动判断是否为单楼层
        min_floor_height = float(self.cfg.pipeline.get("min_floor_height", 0.5))
        force_single_floor = bool(self.cfg.pipeline.get("force_single_floor", False))

        if force_single_floor or y_range < min_floor_height:
            floor_zero_level, floor_upper_level = self._robust_single_floor_bounds(
                points
            )
            reason = (
                "force_single_floor=true"
                if force_single_floor
                else (
                    f"Y-axis range ({y_range:.3f}m) < threshold "
                    f"({min_floor_height:.3f}m)"
                )
            )
            return self._create_single_floor(
                floor_zero_level,
                floor_upper_level,
                reason,
            )
        # ========== 原有多楼层分割逻辑 ==========
        print_status(
            "FLOOR_DETECTION",
            "INFO",
            mode="multi_floor_detection",
            message="Y-axis range sufficient for multi-floor detection",
        )

        # downsample the point cloud
        downpcd = self.full_pcd.voxel_down_sample(voxel_size=0.05)
        # flip the z and y axis
        if flip_zy:
            downpcd.points = o3d.utility.Vector3dVector(
                np.array(downpcd.points)[:, [0, 2, 1]]
            )
            downpcd.transform(np.eye(4) * np.array([1, 1, -1, 1]))
        # rotate the point cloud to align floor with the y axis
        T1 = np.eye(4)
        T1[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()
        downpcd = np.asarray(downpcd.points)
        print("downpcd", downpcd.shape)

        # divide z axis range into 0.01m bin
        reselotion = 0.01
        bins = np.abs(np.max(downpcd[:, 1]) - np.min(downpcd[:, 1])) / reselotion
        print("bins", bins)
        z_hist = np.histogram(downpcd[:, 1], bins=int(bins))
        # smooth the histogram
        z_hist_smooth = gaussian_filter1d(z_hist[0], sigma=2)
        # Find the peaks in this histogram.
        distance = 0.2 / reselotion
        print("distance", distance)
        # set the min peak height based on the histogram
        print(np.mean(z_hist_smooth))
        min_peak_height = np.percentile(z_hist_smooth, 90)
        print("min_peak_height", min_peak_height)
        peaks, _ = find_peaks(z_hist_smooth, distance=distance, height=min_peak_height)

        # plot the histogram
        if self.cfg.pipeline.save_intermediate_results:
            plt.figure()
            plt.plot(z_hist[1][:-1], z_hist_smooth)
            plt.plot(z_hist[1][peaks], z_hist_smooth[peaks], "x")
            plt.hlines(
                min_peak_height, np.min(z_hist[1]), np.max(z_hist[1]), colors="r"
            )
            plt.savefig(os.path.join(self.graph_tmp_folder, "floor_histogram.png"))

        # cluster the peaks using DBSCAN
        peaks_locations = z_hist[1][peaks]
        if peaks_locations.size:
            clustering = DBSCAN(eps=1, min_samples=1).fit(
                peaks_locations.reshape(-1, 1)
            )
            labels = clustering.labels_
        else:
            labels = np.empty((0,), dtype=np.int32)

        # plot the histogram
        if self.cfg.pipeline.save_intermediate_results:
            plt.figure()
            plt.plot(z_hist[1][:-1], z_hist_smooth)
            plt.plot(z_hist[1][peaks], z_hist_smooth[peaks], "x")
            plt.hlines(
                min_peak_height, np.min(z_hist[1]), np.max(z_hist[1]), colors="r"
            )
            # plot the clusters
            for i in range(len(np.unique(labels))):
                plt.plot(
                    z_hist[1][peaks[labels == i]],
                    z_hist_smooth[peaks[labels == i]],
                    "o",
                )
            plt.savefig(
                os.path.join(self.graph_tmp_folder, "floor_histogram_cluster.png")
            )

        # for each cluster find the top 2 peaks
        clustred_peaks = []
        for i in range(len(np.unique(labels))):
            # for first and last cluster, find the top 1 peak
            if i == 0 or i == len(np.unique(labels)) - 1:
                p = peaks[labels == i]
                top_p = p[np.argsort(z_hist_smooth[p])[-1:]].tolist()
                top_p = [z_hist[1][p] for p in top_p]
                clustred_peaks.append(top_p)
                continue
            p = peaks[labels == i]
            top_p = p[np.argsort(z_hist_smooth[p])[-2:]].tolist()
            top_p = [z_hist[1][p] for p in top_p]
            clustred_peaks.append(top_p)
        clustred_peaks = [item for sublist in clustred_peaks for item in sublist]
        clustred_peaks = np.sort(clustred_peaks)
        print("clustred_peaks", clustred_peaks)

        if len(clustred_peaks) < 2:
            floor_peak = (
                float(clustred_peaks[0]) if len(clustred_peaks) == 1 else None
            )
            floor_zero_level, floor_upper_level = self._robust_single_floor_bounds(
                downpcd,
                floor_peak=floor_peak,
            )
            return self._create_single_floor(
                floor_zero_level,
                floor_upper_level,
                reason=(
                    "only_one_height_peak"
                    if floor_peak is not None
                    else "no_height_peaks"
                ),
            )

        floors = []
        # for every two consecutive peaks with 2m distance, assign floor level
        for i in range(0, len(clustred_peaks) - 1, 2):
            floors.append([clustred_peaks[i], clustred_peaks[i + 1]])
        print("floors", floors)
        # for the first floor extend the floor to the ground
        floors[0][0] = (floors[0][0] + np.min(downpcd[:, 1])) / 2
        # for the last floor extend the floor to the ceiling
        floors[-1][1] = (floors[-1][1] + np.max(downpcd[:, 1])) / 2
        print("number of floors: ", len(floors))

        floors_pcd = []
        for i, floor in enumerate(floors):
            floor_obj = Floor(str(i), name="floor_" + str(i))
            floor_pcd = self.full_pcd.crop(
                o3d.geometry.AxisAlignedBoundingBox(
                    min_bound=(-np.inf, floor[0], -np.inf),
                    max_bound=(np.inf, floor[1], np.inf),
                )
            )
            if floor_pcd.is_empty():
                print_status(
                    "FLOOR_SEGMENTATION",
                    "WARN",
                    floor_id=i,
                    reason="empty_floor_interval",
                    lower_bound_m=float(floor[0]),
                    upper_bound_m=float(floor[1]),
                )
                continue
            bbox = floor_pcd.get_axis_aligned_bounding_box()
            floor_obj.vertices = np.asarray(bbox.get_box_points())
            floor_obj.pcd = floor_pcd
            floor_obj.floor_zero_level = float(floor[0])
            floor_obj.floor_height = float(floor[1] - floor[0])
            self.floors.append(floor_obj)
            floors_pcd.append(floor_pcd)
        if not floors_pcd:
            raise RuntimeError("Floor detection produced no non-empty floor point clouds")
        return self.floors

    def segment_rooms(self, floor: Floor, path):
        """
        Segment the rooms from the floor point cloud
        :param floor: Floor, The floor object
        :param path: str, The path to save the intermediate results

        ✅ 修改: 添加覆盖范围检查，避免数据不足时的分割失败
        """

        tmp_floor_path = os.path.join(self.graph_tmp_folder, floor.floor_id)
        if not os.path.exists(tmp_floor_path):
            os.makedirs(tmp_floor_path, exist_ok=True)

        floor_pcd = floor.pcd
        xyz = np.asarray(floor_pcd.points)
        xyz_full = xyz.copy()
        floor_zero_level = floor.floor_zero_level
        floor_height = floor.floor_height

        # ========== 新增: 检查点云覆盖范围 ==========
        x_range = xyz[:, 0].max() - xyz[:, 0].min()
        z_range = xyz[:, 2].max() - xyz[:, 2].min()
        coverage = min(x_range, z_range)
        point_count = len(xyz)

        print_status(
            "ROOM_COVERAGE_CHECK",
            "INFO",
            floor_id=floor.floor_id,
            x_range=f"{x_range:.2f}m",
            z_range=f"{z_range:.2f}m",
            min_coverage=f"{coverage:.2f}m",
            points=point_count,
        )

        # 检查是否跳过房间分割
        min_room_coverage = float(self.cfg.pipeline.get("min_room_coverage", 2.0))
        skip_if_small = bool(self.cfg.pipeline.get("skip_room_segmentation_if_small", True))

        if skip_if_small and (coverage < min_room_coverage or point_count < 1000):
            print_status(
                "ROOM_SEGMENTATION",
                "SKIP",
                floor_id=floor.floor_id,
                reason=f"Insufficient coverage ({coverage:.2f}m < {min_room_coverage:.2f}m) or points ({point_count} < 1000)",
                fallback="Creating single room for entire floor",
            )

            # 创建单个房间包含整个楼层
            room = Room(f"{floor.floor_id}_0", floor.floor_id, name="room_0")
            room.pcd = floor_pcd
            room.vertices = xyz[:, [0, 2]]  # 2D vertices
            room.room_height = floor_height
            room.room_zero_level = floor_zero_level
            room.embeddings = []
            room.represent_images = []
            self.rooms.append(room)
            self.floors[int(floor.floor_id)].add_room(room)

            print_status(
                "ROOM_SEGMENTATION",
                "OK",
                floor_id=floor.floor_id,
                rooms=1,
                mode="single_room_fallback",
            )
            return

        # ========== 原有房间分割逻辑 ==========
        print_status(
            "ROOM_SEGMENTATION",
            "INFO",
            floor_id=floor.floor_id,
            mode="wall_based_segmentation",
            message="Coverage sufficient for room segmentation",
        )
        ## Slice below the ceiling ##
        xyz = xyz[xyz[:, 1] < floor_zero_level + floor_height - 0.3]
        xyz = xyz[xyz[:, 1] >= floor_zero_level + 1.5]
        xyz_full = xyz_full[xyz_full[:, 1] < floor_zero_level + floor_height - 0.2]
        ## Slice above the floor and below the ceiling ##
        # xyz = xyz[xyz[:, 1] < floor_zero_level + 1.8]
        # xyz = xyz[xyz[:, 1] > floor_zero_level + 0.8]
        # xyz_full = xyz_full[xyz_full[:, 1] < floor_zero_level + 1.8]

        # project the point cloud to 2d
        pcd_2d = xyz[:, [0, 2]]
        xyz_full = xyz_full[:, [0, 2]]

        # define the grid size and resolution based on the 2d point cloud
        grid_size = (
            int(np.max(pcd_2d[:, 0]) - np.min(pcd_2d[:, 0])),
            int(np.max(pcd_2d[:, 1]) - np.min(pcd_2d[:, 1])),
        )
        grid_size = (grid_size[0] + 1, grid_size[1] + 1)
        resolution = self.cfg.pipeline.grid_resolution
        print("grid_size: ", resolution)

        # calc 2d histogram of the floor using the xyz point cloud to extract the walls skeleton
        num_bins = (int(grid_size[0] // resolution), int(grid_size[1] // resolution))
        num_bins = (num_bins[1] + 1, num_bins[0] + 1)
        hist, _, _ = np.histogram2d(pcd_2d[:, 1], pcd_2d[:, 0], bins=num_bins)
        if self.cfg.pipeline.save_intermediate_results:
            # plot the histogram
            plt.figure()
            plt.imshow(hist, interpolation="nearest", cmap="jet", origin="lower")
            plt.colorbar()
            plt.savefig(os.path.join(tmp_floor_path, "2D_histogram.png"))

        # applythresholding
        hist = cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        hist = cv2.GaussianBlur(hist, (5, 5), 1)
        hist_threshold = 0.25 * np.max(hist)
        _, walls_skeleton = cv2.threshold(hist, hist_threshold, 255, cv2.THRESH_BINARY)

        # create a bigger image to avoid losing the walls
        walls_skeleton = cv2.copyMakeBorder(
            walls_skeleton, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0
        )

        # apply closing to the walls skeleton
        kernal = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        walls_skeleton = cv2.morphologyEx(
            walls_skeleton, cv2.MORPH_CLOSE, kernal, iterations=1
        )

        # extract outside boundary from histogram of xyz_full
        hist_full, _, _ = np.histogram2d(xyz_full[:, 1], xyz_full[:, 0], bins=num_bins)
        hist_full = cv2.normalize(hist_full, hist_full, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        hist_full = cv2.GaussianBlur(hist_full, (21, 21), 2)
        _, outside_boundary = cv2.threshold(hist_full, 0, 255, cv2.THRESH_BINARY)

        # create a bigger image to avoid losing the walls
        outside_boundary = cv2.copyMakeBorder(
            outside_boundary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0
        )

        # apply closing to the outside boundary
        kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        outside_boundary = cv2.morphologyEx(
            outside_boundary, cv2.MORPH_CLOSE, kernal, iterations=3
        )

        # extract the outside contour from the outside boundary
        contours, _ = cv2.findContours(
            outside_boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        outside_boundary = np.zeros_like(outside_boundary)
        cv2.drawContours(outside_boundary, contours, -1, (255, 255, 255), -1)
        outside_boundary = outside_boundary.astype(np.uint8)

        if self.cfg.pipeline.save_intermediate_results:
            plt.figure()
            plt.imshow(walls_skeleton, cmap="gray", origin="lower")
            plt.savefig(os.path.join(tmp_floor_path, "walls_skeleton.png"))

            plt.figure()
            plt.imshow(outside_boundary, cmap="gray", origin="lower")
            plt.savefig(os.path.join(tmp_floor_path, "outside_boundary.png"))

        # combine the walls skelton and outside boundary
        full_map = cv2.bitwise_or(walls_skeleton, cv2.bitwise_not(outside_boundary))

        # apply closing to the full map
        kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        full_map = cv2.morphologyEx(full_map, cv2.MORPH_CLOSE, kernal, iterations=2)

        if self.cfg.pipeline.save_intermediate_results:
            # plot the full map
            plt.figure()
            plt.imshow(full_map, cmap="gray", origin="lower")
            plt.savefig(os.path.join(tmp_floor_path, "full_map.png"))
        # apply distance transform to the full map
        room_vertices = distance_transform(full_map, resolution, tmp_floor_path)

        # using the 2D room vertices, map the room back to the original point cloud using KDTree
        room_pcds = []
        room_masks = []
        room_2d_points = []
        floor_tree = cKDTree(np.array(floor_pcd.points))
        for i in tqdm(range(len(room_vertices)), desc="Assign floor points to rooms"):
            room = np.zeros_like(full_map)
            room[room_vertices[i][0], room_vertices[i][1]] = 255
            room_masks.append(room)
            room_m = map_grid_to_point_cloud(room, resolution, pcd_2d)
            room_2d_points.append(room_m)
            # extrude the 2D room to 3D room by adding z value from floor zero level to floor zero level + floor height, step by 0.1m
            z_levels = np.arange(
                floor_zero_level, floor_zero_level + floor_height, 0.05
            )
            z_levels = z_levels.reshape(-1, 1)
            z_levels *= -1
            room_m3dd = []
            for z in z_levels:
                room_m3d = np.hstack((room_m, np.ones((room_m.shape[0], 1)) * z))
                room_m3dd.append(room_m3d)
            room_m3d = np.concatenate(room_m3dd, axis=0)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(room_m3d)
            # rotate floor pcd to align with the original point cloud
            T1 = np.eye(4)
            T1[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()
            pcd.transform(T1)
            # find the nearest point in the original point cloud
            _, idx = floor_tree.query(np.array(pcd.points), workers=-1)
            pcd = floor_pcd.select_by_index(idx)
            room_pcds.append(pcd)
        self.room_masks[floor.floor_id] = room_masks

        # compute the features of room: input a list of poses and images, output a list of embeddings list
        rgb_list = []
        pose_list = []
        F_g_list = []

        all_global_clip_feats = dict()
        processing_frame_indices = self._semantic_processing_frame_indices()
        for i, img_id in tqdm(
            enumerate(processing_frame_indices),
            total=len(processing_frame_indices),
            desc="Computing room features",
        ):
            rgb_image, _, pose, _, _ = self.dataset[img_id]
            F_g = get_img_feats(np.array(rgb_image), self.preprocess, self.clip_model)
            all_global_clip_feats[str(img_id)] = F_g
            rgb_list.append(rgb_image)
            pose_list.append(pose)
            F_g_list.append(F_g)
        np.savez(
            os.path.join(self.graph_tmp_folder, "room_views.npz"),
            **all_global_clip_feats,
        )

        pcd_min = np.min(np.array(floor_pcd.points), axis=0)
        pcd_max = np.max(np.array(floor_pcd.points), axis=0)
        assert pcd_min.shape[0] == 3

        repr_embs_list, repr_img_ids_list = compute_room_embeddings(
            room_pcds, pose_list, F_g_list, pcd_min, pcd_max, 10, tmp_floor_path
        )
        assert len(repr_embs_list) == len(room_2d_points)
        assert len(repr_img_ids_list) == len(room_2d_points)

        room_index = 0
        for i in range(len(room_2d_points)):
            room = Room(
                str(floor.floor_id) + "_" + str(room_index),
                floor.floor_id,
                name="room_" + str(room_index),
            )
            room.pcd = room_pcds[i]
            room.vertices = room_2d_points[i]
            self.floors[int(floor.floor_id)].add_room(room)
            room.room_height = floor_height
            room.room_zero_level = floor.floor_zero_level
            room.embeddings = repr_embs_list[i]
            room.represent_images = [
                int(processing_frame_indices[k]) for k in repr_img_ids_list[i]
            ]
            self.rooms.append(room)
            room_index += 1
        print(
            "number of rooms in floor {} is {}".format(
                floor.floor_id, len(self.floors[int(floor.floor_id)].rooms)
            )
        )

    def identify_object(self, object_feat, text_feats, classes):
        """
        Identify the object class by computing the similarity between the object feature and the text features
        we use COCO-Stuff dataset classes as the text features (183) class.
        :param object_feat: np.ndarray, The object feature
        :param text_feats: np.ndarray, The text features
        :param classes: List, The list of classes
        :return: str, The object class
        """
        similarity = np.dot(object_feat, text_feats.T)
        # find the class with the highest similarity
        return classes[np.argmax(similarity)]

    def segment_objects(self, save_dir: str = None):
        """
        Per floor, assign each object to the room with the highest overlap.
        :param save_dir: str, optional, The path to save the intermediate results
        """
        for i, pcd in enumerate(self.mask_pcds):
            self.mask_pcds[i] = pcd_denoise_dbscan(pcd, eps=0.05, min_points=10)
        text_feats, classes = get_label_feats(
            self.clip_model,
            self.clip_feat_dim,
            self.cfg.pipeline.obj_labels,
            self.cfg.main.save_path,
        )

        pbar = tqdm(enumerate(self.floors), total=len(self.floors), desc="Floor: ")
        margin = 0.2
        for f_idx, floor in pbar:
            pbar.set_description(f"Floor: {f_idx}")
            floor_pcd = floor.pcd
            objects_inside_floor = list()
            # assign objects to rooms
            for i, pcd in enumerate(self.mask_pcds):
                min_z = np.min(np.asarray(pcd.points)[:, 1])
                max_z = np.max(np.asarray(pcd.points)[:, 1])
                if min_z > floor.floor_zero_level - margin and max_z < (
                    floor.floor_zero_level + floor.floor_height + margin
                ):
                    objects_inside_floor.append(i)

            # show the second layer of pbar with tqdm
            obj_pbar = tqdm(
                enumerate(objects_inside_floor),
                total=len(objects_inside_floor),
                desc="Object: ",
                leave=False,
            )
            for obj_floor_idx, mask_idx in obj_pbar:
                room_assoc = list()
                for r_idx, room in enumerate(floor.rooms):
                    room_assoc.append(
                        find_intersection_share(
                            room.vertices,
                            np.array(self.mask_pcds[mask_idx].points)[:, [0, 2]],
                            0.2,
                        )
                    )
                # for outlier objects, utilize Euclidean distance between room centers and mask centers
                if np.sum(room_assoc) == 0:
                    for r_idx, room in enumerate(floor.rooms):
                        # use negative distance to align with the similarity metric
                        room_assoc[r_idx] = -1 * np.linalg.norm(
                            np.mean(room.vertices, axis=0)
                            - np.mean(
                                np.array(self.mask_pcds[mask_idx].points)[:, [0, 2]],
                                axis=0,
                            )
                        )
                    if self.cfg.pipeline.save_intermediate_results:
                        plt.clf()
                        fig, ax = plt.subplots()
                        for r_idx, room in enumerate(floor.rooms):
                            if np.argmax(room_assoc) == r_idx:
                                plt.scatter(
                                    room.vertices[:, 0],
                                    room.vertices[:, 1],
                                    color="red",
                                )
                            else:
                                continue
                                # plt.scatter(room.vertices[:, 0], room.vertices[:, 1])
                        plt.scatter(
                            np.array(self.mask_pcds[mask_idx].points)[:, [0, 2]][:, 0],
                            np.array(self.mask_pcds[mask_idx].points)[:, [0, 2]][:, 1],
                            s=0.05,
                            alpha=0.5,
                            color="green",
                        )
                        ax.set_aspect("equal")

                        debug_objects_dir = os.path.join(
                            self.graph_tmp_folder, "objects"
                        )
                        os.makedirs(debug_objects_dir, exist_ok=True)
                        plt.savefig(
                            os.path.join(
                                debug_objects_dir,
                                f"{floor.rooms[np.argmax(room_assoc)].room_id}_{floor.rooms[np.argmax(room_assoc)].object_counter}.png",
                            )
                        )

                closest_room_idx = np.argmax(room_assoc)

                name = self.identify_object(
                    self.mask_feats[mask_idx], text_feats, classes
                )
                # if [i for i in ["wall", "floor", "ceiling", "window", "door", "roof", "railing"] if i in name]:
                #     continue
                parent_room = floor.rooms[closest_room_idx]
                object = Object(
                    parent_room.room_id + "_" + str(parent_room.object_counter),
                    parent_room.room_id,
                )
                parent_room.object_counter += 1
                object.name = name
                obj_pbar.set_description(
                    f"object name: {object.name}, {object.object_id}"
                )
                object.pcd = self.mask_pcds[mask_idx]
                object.vertices = np.array(self.mask_pcds[mask_idx].points)[:, [0, 2]]
                object.embedding = self.mask_feats[mask_idx]
                floor.rooms[closest_room_idx].add_object(object)
                self.objects.append(object)

    def create_graph(self):
        """
        Create the full HOV-SG graph as a networkx graph
        """
        # add nodes to the graph
        for floor in self.floors:
            self.graph.add_node(floor, name="floor", type="floor")
            self.graph.add_edge(0, floor)
            for room in floor.rooms:
                self.graph.add_node(room, name="room", type="room")
                self.graph.add_edge(floor, room)
                for object in room.objects:
                    self.graph.add_node(object, name=object.name, type="object")
                    self.graph.add_edge(room, object)

    def save_graph(self, path):
        """
        Save the HOV-SG graph
        :param path: str, The path to save the graph
        """
        # create a folder for the graph
        if not os.path.exists(path):
            os.makedirs(path)
        # create a folder for floors, rooms and objects
        if not os.path.exists(os.path.join(path, "floors")):
            os.makedirs(os.path.join(path, "floors"))
        if not os.path.exists(os.path.join(path, "rooms")):
            os.makedirs(os.path.join(path, "rooms"))
        if not os.path.exists(os.path.join(path, "objects")):
            os.makedirs(os.path.join(path, "objects"))
        # save the graph
        for i, node in enumerate(self.graph.nodes(data=True)):
            topo_obj, node_dict = node
            if type(topo_obj) == Floor:
                topo_obj.save(os.path.join(path, "floors"))
            elif type(topo_obj) == Room:
                topo_obj.save(os.path.join(path, "rooms"))
            elif type(topo_obj) == Object:
                topo_obj.save(os.path.join(path, "objects"))
        print_status(
            "SAVE_HIERARCHICAL_GRAPH",
            "OK",
            path=path,
            floors=len(self.floors),
            rooms=len(self.rooms),
            objects=len(self.objects),
            floor_files=len(os.listdir(os.path.join(path, "floors"))),
            room_files=len(os.listdir(os.path.join(path, "rooms"))),
            object_files=len(os.listdir(os.path.join(path, "objects"))),
        )

    def load_graph(self, path):
        """
        Load the HOV-SG graph
        :param path: str, The path to load the graph
        """
        print(".. loading predicted graph")
        # load floors
        floor_files = os.listdir(os.path.join(path, "floors"))
        floor_files.sort()
        floor_files = sorted([f for f in floor_files if f.endswith(".ply")])
        for floor_file in floor_files:
            floor_file = floor_file.split(".")[0]
            floor = Floor(str(floor_file), name="floor_" + str(floor_file))
            floor.load(os.path.join(path, "floors"))
            self.floors.append(floor)
            self.graph.add_node(floor, name="floor_" + str(floor_file), type="floor")
            self.graph.add_edge(0, floor)
        print("# pred floors: ", len(self.floors))
        # load rooms
        room_files = os.listdir(os.path.join(path, "rooms"))
        room_files.sort()
        room_files = [f for f in room_files if f.endswith(".ply")]
        for room_file in room_files:
            room_file = room_file.split(".")[0]
            room = Room(str(room_file), room_file.split("_")[0])
            room.load(os.path.join(path, "rooms"))
            self.rooms.append(room)
            self.graph.add_node(room, name="room_" + str(room_file), type="room")
            self.graph.add_edge(self.floors[int(room_file.split("_")[0])], room)
            if isinstance(self.floors[int(room.floor_id)].rooms[0], str):
                self.floors[int(room.floor_id)].rooms = []
            self.floors[int(room.floor_id)].rooms.append(room)
        print("# pred rooms: ", len(self.rooms))
        # load objects
        object_files = os.listdir(os.path.join(path, "objects"))
        object_files.sort()
        object_files = [f for f in object_files if f.endswith(".ply")]
        for object_file in object_files:
            object_file = object_file.split(".")[0]
            room_id = "_".join(object_file.split("_")[:2])
            parent_room = None
            for room in self.rooms:
                if room.room_id == room_id:
                    parent_room = room
                    break
            assert (
                parent_room is not None
            ), f"Couldn't find the room with room id {room_id}"
            objectt = Object(
                str(object_file), room_id, name="object_" + str(object_file)
            )
            objectt.load(os.path.join(path, "objects"))
            objectt.room_id = room_id  # object_file.split("_")[1]
            self.objects.append(objectt)
            self.graph.add_node(
                objectt, name="object_" + str(object_file), type="object"
            )
            self.graph.add_edge(parent_room, objectt)
            # add object to the room
            parent_room.add_object(objectt)
        print("# pred objects: ", len(self.objects))
        print("-------------------")

    def build_graph(self, save_path=None):
        """
        Build the HOV-SG, by segmenting the floors, rooms, and objects and creating the graph.
        :param save_path: str, The path to save the intermediate results
        """
        print_status(
            "GRAPH_BUILD",
            "START",
            save_path=save_path,
            full_points=len(self.full_pcd.points),
            object_masks=len(self.mask_pcds),
        )
        print("segmenting floors...")
        print_status("FLOOR_SEGMENTATION", "START", input_points=len(self.full_pcd.points))
        self.segment_floors(save_path)
        print_status("FLOOR_SEGMENTATION", "OK", floors=len(self.floors))

        print("segmenting rooms...")
        for floor in self.floors:
            print_status(
                "ROOM_SEGMENTATION",
                "START",
                floor_id=floor.floor_id,
                floor_points=len(floor.pcd.points),
            )
            self.segment_rooms(floor, save_path)
            print_status(
                "ROOM_SEGMENTATION",
                "OK",
                floor_id=floor.floor_id,
                floor_rooms=len(floor.rooms),
                total_rooms=len(self.rooms),
            )

        print("segmenting/identifying objects...")
        print_status("OBJECT_SEGMENTATION", "START", masks=len(self.mask_pcds))
        self.segment_objects(save_path)
        print_status("OBJECT_SEGMENTATION", "OK", objects=len(self.objects))

        print("number of objects: ", len(self.objects))
        if self.cfg.pipeline.merge_objects_graph:
            # merge objects that close to each other with same name
            for room in tqdm(self.rooms):
                print("room: ", room.room_id)
                print(" number of objects before merging: ", len(room.objects))
                room.merge_objects()
                print(" number of objects after merging: ", len(room.objects))

        print("creating graph...")
        self.create_graph()
        print_status(
            "NETWORKX_GRAPH",
            "OK",
            nodes=self.graph.number_of_nodes(),
            edges=self.graph.number_of_edges(),
        )

        # create navigation graph for each floor
        if bool(self.cfg.pipeline.get("create_navigation", True)):
            print_status("NAVIGATION_GRAPH", "START", floors=len(self.floors))
            self.create_nav_graph()
            print_status(
                "NAVIGATION_GRAPH",
                "OK",
                path=os.path.join(self.cfg.main.save_path, "graph", "nav_graph"),
            )
        else:
            print_status(
                "NAVIGATION_GRAPH",
                "SKIP",
                reason="pipeline.create_navigation=false",
            )

        # save the graph
        self.save_graph(os.path.join(save_path, "graph"))
        print_status(
            "GRAPH_BUILD",
            "OK",
            floors=len(self.floors),
            rooms=len(self.rooms),
            objects=len(self.objects),
            path=os.path.join(save_path, "graph"),
        )

        print("# floors: ", len(self.floors))
        print("# rooms: ", len(self.rooms))
        print("# objects: ", len(self.objects))
        print("--> HOV-SG representation successfully built")

    def create_nav_graph(self):
        """
        Create the navigation graph for each floor and connect the floors together
        """
        last_nav_graph = None
        global_voronoi = None

        # create a folder for the resulting navigation graph
        nav_dir = os.path.join(self.cfg.main.save_path, "graph", "nav_graph")
        os.makedirs(nav_dir, exist_ok=True)

        # get pose list
        poses_list = []
        rejected_invalid_poses = 0
        rejected_pose_jumps = 0
        previous_pose = None
        max_translation_jump_m = float(
            self.cfg.pipeline.get("max_pose_translation_jump_m", float("inf"))
        )
        max_rotation_jump_deg = float(
            self.cfg.pipeline.get("max_pose_rotation_jump_deg", float("inf"))
        )
        for i in self._processing_frame_indices():
            _, _, pose, _, _ = self.dataset[i]
            pose = np.asarray(pose, dtype=np.float64)
            pose_valid, _ = self._validate_pose(pose)
            if not pose_valid:
                rejected_invalid_poses += 1
                continue
            if previous_pose is not None:
                translation_m, rotation_deg = self._pose_delta(previous_pose, pose)
                if (
                    translation_m > max_translation_jump_m
                    or rotation_deg > max_rotation_jump_deg
                ):
                    rejected_pose_jumps += 1
                    previous_pose = pose
                    continue
            poses_list.append(pose)
            previous_pose = pose

        if not poses_list:
            raise RuntimeError("No valid camera poses are available for navigation")

        print_status(
            "NAVIGATION_POSES",
            "OK",
            accepted=len(poses_list),
            rejected_invalid=rejected_invalid_poses,
            rejected_jump=rejected_pose_jumps,
            max_translation_jump_m=max_translation_jump_m,
            max_rotation_jump_deg=max_rotation_jump_deg,
        )

        if not self.floors:
            raise RuntimeError("Cannot create a navigation graph without floors")

        navigation_cfg = self.cfg.get("navigation", {})
        robot_cfg = self.cfg.get("robot", {})
        cell_size_m = float(navigation_cfg.get("cell_size_m", 0.03))
        pose_region_radius_m = float(
            navigation_cfg.get("pose_region_radius_m", 0.5)
        )
        camera_height_m = float(robot_cfg.get("camera_height_m", 1.5))
        if cell_size_m <= 0.0:
            raise ValueError("navigation.cell_size_m must be positive")
        if pose_region_radius_m <= 0.0:
            raise ValueError("navigation.pose_region_radius_m must be positive")
        if camera_height_m <= 0.0:
            raise ValueError("robot.camera_height_m must be positive")
        single_floor_mode = bool(
            self.cfg.pipeline.get("force_single_floor", False)
        ) or len(self.floors) == 1
        filter_pose_heights = bool(
            navigation_cfg.get("filter_pose_heights", not single_floor_mode)
        )
        infer_stairs = bool(
            navigation_cfg.get("infer_stairs", not single_floor_mode)
        )
        if single_floor_mode:
            filter_pose_heights = False
            infer_stairs = False

        print_status(
            "NAVIGATION_MODE",
            "OK",
            single_floor=single_floor_mode,
            filter_pose_heights=filter_pose_heights,
            infer_stairs=infer_stairs,
            camera_height_m=camera_height_m,
            pose_region_radius_m=pose_region_radius_m,
            cell_size_m=cell_size_m,
            input_poses=len(poses_list),
        )

        for floor_id, floor in enumerate(self.floors):
            print_status(
                "NAVIGATION_FLOOR",
                "START",
                floor_id=floor_id,
                floor_points=len(floor.pcd.points),
                poses=len(poses_list),
            )
            nav_graph = NavigationGraph(floor.pcd, cell_size=cell_size_m)
            upperbound = None
            if floor_id + 1 < len(self.floors):
                upperbound = self.floors[floor_id + 1].floor_zero_level
            floor_poses_list = nav_graph.get_floor_poses(
                floor,
                poses_list,
                upperbound,
                camera_height=camera_height_m,
                filter_by_height=filter_pose_heights,
            )

            sparse_stairs_voronoi = None
            if infer_stairs:
                stairs_candidate = nav_graph.get_stairs_graph_with_poses_v2(
                    floor, floor_id, poses_list, nav_dir
                )
                if stairs_candidate.number_of_nodes() > 0:
                    sparse_stairs_voronoi = stairs_candidate
            sparse_floor_voronoi = nav_graph.get_floor_graph(
                floor,
                floor_poses_list,
                nav_dir,
                cluster_pose_heights=False,
                pose_region_radius=pose_region_radius_m,
            )
            if (
                sparse_stairs_voronoi is not None
                and sparse_stairs_voronoi.number_of_nodes() > 0
            ):
                print(f"connecting stairs and floor {floor_id}")
                sparse_floor_voronoi = nav_graph.connect_stairs_and_floor_graphs(
                    sparse_stairs_voronoi, sparse_floor_voronoi, nav_dir
                )
            NavigationGraph.save_voronoi_graph(
                sparse_floor_voronoi, nav_dir, "sparse_voronoi"
            )
            print_status(
                "NAVIGATION_FLOOR",
                "OK",
                floor_id=floor_id,
                floor_poses=len(floor_poses_list),
                nodes=sparse_floor_voronoi.number_of_nodes(),
                edges=sparse_floor_voronoi.number_of_edges(),
                has_stairs=(
                    sparse_stairs_voronoi is not None
                    and sparse_stairs_voronoi.number_of_nodes() > 0
                ),
                output_directory=nav_dir,
            )

            if last_nav_graph is not None and last_nav_graph.has_stairs:
                print(f"connecting two floors {floor_id}")
                global_voronoi = nav_graph.connect_voronoi_graphs(
                    last_nav_graph.sparse_floor_voronoi, nav_graph.sparse_floor_voronoi
                )
            last_nav_graph = nav_graph

        if global_voronoi is None:
            global_voronoi = last_nav_graph.sparse_floor_voronoi

        NavigationGraph.save_voronoi_graph(global_voronoi, nav_dir, "global_nav_graph")
        print_status(
            "NAVIGATION_GLOBAL",
            "OK",
            nodes=global_voronoi.number_of_nodes(),
            edges=global_voronoi.number_of_edges(),
            output_directory=nav_dir,
        )

    def generate_room_names(
        self,
        generate_method: str = "label",
        default_room_types: List[str] = None,
    ):
        """Generate a name for each room node based on children nodes' embedding

        Args:
            generate_method (str): "label" or "obj_embedding" or "view_embedding"
            default_room_types (List[str]): optionally provide a list of default room types so that the
                                            room names can only be one of the provided options. When the
                                            generate_method is set to "embedding", this list is mandatory.
            clip_model (Any): when the generate_method is set to "embedding", a clip model needs to be
                              provided to the method.
            clip_feat_dim (int): when the generate_method is set to "embedding", the clip features dimension
                                 needs to be provided to this method
        """
        for i in range(len(self.rooms)):
            if generate_method in ["obj_embedding", "view_embedding"]:
                print(default_room_types)
                assert (
                    default_room_types is not None
                ), "You should provide a list of default room types"
                assert self.clip_model is not None, "You should provide a clip model"
                assert (
                    self.clip_feat_dim is not None
                ), "You should provide the clip features dimension"
            self.rooms: List[Room]
            if generate_method in ["obj_embedding", "label"]:
                self.rooms[i].infer_room_type_from_objects(
                    infer_method=generate_method,
                    default_room_types=default_room_types,
                    clip_model=self.clip_model,
                    clip_feat_dim=self.clip_feat_dim,
                )
            elif generate_method in ["view_embedding"]:
                self.rooms[i].infer_room_type_from_view_embedding(
                    default_room_types, self.clip_model, self.clip_feat_dim
                )
            else:
                return NotImplementedError

    def query_graph(self, query):
        """
        search in graph of the openmap with a text query
        """
        text_feats = get_text_feats_multiple_templates(
            [query], self.clip_model, self.clip_feat_dim
        )
        # compute similarity between the text query and the objects embeddings in the graph
        similarity = np.dot(text_feats, np.array([o.embedding for o in self.objects]).T)
        # similarity = compute_similarity(text_feats, np.array([o.embedding for o in self.objects]))
        # find top 5 similar objects
        top_index = np.argsort(similarity[0])[::-1][:5]
        # print the top 5 similar objects
        for i in top_index:
            print(self.objects[i].name, similarity[0])
            print("room: ", self.objects[i].room_id)
            obj_pcd = self.objects[i].pcd.paint_uniform_color([1, 0, 0])
            # find the room with a room id that matches the object's room id
            for room in self.rooms:
                if room.room_id == self.objects[i].room_id:
                    room_pcd = room.pcd
                    break
            o3d.visualization.draw_geometries([room_pcd, obj_pcd])

        # return the object with the highest similarity
        return self.objects[top_index[0]]

    def query_floor(self, query: str, query_method: str = "clip") -> int:
        """search a floor based on the number of the text query

        Args:
            query (str): a number in text format
            query_method (str): "clip" match the clip embeddings of the query text and the text description of all floors.
                                "gpt" provide the floor ids in the graph and the text query to a gpt agent, and ask for the
                                matching floor id.

        Returns:
            int: The target floor id in self.floors
        """
        # TODO: assume that the self.floors are ordered according to the floor level in ascending order. Check again.
        zero_levels_list = [x.floor_zero_level for x in self.floors]
        print("zero_levels_list: ", zero_levels_list)
        zero_level_order_ids = np.argsort(zero_levels_list)

        # check whether the query is a number that is an integer
        try:
            return zero_level_order_ids[int(query) - 1]
        except:
            if query_method == "clip":
                text_feats = get_text_feats_multiple_templates(
                    [query], self.clip_model, self.clip_feat_dim
                )
                floor_names = ["floor " + str(i) for i in range(len(self.floors))]
                floor_embs = get_text_feats_multiple_templates(
                    floor_names, self.clip_model, self.clip_feat_dim
                )
                sim_mat = np.dot(text_feats, floor_embs.T)
                # sim_mat = compute_similarity(text_feats, floor_embs)
                print(sim_mat)
                top_index = np.argsort(sim_mat[0])[::-1][0]
                return zero_level_order_ids[top_index]

            elif query_method == "gpt":
                floor_ids_list = [i + 1 for i in range(len(self.floors))]
                floor_id = infer_floor_id_from_query(floor_ids_list, query)
                return zero_level_order_ids[floor_id - 1]

    def query_room(
        self, query: str, floor_id: int = -1, query_method: str = "view_embedding"
    ) -> List[int]:
        """search a room node with a text query

        Args:
            query (str): a text describing the room
            floor_id (int): -1 means global search. 0-(max_floor - 1) means searching the target room on a specific floor.
            query_method (str): "label" use pre-defined label stored in the room node. "view_embedding" use the room embedding
                                stored in the room node. "children_embedding" use all children objects' embedding and find
                                the most representative one for the room.
        Returns:
            (Room): the target room ids in self.rooms which matches th equery the best
        """
        query_text_feats = get_text_feats_multiple_templates(
            [query], self.clip_model, self.clip_feat_dim
        )

        rooms_list = self.rooms
        if floor_id != -1:
            rooms_list = self.floors[floor_id].rooms
            # for room_id in self.floors[floor_id].rooms:
            #     for room in self.rooms:
            #         if room.room_id == room_id:
            #             rooms_list.append(room)
        rooms_list: List[Room]
        if query_method == "label":
            for room in rooms_list:
                assert (
                    room.name is not None
                ), "The name attribute for the room has not been generated"
            room_names_list = [room.name for room in rooms_list]
            print(room_names_list)
            room_embs = get_text_feats_multiple_templates(
                room_names_list, self.clip_model, self.clip_feat_dim
            )
            similarity = np.dot(query_text_feats, room_embs.T)
            # similarity = compute_similarity(query_text_feats, room_embs)
            top_index = np.argsort(similarity[0])[::-1]
            # print the top 5 matching rooms
            for i in top_index:
                print(rooms_list[i].name, similarity[0][i])
                print("room: ", rooms_list[i].room_id)

            same_sim_indices = []
            tar_sim = similarity[0, top_index[0]]
            same_sim_indices.append(top_index[0])
            for i in top_index[1:]:
                if np.abs(similarity[0, i] - tar_sim) < 1e-3:
                    same_sim_indices.append(i)

            target_rooms = [rooms_list[i] for i in same_sim_indices]
            target_room_ids = [target_room.room_id for target_room in target_rooms]
            target_ids = [
                i for i, x in enumerate(self.rooms) if x.room_id in target_room_ids
            ]
            return target_ids
        elif query_method == "view_embedding":
            room2query_sim = dict()
            for room in self.rooms:
                room_query_sim_median = np.max(
                    np.dot(query_text_feats, np.stack(room.embeddings).T)
                )
                # room_query_sim_median = np.max(compute_similarity(query_text_feats, np.stack(room.embeddings)))
                room2query_sim[room.room_id] = room_query_sim_median
            room2query_sim_sorted = {
                int(k.split("_")[-1]): v
                for k, v in sorted(
                    room2query_sim.items(), key=lambda item: item[1], reverse=True
                )
            }
            return list(room2query_sim_sorted.keys())[
                0 : min(len(room2query_sim_sorted), 3)
            ]  # return three highest-ranking rooms
        elif query_method == "children_embedding":
            return NotImplementedError

    def query_object(
        self,
        query: str,
        room_ids: List[int] = [],
        query_method: str = "clip",
        top_k: int = 1,
        negative_prompt: List[str] = [],
    ) -> Tuple[List[int], List[int]]:
        """search an object (from a room) with a text query

        Args:
            query (str): a description of the object
            room_ids (List[int], optional): The room ids. Defaults to [], which means search from all rooms.
            query_method (str, optional): "clip" means using clip features of the objects and the query to search.
                                          Defaults to "clip".
            top_k (int, optional): The number of top results to return. Default to 1.
            negative_prompt (List[str], optional): A list of categories used as negative prompt.


        Returns:
            Tuple[int, int]: The target object id in self.objects and the corresponding room id in self.rooms.
        """

        if query in negative_prompt:
            query_id = negative_prompt.index(query)
        else:
            query_id = None

        if query_id is None:
            query = [query, *negative_prompt]
            query_id = 0
        else:
            query = negative_prompt

        print(f"query_id: {query_id}")
        print(f"categories list: {query}")

        query_text_feats = get_text_feats_multiple_templates(
            query, self.clip_model, self.clip_feat_dim
        )  # (len(categories), feat_dim)
        print(f"text_feats.shape: {query_text_feats.shape}")
        print(query_text_feats[:, :10])

        room_ids_list = []
        for obj in self.objects:
            for i, room in enumerate(self.rooms):
                if obj.room_id == room.room_id:
                    room_ids_list.append(i)
                    break

        if len(room_ids) != 0:
            objects_list = []
            room_ids_list = []
            for i in room_ids:
                objects_list.extend(self.rooms[i].objects)
                room_ids_list.extend([i] * len(self.rooms[i].objects))
        objects_list: List[Object]
        if query_method == "clip":
            object_embs = np.array([obj.embedding for obj in objects_list])
            sim_mat = np.dot(query_text_feats, object_embs.T)
            # sim_mat = compute_similarity(query_text_feats, object_embs)  # (len(categories), len(objects_list))
            top_index = np.argsort(sim_mat[query_id])[::-1][:10]
            for i in top_index:
                print("object name, score: ", objects_list[i].name, sim_mat[0][i])
                print("object id: ", objects_list[i].object_id)

            # plt.hist(sim_mat.flatten(), bins=100)
            # plt.show()

            top_index = np.argsort(sim_mat[query_id])[::-1][:top_k]
            if len(negative_prompt) > 0:
                cls_ids = np.argmax(sim_mat, axis=0)  # category id for each object
                print(f"cls_ids: {cls_ids}")
                max_scores = np.max(sim_mat, axis=0)  # max scores for each object
                obj_ids = np.where(cls_ids == query_id)[
                    0
                ]  # find the obj ids that assign max score to the target category
                if len(obj_ids) > 0:
                    obj_scores = max_scores[obj_ids]
                    resort_ids = np.argsort(
                        -obj_scores
                    )  # sort the obj ids based on max score (descending)
                    top_index = obj_ids[resort_ids]  # get the top index
                    top_index = top_index[:top_k]

            target_object_id = [objects_list[i].object_id for i in top_index]
            target_room_id = [room_ids_list[i] for i in top_index]
            target_id = []
            for ti in target_object_id:
                target_id.append(
                    [i for i, x in enumerate(self.objects) if x.object_id == ti][0]
                )

            return target_id, target_room_id
        return NotImplementedError

    def query_hierarchy(
        self, query: str, top_k: int = 1
    ) -> Tuple[Floor, Room, List[Object]]:
        """return the target floor, room, and the list of top k objects

        Args:
            query (str): the long query like "object X in room Y on floor Z"
            top_k (int, optional): The number of top results to return. Default to 1.

        Returns:
            Tuple[Floor, List[Room], List[Object]]: return a floor object, a room object and a list of object objects
        """

        negative_labels = ["background"]

        floor_query, room_query, object_query = parse_hier_query(self.cfg, query)
        # log these in a txt file
        # with open("room_obj_query_log.txt", "a") as f:
        #     f.write(f"query: {query} -- {floor_query}, {room_query}, {object_query}\n")

        floor_id = self.query_floor(floor_query) if floor_query is not None else -1
        print(f"floor id: {floor_id}")
        room_ids = (
            self.query_room(room_query, floor_id=floor_id)
            if room_query is not None
            else []
        )
        print(f"room ids: {room_ids}")

        object_ids, room_ids = (
            self.query_object(
                object_query,
                room_ids=room_ids,
                top_k=top_k,
                negative_prompt=negative_labels,
            )
            if object_query is not None
            else ([], [])
        )
        print(f"object id: {object_ids}")

        return (
            self.floors[floor_id] if floor_id != -1 else None,
            [self.rooms[k] for k in room_ids],
            [self.objects[i] for i in object_ids],
        )

    def save_full_pcd(self, path):
        """
        Save the full pcd to disk
        :param path: str, The path to save the full pcd
        """
        if not os.path.exists(path):
            os.makedirs(path)
        output_path = os.path.join(path, "full_pcd.ply")
        success = o3d.io.write_point_cloud(output_path, self.full_pcd)
        print("full pcd saved to disk in {}".format(path))
        print_status(
            "SAVE_FULL_POINT_CLOUD",
            "OK" if success else "ERROR",
            path=output_path,
            points=len(self.full_pcd.points),
            exists=os.path.exists(output_path),
        )
        return None

    def load_full_pcd(self, path):
        """
        Load the full pcd from disk
        :param path: str, The path to load the full pcd
        """
        if not os.path.exists(path):
            print("full pcd not found in {}".format(path))
            return None
        self.full_pcd = o3d.io.read_point_cloud(os.path.join(path, "full_pcd.ply"))
        print(
            "full pcd loaded from disk with shape {}".format(
                np.asarray(self.full_pcd.points).shape
            )
        )
        return self.full_pcd

    def save_full_pcd_feats(self, path):
        """
        Save the full pcd with feats to disk
        :param path: str, The path to save the full pcd feats
        """
        if not os.path.exists(path):
            os.makedirs(path)
        # check if the full pcd feats is empty list
        saved_outputs = []
        if len(self.mask_feats) != 0:
            self.mask_feats = np.array(self.mask_feats)
            mask_feats_path = os.path.join(path, "mask_feats.pt")
            torch.save(
                torch.from_numpy(self.mask_feats), mask_feats_path
            )
            saved_outputs.append(mask_feats_path)
        if len(self.full_feats_array) != 0:
            full_feats_path = os.path.join(path, "full_feats.pt")
            torch.save(
                torch.from_numpy(self.full_feats_array),
                full_feats_path,
            )
            saved_outputs.append(full_feats_path)
        print("full pcd feats saved to disk in {}".format(path))
        print_status(
            "SAVE_FEATURES",
            "OK" if saved_outputs and all(os.path.exists(item) for item in saved_outputs) else "WARN",
            outputs=saved_outputs,
            mask_feature_shape=getattr(self.mask_feats, "shape", None),
            full_feature_shape=getattr(self.full_feats_array, "shape", None),
        )
        return None

    def load_full_pcd_feats(self, path, full_feats=False, normalize=True):
        """
        Load the full pcd with feats from disk
        :param path: str, The path to load the full pcd feats
        :param full_feats: bool, Whether to load the full feats or the mask feats
        :param normalize: bool, Whether to normalize the feats
        """
        if not os.path.exists(path):
            print("full pcd feats not found in {}".format(path))
            return None
        if full_feats:
            self.full_feats_array = torch.load(
                os.path.join(path, "full_feats.pt")
            ).float()
            if normalize:
                self.full_feats_array = (
                    torch.nn.functional.normalize(self.full_feats_array, p=2, dim=-1)
                    .cpu()
                    .numpy()
                )
            else:
                self.full_feats_array = self.full_feats_array.cpu().numpy()
            print(
                "full pcd feats loaded from disk with shape {}".format(
                    self.full_feats_array.shape
                )
            )
            return self.full_feats_array
        else:
            self.mask_feats = torch.load(os.path.join(path, "mask_feats.pt")).float()
            if normalize:
                self.mask_feats = (
                    torch.nn.functional.normalize(self.mask_feats, p=2, dim=-1)
                    .cpu()
                    .numpy()
                )
            else:
                self.mask_feats = self.mask_feats.cpu().numpy()
            print(
                "full pcd feats loaded from disk with shape {}".format(
                    self.mask_feats.shape
                )
            )
            return self.mask_feats

    def print_details(self):
        """
        Print the details of the graph
        """
        print("number of floors: ", len(self.floors))
        print("number of rooms: ", len(self.rooms))
        print("number of objects: ", len(self.objects))
        return None

    def save_masked_pcds(self, path, state="both"):
        """
        Save the masked pcds to disk
        :params state: str 'both' or 'objects' or 'full' to save the full masked pcds or only the objects.
        """
        # # remove any small pcds
        tqdm.write("-- removing small and empty masks --")
        masks_before_save_filter = len(self.mask_pcds)
        if len(self.mask_pcds) != len(self.mask_feats):
            raise RuntimeError(
                "mask point clouds and mask features must have equal length"
            )
        kept_mask_indices = [
            index
            for index, pcd in enumerate(self.mask_pcds)
            if not pcd.is_empty() and len(pcd.points) >= 50
        ]
        self.mask_pcds = [self.mask_pcds[index] for index in kept_mask_indices]
        self.mask_feats = [self.mask_feats[index] for index in kept_mask_indices]
        print_status(
            "MASK_SAVE_FILTER",
            "OK",
            masks_before=masks_before_save_filter,
            masks_after=len(self.mask_pcds),
            minimum_points=50,
        )

        if state == "both":
            if not os.path.exists(path):
                os.makedirs(path)
            objects_path = os.path.join(path, "objects")
            if not os.path.exists(objects_path):
                os.makedirs(objects_path)
            print("number of masked pcds: ", len(self.mask_pcds))
            print("number of mask_feats: ", len(self.mask_feats))
            object_write_results = []
            for i, pcd in enumerate(self.mask_pcds):
                object_path = os.path.join(objects_path, "pcd_{}.ply".format(i))
                object_write_results.append(
                    o3d.io.write_point_cloud(object_path, pcd)
                )

            masked_pcd = o3d.geometry.PointCloud()
            for pcd in self.mask_pcds:
                pcd.paint_uniform_color(np.random.rand(3))
                masked_pcd += pcd
            masked_output_path = os.path.join(path, "masked_pcd.ply")
            masked_write_success = o3d.io.write_point_cloud(
                masked_output_path, masked_pcd
            )
            print("masked pcds saved to disk in {}".format(path))
            print_status(
                "SAVE_MASKED_POINT_CLOUDS",
                (
                    "OK"
                    if masked_write_success and all(object_write_results)
                    else "ERROR"
                ),
                directory=objects_path,
                object_files=len(object_write_results),
                successful_object_files=sum(object_write_results),
                combined_path=masked_output_path,
                combined_points=len(masked_pcd.points),
                exists=os.path.exists(masked_output_path),
            )

        elif state == "objects":
            if not os.path.exists(path):
                os.makedirs(path)
            for i, pcd in enumerate(self.mask_pcds):
                o3d.io.write_point_cloud(
                    os.path.join(objects_path, "pcd_{}.ply".format(i)), pcd
                )
            print("masked pcds saved to disk in {}".format(path))

        elif state == "full":
            if not os.path.exists(path):
                os.makedirs(path)
            masked_pcd = o3d.geometry.PointCloud()
            for pcd in self.mask_pcds:
                pcd.paint_uniform_color(np.random.rand(3))
                masked_pcd += pcd
            o3d.io.write_point_cloud(os.path.join(path, "masked_pcd.ply"), masked_pcd)
            print("masked pcds saved to disk in {}".format(path))

    def load_masked_pcds(self, path):
        """
        Load the masked pcds from disk
        """
        # make sure that self.mask_feats is already loaded
        if len(self.mask_feats) == 0:
            print("load full pcd feats first")
            return None
        if os.path.exists(os.path.join(path, "objects")):
            self.mask_pcds = []
            number_of_pcds = len(os.listdir(os.path.join(path, "objects")))
            not_found = []
            for i in range(number_of_pcds):
                if os.path.exists(
                    os.path.join(path, "objects", "pcd_{}.ply".format(i))
                ):
                    self.mask_pcds.append(
                        o3d.io.read_point_cloud(
                            os.path.join(path, "objects", "pcd_{}.ply".format(i))
                        )
                    )
                else:
                    print("masked pcd {} not found in {}".format(i, path))
                    not_found.append(i)
            print(
                "number of masked pcds loaded from disk {}".format(len(self.mask_pcds))
            )
            # remove masks_feats that are not found
            self.mask_feats = np.delete(self.mask_feats, not_found, axis=0)
            print(
                "number of mask_feats loaded from disk {}".format(len(self.mask_feats))
            )
            return self.mask_pcds
        else:
            print("masked pcds for objects not found in {}".format(path))
            return None

    def transform(self, transform):
        """
        Transform the openmap full pcd and masked pcds
        :param transform: np.ndarray, The transformation matrix
        """
        self.full_pcd.transform(transform)
        for i, pcd in enumerate(self.mask_pcds):
            self.mask_pcds[i].transform(transform)
        return None

    def visualize_instances(self):
        """
        visualize the instance of obejcts in the graph
        """
        all_objects_pcd = o3d.geometry.PointCloud()
        number_of_objects = 0
        for i, node in enumerate(self.graph.nodes):
            if type(node) == Object:
                print("object name: ", node.name, node.object_id)
                print("number of points: ", len(node.pcd.points))
                all_objects_pcd += node.pcd
                number_of_objects += 1
        print("number of objects: ", number_of_objects)
        o3d.visualization.draw_geometries([all_objects_pcd])
        return None
