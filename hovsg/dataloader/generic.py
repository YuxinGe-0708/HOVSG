
import os
from abc import ABC, abstractmethod
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import cv2

from torch.utils.data import Dataset, DataLoader
from hovsg.utils.pose_utils import as_pose_matrix

class RGBDDataset(Dataset, ABC):
    """
    Abstract class for RGBD datasets.

    This class provides a base structure for loading RGBD datasets. Subclasses
    need to implement the abstract methods to handle specific dataset formats.
    """

    def __init__(self, cfg):
        """
        Args:
            root_dir: Path to the root directory containing the dataset.
            mode: "train", "val", or "test" depending on the data split.
            transforms: Optional transformations to apply to the data.
        """
        self.root_dir = cfg["root_dir"]
        self.transforms = cfg["transforms"]
        self.rgb_intrinsics = None
        self.depth_intrinsics = None
        self.scale = None
        self.data_list = self._get_data_list()

    @abstractmethod
    def _get_data_list(self):
        """
        This method should be implemented by subclasses to define how to 
        get a list of data samples (RGB and depth image paths) based on the 
        dataset format and mode (train, val, test).
        """
        pass

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """
            Get a data sample based on the given index.
        """
        pass

    def _load_image(self, path):
        """
        This method should be implemented by subclasses to load the RGB image 
        based on the dataset format (e.g., OpenCV, PIL).
        """
        pass

    def _load_depth(self, path):
        """
        This method should be implemented by subclasses to load the depth image 
        based on the dataset format (e.g., OpenCV, PIL).
        """
        pass
    
    def _load_pose(self, path):
        """
        This method should be implemented by subclasses to load the camera pose 
        based on the dataset format.
        """
        pass
    
    def _load_rgb_intrinsics(self, path):
        """
        This method should be implemented by subclasses to load the RGB camera 
        intrinsics based on the dataset format.
        """
        pass
    
    def _load_depth_intrinsics(self, path):
        """
        This method should be implemented by subclasses to load the depth camera 
        intrinsics based on the dataset format.
        """
        pass
    
    def depth_to_meters(self, depth):
        """Convert raw depth using units per metre (not metres per unit)."""
        if self.scale is None or not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Depth scale must be finite and positive")
        return np.asarray(depth, dtype=np.float32) / self.scale

    def get_valid_depth_mask(self, depth, max_distance=None):
        depth_m = self.depth_to_meters(depth)
        if depth_m.ndim != 2:
            raise ValueError("Depth must be a single-channel image")
        valid = np.isfinite(depth_m) & (depth_m > 0)
        if max_distance is not None:
            if np.isnan(max_distance) or max_distance <= 0:
                raise ValueError("filter_distance must be positive")
            valid &= depth_m <= max_distance
        return depth_m, valid

    def _unproject_pixels(self, x, y, z):
        camera_matrix = np.asarray(self.depth_intrinsics)
        if (camera_matrix.shape not in ((3, 3), (4, 4))
                or not np.isfinite(camera_matrix).all()
                or camera_matrix[0, 0] <= 0 or camera_matrix[1, 1] <= 0):
            raise ValueError("Invalid depth camera intrinsics")
        return np.column_stack((
            (x - camera_matrix[0, 2]) * z / camera_matrix[0, 0],
            (y - camera_matrix[1, 2]) * z / camera_matrix[1, 1],
            z,
        ))

    def create_pcd(self, rgb, depth, camera_pose=None, mask_img=False,
                   filter_distance=np.inf, return_pixel_indices=False):
        """
        Create Open3D point cloud from RGB and depth images, and camera pose. filter_distance is used to filter out
        points that are further than a certain distance.
        :param rgb (pil image): RGB image
        :param depth (pil image): Depth image
        :param camera_pose (np.array): Camera pose
        :param mask_img (bool): Mask image
        :param filter_distance (float): Filter distance
        :param return_pixel_indices (bool): Also return flat row-major source pixels.
        :return: Point cloud, or (point cloud, pixel indices) when requested.
        """
        # Keep the exact row-major pixel -> point mapping for feature lookup.
        rgb = np.asarray(rgb)
        depth_m, valid = self.get_valid_depth_mask(depth, filter_distance)
        if rgb.shape[:2] != depth_m.shape:
            if mask_img:
                raise ValueError("Mask and depth resolutions must match")
            # Legacy benchmark loaders assume registered images at different sizes.
            rgb = cv2.resize(rgb, (depth_m.shape[1], depth_m.shape[0]),
                             interpolation=cv2.INTER_AREA)
        if mask_img:
            if rgb.ndim != 2:
                raise ValueError("Instance mask must be single-channel")
            valid &= rgb.astype(bool)
        elif rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("RGB image must have three channels")
        pixel_indices = np.flatnonzero(valid).astype(np.int64, copy=False)
        y, x = np.unravel_index(pixel_indices, depth_m.shape)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            self._unproject_pixels(x, y, depth_m[valid]))
        if not mask_img:
            pcd.colors = o3d.utility.Vector3dVector(rgb[valid].astype(np.uint8) / 255.0)
        if camera_pose is not None:
            pcd.transform(as_pose_matrix(camera_pose))
        if return_pixel_indices:
            return pcd, pixel_indices
        return pcd
    
    def create_3d_masks(
        self,
        masks,
        depth,
        full_pcd,
        full_pcd_tree,
        camera_pose,
        down_size=0.02,
        filter_distance=None,
        association_distance=None,
    ):
        """
        create 3d masks from 2D masks
        Args:
            masks: list of 2D masks
            depth: depth image
            full_pcd: full point cloud
            full_pcd_tree: KD-Tree of full point cloud
            camera_pose: camera pose
            down_size: voxel size for downsampling
            filter_distance: maximum metric depth used to create each mask
            association_distance: maximum distance from a mask point to its
                nearest point in the global map; None disables this gate
        Returns:
            list of 3D masks as Open3D point clouds
        """
        pcd_list = []
        pcd = np.asarray(full_pcd.points)
        depth = np.array(depth)
        for i in range(len(masks)):
            # get the mask
            mask = masks[i]["segmentation"]
            mask = np.array(mask)
            # create pcd from mask
            pcd_masked = self.create_pcd(
                mask,
                depth,
                camera_pose,
                mask_img=True,
                filter_distance=filter_distance,
            )
            # using KD-Tree to find the nearest points in the point cloud
            pcd_masked = np.asarray(pcd_masked.points)
            if pcd_masked.shape[0] == 0:
                pcd_list.append(o3d.geometry.PointCloud())
                continue

            dist, indices = full_pcd_tree.query(pcd_masked, k=1, workers=-1)
            valid = np.isfinite(dist)
            if association_distance is not None and np.isfinite(
                association_distance
            ):
                if association_distance <= 0:
                    raise ValueError("association_distance must be positive")
                valid &= dist <= association_distance

            indices = np.unique(indices[valid])
            if indices.shape[0] == 0:
                pcd_list.append(o3d.geometry.PointCloud())
                continue

            pcd_masked = pcd[indices]
            pcd_mask = o3d.geometry.PointCloud()
            pcd_mask.points = o3d.utility.Vector3dVector(pcd_masked)
            colors = np.asarray(full_pcd.colors)
            if colors.shape[0] == pcd.shape[0]:
                colors = colors[indices]
                pcd_mask.colors = o3d.utility.Vector3dVector(colors)
            pcd_mask = pcd_mask.voxel_down_sample(voxel_size=down_size)
            pcd_list.append(pcd_mask)
        return pcd_list
