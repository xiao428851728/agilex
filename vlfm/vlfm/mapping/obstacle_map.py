# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Union

import cv2
import numpy as np
from frontier_exploration.frontier_detection import detect_frontier_waypoints
from frontier_exploration.utils.fog_of_war import reveal_fog_of_war

from vlfm.mapping.base_map import BaseMap
from vlfm.utils.geometry_utils import extract_yaw, get_point_cloud, transform_points
from vlfm.utils.img_utils import fill_small_holes

# 在原有代码中插入去噪功能
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree

def sor_filter(point_cloud: np.ndarray, k_neighbors: int = 20, std_dev_thresh: float = 1.0) -> np.ndarray:
    """
    Statistical Outlier Removal (SOR) filter for point cloud denoising.
    
    Args:
        point_cloud (np.ndarray): The input point cloud with shape (N, 3).
        k_neighbors (int): Number of neighbors to consider for each point. Default is 20.
        std_dev_thresh (float): Standard deviation threshold for outlier removal. Default is 1.0.
    
    Returns:
        np.ndarray: The filtered (denoised) point cloud.
    """
    neighbors = NearestNeighbors(n_neighbors=k_neighbors)
    neighbors.fit(point_cloud)
    distances, _ = neighbors.kneighbors(point_cloud)
    
    mean_distances = np.mean(distances, axis=1)
    std_dev_distances = np.std(distances, axis=1)
    
    inlier_mask = std_dev_distances < std_dev_thresh
    return point_cloud[inlier_mask]

def voxel_grid_filter(point_cloud: np.ndarray, voxel_size: float = 0.1) -> np.ndarray:
    """
    Voxel Grid Downsampling filter for point cloud.
    
    Args:
        point_cloud (np.ndarray): The input point cloud with shape (N, 3).
        voxel_size (float): The size of each voxel. Default is 0.1.
    
    Returns:
        np.ndarray: The downsampled point cloud.
    """
    voxel_grid = {}
    for point in point_cloud:
        voxel_idx = tuple(np.floor(point / voxel_size).astype(int))
        if voxel_idx not in voxel_grid:
            voxel_grid[voxel_idx] = []
        voxel_grid[voxel_idx].append(point)
    
    downsampled_points = np.array([np.mean(voxel, axis=0) for voxel in voxel_grid.values()])
    return downsampled_points

def compute_normals(point_cloud: np.ndarray, k_neighbors: int = 20) -> np.ndarray:
    """
    Computes the surface normals for each point in the point cloud.
    
    Args:
        point_cloud (np.ndarray): The input point cloud with shape (N, 3).
        k_neighbors (int): Number of neighbors to consider for normal computation. Default is 20.
    
    Returns:
        np.ndarray: The computed normals (N, 3).
    """
    neighbors = NearestNeighbors(n_neighbors=k_neighbors)
    neighbors.fit(point_cloud)
    _, indices = neighbors.kneighbors(point_cloud)
    
    normals = []
    for idx in range(point_cloud.shape[0]):
        neighborhood = point_cloud[indices[idx]]
        pca = PCA(n_components=3)
        pca.fit(neighborhood - point_cloud[idx])  # Center the neighborhood
        normal = pca.components_[-1]  # Take the last component (smallest eigenvalue)
        normals.append(normal)
    
    return np.array(normals)

def remove_bad_normals(point_cloud: np.ndarray, normals: np.ndarray, angle_thresh: float = 30.0) -> np.ndarray:
    """
    Removes points with normals deviating from the majority normal direction.
    
    Args:
        point_cloud (np.ndarray): The input point cloud with shape (N, 3).
        normals (np.ndarray): The computed normals for the point cloud.
        angle_thresh (float): The angle threshold (in degrees) for normal deviation. Default is 30 degrees.
    
    Returns:
        np.ndarray: The filtered point cloud.
    """
    # Use k-d tree for efficient nearest neighbor search
    tree = cKDTree(point_cloud)
    filtered_points = []
    for i, normal in enumerate(normals):
        # Find neighbors and check if their normal is too different
        neighbors_idx = tree.query(point_cloud[i], k=5)[1]  # Find 5 nearest neighbors
        angle_differences = np.arccos(np.clip(np.dot(normals[neighbors_idx], normal), -1.0, 1.0))
        if np.all(angle_differences < np.radians(angle_thresh)):
            filtered_points.append(point_cloud[i])
    
    return np.array(filtered_points)

def gaussian_filter(point_cloud: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    """
    Applies a Gaussian filter to smooth the point cloud.
    
    Args:
        point_cloud (np.ndarray): The input point cloud with shape (N, 3).
        sigma (float): The standard deviation of the Gaussian filter. Default is 0.1.
    
    Returns:
        np.ndarray: The smoothed point cloud.
    """
    from scipy.ndimage import gaussian_filter1d
    
    smoothed_points = point_cloud.copy()
    smoothed_points[:, 0] = gaussian_filter1d(smoothed_points[:, 0], sigma)
    smoothed_points[:, 1] = gaussian_filter1d(smoothed_points[:, 1], sigma)
    smoothed_points[:, 2] = gaussian_filter1d(smoothed_points[:, 2], sigma)
    
    return smoothed_points

def denoise_point_cloud(point_cloud: np.ndarray, voxel_size: float = 0.3, k_neighbors: int = 15, std_dev_thresh: float = 1.0, angle_thresh: float = 30.0, sigma: float = 0.1) -> np.ndarray:
    """
    Comprehensive point cloud denoising combining multiple techniques.
    
    Args:
        point_cloud (np.ndarray): The input point cloud with shape (N, 3).
        voxel_size (float): Voxel size for downsampling. Default is 0.1.
        k_neighbors (int): Number of neighbors for SOR and normal computation. Default is 20.
        std_dev_thresh (float): Standard deviation threshold for SOR. Default is 1.0.
        angle_thresh (float): Angle threshold for normal deviation. Default is 30 degrees.
        sigma (float): Standard deviation for Gaussian filtering. Default is 0.1.
    
    Returns:
        np.ndarray: The denoised point cloud.
    """
    # Step 1: Remove outliers using SOR
    point_cloud = sor_filter(point_cloud, k_neighbors, std_dev_thresh)
    
    # Step 2: Downsample using Voxel Grid
    point_cloud = voxel_grid_filter(point_cloud, voxel_size)
    
    # Step 3: Compute normals and remove bad normals
    normals = compute_normals(point_cloud, k_neighbors)
    point_cloud = remove_bad_normals(point_cloud, normals, angle_thresh)
    
    # Step 4: Apply Gaussian filter for smoothing
    point_cloud = gaussian_filter(point_cloud, sigma)
    
    return point_cloud


class ObstacleMap(BaseMap):
    """Generates two maps; one representing the area that the robot has explored so far,
    and another representing the obstacles that the robot has seen so far.
    """

    _map_dtype: np.dtype = np.dtype(bool)
    _frontiers_px: np.ndarray = np.array([])
    frontiers: np.ndarray = np.array([])
    radius_padding_color: tuple = (100, 100, 100)

    def __init__(
        self,
        min_height: float,
        max_height: float,
        agent_radius: float,
        area_thresh: float = 3.0,  # square meters
        hole_area_thresh: int = 100000,  # square pixels
        size: int = 1000,
        pixels_per_meter: int = 20,
    ):
        super().__init__(size, pixels_per_meter)
        self.explored_area = np.zeros((size, size), dtype=bool)
        self._map = np.zeros((size, size), dtype=bool)
        self._navigable_map = np.zeros((size, size), dtype=bool)
        self._min_height = min_height
        self._max_height = max_height
        self._area_thresh_in_pixels = area_thresh * (self.pixels_per_meter**2)
        self._hole_area_thresh = hole_area_thresh
        kernel_size = self.pixels_per_meter * agent_radius * 2
        # round kernel_size to nearest odd number
        kernel_size = int(kernel_size) + (int(kernel_size) % 2 == 0)
        self._navigable_kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def reset(self) -> None:
        super().reset()
        self._navigable_map.fill(0)
        self.explored_area.fill(0)
        self._frontiers_px = np.array([])
        self.frontiers = np.array([])

    def update_map(
        self,
        depth: Union[np.ndarray, Any],
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
        topdown_fov: float,
        explore: bool = True,
        update_obstacles: bool = True,
    ) -> None:
        """
        Adds all obstacles from the current view to the map. Also updates the area
        that the robot has explored so far.

        Args:
            depth (np.ndarray): The depth image to use for updating the object map. It
                is normalized to the range [0, 1] and has a shape of (height, width).

            tf_camera_to_episodic (np.ndarray): The transformation matrix from the
                camera to the episodic coordinate frame.
            min_depth (float): The minimum depth value (in meters) of the depth image.
            max_depth (float): The maximum depth value (in meters) of the depth image.
            fx (float): The focal length of the camera in the x direction.
            fy (float): The focal length of the camera in the y direction.
            topdown_fov (float): The field of view of the depth camera projected onto
                the topdown map.
            explore (bool): Whether to update the explored area.
            update_obstacles (bool): Whether to update the obstacle map.
        """
        if update_obstacles:
            if self._hole_area_thresh == -1:
                filled_depth = depth.copy()
                filled_depth[depth == 0] = 1.0
            else:
                filled_depth = fill_small_holes(depth, self._hole_area_thresh)
            scaled_depth = filled_depth * (max_depth - min_depth) + min_depth
            mask = scaled_depth < max_depth
            point_cloud_camera_frame = get_point_cloud(scaled_depth, mask, 388.85, 388.85)  #modified

            #
            print("Fx,Fy:", fx, fy)
            #
            # Modified
            # =========================================================
            # ⚡️ 核心修改：沿着相机朝向（Z轴）拉长 2 倍
            # =========================================================
            # 注意：这里操作的是 index 2 (Z轴)，而不是 index 0 (X轴)
            # 如果你操作 index 0，画面会变宽；操作 index 2，物体会变远
            # point_cloud_camera_frame[:, 2] *= 8.0  
            # point_cloud_camera_frame[:, 0] *= 2.0
            # =========================================================
            point_cloud_episodic_frame = transform_points(tf_camera_to_episodic, point_cloud_camera_frame)
            # 新增去噪功能
            
            # 松灵小车去噪使用下面的参数：
            # point_cloud_episodic_frame = denoise_point_cloud(point_cloud_episodic_frame, voxel_size=0.12, k_neighbors=10, std_dev_thresh=1.5, angle_thresh=30.0, sigma=0.1)
            
            #联想六足狗去噪使用下面的参数：
            point_cloud_episodic_frame = denoise_point_cloud(point_cloud_episodic_frame, voxel_size=0.2, k_neighbors=20, std_dev_thresh=0.3, angle_thresh=30.0, sigma=0.1)

            obstacle_cloud = filter_points_by_height(point_cloud_episodic_frame, 0.74, 0.75) #self._min_height, self._max_height)

            # obstacle_cloud = point_cloud_episodic_frame = denoise_point_cloud(point_cloud_episodic_frame, voxel_size=0.1, k_neighbors=20, std_dev_thresh=1.0, angle_thresh=30.0, sigma=0.1)
            # TODO: 把 cloud 存成 ply 看一看是不是可信
            # ================= [插入这段代码保存 PLY (自动编号版)] =================
            try:
                import os
                save_dir = "vlfm_vis"
                base_name = "debug_obstacle_cloud"
                ext = ".ply"
                os.makedirs(save_dir, exist_ok=True)
                
                # --- 核心逻辑：自动寻找不重复的文件名 ---
                counter = 0
                while True:
                    # 第一张叫 debug_obstacle_cloud.ply
                    # 第二张叫 debug_obstacle_cloud (1).ply ... 以此类推
                    suffix = f" ({counter})" if counter > 0 else ""
                    save_path = os.path.join(save_dir, f"{base_name}{suffix}{ext}")
                    if not os.path.exists(save_path):
                        break
                    counter += 1
                # --------------------------------------

                # 简单的 PLY 头 (顶格写!)
                num_points = obstacle_cloud.shape[0]
                header = f"""ply
format ascii 1.0
element vertex {num_points}
property float x
property float y
property float z
end_header"""
                
                # 保存
                # np.savetxt(save_path, obstacle_cloud, fmt="%.4f", header=header, comments="")
                print(f"💾 Saved PLY: {save_path} ({num_points} points)")
                
            except Exception as e:
                print(f"❌ Save PLY Error: {e}")
            # =================================================================
# ================= [新增调试：保存 Camera 和 Episodic 点云] =================
            try:
                import os
                save_dir = "vlfm_vis"
                os.makedirs(save_dir, exist_ok=True)
                
                # 定义要保存的列表：(点云数据, 文件名前缀)
                debug_clouds = [
                    (point_cloud_camera_frame, "debug_camera_frame"),
                    (point_cloud_episodic_frame, "debug_episodic_frame")
                ]

                for cloud_data, prefix in debug_clouds:
                    num_pts = cloud_data.shape[0]
                    if num_pts > 0:
                        # 1. 自动寻找可用文件名 (防止覆盖)
                        counter = 0
                        while True:
                            file_path = os.path.join(save_dir, f"{prefix}_{counter}.ply")
                            if not os.path.exists(file_path): break
                            counter += 1
                        
                        # 2. 生成 PLY Header (注意：字符串内容必须顶格!)
                        header = f"""ply
format ascii 1.0
element vertex {num_pts}
property float x
property float y
property float z
end_header"""
                        
                        # 3. 保存文件
                        # np.savetxt(file_path, cloud_data, fmt="%.4f", header=header, comments="")
                        print(f"💾 [DEBUG] Saved: {file_path} ({num_pts} pts)")
                    else:
                        print(f"⚠️ [DEBUG] {prefix} 是空的 (0 points)，跳过保存。")

            except Exception as e:
                print(f"❌ Save Debug PLY Error: {e}")
            # ========================================================================

            # Populate topdown map with obstacle locations
            xy_points = obstacle_cloud[:, :2]
            pixel_points = self._xy_to_px(xy_points)
            self._map[pixel_points[:, 1], pixel_points[:, 0]] = 1

            # Update the navigable area, which is an inverse of the obstacle map after a
            # dilation operation to accommodate the robot's radius.
            self._navigable_map = 1 - cv2.dilate(
                self._map.astype(np.uint8),
                self._navigable_kernel,
                iterations=1,
            ).astype(bool)

        if not explore:
            return

        # Update the explored area
        agent_xy_location = tf_camera_to_episodic[:2, 3]
        agent_pixel_location = self._xy_to_px(agent_xy_location.reshape(1, 2))[0]
        new_explored_area = reveal_fog_of_war(
            top_down_map=self._navigable_map.astype(np.uint8),
            current_fog_of_war_mask=np.zeros_like(self._map, dtype=np.uint8),
            current_point=agent_pixel_location[::-1],
            current_angle=-extract_yaw(tf_camera_to_episodic),
            fov=np.rad2deg(topdown_fov),
            max_line_len=max_depth * self.pixels_per_meter,
        )
        new_explored_area = cv2.dilate(new_explored_area, np.ones((3, 3), np.uint8), iterations=1)
        self.explored_area[new_explored_area > 0] = 1
        self.explored_area[self._navigable_map == 0] = 0
        contours, _ = cv2.findContours(
            self.explored_area.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if len(contours) > 1:
            min_dist = np.inf
            best_idx = 0
            for idx, cnt in enumerate(contours):
                dist = cv2.pointPolygonTest(cnt, tuple([int(i) for i in agent_pixel_location]), True)
                if dist >= 0:
                    best_idx = idx
                    break
                elif abs(dist) < min_dist:
                    min_dist = abs(dist)
                    best_idx = idx
            new_area = np.zeros_like(self.explored_area, dtype=np.uint8)
            cv2.drawContours(new_area, contours, best_idx, 1, -1)  # type: ignore
            self.explored_area = new_area.astype(bool)

        # Compute frontier locations
        self._frontiers_px = self._get_frontiers()
        if len(self._frontiers_px) == 0:
            self.frontiers = np.array([])
        else:
            self.frontiers = self._px_to_xy(self._frontiers_px)

    def _get_frontiers(self) -> np.ndarray:
        """Returns the frontiers of the map."""
        # Dilate the explored area slightly to prevent small gaps between the explored
        # area and the unnavigable area from being detected as frontiers.
        explored_area = cv2.dilate(
            self.explored_area.astype(np.uint8),
            np.ones((5, 5), np.uint8),
            iterations=1,
        )
        frontiers = detect_frontier_waypoints(
            self._navigable_map.astype(np.uint8),
            explored_area,
            self._area_thresh_in_pixels,
        )
        return frontiers

    def visualize(self) -> np.ndarray:
        """Visualizes the map."""
        vis_img = np.ones((*self._map.shape[:2], 3), dtype=np.uint8) * 255
        # Draw explored area in light green
        vis_img[self.explored_area == 1] = (200, 255, 200)
        # Draw unnavigable areas in gray
        vis_img[self._navigable_map == 0] = self.radius_padding_color
        # Draw obstacles in black
        vis_img[self._map == 1] = (0, 0, 0)
        # Draw frontiers in blue (200, 0, 0)
        for frontier in self._frontiers_px:
            cv2.circle(vis_img, tuple([int(i) for i in frontier]), 5, (200, 0, 0), 2)

        vis_img = cv2.flip(vis_img, 0)

        if len(self._camera_positions) > 0:
            self._traj_vis.draw_trajectory(
                vis_img,
                self._camera_positions,
                self._last_camera_yaw,
            )

        return vis_img


def filter_points_by_height(points: np.ndarray, min_height: float, max_height: float) -> np.ndarray:
    return points[(points[:, 2] >= min_height) & (points[:, 2] <= max_height)]
