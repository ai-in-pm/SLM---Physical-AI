import torch
import torch.nn as nn
import numpy as np
import open3d as o3d
from typing import Dict, List, Optional, Union, Tuple
import cv2
from ultralytics import YOLO
import logging

class SensorProcessor(nn.Module):
    """
    Processes and fuses data from multiple sensors including LiDAR, depth cameras,
    and other environmental sensors
    """
    def __init__(
        self,
        use_lidar: bool = True,
        use_depth: bool = True,
        use_imu: bool = True,
        yolo_model: str = "yolov8n.pt",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Initialize sensors
        self.use_lidar = use_lidar
        self.use_depth = use_depth
        self.use_imu = use_imu
        
        # Initialize YOLO for object detection and tracking
        if yolo_model:
            self.yolo = YOLO(yolo_model)
        
        # Point cloud processing
        self.pcd_processor = o3d.geometry.PointCloud()
        
        # Sensor calibration matrices
        self.calibration = {}
        
        # Object tracking
        self.tracker = cv2.TrackerCSRT_create()
        self.tracked_objects = {}
    
    def process_lidar(
        self,
        point_cloud: np.ndarray,
        downsample: bool = True
    ) -> Dict[str, Union[np.ndarray, List]]:
        """
        Process LiDAR point cloud data
        Args:
            point_cloud: Input point cloud data
            downsample: Whether to downsample the point cloud
        Returns:
            Processed point cloud data and detected objects
        """
        try:
            # Convert to Open3D format
            self.pcd_processor.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
            
            if downsample:
                self.pcd_processor = self.pcd_processor.voxel_down_sample(
                    voxel_size=0.05
                )
            
            # Remove outliers
            cl, ind = self.pcd_processor.remove_statistical_outlier(
                nb_neighbors=20,
                std_ratio=2.0
            )
            
            # Ground plane detection
            plane_model, inliers = self.pcd_processor.segment_plane(
                distance_threshold=0.1,
                ransac_n=3,
                num_iterations=1000
            )
            
            # Cluster objects
            clusters = self.pcd_processor.cluster_dbscan(
                eps=0.3,
                min_points=10
            )
            
            return {
                "processed_cloud": np.asarray(self.pcd_processor.points),
                "ground_plane": plane_model,
                "objects": clusters
            }
            
        except Exception as e:
            self.logger.error(f"Error processing LiDAR data: {str(e)}")
            return {"error": str(e)}
    
    def process_depth(
        self,
        depth_image: np.ndarray,
        camera_intrinsics: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Process depth camera data
        Args:
            depth_image: Input depth image
            camera_intrinsics: Camera intrinsic parameters
        Returns:
            Processed depth data and 3D points
        """
        try:
            # Convert depth to 3D points
            rows, cols = depth_image.shape
            c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
            
            if camera_intrinsics is not None:
                fx = camera_intrinsics[0, 0]
                fy = camera_intrinsics[1, 1]
                cx = camera_intrinsics[0, 2]
                cy = camera_intrinsics[1, 2]
            else:
                # Use default values
                fx = fy = 525.0
                cx = cols / 2
                cy = rows / 2
            
            z = depth_image
            x = (c - cx) * z / fx
            y = (r - cy) * z / fy
            
            points_3d = np.stack((x, y, z), axis=-1)
            
            # Filter invalid points
            valid_mask = z > 0
            filtered_points = points_3d[valid_mask]
            
            return {
                "points_3d": filtered_points,
                "depth_filtered": depth_image * valid_mask
            }
            
        except Exception as e:
            self.logger.error(f"Error processing depth data: {str(e)}")
            return {"error": str(e)}
    
    def track_objects(
        self,
        frame: np.ndarray,
        detections: Optional[List] = None
    ) -> Dict[str, List]:
        """
        Track objects across frames
        Args:
            frame: Current video frame
            detections: Optional list of object detections
        Returns:
            Dictionary of tracked objects and their trajectories
        """
        try:
            if detections is None:
                # Use YOLO to detect objects
                detections = self.yolo(frame)[0]
            
            current_objects = {}
            
            for det in detections.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = det
                
                if conf < 0.5:  # Confidence threshold
                    continue
                
                bbox = [x1, y1, x2 - x1, y2 - y1]  # Convert to [x, y, w, h]
                
                # Initialize or update tracker
                if cls not in self.tracked_objects:
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, bbox)
                    self.tracked_objects[cls] = {
                        "tracker": tracker,
                        "trajectory": [(bbox, conf)]
                    }
                else:
                    success, new_bbox = self.tracked_objects[cls]["tracker"].update(frame)
                    if success:
                        self.tracked_objects[cls]["trajectory"].append((new_bbox, conf))
                
                current_objects[cls] = self.tracked_objects[cls]["trajectory"]
            
            return {
                "tracked_objects": current_objects,
                "frame_detections": detections
            }
            
        except Exception as e:
            self.logger.error(f"Error tracking objects: {str(e)}")
            return {"error": str(e)}
    
    def calibrate_sensors(
        self,
        sensor_data: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Calibrate multiple sensors
        Args:
            sensor_data: Dictionary of sensor data for calibration
        Returns:
            Calibration matrices for each sensor
        """
        try:
            calibration_results = {}
            
            # Implement sensor calibration logic here
            # This would typically involve:
            # 1. Collecting corresponding points from different sensors
            # 2. Computing transformation matrices
            # 3. Validating calibration accuracy
            
            return calibration_results
            
        except Exception as e:
            self.logger.error(f"Error calibrating sensors: {str(e)}")
            return {"error": str(e)}
    
    def fuse_sensor_data(
        self,
        sensor_data: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Fuse data from multiple sensors
        Args:
            sensor_data: Dictionary of data from different sensors
        Returns:
            Fused sensor data
        """
        try:
            fused_data = {}
            
            # Implement sensor fusion logic here
            # This could involve:
            # 1. Temporal alignment of sensor data
            # 2. Spatial alignment using calibration matrices
            # 3. Probabilistic fusion using Kalman filtering
            
            return fused_data
            
        except Exception as e:
            self.logger.error(f"Error fusing sensor data: {str(e)}")
            return {"error": str(e)}
    
    def get_spatial_map(
        self,
        sensor_data: Dict[str, np.ndarray]
    ) -> Dict[str, Union[np.ndarray, List]]:
        """
        Generate spatial map from sensor data
        Args:
            sensor_data: Dictionary of sensor data
        Returns:
            Spatial map and detected objects
        """
        try:
            # Process LiDAR data if available
            if self.use_lidar and "lidar" in sensor_data:
                lidar_results = self.process_lidar(sensor_data["lidar"])
                if "error" in lidar_results:
                    return lidar_results
            
            # Process depth data if available
            if self.use_depth and "depth" in sensor_data:
                depth_results = self.process_depth(
                    sensor_data["depth"],
                    sensor_data.get("camera_intrinsics", None)
                )
                if "error" in depth_results:
                    return depth_results
            
            # Fuse sensor data
            fused_data = self.fuse_sensor_data(sensor_data)
            
            return {
                "spatial_map": fused_data,
                "sensor_data": {
                    "lidar": lidar_results if self.use_lidar else None,
                    "depth": depth_results if self.use_depth else None
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating spatial map: {str(e)}")
            return {"error": str(e)}
