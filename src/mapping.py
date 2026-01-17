"""
Mapping Module
Builds and maintains an environment representation (occupancy grid, feature map, etc.)
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from collections import deque
try:
    from localization_vo import VisualOdometry
except ImportError:
    from src.localization_vo import VisualOdometry
from dataclasses import dataclass
from enum import Enum


class MapType(Enum):
    """Types of maps."""
    OCCUPANCY_GRID = "OCCUPANCY_GRID"
    FEATURE_MAP = "FEATURE_MAP"
    SEMANTIC_MAP = "SEMANTIC_MAP"


@dataclass
class MapPoint:
    """A point in the map."""
    x: float
    y: float
    z: float
    confidence: float
    semantic_label: Optional[str] = None
    timestamp: float = 0.0


@dataclass
class ObstacleMapEntry:
    """Entry for obstacle in map."""
    position: Tuple[float, float, float]
    size: Tuple[float, float, float]  # width, height, depth
    obstacle_type: str
    confidence: float
    timestamp: float


class EnvironmentMap:
    """
    Maintains a map of the environment.
    Can represent obstacles, free space, and semantic features.
    """
    
    def __init__(
        self,
        map_size: Tuple[int, int] = (200, 200),  # Grid cells
        resolution: float = 0.5,  # meters per cell
        map_center: Tuple[float, float] = (0.0, 0.0)  # meters
    ):
        """
        Initialize environment map.
        
        Args:
            map_size: Size of occupancy grid (width, height)
            resolution: Meters per grid cell
            map_center: Center position of map in world coordinates
        """
        self.map_size = map_size
        self.resolution = resolution
        self.map_center = map_center
        
        # Occupancy grid: -1 = unknown, 0 = free, 1 = occupied
        self.occupancy_grid = np.full(map_size, -1, dtype=np.float32)
        
        # Feature map: stores detected features/landmarks
        self.feature_map: List[MapPoint] = []
        
        # Obstacle map: stores dynamic obstacles
        self.obstacle_map: List[ObstacleMapEntry] = []
        
        # Semantic map: stores semantic information
        self.semantic_map: Dict[Tuple[int, int], str] = {}
        
        # Map bounds
        self.world_width = map_size[0] * resolution
        self.world_height = map_size[1] * resolution
        
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert world coordinates to grid coordinates.
        
        Args:
            x: World X coordinate (meters)
            y: World Y coordinate (meters)
        
        Returns:
            Grid coordinates (col, row)
        """
        # Center the map
        grid_x = int((x - self.map_center[0] + self.world_width / 2) / self.resolution)
        grid_y = int((y - self.map_center[1] + self.world_height / 2) / self.resolution)
        
        # Clamp to map bounds
        grid_x = max(0, min(grid_x, self.map_size[0] - 1))
        grid_y = max(0, min(grid_y, self.map_size[1] - 1))
        
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """
        Convert grid coordinates to world coordinates.
        
        Args:
            grid_x: Grid X coordinate
            grid_y: Grid Y coordinate
        
        Returns:
            World coordinates (x, y) in meters
        """
        x = (grid_x * self.resolution) - (self.world_width / 2) + self.map_center[0]
        y = (grid_y * self.resolution) - (self.world_height / 2) + self.map_center[1]
        return x, y
    
    def update_occupancy(
        self,
        position: Tuple[float, float],
        is_occupied: bool,
        confidence: float = 1.0
    ):
        """
        Update occupancy grid at a position.
        
        Args:
            position: World position (x, y)
            is_occupied: True if occupied, False if free
            confidence: Confidence of the observation (0-1)
        """
        grid_x, grid_y = self.world_to_grid(position[0], position[1])
        
        # Update occupancy (simple: 1 = occupied, 0 = free)
        if is_occupied:
            self.occupancy_grid[grid_y, grid_x] = min(1.0, self.occupancy_grid[grid_y, grid_x] + confidence * 0.1)
        else:
            self.occupancy_grid[grid_y, grid_x] = max(0.0, self.occupancy_grid[grid_y, grid_x] - confidence * 0.1)
    
    def add_obstacle(
        self,
        position: Tuple[float, float, float],
        size: Tuple[float, float, float],
        obstacle_type: str,
        confidence: float = 1.0,
        timestamp: float = 0.0
    ):
        """
        Add or update an obstacle in the map.
        
        Args:
            position: Obstacle position (x, y, z)
            size: Obstacle size (width, height, depth)
            obstacle_type: Type of obstacle (e.g., 'vehicle', 'pedestrian')
            confidence: Confidence of detection
            timestamp: Timestamp of observation
        """
        # Remove old obstacles of same type at similar position
        self.obstacle_map = [
            obs for obs in self.obstacle_map
            if not (obs.obstacle_type == obstacle_type and
                   np.linalg.norm(np.array(obs.position) - np.array(position)) < 2.0)
        ]
        
        # Add new obstacle
        entry = ObstacleMapEntry(
            position=position,
            size=size,
            obstacle_type=obstacle_type,
            confidence=confidence,
            timestamp=timestamp
        )
        self.obstacle_map.append(entry)
        
        # Update occupancy grid
        x, y = position[0], position[1]
        grid_x, grid_y = self.world_to_grid(x, y)
        
        # Mark surrounding cells as occupied
        radius = int(max(size[0], size[1]) / self.resolution)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                gx, gy = grid_x + dx, grid_y + dy
                if 0 <= gx < self.map_size[0] and 0 <= gy < self.map_size[1]:
                    if dx*dx + dy*dy <= radius*radius:
                        self.occupancy_grid[gy, gx] = 1.0
    
    def add_feature(
        self,
        position: Tuple[float, float, float],
        semantic_label: Optional[str] = None,
        confidence: float = 1.0,
        timestamp: float = 0.0
    ):
        """
        Add a feature/landmark to the map.
        
        Args:
            position: Feature position (x, y, z)
            semantic_label: Semantic label (e.g., 'traffic_light', 'sign')
            confidence: Confidence of detection
            timestamp: Timestamp of observation
        """
        feature = MapPoint(
            x=position[0],
            y=position[1],
            z=position[2],
            confidence=confidence,
            semantic_label=semantic_label,
            timestamp=timestamp
        )
        self.feature_map.append(feature)
    
    def is_occupied(
        self,
        position: Tuple[float, float],
        radius: float = 1.0
    ) -> bool:
        """
        Check if a position is occupied.
        
        Args:
            position: World position (x, y)
            radius: Check radius in meters
        
        Returns:
            True if occupied
        """
        grid_x, grid_y = self.world_to_grid(position[0], position[1])
        check_radius = int(radius / self.resolution)
        
        for dy in range(-check_radius, check_radius + 1):
            for dx in range(-check_radius, check_radius + 1):
                gx, gy = grid_x + dx, grid_y + dy
                if 0 <= gx < self.map_size[0] and 0 <= gy < self.map_size[1]:
                    if dx*dx + dy*dy <= check_radius*check_radius:
                        if self.occupancy_grid[gy, gx] > 0.5:
                            return True
        
        return False
    
    def get_obstacles_in_radius(
        self,
        position: Tuple[float, float],
        radius: float
    ) -> List[ObstacleMapEntry]:
        """
        Get obstacles within a radius.
        
        Args:
            position: Center position (x, y)
            radius: Search radius in meters
        
        Returns:
            List of obstacles
        """
        obstacles = []
        for obs in self.obstacle_map:
            dist = np.sqrt((obs.position[0] - position[0])**2 + (obs.position[1] - position[1])**2)
            if dist <= radius:
                obstacles.append(obs)
        return obstacles
    
    def clear_old_obstacles(self, current_time: float, max_age: float = 5.0):
        """
        Remove obstacles older than max_age.
        
        Args:
            current_time: Current timestamp
            max_age: Maximum age in seconds
        """
        self.obstacle_map = [
            obs for obs in self.obstacle_map
            if (current_time - obs.timestamp) <= max_age
        ]
    
    def get_map_image(self) -> np.ndarray:
        """
        Get visualization of the map.
        
        Returns:
            RGB image of the map
        """
        # Create RGB image
        img = np.zeros((self.map_size[1], self.map_size[0], 3), dtype=np.uint8)
        
        # Color coding:
        # Unknown: gray
        # Free: white
        # Occupied: red
        
        for y in range(self.map_size[1]):
            for x in range(self.map_size[0]):
                val = self.occupancy_grid[y, x]
                if val < 0:
                    img[y, x] = (128, 128, 128)  # Unknown - gray
                elif val < 0.5:
                    img[y, x] = (255, 255, 255)  # Free - white
                else:
                    img[y, x] = (0, 0, 255)  # Occupied - red
        
        # Draw obstacles
        for obs in self.obstacle_map:
            gx, gy = self.world_to_grid(obs.position[0], obs.position[1])
            if 0 <= gx < self.map_size[0] and 0 <= gy < self.map_size[1]:
                cv2.circle(img, (gx, gy), 3, (0, 255, 0), -1)  # Green for obstacles
        
        # Draw features
        for feat in self.feature_map:
            gx, gy = self.world_to_grid(feat.x, feat.y)
            if 0 <= gx < self.map_size[0] and 0 <= gy < self.map_size[1]:
                cv2.circle(img, (gx, gy), 2, (255, 255, 0), -1)  # Cyan for features
        
        return img
