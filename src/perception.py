"""
Perception Module
Estimates relative distance and motion of detected vehicles using bounding box tracking and optical flow.
Includes risk estimation with danger score calculation.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
from enum import Enum
from dataclasses import dataclass


class RiskLevel(Enum):
    """Risk level classification."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class RiskAssessment:
    """Risk assessment for a detected obstacle."""
    danger_score: float  # 0-1, higher = more dangerous
    risk_level: RiskLevel
    area_growth_rate: float  # Percentage change in bounding box area
    relative_speed: float  # Normalized relative speed
    time_to_collision: Optional[float]  # Estimated TTC in seconds (None if not calculable)
    distance_estimate: float  # Normalized distance (0-1, 1 = far, 0 = close)


class VehicleTracker:
    """Tracks vehicles across frames and estimates relative motion."""
    
    def __init__(self, max_track_length: int = 30, iou_threshold: float = 0.3):
        """
        Initialize vehicle tracker.
        
        Args:
            max_track_length: Maximum number of frames to keep in track history
            iou_threshold: IoU threshold for matching detections across frames
        """
        self.max_track_length = max_track_length
        self.iou_threshold = iou_threshold
        self.tracks: Dict[int, deque] = {}  # track_id -> deque of (frame_num, bbox, area)
        self.next_track_id = 0
        self.frame_count = 0
        
    def _calculate_iou(self, box1: Tuple[float, float, float, float], 
                      box2: Tuple[float, float, float, float]) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_area(self, bbox: Tuple[float, float, float, float]) -> float:
        """Calculate area of bounding box."""
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)
    
    def update(self, detections: List[Dict]) -> Dict[int, Dict]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of detection dicts with keys: 'bbox' (x1, y1, x2, y2), 'class_id', 'confidence'
        
        Returns:
            Dictionary mapping track_id to track info with motion estimates
        """
        self.frame_count += 1
        current_boxes = []
        
        # Extract bounding boxes from detections
        for det in detections:
            if 'bbox' in det:
                bbox = det['bbox']
                current_boxes.append({
                    'bbox': bbox,
                    'detection': det
                })
        
        # Match detections to existing tracks
        matched_tracks = set()
        matched_detections = set()
        
        for track_id, track_history in self.tracks.items():
            if len(track_history) == 0:
                continue
            
            # Get last known bbox
            last_frame, last_bbox, last_area = track_history[-1]
            
            # Find best matching detection
            best_iou = 0.0
            best_det_idx = -1
            
            for idx, box_info in enumerate(current_boxes):
                if idx in matched_detections:
                    continue
                
                iou = self._calculate_iou(last_bbox, box_info['bbox'])
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_det_idx = idx
            
            if best_det_idx >= 0:
                # Match found - update track
                matched_tracks.add(track_id)
                matched_detections.add(best_det_idx)
                
                bbox = current_boxes[best_det_idx]['bbox']
                area = self._calculate_area(bbox)
                track_history.append((self.frame_count, bbox, area))
                
                # Keep only recent history
                if len(track_history) > self.max_track_length:
                    track_history.popleft()
        
        # Create new tracks for unmatched detections
        for idx, box_info in enumerate(current_boxes):
            if idx not in matched_detections:
                track_id = self.next_track_id
                self.next_track_id += 1
                
                bbox = box_info['bbox']
                area = self._calculate_area(bbox)
                self.tracks[track_id] = deque([(self.frame_count, bbox, area)])
        
        # Remove old tracks (not seen for too long)
        tracks_to_remove = []
        for track_id, track_history in self.tracks.items():
            if len(track_history) == 0:
                tracks_to_remove.append(track_id)
            else:
                last_frame, _, _ = track_history[-1]
                if self.frame_count - last_frame > 10:  # Remove if not seen for 10 frames
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        # Calculate motion estimates for active tracks
        track_info = {}
        for track_id, track_history in self.tracks.items():
            if len(track_history) < 2:
                continue
            
            # Get recent history
            frames = [f for f, _, _ in track_history]
            areas = [a for _, _, a in track_history]
            
            # Calculate area change rate (indicator of distance change)
            if len(areas) >= 2:
                recent_areas = areas[-5:] if len(areas) >= 5 else areas
                area_change_rate = (recent_areas[-1] - recent_areas[0]) / len(recent_areas) if len(recent_areas) > 1 else 0
                area_change_percent = (area_change_rate / recent_areas[0] * 100) if recent_areas[0] > 0 else 0
            else:
                area_change_rate = 0
                area_change_percent = 0
            
            # Estimate relative speed (positive = approaching, negative = receding)
            # Larger area increase = faster approach
            relative_speed = area_change_percent  # Normalized speed estimate
            
            # Estimate closing distance trend
            # Positive area change = getting closer
            closing_trend = "approaching" if area_change_percent > 2 else ("receding" if area_change_percent < -2 else "stable")
            
            # Get current bbox
            _, current_bbox, current_area = track_history[-1]
            
            track_info[track_id] = {
                'bbox': current_bbox,
                'area': current_area,
                'relative_speed': relative_speed,
                'closing_trend': closing_trend,
                'area_change_percent': area_change_percent,
                'track_length': len(track_history)
            }
        
        return track_info


class OpticalFlowEstimator:
    """Estimates motion using optical flow."""
    
    def __init__(self):
        """Initialize optical flow estimator."""
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        self.prev_gray = None
        self.prev_points = None
    
    def estimate_motion(self, frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> Optional[float]:
        """
        Estimate motion within bounding box using optical flow.
        
        Args:
            frame: Current frame
            bbox: Bounding box (x1, y1, x2, y2)
        
        Returns:
            Average flow magnitude (motion estimate) or None if insufficient data
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Extract ROI
        roi = gray[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None
        
        # Create feature points in ROI center
        h, w = roi.shape
        center_x, center_y = w // 2, h // 2
        
        # Create grid of points
        points = np.array([[center_x + x1, center_y + y1]], dtype=np.float32)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_points = points
            return None
        
        # Calculate optical flow
        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_points, None, **self.lk_params
        )
        
        # Calculate flow magnitude
        if status[0][0] == 1:
            flow = next_points[0] - self.prev_points[0]
            flow_magnitude = np.linalg.norm(flow)
        else:
            flow_magnitude = 0.0
        
        self.prev_gray = gray
        self.prev_points = points
        
        return flow_magnitude


class RiskEstimator:
    """Estimates risk/danger score for detected obstacles."""
    
    @staticmethod
    def compute_danger_score(
        area_growth_rate: float,
        relative_speed: float,
        distance_estimate: float,
        current_speed: float = 15.0
    ) -> RiskAssessment:
        """
        Compute danger score based on multiple signals.
        
        Args:
            area_growth_rate: Percentage change in bounding box area
            relative_speed: Normalized relative speed (positive = approaching)
            distance_estimate: Normalized distance (0-1, 1 = far, 0 = close)
            current_speed: Current vehicle speed in m/s
        
        Returns:
            RiskAssessment with danger score and risk level
        """
        # Normalize inputs
        # Area growth: positive = approaching, higher = faster approach
        area_factor = min(abs(area_growth_rate) / 10.0, 1.0)  # Normalize to 0-1
        
        # Relative speed: positive = approaching
        speed_factor = min(abs(relative_speed) / 10.0, 1.0)  # Normalize to 0-1
        
        # Distance: closer = more dangerous
        distance_factor = 1.0 - distance_estimate  # Invert: close = high factor
        
        # Combine factors with weights
        # Area growth is most important (object getting bigger = closer)
        danger_score = (
            area_factor * 0.5 +      # 50% weight on area growth
            speed_factor * 0.3 +      # 30% weight on relative speed
            distance_factor * 0.2     # 20% weight on distance
        )
        
        # Clamp to 0-1
        danger_score = max(0.0, min(1.0, danger_score))
        
        # Estimate time to collision (simplified)
        # TTC = distance / relative_speed (if both are meaningful)
        ttc = None
        if relative_speed > 0.1 and distance_estimate < 0.5:
            # Rough estimate: if closing fast and close, calculate TTC
            # Assume distance in meters (rough conversion from normalized)
            estimated_distance_m = distance_estimate * 50  # Max 50m
            relative_speed_ms = relative_speed * 5  # Rough conversion
            if relative_speed_ms > 0.1:
                ttc = estimated_distance_m / relative_speed_ms
                ttc = max(0.1, min(ttc, 30.0))  # Clamp to reasonable range
        
        # Determine risk level
        if danger_score >= 0.7:
            risk_level = RiskLevel.CRITICAL
        elif danger_score >= 0.5:
            risk_level = RiskLevel.HIGH
        elif danger_score >= 0.3:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        return RiskAssessment(
            danger_score=danger_score,
            risk_level=risk_level,
            area_growth_rate=area_growth_rate,
            relative_speed=relative_speed,
            time_to_collision=ttc,
            distance_estimate=distance_estimate
        )


class MotionEstimator:
    """Main class for estimating relative motion of detected vehicles."""
    
    def __init__(self):
        """Initialize motion estimator."""
        self.tracker = VehicleTracker()
        self.flow_estimator = OpticalFlowEstimator()
        self.risk_estimator = RiskEstimator()
    
    def estimate_relative_motion(
        self, 
        frame: np.ndarray, 
        detections: List[Dict],
        current_speed: float = 15.0
    ) -> Dict[int, Dict]:
        """
        Estimate relative motion of detected vehicles with risk assessment.
        
        Args:
            frame: Current video frame
            detections: List of detection dicts with 'bbox', 'class_id', 'confidence'
            current_speed: Current vehicle speed in m/s (for risk calculation)
        
        Returns:
            Dictionary mapping track_id to motion info:
            {
                'bbox': (x1, y1, x2, y2),
                'relative_speed': float,  # Normalized speed estimate
                'closing_trend': str,      # 'approaching', 'receding', 'stable'
                'area_change_percent': float,
                'flow_magnitude': float,   # Optical flow estimate
                'risk_assessment': RiskAssessment  # Risk/danger assessment
            }
        """
        # Update tracks
        track_info = self.tracker.update(detections)
        
        # Enhance with optical flow and risk assessment for each tracked vehicle
        for track_id, info in track_info.items():
            bbox = info['bbox']
            flow_mag = self.flow_estimator.estimate_motion(frame, bbox)
            if flow_mag is not None:
                info['flow_magnitude'] = flow_mag
            else:
                info['flow_magnitude'] = 0.0
            
            # Compute risk assessment
            risk_assessment = self.risk_estimator.compute_danger_score(
                area_growth_rate=info.get('area_change_percent', 0.0),
                relative_speed=info.get('relative_speed', 0.0),
                distance_estimate=1.0 - (info.get('area', 1000) / 50000),  # Rough distance estimate
                current_speed=current_speed
            )
            info['risk_assessment'] = risk_assessment
        
        return track_info
