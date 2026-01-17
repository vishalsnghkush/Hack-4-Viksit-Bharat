
"""
Localization Layer: Visual Odometry / Deep SLAM
Target: Navigation in GPS-denied environments (Urban Canyons)
Implementation: Monocular VO using OpenCV (Feature Tracking)
"""
import numpy as np
import cv2
from typing import Tuple, List, Optional

class VisualOdometry:
    """
    Monocular Visual Odometry using OpenCV.
    Tracks features -> Computes Essential Matrix -> Recovers Pose.
    """
    def __init__(self, focal_length: float = 718.8560, pp: Tuple[float, float] = (607.1928, 185.2157)):
        """
        Initialize VO system.
        Args:
            focal_length: Camera focal length (default: KITTI benchmark value as placeholder)
            pp: Principal point (cx, cy)
        """
        print("Initializing Localization Layer (Visual Odometry, Monocular)...")
        self.focal_length = focal_length
        self.pp = pp
        
        # Frame storage
        self.old_frame = None
        self.old_keypoints = None
        
        # State
        self.current_pose = np.eye(4)
        self.trajectory: List[np.ndarray] = []
        
        # Feature Detector (FAST is fast and usually good enough for VO)
        self.detector = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
        
        # LK Optical Flow Parameters
        self.lk_params = dict(winSize=(21, 21), 
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def update(self, frame: np.ndarray, speed_scale: float = 1.0, dynamic_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Estimate motion between frames.
        
        Args:
            frame: Current video frame (BGR)
            speed_scale: Absolute scale factor.
            dynamic_mask: Binary mask where 0 = Dynamic Object (Ignore), 255 = Static (Keep).
            
        Returns:
            New 4x4 pose matrix
        """
        # Convert to grayscale for feature tracking
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply mask to reduce feature tracking on cars (Refinement)
        tracking_mask = None
        if dynamic_mask is not None:
             # Resize mask to match frame if needed
             if dynamic_mask.shape != gray.shape:
                 dynamic_mask = cv2.resize(dynamic_mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
             tracking_mask = dynamic_mask
        
        if self.old_frame is None:
            # First frame, just initialize
            self.old_frame = gray
            keypoints = self.detector.detect(self.old_frame, mask=tracking_mask)
            if keypoints:
                self.old_keypoints = np.array([x.pt for x in keypoints], dtype=np.float32)
            else:
                 self.old_keypoints = np.array([], dtype=np.float32)
            self.trajectory.append(self.current_pose.copy())
            return self.current_pose
        
        # 1. Track features using Optical Flow
        # checks if we have enough points
        if self.old_keypoints is None or len(self.old_keypoints) < 5:
            # Re-detect if lost
            self.old_keypoints = self.detector.detect(self.old_frame, mask=tracking_mask)
            
            if not self.old_keypoints:
                # Still no points found? Skip this frame's motion
                self.trajectory.append(self.current_pose.copy())
                return self.current_pose
                
            self.old_keypoints = np.array([x.pt for x in self.old_keypoints], dtype=np.float32)
            
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_frame, gray, self.old_keypoints, None, **self.lk_params)
        
        # Select good points
        if p1 is not None:
            # Fix: Flatten st to match dimension for indexing
            status = st.ravel() == 1
            good_new = p1[status]
            good_old = self.old_keypoints[status]
            
            # 2. Estimate Motion (Essential Matrix)
            # Need at least 5 points for 5-point algorithm
            if len(good_new) > 5:
                try:
                    E, mask = cv2.findEssentialMat(good_new, good_old, focal=self.focal_length, pp=self.pp, 
                                                  method=cv2.RANSAC, prob=0.999, threshold=1.0)
                    
                    if E is not None:
                        # 3. Recover Pose (R, t) from E
                        _, R, t, mask = cv2.recoverPose(E, good_new, good_old, focal=self.focal_length, pp=self.pp)
                        
                        # 4. Update Global Pose
                        # Monocular VO recovers direction of t, but not magnitude.
                        # We apply the external scale (speed * dt)
                        absolute_scale = speed_scale
                        
                        # Check coordinate system: usually Z is forward in camera
                        if absolute_scale > 0.1: # Only update if moving
                             # T_new = T_old * [R|t]
                             # Translation update: T_new_pos = T_old_pos + T_old_rot * (scale * t)
                             ordering_rotation = self.current_pose[:3, :3]
                             
                             self.current_pose[:3, 3] = self.current_pose[:3, 3] + absolute_scale * ordering_rotation.dot(t).flatten()
                             self.current_pose[:3, :3] = self.current_pose[:3, :3].dot(R)
                except Exception as e:
                    print(f"VO Estimation Error: {e}")
                    pass
            
            # Update history
            self.old_frame = gray.copy()
            
            # Re-detect if we are running low on points
            if len(good_new) < 1000:
                new_kps = self.detector.detect(gray, None)
                new_kps = np.array([x.pt for x in new_kps], dtype=np.float32)
                self.old_keypoints = new_kps
            else:
                self.old_keypoints = good_new
        else:
            self.old_frame = gray.copy()

        self.trajectory.append(self.current_pose.copy())
        return self.current_pose

    def get_position(self) -> Tuple[float, float, float]:
        """Get x, y, z position."""
        return tuple(self.current_pose[0:3, 3])
