"""
Main Integration Module
Glues together video input, perception, GPS monitoring, speed smoothing, metrics,
depth estimation, localization, mapping, and navigation.
"""

import cv2
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

# Handle imports for both script run and module import
try:
    from video_input import load_yolo_model, VEHICLE_CLASSES, TRAFFIC_LIGHT_CLASS
    from perception import MotionEstimator
    from gps_monitor import GPSMonitor, GPSDegradationSimulator
    from speed_smoother import SpeedSmoother, ObstacleInfo, SpeedCommand
    from metrics import MetricsCollector
    from depth_estimation import MonocularDepthEstimator
    from localization_vo import VisualOdometry
    from mapping import EnvironmentMap
    from navigation import GlobalNavigator, Goal
except ImportError:
    from src.video_input import load_yolo_model, VEHICLE_CLASSES, TRAFFIC_LIGHT_CLASS
    from src.perception import MotionEstimator
    from src.gps_monitor import GPSMonitor, GPSDegradationSimulator
    from src.speed_smoother import SpeedSmoother, ObstacleInfo, SpeedCommand
    from src.metrics import MetricsCollector
    from src.depth_estimation import MonocularDepthEstimator
    from src.localization_vo import VisualOdometry
    from src.mapping import EnvironmentMap
    from src.navigation import GlobalNavigator, Goal


class VisionSpeedSmoothingSystem:
    """Main system that integrates all components."""
    
    def __init__(
        self,
        video_path: str,
        model_path: Optional[str] = None,
        enable_gps_degradation: bool = True,
        enable_smoothing: bool = True,
        conf_threshold: float = 0.25
    ):
        """
        Initialize the vision speed smoothing system.
        
        Args:
            video_path: Path to input video
            model_path: Optional path to YOLOv8 model
            enable_gps_degradation: Whether to simulate GPS degradation
            enable_smoothing: Whether to apply speed smoothing
            conf_threshold: YOLOv8 confidence threshold
        """
        self.video_path = video_path
        self.enable_gps_degradation = enable_gps_degradation
        self.enable_smoothing = enable_smoothing
        self.conf_threshold = conf_threshold
        
        # Initialize components
        print("Initializing components...")
        self.model = load_yolo_model(model_path)
        self.motion_estimator = MotionEstimator()
        self.gps_monitor = GPSMonitor()
        self.gps_simulator = GPSDegradationSimulator() if enable_gps_degradation else None
        self.speed_smoother = SpeedSmoother() if enable_smoothing else None
        self.metrics = MetricsCollector()
        
        # NEW: Depth estimation
        print("Initializing depth estimation...")
        self.depth_estimator = MonocularDepthEstimator(focal_length=700.0)
        
        # NEW: Visual Odometry (Localization)
        print("Initializing visual odometry...")
        h, w = 720, 1280  # Default, will be updated from video
        self.vo = VisualOdometry(focal_length=700.0, pp=(w/2, h/2))
        
        # NEW: Environment Mapping
        print("Initializing environment map...")
        self.env_map = EnvironmentMap(
            map_size=(200, 200),
            resolution=0.5,
            map_center=(0.0, 0.0)
        )
        
        # NEW: Global Navigation
        print("Initializing global navigation...")
        self.navigator = GlobalNavigator(self.env_map)
        
        # State
        self.current_speed = 15.0  # m/s, starting speed
        self.current_position = (0.0, 0.0)  # World position (x, y)
        self.frame_count = 0
        self.last_time = time.time()
        
    def _extract_detections(self, results) -> List[Dict]:
        """Extract detection information from YOLOv8 results."""
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # Only process vehicles and traffic lights
                if class_id in VEHICLE_CLASSES or class_id == TRAFFIC_LIGHT_CLASS:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    detections.append({
                        'bbox': (float(x1), float(y1), float(x2), float(y2)),
                        'class_id': class_id,
                        'confidence': confidence,
                        'is_traffic_light': class_id == TRAFFIC_LIGHT_CLASS
                    })
        
        return detections
    
    def _convert_motion_to_obstacles(self, motion_info: Dict[int, Dict]) -> List[ObstacleInfo]:
        """Convert motion estimation results to obstacle info."""
        obstacles = []
        
        for track_id, info in motion_info.items():
            # Estimate distance from area (larger area = closer)
            # Normalize area to 0-1 distance estimate
            area = info.get('area', 1000)
            # Rough heuristic: larger area = closer
            # This is simplified - in reality would need calibration
            max_area = 50000  # Approximate max area for close objects
            distance_estimate = max(0, min(1, 1 - (area / max_area)))
            
            obstacles.append(ObstacleInfo(
                relative_speed=info.get('relative_speed', 0.0),
                closing_trend=info.get('closing_trend', 'stable'),
                distance_estimate=distance_estimate,
                is_traffic_light=False,  # Would need separate detection
                traffic_light_state=None
            ))
        
        return obstacles
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame through the entire pipeline.
        
        Args:
            frame: Video frame
        
        Returns:
            Dictionary with processing results
        """
        current_time = time.time()
        dt = current_time - self.last_time if self.last_time > 0 else 0.033  # Default to ~30fps
        self.last_time = current_time
        
        # Step 1: Object detection
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        detections = self._extract_detections(results)
        
        # Step 2: Motion estimation with risk assessment
        motion_info = self.motion_estimator.estimate_relative_motion(frame, detections, self.current_speed)
        obstacles = self._convert_motion_to_obstacles(motion_info)
        
        # Extract risk assessments
        risk_assessments = []
        for track_id, info in motion_info.items():
            if 'risk_assessment' in info:
                risk_assessments.append(info['risk_assessment'])
        
        # NEW: Step 2.5: Depth estimation
        motion_scale = self.current_speed * dt
        depth_map = self.depth_estimator.estimate_depth(frame, motion_scale)
        
        # NEW: Step 2.6: Visual Odometry (Localization)
        vo_pose = self.vo.update(frame, speed_scale=motion_scale)
        vo_position = self.vo.get_position()
        # Use X and Z for 2D navigation (Y is typically up in camera coordinates)
        self.current_position = (float(vo_position[0]), float(vo_position[2]))
        
        # NEW: Step 2.7: Update environment map with obstacles
        current_time = time.time()
        for track_id, info in motion_info.items():
            bbox = info.get('bbox')
            if bbox and depth_map is not None:
                # Get depth at obstacle
                depth = self.depth_estimator.get_depth_at_point(depth_map, 0, 0, bbox)
                if depth:
                    # Convert to world position (simplified)
                    x = self.current_position[0] + depth * 0.1  # Rough conversion
                    y = self.current_position[1]
                    z = depth
                    
                    # Get confidence from risk assessment
                    confidence = 0.5
                    if 'risk_assessment' in info:
                        risk = info['risk_assessment']
                        if hasattr(risk, 'danger_score'):
                            confidence = risk.danger_score
                    
                    # Add to map
                    self.env_map.add_obstacle(
                        position=(x, y, z),
                        size=(2.0, 1.5, 4.0),  # Estimated vehicle size
                        obstacle_type='vehicle',
                        confidence=confidence,
                        timestamp=current_time
                    )
        
        # NEW: Update navigator position
        self.navigator.update_position(self.current_position)
        
        # NEW: Update navigation state
        nav_status = self.navigator.update()
        
        # Clean old obstacles from map
        self.env_map.clear_old_obstacles(current_time, max_age=5.0)
        
        # Step 3: GPS monitoring
        if self.gps_simulator:
            gps_reading = self.gps_simulator.update(dt)
            gps_status = self.gps_monitor.update(gps_reading)
        else:
            # Simulate normal GPS with actual movement
            from gps_monitor import GPSReading
            import math
            
            # Simulate movement based on current speed
            # Update position slightly each frame to avoid "frozen" detection
            if not hasattr(self, '_gps_lat'):
                self._gps_lat = 37.7749
                self._gps_lon = -122.4194
            
            # Move position based on speed (simplified)
            speed_ms = self.current_speed
            heading_rad = 0.0  # Assume heading north
            lat_change = (speed_ms * dt * math.cos(heading_rad)) / 111320.0
            lon_change = (speed_ms * dt * math.sin(heading_rad)) / (111320.0 * math.cos(math.radians(self._gps_lat)))
            
            self._gps_lat += lat_change
            self._gps_lon += lon_change
            
            gps_reading = GPSReading(
                latitude=self._gps_lat,
                longitude=self._gps_lon,
                timestamp=current_time,
                speed=self.current_speed,
                heading=0.0
            )
            gps_status = self.gps_monitor.update(gps_reading)
        
        # Step 4: Speed smoothing with state machine
        speed_command = None
        if self.speed_smoother:
            speed_command = self.speed_smoother.compute_speed_command(
                self.current_speed,
                obstacles,
                gps_status['gps_ok'],
                dt,
                risk_assessments=risk_assessments
            )
            self.current_speed = speed_command.target_speed
        else:
            # No smoothing - reactive behavior
            prev_speed = self.current_speed
            
            # Find max risk level for display (even in reactive mode)
            max_risk_level = "LOW"
            max_danger_score = 0.0
            if risk_assessments:
                for risk in risk_assessments:
                    if hasattr(risk, 'danger_score'):
                        if risk.danger_score > max_danger_score:
                            max_danger_score = risk.danger_score
                            max_risk_level = risk.risk_level.value if hasattr(risk.risk_level, 'value') else str(risk.risk_level)
            
            # Determine state based on risk (for display, but use reactive control)
            from speed_smoother import DrivingState
            if max_danger_score >= 0.7:
                display_state = DrivingState.DANGER
                action = "IMMEDIATE BRAKING"
            elif max_danger_score >= 0.3:
                display_state = DrivingState.CAUTION
                action = "REACTIVE SLOWDOWN"
            else:
                display_state = DrivingState.NORMAL
                action = "MAINTAIN SPEED"
            
            if obstacles:
                # Simple reactive: slow down if obstacle approaching
                for obstacle in obstacles:
                    if obstacle.closing_trend == "approaching":
                        self.current_speed = max(0, self.current_speed - 2.0 * dt)
                        break
            else:
                # Accelerate if no obstacles
                self.current_speed = min(20.0, self.current_speed + 1.0 * dt)
            
            # Calculate acceleration for reactive mode
            acceleration = (self.current_speed - prev_speed) / dt if dt > 0 else 0.0
            brake_pressure = abs(min(0, acceleration)) / 3.0  # Normalize brake pressure
            
            speed_command = SpeedCommand(
                target_speed=self.current_speed,
                acceleration=acceleration,
                brake_pressure=brake_pressure,
                reason='Reactive (no smoothing)',
                state=display_state,
                risk_level=max_risk_level,
                action=action
            )
        
        # Step 5: Collect metrics
        self.metrics.add_data_point(
            self.current_speed,
            speed_command.acceleration if speed_command else 0.0,
            speed_command.brake_pressure if speed_command else 0.0,
            current_time
        )
        
        return {
            'detections': detections,
            'motion_info': motion_info,
            'obstacles': obstacles,
            'gps_status': gps_status,
            'speed_command': speed_command,
            'current_speed': self.current_speed,
            'depth_map': depth_map,
            'vo_position': vo_position,
            'navigation_status': nav_status,
            'map_image': self.env_map.get_map_image()
        }
    
    def draw_overlay(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """Draw system information overlay on frame with state, risk, and action."""
        overlay = frame.copy()
        h, w = overlay.shape[:2]
        
        # Draw detections
        for det in results['detections']:
            x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
            color = (0, 255, 0) if not det['is_traffic_light'] else (0, 0, 255)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        
        # Draw motion info with risk assessment
        y_offset = 30
        for track_id, info in results['motion_info'].items():
            risk_text = ""
            if 'risk_assessment' in info:
                risk = info['risk_assessment']
                risk_text = f" | Risk: {risk.risk_level.value} ({risk.danger_score:.2f})"
            
            text = f"Track {track_id}: {info['closing_trend']} (speed: {info['relative_speed']:.1f}){risk_text}"
            cv2.putText(overlay, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
        
        speed_cmd = results['speed_command']
        
        # Draw STATE, RISK LEVEL, and ACTION prominently (top-right)
        if speed_cmd:
            # State box with color coding
            state = speed_cmd.state.value if hasattr(speed_cmd.state, 'value') else str(speed_cmd.state)
            risk_level = speed_cmd.risk_level
            action = speed_cmd.action
            
            # Color based on state
            if state == "NORMAL":
                state_color = (0, 255, 0)  # Green
            elif state == "CAUTION":
                state_color = (0, 165, 255)  # Orange
            else:  # DANGER
                state_color = (0, 0, 255)  # Red
            
            # Draw state box background
            box_x = w - 300
            box_y = 10
            box_w = 290
            box_h = 100
            
            cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
            cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), state_color, 2)
            
            # Draw text
            cv2.putText(overlay, f"STATE: {state}", (box_x + 10, box_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
            cv2.putText(overlay, f"RISK LEVEL: {risk_level}", (box_x + 10, box_y + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(overlay, f"ACTION: {action}", (box_x + 10, box_y + 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw GPS status
        gps_status = results['gps_status']
        gps_color = (0, 255, 0) if gps_status['gps_ok'] else (0, 0, 255)
        gps_text = f"GPS: {gps_status['status']}"
        cv2.putText(overlay, gps_text, (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, gps_color, 2)
        
        # NEW: Draw navigation status
        nav_status = results.get('navigation_status', {})
        if nav_status:
            nav_text = f"Nav: {nav_status.get('state', 'UNKNOWN')}"
            next_wp = nav_status.get('next_waypoint')
            if next_wp and len(next_wp) >= 2:
                nav_text += f" | WP: ({next_wp[0]:.1f}, {next_wp[1]:.1f})"
            cv2.putText(overlay, nav_text, (10, h - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # NEW: Draw depth map (small overlay)
        depth_map = results.get('depth_map')
        if depth_map is not None and depth_map.size > 0:
            # Normalize depth map for display
            depth_vis = depth_map.copy()
            max_val = depth_vis.max()
            if max_val > 0:
                depth_vis = (depth_vis / max_val * 255).astype(np.uint8)
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                depth_vis = cv2.resize(depth_vis, (160, 120))
                overlay[10:130, w-170:w-10] = depth_vis
                cv2.putText(overlay, "Depth", (w-170, 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # NEW: Draw environment map (small overlay)
        map_image = results.get('map_image')
        if map_image is not None:
            map_vis = cv2.resize(map_image, (160, 160))
            overlay[h-180:h-20, 10:170] = map_vis
            cv2.putText(overlay, "Map", (10, h-185), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw speed info
        speed_text = f"Speed: {results['current_speed']:.1f} m/s ({results['current_speed']*3.6:.1f} km/h)"
        cv2.putText(overlay, speed_text, (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw brake pressure if braking
        if speed_cmd and speed_cmd.brake_pressure > 0.1:
            brake_text = f"Brake: {speed_cmd.brake_pressure*100:.0f}%"
            cv2.putText(overlay, brake_text, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        reason_text = f"Reason: {speed_cmd.reason if speed_cmd else 'N/A'}"
        cv2.putText(overlay, reason_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return overlay
    
    def set_navigation_goal(self, goal_x: float, goal_y: float, tolerance: float = 2.0):
        """
        Set a navigation goal.
        
        Args:
            goal_x: Goal X position in meters
            goal_y: Goal Y position in meters
            tolerance: Goal tolerance in meters
        """
        goal = Goal(
            position=(goal_x, goal_y),
            goal_type='destination',
            tolerance=tolerance
        )
        self.navigator.set_goal(goal)
        self.navigator.plan()
        print(f"Navigation goal set: ({goal_x:.1f}, {goal_y:.1f})")
    
    def run(self, display: bool = True, max_frames: Optional[int] = None):
        """Run the system on the video."""
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {self.video_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nProcessing video: {self.video_path}")
        print(f"Total frames: {total_frames}")
        print(f"GPS degradation: {'Enabled' if self.enable_gps_degradation else 'Disabled'}")
        print(f"Speed smoothing: {'Enabled' if self.enable_smoothing else 'Disabled'}")
        print("\nPress 'q' to quit\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            if max_frames and self.frame_count >= max_frames:
                print(f"Reached max frames ({max_frames})")
                break
            
            # Process frame
            results = self.process_frame(frame)
            
            # Draw overlay
            if display:
                overlay = self.draw_overlay(frame, results)
                cv2.imshow("Vision Speed Smoothing System", overlay)
                
                # Calculate proper delay based on video FPS
                delay = int(1000 / fps) if fps > 0 else 33  # Default to ~30fps
                key = cv2.waitKey(delay) & 0xFF
                if key == ord('q'):
                    break
            
            # Progress update
            if self.frame_count % 30 == 0:
                progress = (self.frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({self.frame_count}/{total_frames} frames)")
        
        cap.release()
        if display:
            cv2.destroyAllWindows()
        
        # Compute final metrics
        metrics = self.metrics.compute_overall_metrics()
        
        print("\n" + "="*60)
        print("FINAL METRICS")
        print("="*60)
        print(f"Speed Variance: {metrics.speed_metrics.variance:.4f} m²/s²")
        print(f"Brake Events: {metrics.brake_metrics.total_brake_events}")
        print(f"Acceleration Peaks: {metrics.acceleration_metrics.acceleration_peaks}")
        print(f"Energy Efficiency Score: {metrics.energy_efficiency_score:.4f}")
        print("="*60)
        
        return metrics


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Vision-based Speed Smoothing System")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--model", type=str, help="Path to YOLOv8 model")
    parser.add_argument("--no-gps-degradation", action="store_true", help="Disable GPS degradation simulation")
    parser.add_argument("--no-smoothing", action="store_true", help="Disable speed smoothing (reactive mode)")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLOv8 confidence threshold")
    parser.add_argument("--no-display", action="store_true", help="Disable video display")
    
    args = parser.parse_args()
    
    # Get project root
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    
    # Default video
    if args.video:
        video_path = args.video
        if not Path(video_path).is_absolute():
            video_path = project_root / video_path
    else:
        video_path = data_dir / "8359-208052066_small.mp4"
    
    # Create and run system
    system = VisionSpeedSmoothingSystem(
        video_path=str(video_path),
        model_path=args.model,
        enable_gps_degradation=not args.no_gps_degradation,
        enable_smoothing=not args.no_smoothing,
        conf_threshold=args.conf
    )
    
    system.run(display=not args.no_display)


if __name__ == "__main__":
    main()
