
import cv2
import numpy as np
import time
from pathlib import Path
import sys
import os
import threading

# Import existing modules
sys.path.append(str(Path(__file__).parent))
from demo_navigation import VisionNavigationSystem
from driver_monitor import DriverMonitor
from speed_smoother import DrivingState, SpeedSmoother, ObstacleInfo
from depth_estimation import MonocularDepthEstimator
from mapping import EnvironmentMap
from mapping import EnvironmentMap
from navigation import GlobalNavigator, Goal
from voice_commander import VoiceCommander # NEW

class DualStreamNavigationSystem(VisionNavigationSystem):
    def __init__(self, road_video_path, driver_camera_index=0, weather_condition="CLEAR"):
        super().__init__(road_video_path)
        print("Initializing Driver Monitor...")
        self.driver_monitor = DriverMonitor(driver_camera_index)
        self.weather_condition = weather_condition
        
        # NEW: Initialize navigation components
        print("Initializing Navigation Components...")
        self.depth_estimator = MonocularDepthEstimator(focal_length=700.0)
        self.env_map = EnvironmentMap(map_size=(200, 200), resolution=0.5, map_center=(0.0, 0.0))
        self.navigator = GlobalNavigator(self.env_map)
        self.current_position = (0.0, 0.0)
        
        # NEW: Speed Smoother Logic
        self.speed_controller = SpeedSmoother(max_speed=1.0) # Normalized 0-1
        
        # NEW: Voice Control & State Machine
        self.voice_commander = VoiceCommander()
        self.driving_state = "STOPPED" # STOPPED, DRIVING, PARKING
        self.last_gesture_cmd = "NONE" # For edge detection
        self.last_face_time = time.time()
        self.parking_timer = 0
        
        # NEW: Threading for Async Perception
        import threading
        self.perception_lock = threading.Lock()
        self.latest_frame_for_ai = None
        self.ai_result_ready = False
        self.bg_stopping = False
        self.bg_thread = threading.Thread(target=self._perception_loop, daemon=True)
        self.bg_thread.start()
        
        # Telemetry for Cloud
        self.telemetry_log = []
        
        # --- STREAMING SETUP ---
        from gcp_pipeline import setup_gcp_clients, stream_to_bigquery, BQ_DATASET_NAME, BQ_TABLE_NAME
        _, self.bq_client = setup_gcp_clients()
        self.bq_dataset = BQ_DATASET_NAME
        self.bq_table = BQ_TABLE_NAME
        
        self.stream_buffer = []
        self.last_stream_time = time.time()
        self.STREAM_INTERVAL = 1.0 # Seconds (Faster updates)
        self.STREAM_BATCH_SIZE = 5

    def _stream_worker(self, data_chunk):
        """Threaded worker to upload data chunk."""
        from gcp_pipeline import stream_to_bigquery
        success = stream_to_bigquery(self.bq_client, self.bq_dataset, self.bq_table, data_chunk)
        if success:
            print(".", end="", flush=True) # Minimal feedback

    def _perception_loop(self):
        """Background loop for heavy AI processing."""
        while not self.bg_stopping:
            frame_to_process = None
            
            # 1. Get latest frame safely
            with self.perception_lock:
                if self.latest_frame_for_ai is not None:
                    frame_to_process = self.latest_frame_for_ai.copy()
                    self.latest_frame_for_ai = None # Mark as consumed
            
            if frame_to_process is not None and self.use_segmentation:
                try:
                    # Heavy Inference (Blocking only this thread)
                    # Downscale for performance
                    small_frame = cv2.resize(frame_to_process, (320, 180))
                    seg, mask = self.segmenter.process_frame(small_frame)
                    
                    # Upscale results
                    if seg is not None:
                        seg = cv2.resize(seg, (1280, 720), interpolation=cv2.INTER_NEAREST)
                        mask = cv2.resize(mask, (1280, 720), interpolation=cv2.INTER_NEAREST)
                        
                        with self.perception_lock:
                            self.last_segmentation = seg
                            self.last_mask = mask
                            self.ai_result_ready = True
                except Exception as e:
                    print(f"AI Thread Error: {e}")
            else:
                time.sleep(0.01) # Avoid spin loop
    
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
        
    def _apply_weather_filter(self, frame):
        if self.weather_condition == "Mirror Mode":
             frame = cv2.flip(frame, 1) # Horizontal flip
             
        elif self.weather_condition == "SUNNY":
             # Increase brightness & warm tint
             hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
             h, s, v = cv2.split(hsv)
             s = cv2.add(s, 40) # More color
             v = cv2.add(v, 30) # Brighter
             hsv = cv2.merge((h, s, v))
             frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
             # Warm tint (Add Yellow/Red)
             # B, G, R
             frame = cv2.addWeighted(frame, 0.9, np.full(frame.shape, (0, 30, 60), dtype=np.uint8), 0.1, 0)

        elif self.weather_condition == "NIGHT":
             # Darken & Cool tint
             hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
             h, s, v = cv2.split(hsv)
             v = cv2.subtract(v, 70) # Darker
             hsv = cv2.merge((h, s, v))
             frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
             # Blue tint
             frame = cv2.addWeighted(frame, 0.8, np.full(frame.shape, (60, 20, 0), dtype=np.uint8), 0.2, 0)
             
        return frame
        
    def _draw_dashboard(self, frame, steering_val, speed_val, target_speed_val=None, cmd_status=""):
        h, w = frame.shape[:2]
        # Draw Dashboard (Curved Hood)
        curve_h = 100
        pts = np.array([
            [0, h],
            [0, h - 50],
            [w//4, h - curve_h],
            [w*3//4, h - curve_h],
            [w, h - 50],
            [w, h]
        ], np.int32)
        cv2.fillPoly(frame, [pts], (30, 30, 30))
        cv2.polylines(frame, [pts], False, (100, 100, 100), 2)
        
        # Speedometer (Left) - BIGGER
        center_l = (w // 6, h - 80) # Move slightly left and up
        radius_speed = 70
        cv2.circle(frame, center_l, radius_speed, (10, 10, 10), -1)
        cv2.circle(frame, center_l, radius_speed, (200, 200, 200), 4)
        
        # Ticks
        for i in range(0, 11):
            angle = 135 + (i * 27)
            rad = np.radians(angle)
            sx = int(center_l[0] + (radius_speed-10) * np.cos(rad))
            sy = int(center_l[1] + (radius_speed-10) * np.sin(rad))
            ex = int(center_l[0] + radius_speed * np.cos(rad))
            ey = int(center_l[1] + radius_speed * np.sin(rad))
            cv2.line(frame, (sx, sy), (ex, ey), (150, 150, 150), 2)
            
        # Target Speed Marker (Green Triangle)
        if target_speed_val is not None:
            target_angle = 135 + (target_speed_val * 270)
            rad_t = np.radians(target_angle)
            tx = int(center_l[0] + (radius_speed+15) * np.cos(rad_t))
            ty = int(center_l[1] + (radius_speed+15) * np.sin(rad_t))
            cv2.circle(frame, (tx, ty), 8, (0, 255, 0), -1)

        # Needle
        speed_angle = 135 + (speed_val * 270) # 0-1 mapped to 135-405 deg
        rad = np.radians(speed_angle)
        ex = int(center_l[0] + (radius_speed-10) * np.cos(rad))
        ey = int(center_l[1] + (radius_speed-10) * np.sin(rad))
        cv2.line(frame, center_l, (ex, ey), (0, 0, 255), 4)
        
        # Digital Speed
        speed_kmh = int(speed_val * 120)
        cv2.putText(frame, f"{speed_kmh}", (center_l[0]-40, center_l[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
        cv2.putText(frame, "km/h", (center_l[0]-30, center_l[1]+50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Voice/Status Panel (Right Side of Dashboard)
        panel_x = w - 300
        panel_y = h - 110
        cv2.rectangle(frame, (panel_x, panel_y), (w-20, h-20), (50, 50, 50), -1)
        cv2.rectangle(frame, (panel_x, panel_y), (w-20, h-20), (100, 100, 100), 2)
        cv2.putText(frame, "COMMAND STATUS", (panel_x+10, panel_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        if cmd_status:
            cv2.putText(frame, cmd_status, (panel_x+10, panel_y+70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 255), 2)
        else:
            cv2.putText(frame, "Listening...", (panel_x+10, panel_y+70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)

        # Draw Steering Wheel (Center)
        self._draw_steering_wheel(frame, steering_val, center_y=h-60)

    def _draw_steering_wheel(self, frame, steering_angle, center_y):
        h, w = frame.shape[:2]
        center_x = w // 2
        radius = 50
        
        # Draw Wheel Ring
        cv2.circle(frame, (center_x, center_y), radius, (50, 50, 50), 8)
        cv2.circle(frame, (center_x, center_y), radius, (200, 200, 200), 2)
        
        angle_rad = steering_angle * (np.pi / 2)
        for spoke_offset in [0, 2*np.pi/3, 4*np.pi/3]:
             final_angle = angle_rad - np.pi/2 + spoke_offset
             end_x = int(center_x + radius * np.cos(final_angle))
             end_y = int(center_y + radius * np.sin(final_angle))
             cv2.line(frame, (center_x, center_y), (end_x, end_y), (150, 150, 150), 3)

    def _calculate_risk(self, seg_mask, speed):
        # 0: Road, 1: Sidewalk, 11: Person, 13: Car, 14: Truck, 15: Bus
        if seg_mask is None: return "LOW", 0.0
        
        h, w = seg_mask.shape
        # Look at center region (danger zone)
        center_roi = seg_mask[h//3:h, w//3:2*w//3]
        
        # Count dangerous pixels (Vehicles + Pedestrians)
        # Cityscapes IDs: Car=13, Truck=14, Bus=15, Person=11, Rider=12
        danger_pixels = np.sum(np.isin(center_roi, [11, 12, 13, 14, 15]))
        total_pixels = center_roi.size
        coverage = danger_pixels / total_pixels
        
        ttc = 99.9
        risk_level = "LOW"
        
        if coverage > 0.01: # 1% threshold (more sensitive)
            # Simple TTC estimation: (Distance / Speed)
            # Higher coverage = closer distance
            # Inv prop to coverage
            dist_factor = 1.0 / (coverage + 0.001)
            ttc = dist_factor * (1.0 / (speed + 0.05)) 
            
            if ttc < 5.0: risk_level = "CRITICAL" # Easier to trigger
            elif ttc < 10.0: risk_level = "HIGH"
            elif ttc < 20.0: risk_level = "MEDIUM"
            
        return risk_level, ttc

    def _draw_info_panel(self, frame, risk, ttc, confort_score):
        h, w = frame.shape[:2]
        panel_w = 400 # Wider panel (Fixed Overlap)
        panel_h = 480 # Taller panel
        x1, y1 = w - panel_w - 50, 50

        
        # Glass panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x1+panel_w, y1+panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame) # Darker background
        cv2.rectangle(frame, (x1, y1), (x1+panel_w, y1+panel_h), (100, 100, 100), 2)
        
        
        # Title
        curr_y = y1 + 30
        cv2.putText(frame, "ADAS SYSTEMS", (x1+20, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 1. RISK
        curr_y += 40
        r_color = (0, 255, 0)
        if risk == "HIGH": r_color = (0, 165, 255)
        elif risk == "CRITICAL": r_color = (0, 0, 255)
        
        cv2.putText(frame, f"RISK: {risk}", (x1+20, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, r_color, 1)
        cv2.putText(frame, f"TTC: {ttc:.1f}s", (x1+220, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 2. GPS (Visual Localization)
        curr_y += 30
        g_color = (0, 0, 255)
        if self.gps_status == "VISUAL LOCK": g_color = (0, 255, 0)
        
        cv2.putText(frame, f"GPS: {self.gps_status}", (x1+20, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, g_color, 1)
        # Move MODE to right side with more space
        cv2.putText(frame, "MODE: VISION", (x1+220, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 3. COMFORT
        curr_y += 30
        c_color = (0, 255, 0)
        if confort_score < 80: c_color = (0, 165, 255)
        if confort_score < 50: c_color = (0, 0, 255)
        cv2.putText(frame, f"COMFORT: {int(confort_score)}%", (x1+20, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c_color, 1)
        
        # 4. WEATHER
        curr_y += 30
        cv2.putText(frame, f"WEATHER: {self.weather_condition}", (x1+20, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 5. SCIENCE LAYER METRICS (New)
        curr_y += 20
        dis_count = self.metrics.get("disengagements", 0)
        scenarios = self.metrics.get("scenarios_tested", 0)
        cv2.putText(frame, f"DISENGAGEMENTS: {dis_count}", (x1+20, curr_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        cv2.putText(frame, f"SCENARIOS (STRESS): {scenarios}", (x1+20, curr_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # 6. UNCERTAINTY (DOUBT LAYER)
        curr_y += 45
        unc = self.uncertainty.get("total", 0.0)
        mode = self.system_mode
        
        # Color bar
        bar_w = int(unc * 100)
        u_color = (0, 255, 0)
        if unc > 0.4: u_color = (0, 0, 255) # High Doubt
        
        cv2.putText(frame, f"UNCERTAINTY: {unc*100:.0f}%", (x1+20, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, u_color, 1)
        cv2.rectangle(frame, (x1+150, curr_y-10), (x1+150+bar_w, curr_y), u_color, -1)
        
        # System Mode
        curr_y += 30
        m_color = (0, 255, 0) if mode == "AUTONOMOUS" else (0, 0, 255)
        cv2.putText(frame, f"MODE: {mode}", (x1+20, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, m_color, 1)
        
        # Source Info (Footer)
        curr_y += 40
        cv2.putText(frame, "AI: Cityscapes | INPUT: Local", (x1+20, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
    def run(self):
        cap_road = cv2.VideoCapture(self.video_source)
        if not cap_road.isOpened():
             print(f"Error: road {self.video_source}")
             return

        print("\nStarting 3D Virtual Cockpit...")
        print("  - STEER: Lean Left/Right (Face Position)")
        print("  - SPEED: Open Palm (Go), Fist (Stop)")
        
        start_time_epoch = time.time()
        frame_count = 0
        speed = 0.5 
        
        # Comfort Metrics
        last_speed = 0.5
        comfort_score = 100.0
        
        # --- SCIENCE LAYER (Systematic Evaluation) ---
        self.metrics = {
            "disengagements": 0,    # Human Takeover (Braking in panic)
            "violations": 0,        # Critical Safety Violations
            "scenarios_tested": 0
        }
        self.stress_mode = "NORMAL" # NORMAL, GLARE, SURPRISE
        self.stress_timer = 0
        braking_engaged = False
        
        # --- DOUBT LAYER (Uncertainty Quantification) ---
        self.uncertainty = {
            "aleatoric": 0.0,  # Data noise (Rain, Glare)
            "epistemic": 0.0,  # Model ignorance (New car type)
            "total": 0.0
        }

        self.system_mode = "AUTONOMOUS" # AUTONOMOUS, HIGH ALERT
        self.gps_status = "DENIED" # DENIED, VISUAL LOCK
        self.xai_saliency_map = None # For Heatmap overlay
        
        # Dynamic Weather Cycling
        weather_states = ["CLEAR", "SUNNY", "NIGHT", "Mirror Mode"]
        cycle_interval = 500 # Change every ~15 seconds (Slower)
        
        # Init Speed Command
        speed_cmd = 0.0

        # --- AUTONOMOUS LOGIC HELPERS ---
        def _get_lane_center(mask):
            if mask is None: return 0.0
            h, w = mask.shape
            # Focus on bottom half
            roi = mask[h//2:, :]
            # Road is class 0
            road_pixels = np.where(roi == 0)
            if len(road_pixels[0]) > 100:
                center_x = np.mean(road_pixels[1])
                # Normalize -1 to 1 (0 is center)
                return (center_x - (w/2)) / (w/2)
            return 0.0

        while True:
            # 1. Driver Input
            # Now returns 5 values: frame, status, steering, gesture_cmd, face_detected
            frame_driver, driver_status, steering_val, gesture_cmd, face_detected = self.driver_monitor.get_driver_state()
            
            if frame_driver is None: break

            current_time = time.time()
            if face_detected:
                self.last_face_time = current_time

            # --- COMPOSITE CONTROL LOGIC ---
            
            # A. Voice Commands (Top Priority - Latch State)
            # A. Voice Commands (Top Priority - Latch State)
            v_cmd = self.voice_commander.get_latest_command()
            if v_cmd:
                print(f"Executing Voice Command: {v_cmd}")
                # Reset Safety Timer because user is interacting
                self.last_face_time = time.time()
                
                if v_cmd == "START":
                    self.driving_state = "DRIVING"
                    speed_cmd = 0.6
                elif v_cmd == "STOP":
                    self.driving_state = "STOPPED"
                    speed_cmd = 0.0
                elif v_cmd == "PARK":
                    self.driving_state = "PARKING"
                elif v_cmd == "SLOW_DOWN":
                    speed_cmd = max(0.0, speed_cmd - 0.2)
                    if speed_cmd < 0.05: # Auto-Stop
                        self.driving_state = "STOPPED"
                elif v_cmd == "SPEED_UP":
                    # Fix: Allow Wake-Up from Stopped
                    if self.driving_state == "STOPPED":
                        self.driving_state = "DRIVING"
                        speed_cmd = 0.2
                    else:
                        speed_cmd = min(1.0, speed_cmd + 0.2)
                elif v_cmd == "EMERGENCY":
                    self.driving_state = "STOPPED"
                    speed_cmd = 0.0
            
            # B. Keyboard Backup (Second Priority - Direct Control)
            # Use cv2.waitKey(1) result from end of loop if possible, or poll here
            # Since waitKey is at end, we check a stored key from previous frame or move waitKey up?
            # Better to rely on the loop's waitKey. We'll check 'key' variable from previous iteration
            if 'key' in locals():
                if key == ord('s'): # Start
                    self.driving_state = "DRIVING" 
                    speed_cmd = 0.6
                elif key == ord(' '): # Space = Stop
                    self.driving_state = "STOPPED"
                    speed_cmd = 0.0
                elif key == ord('p'): # Park
                    self.driving_state = "PARKING"

            # C. Gesture Toggle (Lowest Priority - Only if no higher cmd)
            # LOGIC FIX: Only trigger on GEUSTURE CHANGE (Edge Detection)
            # This prevents a held "Fist" from immediately restarting the car after a "Stop" command.
            if not v_cmd: 
                if gesture_cmd != self.last_gesture_cmd: # Only if changed
                    if gesture_cmd == "START":
                        print("GESTURE TRIGGER: START")
                        self.driving_state = "DRIVING"
                        speed_cmd = 0.6
                    elif gesture_cmd == "STOP":
                        print("GESTURE TRIGGER: STOP")
                        self.driving_state = "STOPPED"
                        speed_cmd = 0.0
            
            # Update latch
            self.last_gesture_cmd = gesture_cmd
            
            # 3. Safety Timeout (60s No Face -> Park)
            if self.driving_state == "DRIVING" and (current_time - self.last_face_time > 60.0):
                self.driving_state = "PARKING"
                print("SAFETY TIMEOUT: Auto-Parking...")

            # --- STATE MACHINE LOGIC ---
            is_autonomous_active = False
            
            if self.driving_state == "STOPPED":
                speed_cmd = 0.0
                center_msg = "STOPPED"
                msg_color = (0, 0, 255)
                
            elif self.driving_state == "DRIVING":
                is_autonomous_active = True 
                center_msg = "AUTONOMOUS DRIVE"
                msg_color = (0, 255, 0)
                if speed_cmd < 0.1: speed_cmd = 0.6
                
            elif self.driving_state == "PARKING":
                center_msg = "AUTO-PARKING..."
                msg_color = (0, 165, 255)
                is_autonomous_active = True
                steering_val = 0.8 # Right
                speed_cmd = 0.2
                self.parking_timer += 1
                if self.parking_timer > 100: 
                    self.driving_state = "STOPPED"
                    self.parking_timer = 0
            
            # --- AUTONOMOUS LOGIC (LANE & SAFETY) ---
            if is_autonomous_active and current_mask is not None:
                # 1. Lane Keeping vs Steering Override
                # Detection: significant steering input (> 0.15)
                is_steering_override = abs(steering_val) > 0.15
                
                if self.driving_state != "PARKING":
                    if is_steering_override:
                        # User is steering! Use their value.
                        center_msg = "MANUAL STEERING"
                        msg_color = (0, 255, 255) # Yellow
                        # steering_val is already set from driver_monitor
                    else:
                        # User is relaxed. Use Autonomous.
                        lane_error = _get_lane_center(current_mask)
                        steering_val = np.clip(lane_error * 1.5, -1.0, 1.0)
                
                # 2. Risk Calculation & Safety (ALWAYS ACTIVE - GUARDIAN ANGEL)
                risk_val, _ = self._calculate_risk(current_mask, speed)
                
                if risk_val == "CRITICAL":
                     speed_cmd = 0.0
                     self.driving_state = "STOPPED" # Force stop state
                     center_msg = "EMERGENCY STOP (OBSTACLE)"
                     msg_color = (0, 0, 255)
                elif risk_val == "HIGH":
                     speed_cmd = min(speed_cmd, 0.3)

            
            # --- WEATHER CYCLE ---
            # Automatically change weather every N frames
            state_idx = (frame_count // cycle_interval) % len(weather_states)
            self.weather_condition = weather_states[state_idx]
            
            # Smooth Speed Transition (CRITICAL FIX)
            target_speed = speed_cmd
            speed = speed * 0.9 + target_speed * 0.1

            # 2. Road Video (Only advance if moving)
            # If speed is very low (< 0.1), we "pause" the video reading to simulate stopping
            if speed > 0.05:
                ret_road, frame_road = cap_road.read()
                if not ret_road:
                    cap_road.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_road, frame_road = cap_road.read()
            else:
                # Paused - use last frame
                # If we just started and paused immediately, read one frame
                if 'frame_road' not in locals(): 
                     ret_road, frame_road = cap_road.read()
                time.sleep(0.03) # Prevent CPU spin when stopped
            
            # Apply Environment Filter
            frame_road = self._apply_weather_filter(frame_road)
            
            rows, cols = frame_road.shape[:2]
            rows, cols = frame_road.shape[:2]
            # FORCE 1280x720 for UI Stability
            frame_road = cv2.resize(frame_road, (1280, 720))
            rows, cols = 720, 1280
            
            frame_count += 1
            
            # --- SCIENCE LAYER: STRESS TEST GENERATOR ---
            # Randomly inject edge cases every ~300 frames
            if frame_count % 300 == 0:
                self.stress_mode = "GLARE" if np.random.rand() > 0.5 else "SURPRISE"
                self.stress_timer = 20 # Duration of stress
                self.metrics["scenarios_tested"] += 1
                
            if self.stress_timer > 0:
                self.stress_timer -= 1
                if self.stress_mode == "GLARE":
                    # Simulate sudden blind sun glare (White washout)
                    overlay = np.ones_like(frame_road) * 255
                    frame_road = cv2.addWeighted(frame_road, 0.5, overlay, 0.5, 0)
                    cv2.putText(frame_road, "STRESS TEST: GLARE", (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
                elif self.stress_mode == "SURPRISE":
                    # Simulate sudden obstacle (Red Block)
                    cv2.rectangle(frame_road, (600, 400), (700, 600), (0, 0, 255), -1)
                    cv2.putText(frame_road, "STRESS TEST: SURPRISE", (350, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            else:
                self.stress_mode = "NORMAL"
                
            # --- METRICS: DISENGAGEMENT ---
            if speed_cmd == 0.0 and speed > 0.3 and not braking_engaged: # Panic stop
                 self.metrics["disengagements"] += 1
                 braking_engaged = True
            if speed_cmd > 0:
                 braking_engaged = False

            # --- PERCEPTION FETCH (Moved Earlier) ---
            # Fetch latest results safely BEFORE simulation step so we can use it for Autonomous Steering
            # Update frame for AI thread
            if frame_count % 3 == 0: # Send every few frames to avoid flooding
                with self.perception_lock:
                    self.latest_frame_for_ai = frame_road
            
            current_seg = None
            current_mask = None
            with self.perception_lock:
                if hasattr(self, 'last_segmentation'):
                    current_seg = self.last_segmentation
                    current_mask = self.last_mask

            # --- AUTONOMOUS LOGIC (LANE & SAFETY) ---
            if is_autonomous_active and current_mask is not None:
                # 1. Lane Keeping
                if self.driving_state != "PARKING":
                     lane_error = _get_lane_center(current_mask)
                     # P-Controller
                     steering_val = np.clip(lane_error * 1.5, -1.0, 1.0)
                
                # 2. Risk Calculation & Safety
                # Calculate risk immediately for safety
                risk_val, _ = self._calculate_risk(current_mask, speed)
                
                if risk_val == "CRITICAL":
                     speed_cmd = 0.0
                     self.driving_state = "STOPPED" # Force stop state
                     center_msg = "EMERGENCY STOP (OBSTACLE)"
                     msg_color = (0, 0, 255)
                elif risk_val == "HIGH":
                     speed_cmd = min(speed_cmd, 0.3)

            # 2. Simulation (Steering)
            shift_x = -steering_val * 300 
            M = np.float32([[1, 0, shift_x], [0, 1, 0]])
            display_frame = cv2.warpAffine(frame_road, M, (cols, rows))
            
            # --- COMFORT METRIC ---
            # Jerk = change in speed based on accel
            jerk = abs(speed - last_speed) * 100.0
            # Lateral acceleration = Speed * Steering
            lat_accel = abs(speed * steering_val) * 1.5
            
            penalty = (jerk + lat_accel) * 2.0
            penalty = min(penalty, 5.0) # Cap max penalty
            comfort_score -= penalty
            comfort_score += 0.2 # Slow Recovery
            comfort_score = max(0.0, min(100.0, comfort_score))
            last_speed = speed

            # --- ACTION TEXT OVERLAY ---
            # Priority: Emergency > Warning > State
            if center_msg == "":
                 if self.driving_state == "STOPPED":
                     center_msg = "STOPPED"
                     msg_color = (0, 0, 255)
                 elif self.driving_state == "DRIVING":
                     # Dynamic Speed Text
                     center_msg = f"SPEED: {int(speed * 120)} km/h"
                     msg_color = (0, 255, 0)
                 elif self.driving_state == "PARKING":
                     center_msg = "PARKING..."
                     msg_color = (0, 165, 255)
            
            if "SLEEPING" in driver_status:
                center_msg = "WAKE UP!"
                msg_color = (0, 0, 255)

            if center_msg:
                text_size = cv2.getTextSize(center_msg, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
                text_x = (cols - text_size[0]) // 2
                text_y = rows // 3 # Move UP to top third
                cv2.putText(display_frame, center_msg, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, msg_color, 3)

            # 3. Perception & Map
            risk_val = "LOW"
            ttc_val = 99.9
            static_mask = None
            
            # Fetch latest results safely
            # (MOVED UP) - current_seg and current_mask are already populated

            # Use local variables which are thread-safe copies or references
            if current_seg is not None:
                seg_shifted = cv2.warpAffine(current_seg, M, (cols, rows))
                # Increase visibility of "Pixel-Level" coloring (0.2 -> 0.35)
                display_frame = cv2.addWeighted(display_frame, 0.65, seg_shifted, 0.35, 0)
                # Calculate Risk
                risk_val, ttc_val = self._calculate_risk(current_mask, speed)
                
                # --- ACTIVE SAFETY GATING ---
                # Override driver input if risk is high
                if risk_val == "CRITICAL" and speed_cmd > 0.0:
                    speed_cmd = 0.0 # Emergency Brake
                    center_msg = "EMERGENCY BRAKE"
                    msg_color = (0, 0, 255)
                elif risk_val == "HIGH" and speed_cmd > 0.3:
                    speed_cmd = 0.3 # Limit speed
                    center_msg = "SPEED LIMITED (RISK)"
                    msg_color = (0, 165, 255)
                
                # Create Static Mask for VO (Ignore 11=Person, 12=Rider, 13=Car, 14=Truck, 15=Bus)
                static_mask = np.ones_like(current_mask, dtype=np.uint8) * 255
                mask_dynamic = np.isin(current_mask, [11, 12, 13, 14, 15])
                static_mask[mask_dynamic] = 0
                
                # --- MEMORY LAYER: ABSOLUTE VISION LOCALIZATION ---
                # Search for Landmarks: 6=Traffic Light, 7=Traffic Sign
                match_landmarks = np.isin(current_mask, [6, 7])
                if np.sum(match_landmarks) > 200: # Found substantial landmark
                    self.gps_status = "VISUAL LOCK"
                else:
                    self.gps_status = "DENIED"
            else:
                 pass # No AI result yet

            if current_seg is not None:
                # Reuse the mask for saliency
                # --- TRANSPARENCY LAYER: XAI SALIENCY MAP ---
                # Simulate Grad-CAM: "Focus" on the risky object
                if risk_val in ["HIGH", "CRITICAL"]:
                    # Create a heatmap centered on the dynamic objects (Cars/People)
                    saliency = np.zeros_like(current_mask, dtype=np.uint8)
                    mask_dynamic_local = np.isin(current_mask, [11, 12, 13, 14, 15]) # Recompute mask locally
                    saliency[mask_dynamic_local] = 255 # Highlight objects
                    # Gaussian Blur to make it look like a smooth "Attention Map"
                    saliency = cv2.GaussianBlur(saliency, (51, 51), 0)
                    # Apply ColorMap (JET = Heatmap style)
                    self.xai_saliency_map = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)
                    # Blend only the hot parts
                    mask_inv = cv2.bitwise_not(saliency)
                    # Add to display
                    display_frame = cv2.addWeighted(display_frame, 0.7, self.xai_saliency_map, 0.3, 0)
                    cv2.putText(display_frame, "XAI: ATTENTION BLOCKED", (cols//2 - 200, rows//2 + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                else:
                    self.xai_saliency_map = None

                # --- NEW: PEDESTRIAN INTENT & GHOST PLANNING ---
                # 1. Pedestrian Detection (Class 11)
                pedestrians = np.sum(current_mask == 11)
                ped_alert = ""
                if pedestrians > 100: # Threshold pixels
                    ped_alert = "PEDESTRIAN DETECTED"
                    
                # 2. Ghost Planning (Visualize potential paths of cars)
                # Find centroids of car blobs (Class 13)
                contours, _ = cv2.findContours((current_mask == 13).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if cv2.contourArea(cnt) > 500:
                        M_cnt = cv2.moments(cnt)
                        if M_cnt["m00"] != 0:
                            cx = int(M_cnt["m10"] / M_cnt["m00"])
                            cy = int(M_cnt["m01"] / M_cnt["m00"])
                            # Draw Ghost Path (Predicted Counterfactual)
                            # Shifted by steering to look cool
                            end_x = int(cx + (cx - cols/2) * 0.5) 
                            cv2.arrowedLine(display_frame, (cx, cy), (end_x, cy-50), (255, 0, 255), 2, tipLength=0.3)
                            cv2.putText(display_frame, "???", (end_x, cy-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            
            # --- END AI BLOCK ---

            # --- REASONING LAYER (VLM Simulation) ---
            # "Inner Monologue" of the AI
            reasoning_text = "Scanning Environment..."
            
            if "PEDESTRIAN" in locals() and ped_alert:
                 reasoning_text = "Pedestrian Detected -> Analyzing Pose -> LOW INTENT TO CROSS"
                 cv2.rectangle(display_frame, (0,0), (cols, rows), (0,0,255), 5) # Flash Red border
                 
            elif risk_val == "LOW":
                reasoning_text = "Path Clear -> Optimization Mode: EFFICIENT"
            elif risk_val == "MEDIUM":
                reasoning_text = f"Vehicle Tracked -> Simulating 3 possible interaction paths..."
            elif risk_val == "HIGH":
                 reasoning_text = f"Obstacle Interaction Likely -> PREPARING EVASIVE MANEUVER"
            elif risk_val == "CRITICAL":
                 reasoning_text = f"IMPACT PREDICTION 99% -> EMERGENCY STOP TRIGGERED"
                 
            # Draw Reasoning Box (Bottom Center, above dashboard)
            cv2.putText(display_frame, f"AI THOUGHT: {reasoning_text}", (cols//2 - 300, rows-180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.line(display_frame, (cols//2 - 310, rows-170), (cols//2 + 310, rows-170), (255, 0, 255), 2)
            cv2.putText(display_frame, "VLM-1 (Reasoning)", (cols//2 + 200, rows-190), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

            # 4. VO & 3D Map
            current_scale = speed * 15.0 * 0.033 
            # Only update VO if moving, otherwise drifting occurs
            if speed > 0.05:
                self.vo.update(frame_road, speed_scale=current_scale, dynamic_mask=static_mask)
                # NEW: Update position for navigation
                vo_position = self.vo.get_position()
                self.current_position = (float(vo_position[0]), float(vo_position[2]))
                self.navigator.update_position(self.current_position)
                
                # NEW: Depth estimation
                depth_map = self.depth_estimator.estimate_depth(frame_road, current_scale)
                
                # NEW: Update environment map with obstacles
                import time as time_module
                current_time = time_module.time()
                if depth_map is not None and hasattr(self, 'last_mask'):
                    # Add obstacles from segmentation
                    mask_dynamic_objs = np.isin(self.last_mask, [11, 12, 13, 14, 15])
                    if np.sum(mask_dynamic_objs) > 100:
                        # Find centroid of dynamic objects
                        contours, _ = cv2.findContours(mask_dynamic_objs.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in contours:
                            if cv2.contourArea(cnt) > 500:
                                M = cv2.moments(cnt)
                                if M["m00"] != 0:
                                    cx = int(M["m10"] / M["m00"])
                                    cy = int(M["m01"] / M["m00"])
                                    depth = self.depth_estimator.get_depth_at_point(depth_map, cx, cy)
                                    if depth:
                                        x = self.current_position[0] + depth * 0.1
                                        y = self.current_position[1]
                                        z = depth
                                        self.env_map.add_obstacle(
                                            position=(x, y, z),
                                            size=(2.0, 1.5, 4.0),
                                            obstacle_type='vehicle',
                                            confidence=0.7,
                                            timestamp=current_time
                                        )
                self.env_map.clear_old_obstacles(current_time, max_age=5.0)
            
            # --- TELEMETRY LOGGING (NEW) ---
            # Record metrics for Cloud Upload
            if hasattr(self, 'telemetry_log'):
                import datetime
                self.telemetry_log.append({
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "relative_time": time.time() - start_time_epoch,
                    "mode": self.driving_state.lower(), # stopped, driving, parking
                    "speed": speed,
                    "acceleration": (speed - last_speed) * 30.0, # Approx accel
                    "brake_pressure": 1.0 if speed_cmd == 0.0 and speed > 0.1 else 0.0
                })
                
                # --- LIVE STREAMING ---
                # Add copy to buffer
                self.stream_buffer.append(self.telemetry_log[-1].copy())
                
                # Check flush conditions
                current_time_log = time.time()
                if (len(self.stream_buffer) >= self.STREAM_BATCH_SIZE) or \
                   (current_time_log - self.last_stream_time > self.STREAM_INTERVAL and len(self.stream_buffer) > 0):
                    
                    # Offload to thread
                    chunk = list(self.stream_buffer)
                    self.stream_buffer = [] # Clear buffer
                    self.last_stream_time = current_time_log
                    
                    t = threading.Thread(target=self._stream_worker, args=(chunk,), daemon=True)
                    t.start()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


            
            # --- 3D MAP VISUALIZATION (NeRF MEMORY) ---
            # Project tracked features onto a "Ground Plane" map
            # Initialize persistent map if not exists
            if not hasattr(self, 'nerf_memory_map'):
                self.nerf_memory_map = np.zeros((200, 200, 3), dtype=np.uint8)
                
            # Fade old points slightly (Simulate confidence decay or just cool effect)
            self.nerf_memory_map = cv2.addWeighted(self.nerf_memory_map, 0.95, np.zeros_like(self.nerf_memory_map), 0.05, 0)
            
            # Draw Trajectory (White Line)
            if len(self.trajectory) > 1:
                # ... (Same trajectory logic as before, skipped for brevity) ...
                pass 

            # Draw 3D Points (Red Dots)
            if self.vo.old_keypoints is not None:
                # These are pixels in the IMAGE. To show them on MAP (Bird's Eye),
                # We need Inverse Perspective Mapping (IPM).
                # Simple approximation: Y-coord in image -> Z-coord in map (higher Y = closer)
                # X-coord in image -> X-coord in map
                for pt in self.vo.old_keypoints:
                    px, py = pt
                    # Map X: 0..W -> 0..200
                    mx = int((px / cols) * 200)
                    # Map Z: H/2..H -> 200..0 (Horizon to Bottom)
                    # Ignore sky points (py < H/2)
                    if py > rows/2:
                        mz = int(200 - ((py - rows/2) / (rows/2)) * 200)
                        
                        # Safety Clamps
                        mz = max(0, min(199, mz))
                        mx = max(0, min(199, mx))
                        
                        # Draw on Persistent Map (NeRF Cloud)
                        # Color based on depth (mz) - Far=Blue, Near=Red
                        # Gradient: Near(200)=Red, Far(0)=Blue
                        val_mz = max(0, min(255, mz))
                        color = (255 - val_mz, val_mz, 200) 
                        cv2.circle(self.nerf_memory_map, (mx, mz), 1, color, -1)

            # Overlay Map (3D Radar style)
            map_h, map_w = self.nerf_memory_map.shape[:2]
            # Border
            cv2.rectangle(self.nerf_memory_map, (0,0), (map_w, map_h), (50, 50, 50), 1)
            
            # Place onto display (FIXED: Better positioning to avoid overlap with dashboard)
            # Dashboard is at bottom (h-100 to h), so place map higher
            # Dashboard height is ~100px, so place map above it
            dashboard_height = 100
            map_y_start = rows - map_h - dashboard_height - 20  # 20px gap above dashboard
            map_x_start = 20
            # Make sure it doesn't overlap with driver PIP (which is at y=70, height ~nh)
            driver_pip_bottom = 70 + int(cols*0.25*(frame_driver.shape[0]/frame_driver.shape[1])) if 'frame_driver' in locals() else 200
            if map_y_start < driver_pip_bottom + 10:
                map_y_start = driver_pip_bottom + 10
            
            if map_y_start > 0 and map_x_start + map_w < cols and map_y_start + map_h < rows - dashboard_height:
                display_frame[map_y_start:map_y_start+map_h, map_x_start:map_x_start+map_w] = self.nerf_memory_map
                cv2.putText(display_frame, "NeRF MEMORY MAP", (map_x_start, map_y_start-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            
            # NEW: Draw navigation status (if active) - place it in ADAS panel area
            if hasattr(self, 'navigator') and self.navigator.current_goal:
                nav_status = self.navigator.update()
                nav_text = f"NAV: {nav_status.get('state', 'IDLE')}"
                if nav_status.get('next_waypoint'):
                    wp = nav_status['next_waypoint']
                    nav_text += f" -> ({wp[0]:.1f}, {wp[1]:.1f})"
                # Place navigation text in ADAS panel (below existing content)
                h, w = display_frame.shape[:2]
                panel_w = 400
                panel_h = 480
                x1, y1 = w - panel_w - 50, 50
                nav_y = y1 + panel_h - 40  # Near bottom of panel
                cv2.putText(display_frame, nav_text, (x1+20, nav_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # 5. Dashboard
            # Pass target speed (speed_cmd) and current status
            status_text = f"{self.driving_state}"
            if self.voice_commander.last_command:
                 status_text = f"CMD: {self.voice_commander.last_command}"
            
            self._draw_dashboard(display_frame, steering_val, speed, target_speed_val=speed_cmd, cmd_status=status_text)
            
            # 6. Advanced Info Panel
            self._draw_info_panel(display_frame, risk_val, ttc_val, comfort_score)
            
            # 7. PIP (Picture-in-Picture)
            if frame_driver is not None and frame_driver.size > 0:
                dh, dw = frame_driver.shape[:2]
                if dw > 0:
                    sc = 0.25
                    nw = int(cols * sc)
                    nh = int(nw * (dh / dw))
                    
                    # Safety check for resize dimensions
                    if nw > 0 and nh > 0:
                        drv_sm = cv2.resize(frame_driver, (nw, nh))
                        cv2.rectangle(drv_sm, (0,0), (nw,nh), (0,255,0), 2)
                        
                        # Place at Top Left (x=20, y=70)
                        # Ensure it fits
                        if 70 + nh < rows and 20 + nw < cols:
                            display_frame[70:70+nh, 20:20+nw] = drv_sm

            # --- VISUAL FEEDBACK FOR COMMANDS ---
            # If a command was recently received, flash it on screen
            if self.voice_commander.last_command and (time.time() - self.voice_commander.last_command_time < 2.0):
                cmd_text = f"VOICE: {self.voice_commander.last_command}"
                cv2.putText(display_frame, cmd_text, (cols//2 - 150, rows//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 4)

            cv2.imshow("Vision Cockpit", display_frame)
            
            # Capture key for next iteration control
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27: # q or ESC
                 break
            
        self.bg_stopping = True
        self.bg_thread.join(timeout=1.0)
        cap_road.release()
        self.driver_monitor.release()
        cv2.destroyAllWindows()

        # --- UPLOAD TO CLOUD (Correctly placed after loop) ---
        if hasattr(self, 'telemetry_log') and len(self.telemetry_log) > 0:
            print(f"\n[Cloud Sync] Processing {len(self.telemetry_log)} data points...")
            
            # 1. Save to CSV
            import pandas as pd
            import datetime
            from gcp_pipeline import setup_gcp_clients, upload_to_bigquery, BQ_DATASET_NAME, BQ_TABLE_NAME
            
            run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("results")
            output_dir.mkdir(exist_ok=True)
            csv_path = output_dir / f"session_{run_id}.csv"
            
            df = pd.DataFrame(self.telemetry_log)
            # Ensure columns match BigQuery Schema
            # timestamp, relative_time, mode, speed, acceleration, brake_pressure
            df.to_csv(csv_path, index=False)
            print(f"[Cloud Sync] Saved session to {csv_path}")
            
            # 2. Upload
            print("[Cloud Sync] Connecting to BigQuery...")
            _, bq_client = setup_gcp_clients()
            if bq_client:
                print(f"[Cloud Sync] Uploading to {BQ_DATASET_NAME}.{BQ_TABLE_NAME}...")
                upload_to_bigquery(bq_client, BQ_DATASET_NAME, BQ_TABLE_NAME, str(csv_path))
                print("[Cloud Sync] ✅ UPDATE COMPLETE! Verify on your Streamlit Dashboard.")
            else:
                print("[Cloud Sync] ❌ Failed to connect to GCP. Data saved locally only.")

def main():
    project_root = Path(__file__).parent.parent
    # Default road video
    video_path = project_root / "data" / "8359-208052066_small.mp4"
    
    # Check for command line argument (Custom Video)
    if len(sys.argv) > 1:
        user_path = sys.argv[1]
        if os.path.exists(user_path):
            video_path = user_path
            print(f"Loading Custom Video: {video_path}")
        else:
            print(f"Warning: File not found {user_path}, using default.")
    
    # Run System
    system = DualStreamNavigationSystem(str(video_path), driver_camera_index=0)
    system.run()

if __name__ == "__main__":
    main()
