
import cv2
import numpy as np
import time
import mediapipe as mp
import os

# New MediaPipe Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class DriverMonitor:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        # Use DirectShow (CAP_DSHOW) to avoid MSMF errors on Windows
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
             print(f"Warning: Camera {camera_index} failed to open with DSHOW. Trying default.")
             self.cap = cv2.VideoCapture(camera_index)
        
        # Enforce MJPG format to fix green/striping artifacts
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Debug: Print actual resolution
        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Driver Camera initialized at {actual_w}x{actual_h}")

        # --- FACE DETECTION (For Steering) ---
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # --- HAND DETECTION (New Tasks API) ---
        model_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                               num_hands=1)
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        self.status = "NORMAL"
        self.steering = 0.0
        self.speed_cmd = 0.0 # Default to Stop if no hand

        # Connections for drawing (standard hand connections)
        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]

    def count_fingers(self, hand_landmarks):
        # hand_landmarks is a list of NormalizedLandmark objects
        
        lm_list = hand_landmarks # In new API, this IS the list
            
        if not lm_list: return 0
        
        # Thumb: Check if tip is to the right/left of IP joint (depending on hand)
        # Simplifying: Checks distance to Wrist (0). If Tip is further than IP, it's extended.
        # WRIST = 0
        wrist = lm_list[0]
        
        # Tips: 4, 8, 12, 16, 20
        # IP/PIP: 3, 6, 10, 14, 18 (Using simplistic joints)
        
        tip_ids = [4, 8, 12, 16, 20]
        pip_ids = [3, 6, 10, 14, 18] # Using lower joint for stability
        
        count = 0
        for i in range(5):
            # Calculate distance to wrist (invariant to rotation)
            tip_dist = np.hypot(lm_list[tip_ids[i]].x - wrist.x, lm_list[tip_ids[i]].y - wrist.y)
            pip_dist = np.hypot(lm_list[pip_ids[i]].x - wrist.x, lm_list[pip_ids[i]].y - wrist.y)
            
            if tip_dist > pip_dist:
                count += 1
                
        return count

    def draw_hand_landmarks(self, image, landmarks):
        h, w, _ = image.shape
        
        # Draw connections
        for p1_idx, p2_idx in self.HAND_CONNECTIONS:
            p1 = landmarks[p1_idx]
            p2 = landmarks[p2_idx]
            x1, y1 = int(p1.x * w), int(p1.y * h)
            x2, y2 = int(p2.x * w), int(p2.y * h)
            cv2.line(image, (x1, y1), (x2, y2), (200, 200, 200), 2)

        # Draw points
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1) # Red points
            cv2.circle(image, (cx, cy), 3, (255, 255, 255), -1)

    def get_driver_state(self):
        if self.cap is None or not self.cap.isOpened():
            return None, "CAM_OFF", 0.0, "NONE", False

        try:
            ret, frame = self.cap.read()
        except Exception as e:
            print(f"Camera Read Error: {e}")
            return None, "CAM_ERR", 0.0, "NONE", False

        if not ret:
            return None, "CAMERA_FAIL", 0.0, "NONE", False

        frame = cv2.flip(frame, 1) # Mirror view
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. STEERING (Face Detection)
        self.steering = 0.0
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4) # Slightly stricter
        
        face_detected = False
        if len(faces) > 0:
            face_detected = True
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            (x, y, face_w, face_h) = largest_face
            cv2.rectangle(frame, (x, y), (x+face_w, y+face_h), (255, 0, 0), 2)
            
            face_center_x = x + face_w / 2
            img_center_x = w / 2
            diff_x = face_center_x - img_center_x
            max_deflection_x = w / 3.0 
            
            if abs(diff_x) > (w * 0.05): # Deadzone
                self.steering = diff_x / max_deflection_x
                self.steering = max(-1.0, min(1.0, self.steering))

        # 2. SPEED (Hand Gestures - TOGGLE LOGIC)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image)
        
        gesture_cmd = "NONE" # NONE, START, STOP
        self.status = "WAITING"
        
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # Draw
                self.draw_hand_landmarks(frame, hand_landmarks)
                
                # Logic
                finger_count = self.count_fingers(hand_landmarks)
                
                if finger_count >= 4: # Palm -> STOP
                    gesture_cmd = "STOP"
                    self.status = "CMD: STOP (PALM)"
                elif finger_count <= 1: # Fist -> START
                    gesture_cmd = "START"
                    self.status = "CMD: START (FIST)"
                else:
                     self.status = "HAND DETECTED"
        
        # NOTE: Drawing removed to avoid conflict with Main Dashboard overlay. 
        # This module only returns the frame and data now.

        # Return gesture_cmd instead of continuous speed
        return frame, self.status, self.steering, gesture_cmd, face_detected

    def release(self):
        self.cap.release()
        try:
             self.detector.close()
        except:
             pass

if __name__ == "__main__":
    monitor = DriverMonitor()
    while True:
        frame, status, steer, speed = monitor.get_driver_state()
        if frame is None: break
        
        cv2.imshow("Driver Monitor (MediaPipe)", frame)
        if cv2.waitKey(1) == ord('q'): break
    
    monitor.release()
    cv2.destroyAllWindows()
