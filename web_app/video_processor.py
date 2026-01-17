"""
Video Processor Module for Web App
Adapts the VisionSpeedSmoothingSystem to yield frames for Streamlit
and batch uploads metrics to BigQuery.
"""
import av
import cv2
import numpy as np
import threading
import queue
import time
from datetime import datetime
import pandas as pd
from google.cloud import bigquery
import os
import streamlit as st

# Import the core system
from src.main import VisionSpeedSmoothingSystem
from src.metrics import OverallMetrics

class WebVisionSystem(VisionSpeedSmoothingSystem):
    """
    Subclass that overrides display methods to yield frames instead of using cv2.imshow
    """
    def __init__(self, video_path, project_id, dataset_name):
        # We need to manually initialize the parent logic since we might be overriding it too much
        # Or better, let's call super properly
        from src.perception import MotionEstimator
        from src.speed_smoother import SpeedSmoother
        from src.driver_monitor import DriverMonitor
        from src.gps_monitor import GPSMonitor
        from src.metrics import MetricsCollector
        from src.video_input import VideoFrameSkipper, load_yolo_model, VEHICLE_CLASSES, TRAFFIC_LIGHT_CLASS

        # Explicitly initialize components instead of relying on super() which might fail due to imports
        self.video_path = video_path
        self.enable_gps_degradation = True
        self.enable_smoothing = True
        
        # Initialize Core Components
        self.perception = MotionEstimator()
        self.gps_monitor = GPSMonitor() 
        self.smoother = SpeedSmoother()
        self.driver_monitor = DriverMonitor()
        self.metrics_collector = MetricsCollector()
        self.frame_skipper = VideoFrameSkipper(skip_n=2)
        
        # Initialize YOLO
        self.yolo_model = load_yolo_model()
        self.vehicle_classes = VEHICLE_CLASSES
        self.traffic_light_class = TRAFFIC_LIGHT_CLASS
        
        self.frame_queue = queue.Queue(maxsize=10)
        self.metric_queue = queue.Queue()
        self.stop_event = threading.Event()
        
        # GCP Config
        self.project_id = project_id
        self.dataset_id = f"{project_id}.{dataset_name}"
        self.table_id = f"{self.dataset_id}.metrics"
        self.bq_client = None
        
        # Buffer for batch upload
        self.metrics_buffer = []
        self.upload_thread = threading.Thread(target=self._upload_worker)
        self.upload_thread.daemon = True
        self.upload_thread.start()
        
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _upload_worker(self):
        """Background worker to batch upload metrics to BQ."""
        try:
            self.bq_client = bigquery.Client(project=self.project_id)
        except Exception as e:
            print(f"BQ Client Init Failed: {e}")
            return

        while not self.stop_event.is_set():
            time.sleep(5) # Upload every 5 seconds
            self._flush_buffer()
            
        # Final flush
        self._flush_buffer()

    def _flush_buffer(self):
        if not self.metrics_buffer:
            return
            
        # Swap buffer
        rows_to_upload = list(self.metrics_buffer)
        self.metrics_buffer.clear()
        
        try:
            errors = self.bq_client.insert_rows_json(self.table_id, rows_to_upload)
            if errors:
                print(f"BQ Insert Errors: {errors}")
            else:
                print(f"Uploaded {len(rows_to_upload)} rows to BQ.")
        except Exception as e:
            print(f"BQ Upload Exception: {e}")

    def process_video_stream(self):
        """
        Generator that yields (frame, stats_dict)
        """
        cap = cv2.VideoCapture(self.video_path)
        
        frame_count = 0
        
        while cap.isOpened() and not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # 1. Process Frame (Core Logic)
            # 1. Process Frame (Core Logic)
            if self.frame_skipper.should_process(frame_count):
                # Run YOLO Inference
                results = self.yolo_model(frame, verbose=False)
                
                # Format detections for Perception Layer
                detections = []
                for result in results:
                     boxes = result.boxes
                     for box in boxes:
                         cls = int(box.cls[0])
                         if cls in self.vehicle_classes:
                             x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                             conf = float(box.conf[0])
                             detections.append({
                                 'bbox': (float(x1), float(y1), float(x2), float(y2)),
                                 'class_id': cls,
                                 'confidence': conf
                             })

                # Estimate Relative Motion (Perception)
                # track_info is Dict[int, Dict]
                track_info = self.perception.estimate_relative_motion(frame, detections)
                
                # Convert track_info to list of ObstacleInfo for SpeedSmoother
                # We need to import ObstacleInfo first? It's in src.speed_smoother
                from src.speed_smoother import ObstacleInfo
                
                obstacles = []
                for t_id, info in track_info.items():
                    # Extract fields safely
                    rel_speed = info.get('relative_speed', 0.0)
                    trend = info.get('closing_trend', 'stable')
                    # distance_estimate was calculated in perception usually (1.0 - area/max)
                    # Let's re-use what perception calculated if available or derive it
                    area = info.get('area', 0.0)
                    dist_est = max(0.0, min(1.0, 1.0 - (area / 50000.0)))
                    
                    obstacles.append(ObstacleInfo(
                        relative_speed=rel_speed,
                        closing_trend=trend,
                        distance_estimate=dist_est,
                        is_traffic_light=False, # TODO: add traffic light logic if needed
                        traffic_light_state=None
                    ))
                
                # Simulate a noisy GPS speed signal (around 40 km/h = ~11 m/s)
                base_speed = 11.1 
                noise = np.random.normal(0, 1.5)
                raw_speed = max(0, base_speed + noise)
                
                # Compute Speed Command (Smoothing)
                # gps_ok=True for simulation
                speed_cmd = self.smoother.compute_speed_command(
                    current_speed=raw_speed,
                    obstacles=obstacles,
                    gps_ok=True,
                    dt=0.1
                )
                
                smoothed_speed = speed_cmd.target_speed
                accel = speed_cmd.acceleration
                brake_pressure = speed_cmd.brake_pressure
                
                # Monitor Driver (Simulated/Real)
                # get_driver_state returns: frame, status, steering, gesture_cmd, face_detected
                _, driver_status, steering, gesture_cmd, face_detected = self.driver_monitor.get_driver_state()
                
                # Format driver state for overlay
                driver_state = {
                    "status": driver_status,
                    "steering": steering,
                    "gesture": gesture_cmd
                }
                
                # Update Metrics
                row = {
                    "run_id": self.run_id,
                    "timestamp": time.time(),
                    "relative_time": frame_count / 30.0, 
                    "mode": "live_web",
                    "speed": float(smoothed_speed),
                    "acceleration": float(accel),
                    "brake_pressure": float(brake_pressure)
                }
                self.metrics_buffer.append(row)
                
                # Overlay Info for Display
                output_frame = self._draw_overlay(frame, detections, smoothed_speed, driver_state)
                
                # Convert BGR to RGB for Streamlit
                output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                
                yield output_frame, row
            
            else:
                 # Skip frame but maybe yield previous? 
                 # For smooth video, we just continue
                 continue
                 
        cap.release()
        self.stop_event.set()

    def _draw_overlay(self, frame, detections, speed, driver_state):
        # Re-use the overlay logic from main.py or simplify
        # For now, simple CV2 drawing
        img = frame.copy()
        
        # Bounding Boxes
        for det in detections:
            # detections is a list of dicts now
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
        # Speed HUD
        cv2.putText(img, f"SPEED: {speed:.1f} m/s", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                   
        # Overlay a "REC" symbol
        cv2.circle(img, (img.shape[1]-50, 50), 10, (0, 0, 255), -1)
        cv2.putText(img, "LIVE CLOUD SYNC", (img.shape[1]-200, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                   
        return img
