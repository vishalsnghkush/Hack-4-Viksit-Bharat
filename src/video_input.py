"""
Video Input Module
Reads and displays video files using OpenCV with YOLOv8 object detection.
Detects vehicles and traffic lights in video frames.
"""

import cv2
import os
import sys
from pathlib import Path
from ultralytics import YOLO
import numpy as np


# YOLOv8 COCO class IDs for vehicles and traffic lights
VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat'
}
TRAFFIC_LIGHT_CLASS = 9  # traffic light

# Colors for bounding boxes (BGR format)
VEHICLE_COLOR = (0, 255, 0)  # Green for vehicles
TRAFFIC_LIGHT_COLOR = (0, 0, 255)  # Red for traffic lights


def load_yolo_model(model_path: str = None) -> YOLO:
    """
    Load YOLOv8 model. Downloads if not available.
    
    Args:
        model_path: Optional path to custom model. If None, uses pretrained 'yolov8n.pt'
    
    Returns:
        Loaded YOLO model
    """
    if model_path is None:
        print("Loading YOLOv8 model (this will download on first run)...")
        model = YOLO('yolov8n.pt')  # nano model for speed, use 'yolov8s.pt' or 'yolov8m.pt' for better accuracy
    else:
        print(f"Loading YOLOv8 model from {model_path}...")
        model = YOLO(model_path)
    
    print("YOLOv8 model loaded successfully!")
    return model


def draw_detections(frame: np.ndarray, results, vehicle_classes: dict, traffic_light_class: int) -> np.ndarray:
    """
    Draw bounding boxes and labels for detected vehicles and traffic lights.
    
    Args:
        frame: Input frame
        results: YOLOv8 detection results
        vehicle_classes: Dictionary mapping class IDs to vehicle names
        traffic_light_class: Class ID for traffic lights
    
    Returns:
        Frame with drawn detections
    """
    annotated_frame = frame.copy()
    
    # Process detections
    for result in results:
        boxes = result.boxes
        
        for box in boxes:
            # Get class ID and confidence
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            # Only process vehicles and traffic lights
            if class_id in vehicle_classes or class_id == traffic_light_class:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Choose color and label based on class
                if class_id in vehicle_classes:
                    color = VEHICLE_COLOR
                    label = f"{vehicle_classes[class_id]} {confidence:.2f}"
                else:
                    color = TRAFFIC_LIGHT_COLOR
                    label = f"traffic light {confidence:.2f}"
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label background
                (label_width, label_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1 - label_height - 10),
                    (x1 + label_width, y1),
                    color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
    
    return annotated_frame


def read_and_display_video(
    video_path: str,
    window_name: str = "Video Player with YOLOv8 Detection",
    model_path: str = None,
    conf_threshold: float = 0.25
) -> None:
    """
    Read a video file and display frames with YOLOv8 object detection.
    
    Args:
        video_path: Path to the video file
        window_name: Name of the display window
        model_path: Optional path to custom YOLOv8 model
        conf_threshold: Confidence threshold for detections (0.0-1.0)
    """
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
    
    # Load YOLOv8 model
    try:
        model = load_yolo_model(model_path)
    except Exception as e:
        print(f"Error loading YOLOv8 model: {e}")
        print("Make sure ultralytics is installed: pip install ultralytics")
        return
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo Properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Detection Threshold: {conf_threshold}")
    print(f"\nDetecting: Vehicles and Traffic Lights")
    print(f"\nControls:")
    print(f"  Press 'q' or ESC to quit")
    print(f"  Press SPACE to pause/resume")
    
    paused = False
    frame_count = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            
            if not ret:
                print("End of video reached")
                break
            
            frame_count += 1
            
            # Run YOLOv8 inference
            results = model(frame, conf=conf_threshold, verbose=False)
            
            # Draw detections on frame
            annotated_frame = draw_detections(
                frame, results, VEHICLE_CLASSES, TRAFFIC_LIGHT_CLASS
            )
            
            # Add frame counter and FPS info
            info_text = f"Frame: {frame_count}/{total_frames}"
            cv2.putText(
                annotated_frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Display the annotated frame
            cv2.imshow(window_name, annotated_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(int(1000 / fps) if fps > 0 else 30) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif key == ord(' '):  # SPACE
            paused = not paused
            if paused:
                print("Paused - Press SPACE to resume")
            else:
                print("Resumed")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nVideo playback finished. Processed {frame_count} frames.")


def main():
    """Main function to run video input with YOLOv8 detection."""
    # Get project root directory (parent of src)
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    models_dir = project_root / "models"
    
    # Default video file
    default_video = data_dir / "8359-208052066_small.mp4"
    
    # Parse command line arguments
    video_path = None
    model_path = None
    conf_threshold = 0.25
    
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--model" or arg == "-m":
            if i + 1 < len(sys.argv):
                model_path = sys.argv[i + 1]
                i += 2
            else:
                print("Error: --model requires a path")
                return
        elif arg == "--conf" or arg == "-c":
            if i + 1 < len(sys.argv):
                try:
                    conf_threshold = float(sys.argv[i + 1])
                    i += 2
                except ValueError:
                    print("Error: --conf requires a float value")
                    return
            else:
                print("Error: --conf requires a value")
                return
        elif arg == "--help" or arg == "-h":
            print("Usage: python video_input.py [video_path] [options]")
            print("\nOptions:")
            print("  --model, -m PATH    Path to custom YOLOv8 model")
            print("  --conf, -c FLOAT    Confidence threshold (default: 0.25)")
            print("  --help, -h          Show this help message")
            return
        else:
            video_path = arg
            i += 1
    
    # Set default video if not provided
    if video_path is None:
        video_path = str(default_video)
    else:
        # If relative path, try relative to project root
        if not os.path.isabs(video_path):
            video_path = str(project_root / video_path)
    
    # Check if custom model path is relative
    if model_path and not os.path.isabs(model_path):
        model_path = str(models_dir / model_path)
    
    print(f"Reading video from: {video_path}")
    read_and_display_video(
        str(video_path),
        model_path=model_path if model_path else None,
        conf_threshold=conf_threshold
    )


if __name__ == "__main__":
    main()
