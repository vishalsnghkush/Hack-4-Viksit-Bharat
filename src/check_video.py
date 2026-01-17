import cv2
import sys
from pathlib import Path

video_path = r"c:/Users/VISHAL KUSHWAHA/OneDrive/Desktop/vision_speed_smoothing_ai/data/istockphoto-2159760544-640_adpp_is.mp4"

def check_video():
    if not Path(video_path).exists():
        print(f"File not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video")
        return

    ret, frame = cap.read()
    if ret:
        print(f"Shape: {frame.shape}")
        print(f"Dtype: {frame.dtype}")
    else:
        print("Failed to read first frame")
    cap.release()

if __name__ == "__main__":
    check_video()
