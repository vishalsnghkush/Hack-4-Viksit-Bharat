
"""
Vision-Only Navigation Demo
Implements Phase 1 of the new architecture:
- Perception: Semantic Segmentation (SegFormer/Cityscapes)
- Localization: Visual Odometry (Placeholder/Simulated)
- Reasoning: VLM (Placeholder)
"""
import cv2
import time
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from perception_transformer import SemanticSegmenter
from localization_vo import VisualOdometry
from reasoning_core import SpatialReasoning


class VisionNavigationSystem:
    def __init__(self, video_source):
        self.video_source = video_source
        
        # Load Perception (SegFormer - Cityscapes)
        print("Initializing Perception Layer (Cityscapes)...")
        try:
            self.segmenter = SemanticSegmenter()
            self.use_segmentation = True
        except ImportError as e:
            print(f"Warning: Could not load SegFormer: {e}")
            print("Running in 'Blind' mode.")
            self.use_segmentation = False
        except Exception as e:
            print(f"Error loading SegFormer: {e}")
            self.use_segmentation = False

        # Visual Odometry State (Phase 2: Active)
        self.vo = VisualOdometry(focal_length=700.0, pp=(640, 360)) # Approximate for 720p/1080p video
        self.reasoning = SpatialReasoning()
        self.position = [0.0, 0.0, 0.0]
        self.trajectory = []
        
    def run(self):
        # Handle webcam (int) or video file (str)
        source = self.video_source
        if str(source).isdigit():
            source = int(source)
            print(f"Opening camera {source}...")
        else:
            print(f"Opening video file {source}...")

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Could not open {source}")
            return

        print("\nStarting Vision-Only Navigation System...")
        print("Press 'q' to quit")
        
        # Output saver
        project_root = Path(__file__).parent.parent
        output_path = project_root / "vision_navigation_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = None
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if out_writer is None:
                 h, w = frame.shape[:2]
                 out_writer = cv2.VideoWriter(str(output_path), fourcc, 30.0, (w, h))

            start_time = time.time()
            
            # 1. Perception Step
            display_frame = frame.copy()
            if self.use_segmentation:
                # Process every Nth frame to maintain interactivity on CPU
                if frame_count % 3 == 0: 
                    self.last_segmentation, self.last_mask = self.segmenter.process_frame(frame)
                
                if hasattr(self, 'last_segmentation'):
                    display_frame = self.last_segmentation
                    # 1.5 Reasoning Step (The "Brain")
                    # Using the perception result to make decisions
                    # Use last_mask (raw IDs) for reasoning, NOT last_segmentation (image)
                    self.current_thought = self.reasoning.think(self.last_mask)

            # 2. Localization Step (Real Monocular VO)
            # We assume a scalar speed for scale ambiguity resolution
            current_scale = 10.0 * 0.033 # Speed (10m/s) * dt
            
            self.vo.update(frame, speed_scale=current_scale)
            x, y, z = self.vo.get_position()
            
            # Map X/Z to 2D trajectory (Bird's eye view usually X-Z in camera coords)
            self.trajectory.append((x, z))
            
            # Draw UI
            self._draw_hud(display_frame)
            
            # Save frame
            if out_writer:
                out_writer.write(display_frame)
            
            cv2.imshow("Vision-Only Navigation (Cityscapes + UrbanNav)", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        if out_writer:
            out_writer.release()
            print(f"\nSaved output video to: {output_path}")
        cv2.destroyAllWindows()

    def _draw_hud(self, frame):
        h, w = frame.shape[:2]
        
        # --- 1. Reasoning Display ("The Brain") ---
        if hasattr(self, 'current_thought'):
            # Text Box Background
            box_h = 80
            cv2.rectangle(frame, (0, 0), (w, box_h), (0, 0, 0), -1)
            cv2.rectangle(frame, (0, 0), (w, box_h), (0, 255, 0), 2)
            
            lines = self.current_thought.split('\n')
            for i, line in enumerate(lines):
                color = (0, 255, 255) if "DECISION" in line else (200, 200, 200)
                cv2.putText(frame, line, (20, 30 + i*30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # --- 2. Localization Map (Visual Odometry) ---
        map_size = 200
        map_img = np.zeros((map_size, map_size, 3), dtype=np.uint8)
        cv2.rectangle(map_img, (0,0), (map_size, map_size), (30,30,30), -1)
        
        if len(self.trajectory) > 1:
            points = np.array(self.trajectory)
            # Dynamic Scaling
            min_x, max_x = np.min(points[:,0]), np.max(points[:,0])
            min_z, max_z = np.min(points[:,1]), np.max(points[:,1])
            
            range_x = max(1.0, max_x - min_x)
            range_z = max(1.0, max_z - min_z)
            max_range = max(range_x, range_z)
            
            # Scale to fit 80% of map
            scale = (map_size * 0.8) / max_range
            
            # Center map
            center_x = (min_x + max_x) / 2
            center_z = (min_z + max_z) / 2
            
            # Transform points
            # Map X -> Image X, Map Z -> Image Y (inverted)
            # Normalized (-0.5 to 0.5) * size + offset
            norm_x = (points[:,0] - center_x) * scale + (map_size/2)
            norm_z = (map_size/2) - (points[:,1] - center_z) * scale # Invert Z for display
            
            # Ensure points are integers
            draw_points = np.stack([norm_x, norm_z], axis=1).astype(np.int32)
            
            cv2.polylines(map_img, [draw_points], False, (0, 255, 0), 2)
            # Draw current pos (red dot)
            cv2.circle(map_img, tuple(draw_points[-1]), 4, (0, 0, 255), -1)
            
        # Overlay map (Bottom Left)
        frame[h-map_size-20:h-20, 20:20+map_size] = map_img
        cv2.putText(frame, "Visual Odometry", (20, h-map_size-25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Vision-Only Navigation Demo")
    parser.add_argument("--source", type=str, default=None, help="Video source (path or 0 for cam)")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    
    if args.source:
        source = args.source
    else:
        # Default to the sample video
        source = project_root / "data" / "8359-208052066_small.mp4"

    system = VisionNavigationSystem(str(source))
    system.run()

if __name__ == "__main__":
    main()
