"""
Perception Layer: Semantic Segmentation using DeepLabV3-MobileNetV3 (Torchvision)
Optimized for CPU/Real-Time Performance
"""
from typing import Tuple
import torch
import numpy as np
import cv2
from torchvision import models, transforms
from PIL import Image

class SemanticSegmenter:
    """
    Semantic Segmentation using DeepLabV3 with MobileNetV3 backbone.
    Much faster than Transformer-based models on CPU.
    """
    def __init__(self):
        print("Loading MobileNetV3 Segmentation model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load pre-trained DeepLabV3 with MobileNetV3 backbone
        self.model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
        self.model.eval()
        self.model.to(self.device)
        print("Model loaded successfully.")
        
        # Standard ImageNet normalization
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # COCO/Pascal VOC Palette (Simplified for common road objects)
        # Class 15 = Person, 7 = Car, 3 = Motorcycle, 2 = Bicycle, 6 = Bus, 8 = Cat/Dog...
        # We need to map these to our own simple IDs or just use them directly.
        # DeepLabV3 (COCO) classes:
        # 0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
        # 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=diningtable
        # 12=dog, 13=horse, 14=motorbike, 15=person, 16=pottedplant
        # 17=sheep, 18=sofa, 19=train, 20=tvmonitor
        
        # Mapping for colors (BGR)
        self.colors = np.zeros((21, 3), dtype=np.uint8)
        self.colors[0] = [0, 0, 0]       # Background
        self.colors[15] = [0, 0, 255]    # Person (Red)
        self.colors[7] = [255, 0, 0]     # Car (Blue)
        self.colors[6] = [255, 165, 0]   # Bus (Cyan-ish)
        self.colors[14] = [0, 255, 255]  # Motorbike (Yellow)
        self.colors[2] = [0, 255, 0]     # Bicycle (Green)
        # Others gray
        for i in range(21):
            if np.sum(self.colors[i]) == 0 and i != 0:
                self.colors[i] = [100, 100, 100]

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment a single video frame.
        """
        # Resize for speed (MobileNet likes 520 roughly, but 320 is faster)
        input_h, input_w = 320, 320
        frame_resized = cv2.resize(frame, (input_w, input_h))
        rgb_image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        try:
            # Preprocess
            input_tensor = self.preprocess(Image.fromarray(rgb_image))
            input_batch = input_tensor.unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                output = self.model(input_batch)['out'][0]
            
            # Post-process
            output_predictions = output.argmax(0).byte().cpu().numpy()
            
            # Map COCO classes to our expected "Cityscapes-like" IDs for logic compatibility if needed
            # Current Logic expects: 11=Person, 13=Car, 14=Truck, 15=Bus
            # COCO: 15=Person, 7=Car, 6=Bus, 14=Motorbike
            # We will remap them for the logic:
            
            remapped_mask = np.zeros_like(output_predictions)
            remapped_mask[output_predictions == 15] = 11 # Person -> 11
            remapped_mask[output_predictions == 7] = 13  # Car -> 13
            remapped_mask[output_predictions == 6] = 15  # Bus -> 15
            remapped_mask[output_predictions == 14] = 12 # Motorbike -> 12 (Rider)
            
            # Create overlay
            # Apply colors to the raw mask first for visualization
            seg_color = self.colors[output_predictions]
            seg_bgr = seg_color # Already BGR in self.colors init
            
            # Upscale result to original frame size
            # Resize 'seg_bgr' to match input 'frame'
            seg_bgr_large = cv2.resize(seg_bgr, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            output_predictions_large = cv2.resize(remapped_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # Blend
            overlay = cv2.addWeighted(frame, 0.7, seg_bgr_large, 0.3, 0)
            
            return overlay, output_predictions_large
            
        except Exception as e:
            print(f"Perception Error: {e}")
            return frame, np.zeros(frame.shape[:2], dtype=np.uint8)

if __name__ == "__main__":
    segmenter = SemanticSegmenter()
    # Dummy run
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    segmenter.process_frame(img)
    print("Test run complete.")
