
"""
Reasoning Layer: Spatial Understanding & Decision Making
Uses "Cityscapes" semantic classes to make "Alpamayo"-style reasoning decisions.
"""
import numpy as np

class SpatialReasoning:
    """
    The 'Brain' of the agent.
    Analyzes the 'Perception' (Segmentation) to output 'Action' (Text/Control).
    """
    def __init__(self):
        print("Initializing Reasoning Layer (Rule-Based VLM)...")
        # Cityscapes Class IDs
        self.ROAD = 0
        self.SIDEWALK = 1
        self.VEHICLE = 13
        self.PERSON = 11
        
    def think(self, segmentation_mask: np.ndarray) -> str:
        """
        Analyze the scene and generate a 'thought' (reasoning chain).
        """
        h, w = segmentation_mask.shape
        total_pixels = h * w
        
        # 1. Analyze Core Areas
        road_pixels = np.sum(segmentation_mask == self.ROAD)
        person_pixels = np.sum(segmentation_mask == self.PERSON)
        vehicle_pixels = np.sum(segmentation_mask == self.VEHICLE)
        
        road_ratio = road_pixels / total_pixels
        risk_level = "LOW"
        
        # 2. Formulate Safety Check
        thought = ""
        
        # Check Road Scan
        if road_ratio > 0.3:
            thought += f"Drivable path clear ({int(road_ratio*100)}%). "
        else:
            thought += "Narrow drivable path detected. "
            risk_level = "MEDIUM"

        # Check for Hazards
        if person_pixels > (total_pixels * 0.005): # > 0.5% of screen
            thought += "CAUTION: Pedestrian detected. "
            risk_level = "HIGH"
        
        if vehicle_pixels > (total_pixels * 0.1):
            thought += "Vehicle ahead. "
            
        # 3. Final Decision
        if risk_level == "HIGH":
            decision = "DECISION: SLOW DOWN (Precision Mode)"
        elif risk_level == "MEDIUM":
            decision = "DECISION: CAUTION (Scan Mode)"
        else:
            decision = "DECISION: PROCEED (Cruise Mode)"
            
        return f"[{risk_level}] {thought}\n{decision}"
