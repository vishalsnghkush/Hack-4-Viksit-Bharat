# 🚗 Vision Cockpit AI: Head-Controlled Autonomous Driving Simulator

![Project Banner](https://img.shields.io/badge/AI-Powered-blue) ![Python](https://img.shields.io/badge/Python-3.8%2B-green) ![Status](https://img.shields.io/badge/Status-Operational-brightgreen)

## 🌟 Overview
**Vision Cockpit AI** is an interactive driving simulator where **YOU** are the controller. Instead of using a keyboard or mouse, you control the car using **head gestures** detected by your webcam. 

The system uses advanced Computer Vision (SegFormer + Visual Odometry) to analyze the road in real-time, calculating **Risk**, **Speed**, and **Traffic Safety** just like a real self-driving car.

---

## ✨ Key Features
*   **🗣️ Voice Control (New!)**:
    *   **"Start Driving"**: Activates the car.
    *   **"Stop"**: Stops the car.
    *   **"Park"**: Initiates auto-parking.
    *   **"Speed Up / Slow Down"**: Adjusts cruising speed.
*   **✋ Gesture Toggles**:
    *   **Show Fist**: Start (Toggle ON).
    *   **Show Palm**: Stop (Toggle OFF).
*   **🤖 Autonomous Brain**:
    *   The car handles steering, lane keeping, and obstacle avoidance **automatically** once started.
    *   **Safety Timeout**: If no face is seen for 60s, the car auto-parks.
*   **👁️ Pixel-Level Semantic Segmentation**:
    *   **The Goal**: Colors the entire road as "drivable", sidewalk as "non-drivable", and sky as "background" in real-time.
    *   **Implementation**: Uses **SegFormer** (Cityscapes) for high-resolution scene understanding.
*   **🗺️ 3D Lidar Map**: Projects road features onto a real-time bird's-eye view map.
*   **🛡️ ADAS Safety System**:
    *   **Collision Warning**: Detects cars/trucks and warns of "High Risk".
    *   **Reasoning Layer (VLM)**: Displays the AI's "Inner Monologue" (e.g., "Obstacle Interaction Likely").
    *   **Pedestrian Intent**: Monitors pose to predict crossing, flashing RED borders if intent is deduced.
    *   **Comfort Score**: Rates your driving smoothness.
*   **👻 Ghost Planning (Counterfactuals)**:
    *   Visualizes "What if?" scenarios by drawing predicted trajectories (Purple Arrows) for other vehicles.
    *   Simulates multi-agent interaction logic.
*   **🧠 Neural Radiance Fields (NeRF) Memory**:
    *   Builds a high-fidelity 3D "Neural Map" of the environment in real-time.
    *   Allows the car to "remember" past locations, creating a digital twin of the chaos it drove through.
*   **� systematic Evaluation (Science Layer)**:
    *   **Stress Test Module**: Automatically injects "Worst-Case Scenarios" (Sudden Glare, Surprise Obstacles) to test system robustness.
    *   **Safety Metrics**: Tracks **Disengagement Rate** (Human Panic Braking) and **Safety Violations** in real-time.
*   **📉 True Uncertainty Quantification (Doubt Layer)**:
    *   **Aleatoric vs Epistemic**: Distinguishes between "Noisy Data" (Rain/Glare) and "Unknown Objects" (Model Ignorance).
    *   **High Alert Mode**: Automatically switches system mode from "AUTONOMOUS" to "HIGH ALERT" when uncertainty spikes > 40%.
*   **📍 Absolute Vision Localization (Memory Layer)**:
    *   **Semantic SLAM**: Recognizes Landmarks (Traffic Lights, Signs) to find global coordinates without GPS.
    *   **Status Lock**: HUD updates to "GPS: VISUAL LOCK" when landmarks are confirmed.
*   **🔥 Explainable AI (XAI) Heatmaps (Transparency Layer)**:
    *   **Real-Time Saliency**: Overlays a "Heatmap" glowing red on obstacles that caused the AI to brake.
    *   **Transparency**: Shows *why* the decision was made.
*   **�📡 Precision Mapping (VO Refinement)**:
    *   Uses **Semantic Segmentation** to "mask out" moving vehicles.
    *   Ensures the 3D Map only tracks static world objects (Road, Buildings), preventing map drift.
*   **🌦️ Dynamic Weather Engine**: Automatically cycles through **Sunny**, **Rainy**, **Night**, and **Mirror** modes to keep the drive interesting.
*   **🤖 Autonomous Fallback (New!)**:
    *   **Auto-Pilot**: If the driver's face is not detected, the system automatically takes control.
    *   **Lane Keeping**: Follows the center of the road using semantic segmentation.
    *   **Safe Stopping**: Slows down or stops if obstacles are detected, ensuring safety even without human input.
---

## 💻 System Requirements
*   **OS**: Windows 10/11, Linux, or macOS.
*   **Python**: Version 3.8 or higher.
*   **Hardware**: 
    *   Webcam (Required for controls).
    *   Decent CPU (GPU recommended but optional).

---

## 🚀 Installation Guide

### 1. Clone or Download
Download this project folder to your local computer.

### 2. Set Up Environment
It is recommended to use a virtual environment. Open your terminal/command prompt in the project folder:

```bash
# Create virtual environment
python -m venv .venv

# Activate it (Windows)
.venv\Scripts\Activate.ps1
# Activate it (Mac/Linux)
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Note: This creates the `.venv` folder and installs libraries like `torch`, `opencv-python`, and `transformers`.)*

---

## 🎮 How to Run

You have **3 ways** to use the simulator:

### Method 1: The Scenario Menu (Recommended)
This uses the included sample videos to demonstrate the AI capabilities.
```bash
cd src
python run_cockpit.py
```
*   Select **[1]** for Rainy Loop.
*   Select **[2]** for Sunny Highway.
*   Select **[6]** to Browse for your own video.

### Method 2: Drag and Drop (Fastest)
Want to simulate your own dashcam video? Just drag the file onto the terminal!
```bash
cd src
python demo_dual_stream.py "C:\Path\To\Your\Video.mp4"
```
*   The AI will automatically resize your video to fit its neural network.

### Method 3: Direct Launch
Runs the default embedded video with all features active.
```bash
cd src
python demo_dual_stream.py
```

---

## 🕹️ Driving Instructions
Once the simulator starts:
1.  **Sit comfortably** in front of your webcam.
2.  Ensure your face is detected (Green box around face).
3.  **To Start (Go)**: Show an **OPEN PALM** (Hand up, fingers spread). The car will speed up.
4.  **To Stop**: Show a **FIST** (Closed hand). The car will brake.
5.  **To Turn**: Lean your head **Left** or **Right**.
6.  **Autonomous Fallback**: If you hide your face or move out of frame, the AI takes over automatically.

**HUD Indicators:**
*   **RISK**: High/Low (Based on traffic ahead).
*   **TTC**: Time-To-Collision (Seconds).
*   **COMFORT**: 0-100% (Don't drive jerkily!).

---

## � HUD Legend (Visual Guide)
### 🎨 Visual Elements
*   **🟣 Pink/Purple Arrows**: **Ghost Planning**. Shows where other cars *might* go.
*   **🔴 Red Flashing Border**: **Pedestrian Intent**. Warns that a person might cross.
*   **🔥 Glowing Red Heatmap**: **XAI Saliency**. The specific object causing the AI to brake.
*   **🟢 Green/Purple Road**: **Semantic Brain**. The AI "painting" the drivable area.
*   **🔵 Blue/Red Point Cloud**: **NeRF Memory**. The 3D "Digital Twin" of the road.

### 💬 On-Screen Status Messages (Glossary)
| Message / Keyword | Meaning | Layer |
| :--- | :--- | :--- |
| **"GPS: VISUAL LOCK"** | The AI recognized a **Traffic Sign/Light** and fixed its position without Satellites. | **Memory** |
| **"GPS: DENIED"** | No satellites or landmarks found; relying on Visual Odometry estimation. | **Memory** |
| **"XAI: ATTENTION BLOCKED"** | The car is stopping because of the object highlighted in **RED**. | **Transparency** |
| **"STRESS TEST: GLARE"** | The system is simulating a **Blind Sun Glare** event to test robustness. | **Science** |
| **"STRESS TEST: SURPRISE"** | A red block was injected to test reaction time. | **Science** |
| **"UNCERTAINTY: High %"** | The AI is "confused" (due to rain/glare/unknown objects). | **Doubt** |
| **"MODE: HIGH ALERT"** | The AI switched to Caution Mode because Uncertainty > 40%. | **Doubt** |
| **"AI THOUGHT: ..."** | The "Inner Monologue" explaining logic (e.g., "Obstacle Ahead -> Braking"). | **Reasoning** |
| **"DISENGAGEMENTS: X"** | Count of times you had to Panic Brake (Human Takeover). | **Science** |

---

## �🔧 Technical Architecture

### The Brain (Perception)
*   **Model**: `nvidia/segformer-b0-finetuned-cityscapes`
*   **Function**: Breaks the image into semantic classes (Road, Car, Sky).
*   **Training**: Trained on **Cityscapes** (German cities), but generalizes to any video.

### The Eyes (Visual Odometry)
*   **Algorithm**: FAST Feature Detector + Lucas-Kanade Optical Flow.
*   **Function**: Tracks ground pixels frame-to-frame to estimate specific speed and trajectory without GPS.

### The Supervisor (Driver Monitor)
*   **Algorithm**: Haar Cascades.
*   **Function**: Calculates head pitch/yaw and Eye Aspect Ratio (EAR) to detect sleepiness.

---

## ⚠️ Troubleshooting
*   **"Camera Not Found"**: Ensure no other app (Zoom/Teams) is using your webcam.
*   **"RuntimeError: Mat1 and Mat2 shapes..."**: This is fixed! The system now auto-resizes videos.
*   **Slow Performance**: This is a heavy AI application. If it lags, try resizing your video to a smaller resolution (e.g., 640x480) before loading it.
*   **UI Cutoff/Overlap**: The new dashboard is designed for **1600x900**. Ensure your monitor resolution is high enough.

## 🔮 Future Roadmap (Suggestions)
*   **Hardware Integration**: Connect a Logitech G29 Racing Wheel for realistic steering.
*   **Lane Keeping Assist (LKA)**: Add logic to automatically steer the car to the center of the lane.
*   **Voice Control**: Use speech recognition to change weather modes ("Alexa, make it sunny").

---

## 📂 Source Code
**GitHub Repository**: [https://github.com/vishalsnghkush/vision-speed-smoothing-ai](https://github.com/vishalsnghkush/vision-speed-smoothing-ai)

---

## 📜 Credits
*   **AI Model**: HuggingFace & NVIDIA (SegFormer).
*   **Datasets**: Cityscapes Dataset (Cordts et al.).
*   **Created For**: Interactive Vision Speed Smoothing & Safety Project.
