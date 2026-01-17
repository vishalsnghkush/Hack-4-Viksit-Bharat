# GeeksForGeeks Resources

Here is a list of GeeksForGeeks articles that cover the core concepts and algorithms used in this project (`vision_speed_smoothing_ai`).

## 1. Computer Vision & Visual Odometry
**Code Reference:** `src/localization_vo.py`
*   **Topic:** Optical Flow (Lucas-Kanade Method)
    *   **Link:** [Optical Flow in OpenCV (Python) - GeeksForGeeks](https://www.geeksforgeeks.org/python-opencv-optical-flow-with-lucas-kanade-method/)
    *   **Description:** Learn how to track feature points between frames using the Lucas-Kanade algorithm, which is the foundation of the Visual Odometry system in this project.

## 2. Object Detection & Driver Monitoring
**Code Reference:** `src/driver_monitor.py`
*   **Topic:** Face Detection (Haar Cascades)
    *   **Link:** [Face Detection using Haar Cascades in OpenCV - GeeksForGeeks](https://www.geeksforgeeks.org/face-detection-using-cascade-classifier-using-opencv-python/)
    *   **Description:** Explains how to use pre-trained Haar Cascade XML files to detect faces in real-time video, used here for the driver monitoring system.
*   **Topic:** Head Pose Estimation
    *   **Link:** [Head Pose Estimation using OpenCV - GeeksForGeeks](https://www.geeksforgeeks.org/head-pose-estimation-using-opencv-and-dlib/)
    *   **Description:** (General Concept) Understanding how to estimate the orientation of the head (Yaw/Pitch/Roll) from 2D facial landmarks.

## 3. Deep Learning & Perception
**Code Reference:** `src/perception.py`
*   **Topic:** Semantic Segmentation
    *   **Link:** [Introduction to Semantic Segmentation - GeeksForGeeks](https://www.geeksforgeeks.org/introduction-to-semantic-segmentation/)
    *   **Description:** Covers the concept of classifying every pixel in an image (e.g., Road vs. Sky), which is implemented in this project using the SegFormer model.

## 4. State Estimation & Smoothing
**Code Reference:** `src/speed_smoother.py`
*   **Topic:** Kalman Filter
    *   **Link:** [Kalman Filter in Python - GeeksForGeeks](https://www.geeksforgeeks.org/kalman-filter-in-python/)
    *   **Description:** A detailed guide on implementing Kalman Filters for predicting and smoothing noisy data, essential for the "Speed Smoothing" aspect of the project.

## 5. Python & OpenCV Basics
**Code Reference:** `src/demo_dual_stream.py`
*   **Topic:** Image Processing (Resizing, Colors)
    *   **Link:** [Image Processing in Python (OpenCV) - GeeksForGeeks](https://www.geeksforgeeks.org/image-processing-in-python/)
