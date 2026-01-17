# 🚀 Final Deployment Roadmap: GhostJam Buster

**Project**: Vision Speed Smoothing AI ("GhostJam Buster")  
**Platform**: Google Cloud Run  
**Architecture**: Hybrid (Local Edge AI + Cloud Serverless)  
**Version**: 1.1 (Live Patch)

---

## 🗺️ System Architecture

This project demonstrates a high-performance **Hybrid Edge-Cloud Digital Twin** architecture, designed to minimize latency while maximizing accessibility.

### 1. Edge Layer (The "Brain")
*   **Location**: Running locally on the user's machine (or edge device like NVIDIA Jetson).
*   **Function**:
    *   Executes the **computer vision pipeline** (YOLOv8/SegFormer) to analyze video frames in real-time.
    *   Calculates **speed commands** and **safety metrics** (TTC, brake pressure) locally to ensure sub-millisecond control loop latency.
*   **Data Egress**: Instead of processing heavy video in the cloud (high bandwidth/latency), we upload only the **lightweight telemetry payload** (JSON) to the cloud.

### 2. Streaming Layer (The "Bridge")
*   **Protocol**: Direct streaming to **Google BigQuery** using the Storage Write API.
*   **Frequency**: Batched uploads every **1.0 second** to balance network load and real-time visualization.
*   **Optimizations**: Data is verified and structured into a time-series schema before upload.

### 3. Cloud Layer (The "View")
*   **Host**: **Google Cloud Run** containers.
*   **Frontend**: A responsive **Streamlit** dashboard.
*   **Logic**: 
    1.  Fetches the latest telemetry from BigQuery (ordered by `timestamp DESC`).
    2.  **Visualizes** speed smoothing effects, acceleration jitter, and driver safety profiles.
    3.  **Auto-refreshes** to act as a near real-time "Digital Twin" of the local simulation.

```mermaid
graph LR
    User[🚗 User/Driver] -->|Input| LocalApp[🖥️ Edge AI Cockpit]
    LocalApp -->|Computer Vision Processing| LocalApp
    LocalApp -->|Telemetry Stream (JSON)| BQ[(🔍 BigQuery)]
    BQ -->|SQL Query| CloudRun[☁️ Cloud Run Service]
    CloudRun -->|HTTPS / WebSocket| Web[📱 Live Dashboard]
```

---

## 🛠️ Technology Stack & Cloud Integration

We leveraged a modern, serverless stack to ensure the solution is both **scalable** and **cost-effective**.

| Component | Technology | Detailed Role |
| :--- | :--- | :--- |
| **Edge Compute** | **Python + PyTorch** | Runs the neural networks for perception and control logic. |
| **Hosting** | **Google Cloud Run** | Hosts the web application. Key benefit: **Scales to Zero** (costs $0 when not viewed) and handles auto-scaling for traffic spikes. |
| **Data Warehouse** | **Google BigQuery** | Acts as the real-time message broker and storage. Selected for its ability to ingest streaming data and run SQL queries instantly. |
| **CI/CD** | **Google Cloud Build** | Automates the build pipeline. It creates the Docker image from our source code and saves it to the registry. |
| **Containerization** | **Docker** | Ensures environment consistency (Python 3.9, dependencies) between development and production. |

---

## 🐳 Docker Configuration (Production Optimized)

The `Dockerfile` is engineered for a **production-grade** deployment, aiming for a small footprint and security.

### Key Optimizations:
1.  **Base Image**: `python:3.9-slim`. We chose the "slim" variant to keep the image size small (<1GB) vs the full image, leading to faster cold-start times.
2.  **Headless OpenCV**: We explicitly install `opencv-python-headless`. The standard `opencv-python` includes GUI dependencies (Qt, X11) which crash serverless environments.
3.  **Port Mapping**: Cloud Run dynamically assigns a port (injecting it as `$PORT` env var). Our configuration respects this:
    ```dockerfile
    CMD ["streamlit", "run", "web_app/app.py", "--server.port=8080", "--server.address=0.0.0.0"]
    ```

---

## 🚀 Deployment Strategy (CI/CD Pipeline)

We utilized **Google Cloud Build** to act as our CI/CD runner. This prevents "it works on my machine" issues by building the container in the cloud environment itself.

### Step 1: Remote Build
We submit the source code to Cloud Build, which provisions a high-CPU runner to install PyTorch and build the image.

```powershell
gcloud builds submit --tag gcr.io/[PROJECT_ID]/vision-dashboard .
```

### Step 2: Serverless Deployment
We deploy the verified image to Cloud Run as a public service.

```powershell
gcloud run deploy vision-dashboard \
  --image gcr.io/[PROJECT_ID]/vision-dashboard \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```
*   **`--allow-unauthenticated`**: Configuring the service as a public web app.

---

## 🔧 Troubleshooting & Performance Tuning

During the development, we encountered and solved the following critical challenges:

### 1. The "Stale Data" Caching Issue
*   **Symptom**: The dashboard would sometimes freeze or show data from minutes ago, despite the simulation running.
*   **Root Cause**: BigQuery, by default, caches query results for 24 hours to save costs and improve performance. For a real-time app, this is fatal.
*   **Solution**: We explicitly disabled the query cache in our SQL client configuration:
    ```python
    job_config = bigquery.QueryJobConfig(use_query_cache=False)
    client.query(query, job_config=job_config)
    ```

### 2. The "Blinking Dashboard" Loop
*   **Symptom**: The entire UI would flash white every second.
*   **Root Cause**: Calling `st.rerun()` at the top of the script interrupted the rendering process before the UI could even draw.
*   **Solution**: Moved the auto-refresh logic to the **end** of the execution flow, ensuring the charts render fully before the next update cycle begins.

---

## 🟢 Monitoring & Status

*   **Service URL**: [Live Dashboard Link](https://vision-dashboard-1023961337200.us-central1.run.app)
*   **Status Endpoint**: `v1.1 (Live Patch)` (Visible in Sidebar)
*   **Region**: `us-central1` (Iowa)

*Documentation generated by the Vision Speed Smoothing AI Team*
