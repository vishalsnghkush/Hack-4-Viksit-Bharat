# Google Cloud & BigData Integration Strategy

This document outlines how the **Vision Speed Smoothing AI** integrates with Google Cloud Platform (GCP) for scalable data processing and analytics.

## Architecture Overview

```mermaid
graph TD
    User[React Frontend] -->|WebSocket (Real-time)| CloudRun[FastAPI Backend on Cloud Run]
    CloudRun -->|1. Process Video| AI[Vision AI Logic]
    AI -->|2. Stream Metrics| User
    AI -->|3. Batch Insert| BigQuery[Google BigQuery]
    
    subgraph Google Cloud Platform
        CloudRun
        BigQuery
    end
```

## 1. Google Cloud Run (Compute)
**Why?** Serverless, auto-scaling container hosting.
*   The FastAPI backend is Dockerized and deployed here.
*   It handles the heavy lifting: OpenCV video processing and YOLO inference.
*   **Integration**:
    *   Build Docker image: `gcloud builds submit --tag gcr.io/PROJECT_ID/vision-backend`
    *   Deploy: `gcloud run deploy --image gcr.io/PROJECT_ID/vision-backend`

## 2. Google BigQuery (Big Data)
**Why?** Storing massive amounts of telemetry data for "Historical Analysis".
*   Every simulation run generates thousands of data points (speed, acceleration, risk levels).
*   BigQuery allows us to query this data in milliseconds to generate "Driver Profiles".

### Integration Code Pattern
In `src/gcp_pipeline.py`:
```python
from google.cloud import bigquery

client = bigquery.Client()
table_id = "your-project.vision_ai_dataset.telemetry"

rows_to_insert = [
    {"timestamp": "2023-10-27T10:00:00", "speed": 12.5, "driver": "Alex", "risk": "LOW"},
    # ...
]

errors = client.insert_rows_json(table_id, rows_to_insert)
if errors == []:
    print("New rows have been added.")
else:
    print("Encountered errors while inserting rows: {}".format(errors))
```

## 3. How to Integrate (Step-by-Step)

### Step 1: Create GCP Project
1.  Go to [Google Cloud Console](https://console.cloud.google.com/).
2.  Create a new project (e.g., `vision-speed-ai`).
3.  Enable APIs: **Cloud Run API**, **BigQuery API**, **Container Registry API**.

### Step 2: Set up BigQuery
1.  Go to BigQuery in the console.
2.  Create a Dataset: `vision_ai_data`.
3.  Create a Table `telemetry` with schema:
    *   `timestamp` (TIMESTAMP)
    *   `session_id` (STRING)
    *   `driver_id` (STRING)
    *   `speed` (FLOAT)
    *   `acceleration` (FLOAT)
    *   `brake_pressure` (FLOAT)
    *   `weather_condition` (STRING)

### Step 3: Service Account Authentication
To allow your local app (or Cloud Run) to talk to BigQuery:
1.  Go to **IAM & Admin** > **Service Accounts**.
2.  Create a service account (e.g., `vision-sa`).
3.  Grant Code: `BigQuery Data Editor`.
4.  Create Key (JSON) and save it as `credentials.json` in your project root.
5.  Set env var: `export GOOGLE_APPLICATION_CREDENTIALS="credentials.json"`.

## 4. Frontend Integration
The React Frontend does **not** talk to BigQuery directly (security risk).
*   It requests historical data from the **FastAPI Backend**.
*   The Backend queries BigQuery and returns JSON to the Frontend.

```javascript
// React fetching history
const response = await fetch('https://api-url.com/api/history?driver=Alex');
const data = await response.json();
// Render Recharts
```
