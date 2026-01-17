# Deployment Guide for Vision Speed AI

You have two primary options for deploying your Streamlit dashboard.

---

## Option 1: Streamlit Community Cloud (Easiest)
Best for sharing demos quickly and for free.

### Steps
1.  **Push to GitHub**:
    *   Ensure your project is in a GitHub repository.
    *   Make sure `requirements.txt` is in the root directory.
    *   **Crucial**: Ensure `packages.txt` exists if you need system packages (like `libgl1` for OpenCV), though `opencv-python-headless` usually works without it on Streamlit Cloud.

2.  **Deploy**:
    *   Go to [share.streamlit.io](https://share.streamlit.io/).
    *   Click **"New app"**.
    *   Select your Repository, Branch (`main`), and Main File Path (`web_app/app.py`).
    *   Click **"Deploy"**.

3.  **Secrets (env vars)**:
    *   Go to your app dashboard -> Settings -> Secrets.
    *   Paste the content of your `.env` file there (e.g., `GCP_PROJECT_ID`, `BQ_DATASET_NAME`).
    *   For the Service Account JSON, paste the *entire content* of the JSON file into a secret named `GCP_TYPE_SERVICE_ACCOUNT` or similar, and update your code to read from `st.secrets`.

---

## Option 2: Google Cloud Run (Robust & Scalable)
Best for production apps that need custom environments or powerful compute.

### 1. Create a Dockerfile
Create a file named `Dockerfile` in the root of your project:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose Streamlit port
EXPOSE 8080

# Run Streamlit
CMD ["streamlit", "run", "web_app/app.py", "--server.port=8080", "--server.address=0.0.0.0"]
```

### 2. Build & Deploy
Run these commands in your terminal (ensure `gcloud` CLI is installed):

```bash
# 1. Set Project
gcloud config set project YOUR_PROJECT_ID

# 2. Build Image (Cloud Build)
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/vision-dashboard

# 3. Deploy to Cloud Run
gcloud run deploy vision-dashboard \
  --image gcr.io/YOUR_PROJECT_ID/vision-dashboard \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### 3. Environment Variables
When deploying to Cloud Run, you can set environment variables via the console or CLI flags:
`--set-env-vars GCP_PROJECT_ID=...,BQ_DATASET_NAME=...`

---

## Recommended Choice
*   **Use Option 1 (Streamlit Cloud)** if you just want to share a link in 5 minutes.
*   **Use Option 2 (Cloud Run)** if you need the app to be part of your existing Google Cloud infrastructure (BigQuery access is faster/secure internally).
