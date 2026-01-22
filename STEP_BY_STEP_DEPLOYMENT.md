# Step-by-Step Deployment Guide: Docker & Google Cloud

This guide provides the exact steps to deploy your **Vision Speed Smoothing AI** application to **Google Cloud Run** using Docker.

## Prerequisites

1.  **Google Cloud Account**: Ensure you have an active GCP account and a project created.
2.  **Google Cloud SDK**: Install the [gcloud CLI](https://cloud.google.com/sdk/docs/install) on your local machine.
3.  **Billing Enabled**: Ensure billing is enabled for your Google Cloud Project.

---

## Phase 1: File Preparation

Before running commands, ensure your project files are set up correctly.

### 1. Verify `Dockerfile`
Ensure your `Dockerfile` in the root directory contains exactly this optimized configuration:

```dockerfile
# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (minimized for production)
# We use opencv-python-headless, so no heavy GL libs needed
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Expose port 8080 (Cloud Run default)
EXPOSE 8080

# Run Streamlit when the container launches
CMD ["streamlit", "run", "web_app/app.py", "--server.port=8080", "--server.address=0.0.0.0"]
```

### 2. Create/Verify `.dockerignore`
Create a file named `.dockerignore` in the root directory to prevent huge/unnecessary files from slowing down your build.

**File: `.dockerignore`**
```text
.git
.venv
__pycache__
results/
runs/
data/
*.mp4
*.env
```

### 3. Verify `requirements.txt`
Ensure `opencv-python-headless` is listed *instead of* `opencv-python` to avoid server crashes.

```text
opencv-python-headless>=4.8.0
ultralytics>=8.0.0
numpy>=1.24.0
# ... rest of your dependencies
```

---

## Phase 2: Google Cloud Setup

Open your terminal (PowerShell or Command Prompt) and run these commands one by one.

### 1. Login to Google Cloud
```powershell
gcloud auth login
```
*A browser window will open. Sign in with your Google account.*

### 2. Set your Project ID
Replace `YOUR_PROJECT_ID` with your actual GCP Project ID (not the name).
```powershell
gcloud config set project YOUR_PROJECT_ID
```

### 3. Enable Required APIs
Enable the services needed for deployment.
```powershell
gcloud services enable cloudbuild.googleapis.com run.googleapis.com containerregistry.googleapis.com
```

---

## Phase 3: Build and Deploy

We will use **Google Cloud Build** to build your Docker container in the cloud. This avoids needing to install Docker locally and ensures it works in the cloud environment.

### 1. Build the Container Image
Run this command from your project root. (Replace `YOUR_PROJECT_ID` again).

```powershell
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/vision-smoothing-app .
```
*Wait for the build to complete. It may take 2-3 minutes. You should see "SUCCESS" at the end.*

### 2. Deploy to Cloud Run
Now, deploy the image you just built.

```powershell
gcloud run deploy vision-smoothing-app `
  --image gcr.io/YOUR_PROJECT_ID/vision-smoothing-app `
  --platform managed `
  --region us-central1 `
  --allow-unauthenticated `
  --memory 2Gi
```
*(Note: The backticks ` allow for multi-line commands in PowerShell. If using CMD, remove them and put everything on one line).*

*   **--allow-unauthenticated**: Makes the web app publicly accessible.
*   **--memory 2Gi**: allocating 2GB RAM (recommended for YOLO/Streamlit).

---

## Phase 4: Post-Deployment Configuration

After the command finishes, it will print a **Service URL** (e.g., `https://vision-smoothing-app-xyz.a.run.app`).

### 1. Set Environment Variables
Your app likely needs credentials (like BigQuery). You shouldn't upload `.env` files. Instead, set them in Cloud Run:

1.  Go to the [Google Cloud Console](https://console.cloud.google.com/).
2.  Navigate to **Cloud Run**.
3.  Click on your service (`vision-smoothing-app`).
4.  Click **Edit & Deploy New Revision**.
5.  Go to the **Variables & Secrets** tab.
6.  Add your variables manually (e.g., `BQ_DATASET_NAME`, `GCP_PROJECT_ID`).
7.  Click **Deploy**.

### 2. Verify
Open the Service URL in your browser. Your Streamlit app should load!

---

## Troubleshooting

*   **"Error: Image not found"**: Ensure the `gcloud builds submit` step finished successfully and you used the exact same image tag in the deploy command.
*   **"App crashes immediately"**: Check the `Logs` tab in the Cloud Run console. It usually indicates a missing environment variable or a `requirements.txt` issue.
