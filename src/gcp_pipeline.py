"""
GCP Pipeline Script
Runs the Vision Speed Smoothing System, extracts metrics, uploads video/csv to GCS,
and loads data into BigQuery.
"""
import os
import sys
import csv
import json
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

import pandas as pd
from google.cloud import storage
from google.cloud import bigquery

# Add current directory to path to import local modules
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from main import VisionSpeedSmoothingSystem
from metrics import OverallMetrics

# Load environment variables
load_dotenv()

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_CREDENTIALS_PATH = os.getenv("GCP_CREDENTIALS_PATH")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
BQ_DATASET_NAME = os.getenv("BQ_DATASET_NAME")
BQ_TABLE_NAME = os.getenv("BQ_TABLE_NAME", "metrics")


def setup_gcp_clients():
    """Initialize GCS and BigQuery clients."""
    if not GCP_PROJECT_ID:
        print("WARNING: GCP_PROJECT_ID not found in .env. GCP upload will be skipped.")
        return None, None

    try:
        # If credentials path is provided, use it. Otherwise, fallback to ADC.
        if GCP_CREDENTIALS_PATH and os.path.exists(GCP_CREDENTIALS_PATH):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_CREDENTIALS_PATH
            print(f"Using Service Account Key from: {GCP_CREDENTIALS_PATH}")
        else:
            print("No Service Account Key found. Using Application Default Credentials (gcloud auth)...")
            # Unset env var if it might be set to an invalid path
            if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
                del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

        storage_client = storage.Client(project=GCP_PROJECT_ID)
        bq_client = bigquery.Client(project=GCP_PROJECT_ID)
        print("GCP Clients initialized successfully.")
        return storage_client, bq_client
    except Exception as e:
        print(f"Error initializing GCP clients: {e}")
        return None, None


def upload_to_gcs(storage_client, bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    if not storage_client:
        return
    
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        print(f"File {source_file_name} uploaded to gs://{bucket_name}/{destination_blob_name}.")
        return f"gs://{bucket_name}/{destination_blob_name}"
    except Exception as e:
        print(f"Error uploading to GCS: {e}")
        return None


def upload_to_bigquery(bq_client, dataset_id, table_id, csv_file_path):
    """Loads a CSV file into BigQuery."""
    if not bq_client:
        return

    try:
        dataset_ref = bq_client.dataset(dataset_id)
        table_ref = dataset_ref.table(table_id)
        
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.CSV,
            skip_leading_rows=1,
            autodetect=True,
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        )

        with open(csv_file_path, "rb") as source_file:
            job = bq_client.load_table_from_file(source_file, table_ref, job_config=job_config)

        job.result()  # Waits for the job to complete.
        print(f"Loaded {job.output_rows} rows into {dataset_id}.{table_id}.")
        
    except Exception as e:
        print(f"Error uploading to BigQuery: {e}")


def stream_to_bigquery(bq_client, dataset_id, table_id, row_data):
    """
    Streams a list of dictionary rows directly to BigQuery.
    Use this for real-time updates.
    """
    if not bq_client:
        return False

    try:
        table_ref = f"{bq_client.project}.{dataset_id}.{table_id}"
        errors = bq_client.insert_rows_json(table_ref, row_data)
        
        if errors:
            print(f"Encountered errors while streaming: {errors}")
            return False
            
        return True
    except Exception as e:
        print(f"Error streaming to BigQuery: {e}")
        return False


def extract_metrics_csv(metrics_obj: OverallMetrics, run_type: str, run_id: str, output_path: str):
    """
    Extracts time-series metrics from MetricsCollector object and saves to CSV.
    Structure: run_id, timestamp, run_type, speed, acceleration, brake_pressure
    """
    
    # We need to access the raw history from the collector which might not be directly in OverallMetrics
    # But OverallMetrics is a dataclass summary. We need the history data.
    # The 'run' method in main.py returns MetricsCollector (or OverallMetrics? let's check demo.py)
    # demo.py says: metrics_reactive = system.run() 
    # and metrics_reactive is an OverallMetrics object.
    # Wait, looking at metrics.py: SpeedMetrics, AccelerationMetrics are summaries.
    # The HISTORY is in the MetricsCollector instance, not the OverallMetrics return value.
    # We need to modify return of run() or access the system.metrics object directly.
    pass 

def process_and_upload(video_path: str):
    """
    Main processing pipeline.
    """
    
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting Pipeline Run ID: {run_id}")
    
    # Initialize GCP
    storage_client, bq_client = setup_gcp_clients()
    
    # 1. Run Baseline
    print("Running Baseline (Reactive Mode)...")
    system_reactive = VisionSpeedSmoothingSystem(
        video_path=video_path,
        enable_gps_degradation=False,
        enable_smoothing=False
    )
    _ = system_reactive.run(display=False) # We don't need the summary return for CSV
    
    # 2. Run Smoothed
    print("Running Optimized (Smoothed Mode)...")
    system_smooth = VisionSpeedSmoothingSystem(
        video_path=video_path,
        enable_gps_degradation=True,
        enable_smoothing=True
    )
    _ = system_smooth.run(display=False)
    
    # 3. Export CSV Data
    csv_filename = f"metrics_{run_id}.csv"
    csv_path = output_dir / csv_filename
    
    print(f"Exporting metrics to {csv_path}...")
    
    # Aggregate data
    # Timestamps might differ slightly, so we stack them.
    
    data_rows = []
    
    # Reactive Data
    r_timestamps = list(system_reactive.metrics.timestamps)
    r_speeds = list(system_reactive.metrics.speed_history)
    r_accels = list(system_reactive.metrics.acceleration_history)
    r_brakes = list(system_reactive.metrics.brake_pressure_history)
    
    min_len_r = min(len(r_timestamps), len(r_speeds), len(r_accels), len(r_brakes))
    
    for i in range(min_len_r):
        data_rows.append({
            "run_id": run_id,
            "timestamp": r_timestamps[i],
            "relative_time": r_timestamps[i] - r_timestamps[0],
            "mode": "reactive",
            "speed": r_speeds[i],
            "acceleration": r_accels[i],
            "brake_pressure": r_brakes[i]
        })
        
    # Smoothed Data
    s_timestamps = list(system_smooth.metrics.timestamps)
    s_speeds = list(system_smooth.metrics.speed_history)
    s_accels = list(system_smooth.metrics.acceleration_history)
    s_brakes = list(system_smooth.metrics.brake_pressure_history)
    
    min_len_s = min(len(s_timestamps), len(s_speeds), len(s_accels), len(s_brakes))
    
    for i in range(min_len_s):
         data_rows.append({
            "run_id": run_id,
            "timestamp": s_timestamps[i],
            "relative_time": s_timestamps[i] - s_timestamps[0],
            "mode": "smoothed",
            "speed": s_speeds[i],
            "acceleration": s_accels[i],
            "brake_pressure": s_brakes[i]
        })
    
    df = pd.DataFrame(data_rows)
    df.to_csv(csv_path, index=False)
    print("CSV Export Complete.")
    
    # 4. Upload to GCP
    if storage_client and GCS_BUCKET_NAME:
        print("Uploading to GCS...")
        # Upload CSV
        upload_to_gcs(storage_client, GCS_BUCKET_NAME, str(csv_path), f"data/{run_id}/{csv_filename}")
        # Upload Video (optional, might be large)
        # upload_to_gcs(storage_client, GCS_BUCKET_NAME, video_path, f"data/{run_id}/source_video.mp4")
        
    if bq_client and BQ_DATASET_NAME:
        print("Uploading to BigQuery...")
        upload_to_bigquery(bq_client, BQ_DATASET_NAME, BQ_TABLE_NAME, str(csv_path))
        
    print("Pipeline Complete!")
    return str(csv_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, help="Path to video file")
    args = parser.parse_args()
    
    # Default video logic
    if not args.video:
        # Try to find default video
        root = Path(__file__).parent.parent
        default_video = root / "data" / "8359-208052066_small.mp4"
        if default_video.exists():
            video_path = str(default_video)
        else:
            print("No video found. Please provide --video")
            sys.exit(1)
    else:
        video_path = args.video
        
    process_and_upload(video_path)
