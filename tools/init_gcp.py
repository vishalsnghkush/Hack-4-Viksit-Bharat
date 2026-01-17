import os
from google.cloud import storage
from google.cloud import bigquery
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
DATASET_NAME = os.getenv("BQ_DATASET_NAME")

print(f"Initializing Resources for Project: {PROJECT_ID}")

def create_bucket():
    storage_client = storage.Client(project=PROJECT_ID)
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        if not bucket.exists():
            storage_client.create_bucket(bucket, location="US")
            print(f"Created Bucket: {BUCKET_NAME}")
        else:
            print(f"Bucket {BUCKET_NAME} already exists.")
    except Exception as e:
        print(f"Error creating bucket: {e}")

def create_dataset():
    bq_client = bigquery.Client(project=PROJECT_ID)
    dataset_id = f"{PROJECT_ID}.{DATASET_NAME}"
    try:
        bq_client.get_dataset(dataset_id)
        print(f"Dataset {dataset_id} already exists.")
    except Exception:
        print(f"Dataset {dataset_id} not found. Creating...")
        dataset = bigquery.Dataset(dataset_id)
        dataset.location = "US"
        try:
            bq_client.create_dataset(dataset, timeout=30)
            print(f"Created Dataset: {dataset_id}")
        except Exception as e:
             print(f"Error creating dataset: {e}")


def create_table():
    bq_client = bigquery.Client(project=PROJECT_ID)
    table_id = f"{PROJECT_ID}.{DATASET_NAME}.metrics"
    
    schema = [
        bigquery.SchemaField("run_id", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("timestamp", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("relative_time", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("mode", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("speed", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("acceleration", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("brake_pressure", "FLOAT", mode="NULLABLE"),
    ]

    try:
        bq_client.get_table(table_id)
        print(f"Table {table_id} already exists.")
    except Exception:
        print(f"Table {table_id} not found. Creating...")
        table = bigquery.Table(table_id, schema=schema)
        try:
            bq_client.create_table(table)
            print(f"Created Table: {table_id}")
        except Exception as e:
            print(f"Error creating table: {e}")

if __name__ == "__main__":
    create_bucket()
    create_dataset()
    create_table()
