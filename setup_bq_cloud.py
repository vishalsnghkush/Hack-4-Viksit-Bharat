from google.cloud import bigquery
import os

# Configuration matching your new project
project_id = "hack-4-viksit-bharat"
dataset_id = "vision_speed_smoothing_ai"
table_id = "metrics"

print(f"Initializing BigQuery for project: {project_id}")

try:
    # Initialize client
    client = bigquery.Client(project=project_id)

    # 1. Create Dataset
    dataset_ref = f"{project_id}.{dataset_id}"
    try:
        client.get_dataset(dataset_ref)
        print(f"✅ Dataset '{dataset_id}' already exists.")
    except Exception:
        print(f"⚠️ Dataset '{dataset_id}' not found. Creating...")
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"
        client.create_dataset(dataset, timeout=30)
        print(f"✅ Created dataset '{dataset_id}'.")

    # 2. Create Table
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    try:
        client.get_table(table_ref)
        print(f"✅ Table '{table_id}' already exists.")
    except Exception:
        print(f"⚠️ Table '{table_id}' not found. Creating...")
        schema = [
            bigquery.SchemaField("timestamp", "TIMESTAMP"),
            bigquery.SchemaField("relative_time", "FLOAT"),
            bigquery.SchemaField("mode", "STRING"),
            bigquery.SchemaField("speed", "FLOAT"),
            bigquery.SchemaField("acceleration", "FLOAT"),
            bigquery.SchemaField("brake_pressure", "FLOAT"),
        ]
        table = bigquery.Table(table_ref, schema=schema)
        client.create_table(table, timeout=30)
        print(f"✅ Created table '{table_id}'.")

    print("\n🎉 SUCCESS: Database setup complete. The 404 error should disappear.")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("Tip: You might need to authenticate. Run: 'gcloud auth application-default login'")
