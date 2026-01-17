from google.cloud import bigquery
import datetime
import random
import time

project_id = "hack-4-viksit-bharat"
table_id = f"{project_id}.vision_speed_smoothing_ai.metrics"

print(f"Connecting to {table_id}...")
client = bigquery.Client(project=project_id)

# Generate 100 points of data
rows = []
now = datetime.datetime.now(datetime.timezone.utc)

print("Generating sample data...")
for i in range(100):
    # Create a nice sine wave for speed
    t = float(i)
    base_speed = 15.0 + 5.0 * (random.random() - 0.5)
    
    rows.append({
        "timestamp": (now - datetime.timedelta(seconds=100-i)).isoformat(),
        "relative_time": t,
        "mode": "smoothed" if i > 50 else "reactive",
        "speed": base_speed,
        "acceleration": random.uniform(-2.0, 2.0),
        "brake_pressure": max(0.0, random.uniform(-0.5, 0.5))
    })

print(f"Inserting {len(rows)} rows to BigQuery...")
errors = client.insert_rows_json(table_id, rows)

if not errors:
    print("✅ Success! New rows have been added.")
    print("The Dashboard should now show this data instead of 'Mock Data'.")
else:
    print(f"❌ Encountered errors: {errors}")
