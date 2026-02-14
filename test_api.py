import requests
import json
import random
import time

url = "http://localhost:8000/predict"

# Generate 30 seconds of dummy data (1Hz)
data = []
start_time = time.time()

for i in range(30):
    point = {
        "timestamp": start_time + i,
        "heart_rate": 80 + random.uniform(-2, 2),
        "spo2": 98 + random.uniform(-0.5, 0.5),
        "bp_systolic": 120 + random.uniform(-5, 5),
        "bp_diastolic": 80 + random.uniform(-3, 3),
        "vibration": 0.02 + random.uniform(0, 0.01) # Low vibration
    }
    data.append(point)

# Inject a drop (simulated deterioration) in the last few seconds
for i in range(25, 30):
    data[i]["heart_rate"] = 110 + random.uniform(-2, 2) # Rising HR
    data[i]["spo2"] = 92 + random.uniform(-0.5, 0.5)    # Dropping SpO2

print(f"Sending {len(data)} data points to {url}...")

try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
