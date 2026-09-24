"""Test the training endpoint directly"""
import sys
import os
sys.path.insert(0, 'workbench')
os.chdir('/run/media/teerametr/3b44566d-03e0-4d76-8469-1bf9cf418a63/Project/Fine-tuning')

from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

# Test preflight
print("Testing preflight...")
resp = client.post("/training/preflight", json={"config": {"base_model": "NousResearch/Hermes-2-Pro-Llama-3-8B", "dataset_version": "v0001"}})
print(f"Preflight status: {resp.status_code}")
print(f"Preflight response: {resp.json()}")

# Test training runs with preflight
print("\nTesting training runs...")
preflight_report = resp.json()
config = {"base_model": "NousResearch/Hermes-2-Pro-Llama-3-8B", "dataset_version": "v0001", "_preflight": preflight_report}
resp2 = client.post("/training/runs", json={"config": config})
print(f"Training runs status: {resp2.status_code}")
print(f"Training runs response: {resp2.json()}")