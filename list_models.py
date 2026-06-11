"""
Run this to see all models available in your Tinker account.
Find your 20B model and copy its ID into MODEL_ID in train.py.

Usage:
    python list_models.py
"""

import tinker

service_client = tinker.ServiceClient()  # reads TINKER_API_KEY from env

models = service_client.get_models()

print("\nAvailable Tinker models:\n")
for m in models:
    print(f"  ID: {m.tinker_id}")
    print(f"  Name: {m.name}")
    print(f"  Size: {m.size}")
    print()
