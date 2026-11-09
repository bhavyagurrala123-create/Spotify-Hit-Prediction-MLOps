"""Lab 3 entry point: validate data/raw/spotify_tracks.csv against the strict schema.

The schema itself lives in src/mlops_pipeline/validation.py (single source of truth).
Run from the project root:   python -m src.validate_data
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mlops_pipeline"))
from validation import get_spotify_schema, validate_schema  # noqa: E402,F401

if __name__ == "__main__":
    data_path = "data/raw/spotify_tracks.csv"
    if os.path.exists(data_path):
        validate_schema(pd.read_csv(data_path), output_report_name="baseline_validation.csv")
    else:
        print(f"[ERROR] Target file not found at: {data_path}")
