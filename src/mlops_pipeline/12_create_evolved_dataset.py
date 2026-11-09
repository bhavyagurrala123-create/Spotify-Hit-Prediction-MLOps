import numpy as np
import pandas as pd
from pathlib import Path

from spotify_drift import apply_feature_drift

SOURCE_PATH = Path("data/processed/X_test_raw.csv")
OUTPUT_PATH = Path("data/monitoring/drift/evolved_data.csv")
RANDOM_SEED = 42

MODIFIED_FEATURES = ["danceability", "energy", "loudness", "tempo", "duration_ms", "explicit", "track_genre"]


def create_evolved_dataset():
    rng = np.random.default_rng(RANDOM_SEED)
    df = apply_feature_drift(pd.read_csv(SOURCE_PATH), rng)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 60)
    print("EVOLVED DATASET CREATED")
    print("=" * 60)
    print(f"\nSource      : {SOURCE_PATH}")
    print(f"Output      : {OUTPUT_PATH}")
    print(f"Rows        : {len(df)}")
    print(f"Columns     : {len(df.columns)}")
    print("\nModified features:")
    for name in MODIFIED_FEATURES:
        print(f"- {name}")


if __name__ == "__main__":
    create_evolved_dataset()
