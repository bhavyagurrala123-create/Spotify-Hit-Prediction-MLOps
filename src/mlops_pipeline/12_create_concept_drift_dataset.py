import numpy as np
import pandas as pd
from pathlib import Path

from spotify_drift import apply_concept_drift

FEATURES_PATH = Path("data/monitoring/drift/evolved_data.csv")
LABELS_PATH = Path("data/processed/y_test.npy")
OUTPUT_PATH = Path("data/monitoring/drift/evolved_labeled_data.csv")
RANDOM_SEED = 42
TARGET = "Hit"


def create_concept_drift_dataset():
    rng = np.random.default_rng(RANDOM_SEED)

    X = pd.read_csv(FEATURES_PATH)
    y = np.load(LABELS_PATH).astype(int)

    if len(X) != len(y):
        raise ValueError(f"Feature/label mismatch: X={len(X)}, y={len(y)}")

    y_evolved = apply_concept_drift(X, y, rng)

    result = X.copy()
    result[TARGET] = y_evolved

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print("=" * 70)
    print("CONCEPT DRIFT DATASET CREATED")
    print("=" * 70)
    print(f"Feature source : {FEATURES_PATH}")
    print(f"Label source   : {LABELS_PATH}")
    print(f"Output         : {OUTPUT_PATH}")
    print(f"Rows           : {len(result)}")
    print(f"Columns        : {len(result.columns)}")
    print("\nBaseline label distribution:")
    print(pd.Series(y).value_counts().sort_index())
    print("\nEvolved label distribution:")
    print(pd.Series(y_evolved).value_counts().sort_index())
    print(f"\nChanged labels : {(y != y_evolved).sum()}")


if __name__ == "__main__":
    create_concept_drift_dataset()
