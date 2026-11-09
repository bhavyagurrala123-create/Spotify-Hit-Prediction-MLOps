import numpy as np
import pandas as pd
from pathlib import Path

from spotify_drift import apply_concept_drift, apply_feature_drift


TRAIN_FEATURES_PATH = Path("data/processed/X_train_raw.csv")
TRAIN_LABELS_PATH = Path("data/processed/y_train.npy")
TEST_FEATURES_PATH = Path("data/processed/X_test_raw.csv")

NEW_BATCH_PATH = Path("data/monitoring/drift/post_drift_training_data.csv")
UPDATED_DATASET_PATH = Path("data/processed/updated_training_data.csv")
SUMMARY_PATH = Path("outputs/retraining/updated_dataset_summary.csv")

RANDOM_SEED = 42
NEW_BATCH_SIZE = 3000


def load_original_training_data():
    X_train = pd.read_csv(TRAIN_FEATURES_PATH)
    y_train = np.load(TRAIN_LABELS_PATH).astype(int)
    X_test = pd.read_csv(TEST_FEATURES_PATH)

    if len(X_train) != len(y_train):
        raise ValueError(
            f"Training feature/label mismatch: X={len(X_train)}, y={len(y_train)}"
        )

    return X_train, y_train, X_test


def create_post_drift_batch(X_train, y_train):
    rng = np.random.default_rng(RANDOM_SEED)

    sample_indices = rng.choice(len(X_train), size=NEW_BATCH_SIZE, replace=True)

    X_new = X_train.iloc[sample_indices].copy().reset_index(drop=True)
    y_new = y_train[sample_indices].copy()

    X_new = apply_feature_drift(X_new, rng)
    y_new = apply_concept_drift(X_new, y_new, rng, only_non_hits=True)

    return X_new, y_new


def validate_new_batch(X_new, y_new, X_test):
    expected_columns = list(X_test.columns)

    if list(X_new.columns) != expected_columns:
        raise ValueError(
            "Feature schema mismatch between new batch and test feature schema."
        )

    if len(X_new) != len(y_new):
        raise ValueError("New batch feature/label count mismatch.")

    if X_new.isnull().all().any():
        raise ValueError("One or more columns contain only null values.")

    for col in ("danceability", "energy"):
        if col in X_new.columns and ((X_new[col] < 0) | (X_new[col] > 1)).any():
            raise ValueError(f"{col} contains values outside [0, 1].")

    if "tempo" in X_new.columns and ((X_new["tempo"] < 0) | (X_new["tempo"] > 300)).any():
        raise ValueError("tempo contains values outside [0, 300].")

    # Check that the post-drift batch is not an exact copy of the held-out test rows.
    test_hashes = set(
        pd.util.hash_pandas_object(X_test, index=False).astype("uint64")
    )

    new_hashes = pd.util.hash_pandas_object(
        X_new,
        index=False,
    ).astype("uint64")

    exact_overlap = sum(value in test_hashes for value in new_hashes)

    if exact_overlap > 0:
        raise ValueError(
            f"Detected {exact_overlap} exact feature-row overlaps with the test set."
        )

    return exact_overlap


def create_updated_dataset(X_train, y_train, X_new, y_new):
    original_train = X_train.copy()
    original_train["Hit"] = y_train

    new_batch = X_new.copy()
    new_batch["Hit"] = y_new

    updated = pd.concat(
        [original_train, new_batch],
        ignore_index=True,
    )

    return updated, original_train, new_batch


def save_summary(original_train, new_batch, updated):
    rows = [
        {
            "dataset": "original_training",
            "rows": len(original_train),
            "hits": int((original_train["Hit"] == 1).sum()),
            "non_hit": int((original_train["Hit"] == 0).sum()),
            "hit_rate": round(float(original_train["Hit"].mean()), 6),
        },
        {
            "dataset": "post_drift_batch",
            "rows": len(new_batch),
            "hits": int((new_batch["Hit"] == 1).sum()),
            "non_hit": int((new_batch["Hit"] == 0).sum()),
            "hit_rate": round(float(new_batch["Hit"].mean()), 6),
        },
        {
            "dataset": "updated_training",
            "rows": len(updated),
            "hits": int((updated["Hit"] == 1).sum()),
            "non_hit": int((updated["Hit"] == 0).sum()),
            "hit_rate": round(float(updated["Hit"].mean()), 6),
        },
    ]

    summary = pd.DataFrame(rows)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)

    return summary


def main():
    print("=" * 70)
    print("UPDATED TRAINING DATASET CREATION")
    print("=" * 70)

    X_train, y_train, X_test = load_original_training_data()

    print()
    print("ORIGINAL DATA")
    print("-" * 70)
    print(f"Training features : {X_train.shape}")
    print(f"Training labels   : {y_train.shape}")
    print(f"Test features     : {X_test.shape}")

    X_new, y_new = create_post_drift_batch(
        X_train,
        y_train,
    )

    exact_overlap = validate_new_batch(
        X_new,
        y_new,
        X_test,
    )

    print()
    print("POST-DRIFT BATCH")
    print("-" * 70)
    print(f"Rows              : {len(X_new)}")
    print(f"Columns           : {len(X_new.columns)}")
    print(f"Hits              : {(y_new == 1).sum()}")
    print(f"Non-hits          : {(y_new == 0).sum()}")
    print(f"Test-set overlap  : {exact_overlap}")

    updated, original_train, new_batch = create_updated_dataset(
        X_train,
        y_train,
        X_new,
        y_new,
    )

    NEW_BATCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    UPDATED_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

    new_batch.to_csv(
        NEW_BATCH_PATH,
        index=False,
    )

    updated.to_csv(
        UPDATED_DATASET_PATH,
        index=False,
    )

    summary = save_summary(
        original_train,
        new_batch,
        updated,
    )

    print()
    print("UPDATED TRAINING DATASET")
    print("-" * 70)
    print(f"Rows              : {len(updated)}")
    print(f"Columns           : {len(updated.columns)}")

    print()
    print(summary.to_string(index=False))

    print()
    print("=" * 70)
    print("FILES CREATED")
    print("=" * 70)
    print(f"Post-drift batch  : {NEW_BATCH_PATH}")
    print(f"Updated dataset   : {UPDATED_DATASET_PATH}")
    print(f"Summary           : {SUMMARY_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()