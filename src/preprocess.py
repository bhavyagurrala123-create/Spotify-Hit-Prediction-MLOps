import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

TEXT_AND_LEAKAGE = ["track_id", "artists", "album_name", "track_name", "popularity"]
CAT_DTYPES = ["object", "category", "string"]


def load_features_and_target(path):
    """Read the raw Spotify CSV -> (X, y). `popularity` is dropped: the target is derived from it."""
    df = pd.read_csv(path)
    y = df["Hit"].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0).astype(int)
    X = df.drop(columns=[c for c in TEXT_AND_LEAKAGE + ["Hit"] if c in df.columns])
    for col in ("key", "time_signature"):      # integer codes, not quantities
        X[col] = X[col].astype(int).astype(str)
    return X, y


def run_preprocessing():
    print("Starting Preprocessing Pipeline...")

    # 1. Load + clean (ids, text and the popularity leakage column removed, target -> 0/1)
    X, y = load_features_and_target("data/raw/spotify_tracks.csv")

    # 2. Train-Test Split (before any fitting, to prevent leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Separate column types
    cat_cols = X_train.select_dtypes(include=CAT_DTYPES).columns
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns

    # 4. Scale & encode
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(X_train[num_cols])
    x_test_scaled = scaler.transform(X_test[num_cols])

    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    x_train_encoded = ohe.fit_transform(X_train[cat_cols])
    x_test_encoded = ohe.transform(X_test[cat_cols])

    X_train_final = np.hstack((x_train_scaled, x_train_encoded))
    X_test_final = np.hstack((x_test_scaled, x_test_encoded))

    # 5. Save artifacts
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    np.save("data/processed/X_train_final.npy", X_train_final)
    np.save("data/processed/X_test_final.npy", X_test_final)
    np.save("data/processed/y_train.npy", y_train.to_numpy(dtype=np.int64))
    np.save("data/processed/y_test.npy", y_test.to_numpy(dtype=np.int64))
    X_test.to_csv("data/processed/X_test_raw.csv", index=False)   # same row order as X_test_final (used for error analysis)

    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(ohe, "models/ohe.pkl")

    metadata = {
        "dataset_name": "Spotify Hit Prediction",
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "numerical_features": list(num_cols),
        "categorical_features": list(cat_cols),
    }
    with open("data/processed/dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("Preprocessing completed successfully!")


if __name__ == "__main__":
    run_preprocessing()
