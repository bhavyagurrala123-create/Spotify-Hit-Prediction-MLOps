import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
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
    print("[INFO] Starting Sklearn Pipeline Preprocessing...")

    X, y = load_features_and_target("data/raw/spotify_tracks.csv")

    # Split before any transformation to prevent leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    cat_cols = X_train.select_dtypes(include=CAT_DTYPES).columns.tolist()
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("scaler", StandardScaler()),
    ])
    cat_pipeline = Pipeline(steps=[
        ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols),
    ])

    print("[INFO] Fitting and transforming training data...")
    X_train_final = preprocessor.fit_transform(X_train)
    print("[INFO] Transforming test data...")
    X_test_final = preprocessor.transform(X_test)

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    np.save("data/processed/X_train_final.npy", X_train_final)
    np.save("data/processed/X_test_final.npy", X_test_final)
    np.save("data/processed/y_train.npy", y_train.to_numpy(dtype=np.int64))
    np.save("data/processed/y_test.npy", y_test.to_numpy(dtype=np.int64))
    X_test.to_csv("data/processed/X_test_raw.csv", index=False)   # same row order as X_test_final (used for error analysis)
    joblib.dump(preprocessor, "models/preprocessor.pkl")

    metadata = {
        "dataset_name": "Spotify Hit Prediction",
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
    }
    with open("data/processed/dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("[SUCCESS] Preprocessing completed successfully! Pipeline saved as preprocessor.pkl.")


if __name__ == "__main__":
    run_preprocessing()
