"""Train the Spotify Hit model WITHOUT MLflow (no tracking, no registry).

Same model and hyper-parameters as src/mlops_pipeline/05_train_registry.py, written to
paths.model so stages 06 (evaluation) and 07 (quality gate) can run on a machine that has
only scikit-learn installed.  Use 05_train_registry.py for the full MLflow workflow.
"""
import os
import sys

import joblib
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    processed_dir = config["data"]["processed_dir"]
    try:
        X_train = np.load(os.path.join(processed_dir, "X_train_final.npy"))
        y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    except FileNotFoundError:
        print("[ERROR] Processed data not found. Run 03_preprocessing.py first.")
        return False

    model_cfg = config["model"]
    params = {k: model_cfg[k] for k in ("n_estimators", "max_depth", "min_samples_leaf",
                                         "random_state", "class_weight", "n_jobs") if k in model_cfg}

    print(f"[INFO] Training RandomForest on {X_train.shape[0]} tracks, {X_train.shape[1]} features...")
    model = RandomForestClassifier(**params).fit(X_train, y_train)

    os.makedirs(os.path.dirname(config["paths"]["model"]), exist_ok=True)
    joblib.dump(model, config["paths"]["model"])
    print(f"[SUCCESS] Model saved to {config['paths']['model']}")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
