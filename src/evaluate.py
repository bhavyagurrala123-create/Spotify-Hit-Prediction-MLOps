import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def run_evaluation(model_path="models/random_forest_baseline.pkl"):
    print("Starting Model Evaluation...")

    X_test_final = np.load("data/processed/X_test_final.npy")
    y_test = np.load("data/processed/y_test.npy")
    model = joblib.load(model_path)

    y_pred = model.predict(X_test_final)

    print("\n--- Model Evaluation Report ---")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score : {f1_score(y_test, y_pred):.4f}")
    print("-------------------------------\n")

    # Error analysis: rows of the fixed test set, in the same order as X_test_final
    errors_df = pd.read_csv("data/processed/X_test_raw.csv")
    errors_df["Actual_Hit"] = y_test
    errors_df["Predicted_Hit"] = y_pred

    os.makedirs("outputs", exist_ok=True)
    missed_hits = errors_df[(errors_df["Actual_Hit"] == 1) & (errors_df["Predicted_Hit"] == 0)]
    false_alarms = errors_df[(errors_df["Actual_Hit"] == 0) & (errors_df["Predicted_Hit"] == 1)]
    missed_hits.to_csv("outputs/false_negatives.csv", index=False)
    false_alarms.to_csv("outputs/false_positives.csv", index=False)
    print(f"Missed hits (false negatives): {len(missed_hits)}")
    print(f"False alarms (false positives): {len(false_alarms)}")


if __name__ == "__main__":
    run_evaluation()
