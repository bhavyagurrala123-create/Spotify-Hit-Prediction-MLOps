"""Turn the raw Spotify tracks dump (data/raw/dataset.csv) into the files the MLOps pipeline expects.

What this script decides (documented in README "Problem framing"):
  * Target  : Hit = "Yes" when popularity >= HIT_THRESHOLD, else "No"   (Telco's "Churn" equivalent)
  * ID      : track_id (unique after de-duplication)
  * Leakage : `popularity` is kept in the raw file for auditing, but is NEVER used as a feature.

Outputs
  data/raw/spotify_tracks.csv   current training dataset  (~ 90% of unique tracks)
  data/staging/new_tracks.csv   "incoming batch" (~ 10%). Copy it to data/raw/new_tracks.csv to trigger
                                ingestion + retraining; it is kept out of data/raw so a first run is clean.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HIT_THRESHOLD = 50
DROP_COLS = ["Unnamed: 0"]

COLUMN_ORDER = [
    "track_id", "artists", "album_name", "track_name", "track_genre", "popularity",
    "duration_ms", "explicit", "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness", "liveness", "valence",
    "tempo", "time_signature", "Hit",
]


def build_clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns]).copy()
    n0 = len(df)

    # the same track is listed once per genre -> keep one row per track_id
    df = df.drop_duplicates(subset="track_id", keep="first")
    n1 = len(df)

    # rows without descriptive text or with zero duration are unusable
    df = df.dropna(subset=["artists", "album_name", "track_name"])
    df = df[df["duration_ms"] > 0]
    n2 = len(df)

    df["explicit"] = df["explicit"].map({True: "Yes", False: "No", "True": "Yes", "False": "No"})
    df["Hit"] = np.where(df["popularity"] >= HIT_THRESHOLD, "Yes", "No")

    print(f"[INFO] rows in file            : {n0}")
    print(f"[INFO] after track_id de-dupe  : {n1}  (removed {n0 - n1})")
    print(f"[INFO] after null/zero cleanup : {n2}  (removed {n1 - n2})")
    print(f"[INFO] Hit rate (popularity>={HIT_THRESHOLD}) : {(df['Hit'] == 'Yes').mean():.3f}")
    return df[COLUMN_ORDER].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/raw/dataset.csv")
    parser.add_argument("--out-dir", default="data/raw")
    parser.add_argument("--staging-dir", default="data/staging")
    parser.add_argument("--new-batch-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = build_clean_frame(pd.read_csv(args.source))

    shuffled = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n_new = int(len(shuffled) * args.new_batch_fraction)
    new_batch, current = shuffled.iloc[:n_new], shuffled.iloc[n_new:]

    out = Path(args.out_dir)
    staging = Path(args.staging_dir)
    out.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True, exist_ok=True)
    current.to_csv(out / "spotify_tracks.csv", index=False)
    new_batch.to_csv(staging / "new_tracks.csv", index=False)
    print(f"[SUCCESS] wrote {out / 'spotify_tracks.csv'} ({len(current)} rows)")
    print(f"[SUCCESS] wrote {staging / 'new_tracks.csv'} ({len(new_batch)} rows)")


if __name__ == "__main__":
    main()
