"""Shared, reproducible drift scenarios for the Spotify Hit Prediction monitoring labs.

Feature drift  : the "market" shifts toward louder, more danceable, faster, more explicit
                 tracks and toward a few mainstream genres.
Concept drift  : the relationship between features and the label changes - mainstream genres
                 and very danceable tracks become more likely to be hits.
"""
import numpy as np

POPULAR_GENRES = ["pop", "dance", "hip-hop", "latin", "k-pop", "edm"]


def apply_feature_drift(df, rng):
    df = df.copy()

    if "danceability" in df.columns:
        df["danceability"] = (df["danceability"] * rng.normal(1.10, 0.05, len(df))).clip(0.0, 1.0)
    if "energy" in df.columns:
        df["energy"] = (df["energy"] + rng.normal(0.08, 0.05, len(df))).clip(0.0, 1.0)
    if "loudness" in df.columns:
        df["loudness"] = (df["loudness"] + rng.normal(1.5, 1.0, len(df))).clip(-60.0, 0.0)
    if "tempo" in df.columns:
        df["tempo"] = (df["tempo"] * rng.normal(1.08, 0.05, len(df))).clip(40.0, 250.0)
    if "duration_ms" in df.columns:
        df["duration_ms"] = (df["duration_ms"] * rng.normal(0.92, 0.05, len(df))).clip(30_000, None).round().astype(int)
    if "explicit" in df.columns:
        df["explicit"] = rng.choice(["Yes", "No"], size=len(df), p=[0.30, 0.70])
    if "track_genre" in df.columns:
        move = rng.random(len(df)) < 0.40
        df.loc[move, "track_genre"] = rng.choice(POPULAR_GENRES, size=int(move.sum()))
    return df


def apply_concept_drift(X, y, rng, only_non_hits=False):
    """Flip some labels to 1 (Hit) for mainstream-genre and very danceable tracks."""
    y = np.asarray(y).astype(int).copy()

    def flip(mask, share):
        mask = np.asarray(mask)
        if only_non_hits:
            mask = mask & (y == 0)
        idx = np.where(mask)[0]
        k = int(len(idx) * share)
        if k > 0:
            y[rng.choice(idx, size=k, replace=False)] = 1

    if "track_genre" in X.columns:
        flip(X["track_genre"].isin(POPULAR_GENRES).to_numpy(), 0.25)
    if "danceability" in X.columns:
        flip((X["danceability"] >= X["danceability"].quantile(0.75)).to_numpy(), 0.15)
    return y
