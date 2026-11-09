import pandas as pd

# Columns that are not model features. Must match src/mlops_pipeline/03_preprocessing.py
NON_FEATURE_COLUMNS = ["track_id", "artists", "album_name", "track_name", "popularity", "Hit"]
INTEGER_CODE_COLUMNS = ["key", "time_signature"]


def clean_prediction_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw track data before prediction (same rules as training)."""
    df = df.copy()

    # identifiers, free text, and anything derived from the target are not features
    df = df.drop(columns=[c for c in NON_FEATURE_COLUMNS if c in df.columns])

    # key / time_signature are categories encoded as integers
    for col in INTEGER_CODE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(int).astype(str)

    return df
