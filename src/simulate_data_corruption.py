import pandas as pd

from src.validate_data import validate_schema


def simulate_corrupted_data(df):
    """Simulates real-world data corruption scenarios across multiple columns."""
    print("[INFO] Injecting simulated faults into data...")
    corrupted_df = df.copy()

    # 1. Invalid numerical value (negative duration)
    corrupted_df.loc[0, "duration_ms"] = -5

    # 2. Out-of-range audio feature (danceability must be in [0, 1])
    corrupted_df.loc[1, "danceability"] = 1.8

    # 3. Invalid target label
    corrupted_df.loc[2, "Hit"] = "Maybe"

    # 4. Datatype mismatch (string injected into a float column)
    corrupted_df["tempo"] = corrupted_df["tempo"].astype(object)
    corrupted_df.loc[3, "tempo"] = "One Hundred"

    # 5. Schema inconsistency (unexpected extra column)
    corrupted_df["unexpected_column"] = 123

    # 6. Missing required attribute (dropping a column)
    corrupted_df = corrupted_df.drop(columns=["track_genre"])

    return corrupted_df


if __name__ == "__main__":
    data_path = "data/raw/spotify_tracks.csv"

    try:
        raw_df = pd.read_csv(data_path)
        corrupted_df = simulate_corrupted_data(raw_df)

        print("\n--- Running Validation on Corrupted Data ---")
        validate_schema(corrupted_df, output_report_name="corrupted_simulation_errors.csv")

    except FileNotFoundError:
        print(f"[ERROR] Could not find data file at {data_path}.")
