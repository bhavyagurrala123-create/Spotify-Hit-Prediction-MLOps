"""Schema validation for the Spotify Hit Prediction dataset.

One rule table (SCHEMA_RULES) drives two interchangeable validators:
  * pandera  - used when the package is installed (it is listed in requirements.txt)
  * pandas   - built-in fallback so validation still works in a minimal environment
Both write the same failure report (artifacts/<name>.csv).
"""
import os

import pandas as pd

try:  # pragma: no cover - depends on the environment
    import pandera.pandas as pa
    from pandera import Check, Column

    HAVE_PANDERA = True
except ImportError:  # pragma: no cover
    HAVE_PANDERA = False

YES_NO = ["Yes", "No"]

# kind: str | int | float     (nullable defaults to False)
SCHEMA_RULES = {
    "track_id": {"kind": "str"},
    "artists": {"kind": "str"},
    "album_name": {"kind": "str"},
    "track_name": {"kind": "str"},
    "track_genre": {"kind": "str"},
    "popularity": {"kind": "int", "ge": 0, "le": 100},
    "duration_ms": {"kind": "int", "gt": 0},
    "explicit": {"kind": "str", "isin": YES_NO},
    "danceability": {"kind": "float", "ge": 0.0, "le": 1.0},
    "energy": {"kind": "float", "ge": 0.0, "le": 1.0},
    "key": {"kind": "int", "ge": 0, "le": 11},
    "loudness": {"kind": "float", "ge": -60.0, "le": 5.0},
    "mode": {"kind": "int", "isin": [0, 1]},
    "speechiness": {"kind": "float", "ge": 0.0, "le": 1.0},
    "acousticness": {"kind": "float", "ge": 0.0, "le": 1.0},
    "instrumentalness": {"kind": "float", "ge": 0.0, "le": 1.0},
    "liveness": {"kind": "float", "ge": 0.0, "le": 1.0},
    "valence": {"kind": "float", "ge": 0.0, "le": 1.0},
    "tempo": {"kind": "float", "ge": 0.0, "le": 300.0},
    "time_signature": {"kind": "int", "ge": 0, "le": 7},
    "Hit": {"kind": "str", "isin": YES_NO},
}


# --------------------------------------------------------------------------- pandera
def get_spotify_schema():
    if not HAVE_PANDERA:
        raise RuntimeError("pandera is not installed")

    dtypes = {"str": pa.String, "int": pa.Int, "float": pa.Float}
    columns = {}
    for name, rule in SCHEMA_RULES.items():
        checks = []
        if "isin" in rule:
            checks.append(Check.isin(rule["isin"]))
        if "gt" in rule:
            checks.append(Check.gt(rule["gt"]))
        if "ge" in rule:
            checks.append(Check.ge(rule["ge"]))
        if "le" in rule:
            checks.append(Check.le(rule["le"]))
        columns[name] = Column(dtypes[rule["kind"]], checks or None, nullable=False)
    return pa.DataFrameSchema(columns, strict=True)


def _validate_pandera(df):
    schema = get_spotify_schema()
    try:
        schema.validate(df, lazy=True)
        return pd.DataFrame()
    except pa.errors.SchemaErrors as err:
        return err.failure_cases[["schema_context", "column", "check", "failure_case", "index"]]


# --------------------------------------------------------------------------- pandas fallback
_KIND_CHECK = {
    "str": lambda s: pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s),
    "int": pd.api.types.is_integer_dtype,
    "float": pd.api.types.is_float_dtype,
}


def _validate_pandas(df):
    rows = []

    def add(context, column, check, case, index):
        rows.append({"schema_context": context, "column": column, "check": check,
                     "failure_case": case, "index": index})

    expected = list(SCHEMA_RULES)
    for col in expected:
        if col not in df.columns:
            add("DataFrameSchema", col, "column_in_dataframe", col, None)
    for col in df.columns:
        if col not in SCHEMA_RULES:
            add("DataFrameSchema", col, "column_in_schema (strict)", col, None)

    for col, rule in SCHEMA_RULES.items():
        if col not in df.columns:
            continue
        s = df[col]
        if not _KIND_CHECK[rule["kind"]](s):
            add("Column", col, f"dtype('{rule['kind']}')", str(s.dtype), None)
            continue
        for idx in s.index[s.isna()]:
            add("Column", col, "not_nullable", None, idx)
        valid = s.dropna()
        masks = []
        if "isin" in rule:
            masks.append((f"isin({rule['isin']})", ~valid.isin(rule["isin"])))
        if "gt" in rule:
            masks.append((f"greater_than({rule['gt']})", ~(valid > rule["gt"])))
        if "ge" in rule:
            masks.append((f"greater_than_or_equal_to({rule['ge']})", ~(valid >= rule["ge"])))
        if "le" in rule:
            masks.append((f"less_than_or_equal_to({rule['le']})", ~(valid <= rule["le"])))
        for check, bad in masks:
            for idx in valid.index[bad]:
                add("Column", col, check, valid.loc[idx], idx)

    return pd.DataFrame(rows, columns=["schema_context", "column", "check", "failure_case", "index"])


# --------------------------------------------------------------------------- public API
def validate_schema(df, output_report_name="schema_validation_errors.csv"):
    print(f"[INFO] Validating schema (Records: {len(df)}, engine: {'pandera' if HAVE_PANDERA else 'pandas'})...")
    failures = _validate_pandera(df) if HAVE_PANDERA else _validate_pandas(df)

    if failures.empty:
        print("[SUCCESS] Schema Validation PASSED. Dataset is clean.")
        return True

    print("[ERROR] Schema Validation FAILED. Corruptions detected.")
    print(failures.head(50).to_string())
    if len(failures) > 50:
        print(f"... {len(failures) - 50} more failure rows")

    os.makedirs("artifacts", exist_ok=True)
    report_path = os.path.join("artifacts", output_report_name)
    failures.to_csv(report_path, index=False)
    print(f"[INFO] Detailed failure report saved to '{report_path}'.")
    return False
